//! Zero-dependency HuggingFace `tokenizer.json` Byte-Level BPE tokenizer
//! (compatible with Inkling-2-Mini's 200,058-token tiktoken-derived `tokenizer.json`).

use serde::Deserialize;
use smalltide_core::{Result, SmallTideError};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Deserialize)]
struct BpeModel {
    vocab: HashMap<String, u32>,
    merges: Vec<MergeEntry>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum MergeEntry {
    Str(String),
    Pair(String, String),
}

#[derive(Deserialize)]
struct HfTokenizerJson {
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    model: BpeModel,
}

pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    merges: HashMap<(String, String), usize>,
    special_tokens: HashMap<String, u32>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

impl BpeTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = fs::read(path)?;
        let parsed: HfTokenizerJson = serde_json::from_slice(&raw)
            .map_err(|e| SmallTideError::InvalidModel(format!("parse tokenizer.json: {e}")))?;

        let mut vocab = parsed.model.vocab;
        let mut special_tokens = HashMap::new();
        for at in &parsed.added_tokens {
            vocab.insert(at.content.clone(), at.id);
            if at.special {
                special_tokens.insert(at.content.clone(), at.id);
            }
        }

        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        let mut id_to_token = vec![String::new(); max_id + 1];
        for (tok, &id) in &vocab {
            id_to_token[id as usize] = tok.clone();
        }

        let mut merges = HashMap::with_capacity(parsed.model.merges.len());
        for (rank, entry) in parsed.model.merges.into_iter().enumerate() {
            match entry {
                MergeEntry::Pair(a, b) => {
                    merges.insert((a, b), rank);
                }
                MergeEntry::Str(s) => {
                    if let Some((a, b)) = s.split_once(' ') {
                        merges.insert((a.to_string(), b.to_string()), rank);
                    }
                }
            }
        }

        let (byte_to_char, char_to_byte) = build_gpt2_byte_unicode();
        Ok(Self {
            vocab,
            id_to_token,
            merges,
            special_tokens,
            byte_to_char,
            char_to_byte,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Encodes `text` (honoring exact special tokens like `<|message_user|>` and `<|endoftext|>`)
    /// into token IDs using byte-level BPE merges.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // Check if a special token occurs at the start or ahead
            let mut next_special: Option<(usize, &str, u32)> = None;
            for (sp, &id) in &self.special_tokens {
                if let Some(pos) = rest.find(sp.as_str()) {
                    let better = match next_special {
                        None => true,
                        Some((bp, bs, _)) => pos < bp || (pos == bp && sp.len() > bs.len()),
                    };
                    if better {
                        next_special = Some((pos, sp.as_str(), id));
                    }
                }
            }
            match next_special {
                Some((0, sp, id)) => {
                    out.push(id);
                    rest = &rest[sp.len()..];
                }
                Some((pos, _, _)) => {
                    self.encode_plain(&rest[..pos], &mut out);
                    rest = &rest[pos..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        // Split cleanly into word-like chunks (keeping leading spaces attached like tiktoken/GPT-2)
        let chunks = split_bpe_chunks(text);
        for chunk in chunks {
            let mut symbols: Vec<String> = chunk
                .as_bytes()
                .iter()
                .map(|&b| self.byte_to_char[b as usize].to_string())
                .collect();
            if symbols.is_empty() {
                continue;
            }
            loop {
                if symbols.len() < 2 {
                    break;
                }
                let mut best_rank = usize::MAX;
                let mut best_idx = None;
                for i in 0..(symbols.len() - 1) {
                    if let Some(&rank) = self.merges.get(&(symbols[i].clone(), symbols[i + 1].clone()))
                    {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = Some(i);
                        }
                    }
                }
                let Some(idx) = best_idx else {
                    break;
                };
                let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
                symbols[idx] = merged;
                symbols.remove(idx + 1);
            }
            for sym in symbols {
                if let Some(&id) = self.vocab.get(&sym) {
                    out.push(id);
                }
            }
        }
    }

    /// Decodes token IDs back into UTF-8 text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = self.id_to_token.get(id as usize) else {
                continue;
            };
            if self.special_tokens.contains_key(tok) {
                bytes.extend_from_slice(tok.as_bytes());
                continue;
            }
            for ch in tok.chars() {
                if let Some(&b) = self.char_to_byte.get(&ch) {
                    bytes.push(b);
                } else {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn split_bpe_chunks(text: &str) -> Vec<&str> {
    let mut chunks = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\n' || b == b'\r' {
            if i > start {
                chunks.push(&text[start..i]);
            }
            let nl_start = i;
            while i < bytes.len() && (bytes[i] == b'\n' || bytes[i] == b'\r') {
                i += 1;
            }
            chunks.push(&text[nl_start..i]);
            start = i;
        } else if b == b' ' && i > start {
            chunks.push(&text[start..i]);
            start = i;
            i += 1;
        } else {
            i += 1;
        }
    }
    if start < bytes.len() {
        chunks.push(&text[start..]);
    }
    chunks
}

fn build_gpt2_byte_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u32> = Vec::with_capacity(256);
    for b in b'!'..=b'~' {
        bs.push(b as u32);
    }
    for b in 0xA1u32..=0xACu32 {
        bs.push(b);
    }
    for b in 0xAEu32..=0xFFu32 {
        bs.push(b);
    }
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut b2c = ['\0'; 256];
    let mut c2b = HashMap::with_capacity(256);
    for (b, c) in bs.into_iter().zip(cs.into_iter()) {
        let ch = char::from_u32(c).unwrap();
        b2c[b as usize] = ch;
        c2b.insert(ch, b as u8);
    }
    (b2c, c2b)
}
