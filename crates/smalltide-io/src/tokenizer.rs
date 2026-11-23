//! Zero-dependency HuggingFace `tokenizer.json` Byte-Level BPE tokenizer
//! (compatible with Inkling-2-Mini's 200,058-token tiktoken-derived `tokenizer.json`).
//!
//! Matches HuggingFace `tokenizers` exactly (checked by `tools/test_tokenizer.py`):
//! - added/special tokens are split out first (longest match at the earliest position);
//! - the o200k-style `Split` pre-tokenizer regex is implemented by hand (`pretokenize`) with the
//!   same alternation order and backtracking semantics, using generated Unicode general
//!   categories (`unicode_cats.rs`);
//! - byte-level mapping, then BPE with `ignore_merges = true` (a pre-token that is itself in the
//!   vocabulary is emitted as one token without merging).

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
    /// `(left_id, right_id) -> (rank, merged_id)`.
    merges: HashMap<(u32, u32), (u32, u32)>,
    special_tokens: HashMap<String, u32>,
    byte_to_char: [char; 256],
    /// Token id of each single byte (byte-level alphabet).
    byte_ids: [u32; 256],
    /// Raw byte sequence of every token (special tokens: their literal text).
    id_bytes: Vec<Vec<u8>>,
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
            let (a, b) = match entry {
                MergeEntry::Pair(a, b) => (a, b),
                MergeEntry::Str(s) => match s.split_once(' ') {
                    Some((a, b)) => (a.to_string(), b.to_string()),
                    None => continue,
                },
            };
            let merged = format!("{a}{b}");
            if let (Some(&ia), Some(&ib), Some(&im)) =
                (vocab.get(&a), vocab.get(&b), vocab.get(&merged))
            {
                merges.entry((ia, ib)).or_insert((rank as u32, im));
            }
        }

        let (byte_to_char, char_to_byte) = build_gpt2_byte_unicode();
        let mut byte_ids = [0u32; 256];
        for b in 0..256usize {
            let ch = byte_to_char[b].to_string();
            byte_ids[b] = *vocab.get(&ch).ok_or_else(|| {
                SmallTideError::InvalidModel(format!("byte-level token for byte {b} missing"))
            })?;
        }
        let mut id_bytes = vec![Vec::new(); id_to_token.len()];
        for (id, tok) in id_to_token.iter().enumerate() {
            if special_tokens.contains_key(tok) {
                id_bytes[id] = tok.as_bytes().to_vec();
            } else {
                let mut v = Vec::with_capacity(tok.len());
                for ch in tok.chars() {
                    match char_to_byte.get(&ch) {
                        Some(&b) => v.push(b),
                        None => {
                            let mut buf = [0u8; 4];
                            v.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                    }
                }
                id_bytes[id] = v;
            }
        }
        Ok(Self {
            vocab,
            id_to_token,
            merges,
            special_tokens,
            byte_to_char,
            byte_ids,
            id_bytes,
        })
    }

    /// Raw bytes of one token (a token may end in the middle of a UTF-8 character; use
    /// `Utf8Stream` to decode a token stream incrementally).
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.id_bytes.get(id as usize).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// True for added special tokens such as `<|end_message|>`.
    pub fn is_special(&self, id: u32) -> bool {
        self.id_to_token
            .get(id as usize)
            .is_some_and(|t| self.special_tokens.contains_key(t))
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
        self.encode_with(text, None)
    }

    /// `encode` for a prompt whose caller-supplied text carries `guard` characters: a special
    /// token's text with a guard inside it is plain text, and the guards themselves are dropped
    /// before the plain text is tokenized. A web page that quotes `<|end_message|>` then reaches
    /// the model as those characters, not as the control token.
    pub fn encode_guarded(&self, text: &str, guard: char) -> Vec<u32> {
        self.encode_with(text, Some(guard))
    }

    fn encode_with(&self, text: &str, guard: Option<char>) -> Vec<u32> {
        let plain = |seg: &str, out: &mut Vec<u32>| match guard {
            Some(g) if seg.contains(g) => {
                let kept: String = seg.chars().filter(|&c| c != g).collect();
                self.encode_plain(&kept, out);
            }
            _ => self.encode_plain(seg, out),
        };
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
                    plain(&rest[..pos], &mut out);
                    rest = &rest[pos..];
                }
                None => {
                    plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        let mut mapped = String::new();
        for piece in pretokenize(text) {
            // ignore_merges: a pre-token that is itself a vocabulary entry is one token.
            mapped.clear();
            for &b in piece.as_bytes() {
                mapped.push(self.byte_to_char[b as usize]);
            }
            if let Some(&id) = self.vocab.get(&mapped) {
                out.push(id);
                continue;
            }
            let mut syms: Vec<u32> = piece.as_bytes().iter().map(|&b| self.byte_ids[b as usize]).collect();
            while syms.len() >= 2 {
                let mut best: Option<(u32, usize, u32)> = None; // (rank, idx, merged)
                for i in 0..syms.len() - 1 {
                    if let Some(&(rank, merged)) = self.merges.get(&(syms[i], syms[i + 1])) {
                        if best.map_or(true, |(r, _, _)| rank < r) {
                            best = Some((rank, i, merged));
                        }
                    }
                }
                let Some((_, idx, merged)) = best else { break };
                syms[idx] = merged;
                syms.remove(idx + 1);
            }
            out.extend_from_slice(&syms);
        }
    }

    /// Decodes token IDs back into UTF-8 text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(self.token_bytes(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Incremental UTF-8 decoder for streamed tokens: holds back an incomplete trailing character
/// until the next token completes it (a token can end in the middle of a multi-byte character).
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    /// Appends bytes and returns the longest valid UTF-8 prefix now available.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) => match e.error_len() {
                // Incomplete sequence at the end: keep it for the next push.
                None => e.valid_up_to(),
                // Invalid bytes in the middle: decode lossily up to the end of the bad sequence.
                Some(_) => {
                    let s = String::from_utf8_lossy(&self.pending).into_owned();
                    self.pending.clear();
                    return s;
                }
            },
        };
        let s = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
        self.pending.drain(..valid);
        s
    }

    /// Flushes any held-back bytes (lossily).
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

// --- o200k-style pre-tokenizer ------------------------------------------------------------
//
// [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
// |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
// |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+

#[derive(Clone, Copy, PartialEq, Eq)]
enum Cat {
    Other,
    Lu,
    Ll,
    Lt,
    Lm,
    Lo,
    M,
    N,
}

fn cat(ch: char) -> Cat {
    let c = ch as u32;
    if c < 0x80 {
        return match ch {
            'A'..='Z' => Cat::Lu,
            'a'..='z' => Cat::Ll,
            '0'..='9' => Cat::N,
            _ => Cat::Other,
        };
    }
    let r = &crate::unicode_cats::RANGES;
    let idx = r.partition_point(|&(_, hi, _)| hi < c);
    if idx < r.len() && r[idx].0 <= c {
        match r[idx].2 {
            1 => Cat::Lu,
            2 => Cat::Ll,
            3 => Cat::Lt,
            4 => Cat::Lm,
            5 => Cat::Lo,
            6 => Cat::M,
            7 => Cat::N,
            _ => Cat::Other,
        }
    } else {
        Cat::Other
    }
}

#[inline]
fn is_letter(c: Cat) -> bool {
    matches!(c, Cat::Lu | Cat::Ll | Cat::Lt | Cat::Lm | Cat::Lo)
}
#[inline]
fn is_upperish(c: Cat) -> bool {
    matches!(c, Cat::Lu | Cat::Lt | Cat::Lm | Cat::Lo | Cat::M)
}
#[inline]
fn is_lowerish(c: Cat) -> bool {
    matches!(c, Cat::Ll | Cat::Lm | Cat::Lo | Cat::M)
}

struct Chars {
    ch: Vec<char>,
    cat: Vec<Cat>,
}

impl Chars {
    fn ws(&self, i: usize) -> bool {
        self.ch[i].is_whitespace()
    }
    fn crlf(&self, i: usize) -> bool {
        self.ch[i] == '\r' || self.ch[i] == '\n'
    }
    /// `[^\r\n\p{L}\p{N}]`
    fn prefix_ok(&self, i: usize) -> bool {
        !self.crlf(i) && !is_letter(self.cat[i]) && self.cat[i] != Cat::N
    }
    /// `[^\s\p{L}\p{N}]`
    fn punct(&self, i: usize) -> bool {
        !self.ws(i) && !is_letter(self.cat[i]) && self.cat[i] != Cat::N
    }
    fn scan(&self, mut i: usize, f: impl Fn(&Self, usize) -> bool) -> usize {
        while i < self.ch.len() && f(self, i) {
            i += 1;
        }
        i
    }
    /// `(?i:'s|'t|'re|'ve|'m|'ll|'d)?`
    fn contraction(&self, e: usize) -> usize {
        let n = self.ch.len();
        if e + 1 < n && self.ch[e] == '\'' {
            let a = self.ch[e + 1].to_ascii_lowercase();
            if matches!(a, 's' | 't' | 'm' | 'd') {
                return e + 2;
            }
            if e + 2 < n {
                let b = self.ch[e + 2].to_ascii_lowercase();
                if matches!((a, b), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                    return e + 3;
                }
            }
        }
        e
    }

    fn match_at(&self, i: usize) -> usize {
        let n = self.ch.len();
        let starts = |this: &Self| {
            let mut v = Vec::with_capacity(2);
            if this.prefix_ok(i) && i + 1 <= n {
                v.push(i + 1);
            }
            v.push(i);
            v
        };
        // Alt 1: P? U* L+ C?
        for start in starts(self) {
            let j = self.scan(start, |s, k| is_upperish(s.cat[k]));
            let e = if j < n && is_lowerish(self.cat[j]) {
                Some(self.scan(j, |s, k| is_lowerish(s.cat[k])))
            } else {
                (start..j).rev().find(|&k| is_lowerish(self.cat[k])).map(|k| k + 1)
            };
            if let Some(e) = e {
                return self.contraction(e);
            }
        }
        // Alt 2: P? U+ L* C?
        for start in starts(self) {
            let j = self.scan(start, |s, k| is_upperish(s.cat[k]));
            if j > start {
                let e = self.scan(j, |s, k| is_lowerish(s.cat[k]));
                return self.contraction(e);
            }
        }
        // Alt 3: \p{N}{1,3}
        if self.cat[i] == Cat::N {
            let j = self.scan(i, |s, k| s.cat[k] == Cat::N);
            return (i + 3).min(j);
        }
        // Alt 4: ' '?[^\s\p{L}\p{N}]+[\r\n/]*
        let start = if self.ch[i] == ' ' && i + 1 < n && self.punct(i + 1) { i + 1 } else { i };
        if self.punct(start) {
            let j = self.scan(start, |s, k| s.punct(k));
            return self.scan(j, |s, k| s.crlf(k) || s.ch[k] == '/');
        }
        // Whitespace alternatives
        if self.ws(i) {
            let j = self.scan(i, |s, k| s.ws(k));
            // Alt 5: \s*[\r\n]+  -> ends right after the last CR/LF of the run
            if let Some(m) = (i..j).rev().find(|&k| self.crlf(k)) {
                return m + 1;
            }
            // Alt 6: \s+(?!\S)
            if j == n {
                return j;
            }
            if j - i >= 2 {
                return j - 1;
            }
            // Alt 7: \s+
            return j;
        }
        i + 1
    }
}

/// Splits plain text (no special tokens) into pre-tokens exactly like the tokenizer.json regex.
pub fn pretokenize(text: &str) -> Vec<&str> {
    let idx: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
    let ch: Vec<char> = text.chars().collect();
    let cat: Vec<Cat> = ch.iter().map(|&c| cat(c)).collect();
    let cs = Chars { ch, cat };
    let n = cs.ch.len();
    let byte_at = |k: usize| if k < n { idx[k] } else { text.len() };
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < n {
        let e = cs.match_at(i).max(i + 1);
        out.push(&text[byte_at(i)..byte_at(e)]);
        i = e;
    }
    out
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
