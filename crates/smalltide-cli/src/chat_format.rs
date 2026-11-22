//! Inkling-2-Mini chat formatting and answer parsing for the OpenAI server.
//!
//! * [`render_chat`] is a byte-exact port of the checkpoint's `chat_template.jinja` as rendered by
//!   transformers' `apply_chat_template` (sandboxed jinja2, `tojson` = `json.dumps(ensure_ascii=
//!   False)`); verified by `tools/test_chat_template.py`. It keeps the template's value rules that
//!   decide bytes (Python truthiness, `is defined`, `str()` of non-strings, `==` on ids, the
//!   effort default 0.9) and refuses where the template raises. OpenAI clients send tool-call
//!   `arguments` as JSON strings; they are parsed (order-preserving) first, which is the
//!   template's "canonicalize upstream" step.
//! * [`OJson`] is a small order-preserving JSON value whose serializer reproduces Python's
//!   `json.dumps` (escapes, float repr, separators, `sort_keys`), so key order inside tool
//!   arguments survives exactly as the client sent it.
//! * [`AnswerParser`] routes generated token ids: `<|message_model|>` header (a tool call's name)
//!   `<|content_*|>` body `<|end_message|>`; thinking -> reasoning, invoke_tool_json -> tool call,
//!   every other content kind -> content. UTF-8 is decoded incrementally across tokens.

use smalltide_io::tokenizer::Utf8Stream;
use std::fmt::Write as _;

// ------------------------------------------------------------------------------------------
// Ordered JSON
// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum OJson {
    Null,
    Bool(bool),
    /// Integer literal, normalized like Python's `int` (`-0` -> `0`).
    Int(String),
    Float(f64),
    Str(String),
    Arr(Vec<OJson>),
    /// Insertion-ordered object (duplicate keys: first position, last value — like `dict`).
    Obj(Vec<(String, OJson)>),
}

impl OJson {
    pub fn get(&self, key: &str) -> Option<&OJson> {
        match self {
            OJson::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            OJson::Str(s) => Some(s),
            _ => None,
        }
    }
    /// Python truthiness.
    pub fn truthy(&self) -> bool {
        match self {
            OJson::Null => false,
            OJson::Bool(b) => *b,
            OJson::Int(s) => s.trim_start_matches('-').chars().any(|c| c != '0'),
            OJson::Float(f) => *f != 0.0,
            OJson::Str(s) => !s.is_empty(),
            OJson::Arr(a) => !a.is_empty(),
            OJson::Obj(o) => !o.is_empty(),
        }
    }
    fn as_f64(&self) -> Option<f64> {
        match self {
            OJson::Int(s) => s.parse::<f64>().ok(),
            OJson::Float(f) => Some(*f),
            OJson::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
    /// Python `==` (numbers compare by value across int/float/bool).
    pub fn py_eq(&self, other: &OJson) -> bool {
        match (self, other) {
            (OJson::Str(a), OJson::Str(b)) => a == b,
            (OJson::Null, OJson::Null) => true,
            (OJson::Arr(a), OJson::Arr(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.py_eq(y)),
            (OJson::Obj(a), OJson::Obj(b)) => {
                a.len() == b.len() && a.iter().all(|(k, v)| other.get(k).is_some_and(|w| v.py_eq(w)))
            }
            _ => match (self.as_f64(), other.as_f64()) {
                (Some(x), Some(y)) => match (self, other) {
                    (OJson::Int(a), OJson::Int(b)) => a == b,
                    _ => x == y,
                },
                _ => false,
            },
        }
    }
    /// Python `str()` as jinja's `~` applies it.
    pub fn py_str(&self) -> String {
        match self {
            OJson::Str(s) => s.clone(),
            OJson::Null => "None".into(),
            OJson::Bool(b) => if *b { "True".into() } else { "False".into() },
            OJson::Int(s) => s.clone(),
            OJson::Float(f) => py_float(*f),
            OJson::Arr(a) => {
                let parts: Vec<String> = a.iter().map(|x| x.py_repr()).collect();
                format!("[{}]", parts.join(", "))
            }
            OJson::Obj(o) => {
                let parts: Vec<String> =
                    o.iter().map(|(k, v)| format!("{}: {}", py_repr_str(k), v.py_repr())).collect();
                format!("{{{}}}", parts.join(", "))
            }
        }
    }
    fn py_repr(&self) -> String {
        match self {
            OJson::Str(s) => py_repr_str(s),
            other => other.py_str(),
        }
    }
}

fn py_repr_str(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

/// Python `repr(float)`.
pub fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let e = format!("{:e}", f); // shortest round-trip digits, e.g. "-1.2345e-7"
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let e = exp as usize;
            if digits.len() <= e + 1 {
                out.push_str(&digits);
                out.extend(std::iter::repeat('0').take(e + 1 - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..e + 1]);
                out.push('.');
                out.push_str(&digits[e + 1..]);
            }
        } else {
            out.push_str("0.");
            out.extend(std::iter::repeat('0').take((-exp - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    out
}

/// Python `json.dumps(v, ensure_ascii=False, separators=..., sort_keys=...)`.
pub fn py_dumps(v: &OJson, compact: bool, sort_keys: bool) -> String {
    let mut out = String::new();
    dump_into(&mut out, v, compact, sort_keys);
    out
}

fn dump_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn dump_into(out: &mut String, v: &OJson, compact: bool, sort_keys: bool) {
    let (item_sep, kv_sep) = if compact { (",", ":") } else { (", ", ": ") };
    match v {
        OJson::Null => out.push_str("null"),
        OJson::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        OJson::Int(s) => out.push_str(s),
        OJson::Float(f) => out.push_str(&py_float(*f)),
        OJson::Str(s) => dump_str(out, s),
        OJson::Arr(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                dump_into(out, x, compact, sort_keys);
            }
            out.push(']');
        }
        OJson::Obj(o) => {
            let mut items: Vec<&(String, OJson)> = o.iter().collect();
            if sort_keys {
                items.sort_by(|a, b| a.0.cmp(&b.0));
            }
            out.push('{');
            for (i, (k, x)) in items.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                dump_str(out, k);
                out.push_str(kv_sep);
                dump_into(out, x, compact, sort_keys);
            }
            out.push('}');
        }
    }
}

pub fn parse_json(s: &str) -> Result<OJson, String> {
    parse_json_with(s, false)
}

/// Lenient JSON parser used only as a fallback for model-emitted tool calls after strict
/// `parse_json` fails. Tolerates non-standard string escapes (`\'`, `\s`, `\U`, non-hex `\u`),
/// trailing commas, and unescaped inner quotes (e.g. HTML attributes inside large payloads).
pub fn parse_json_lenient(s: &str) -> Result<OJson, String> {
    parse_json_with(s, true)
}

fn parse_json_with(s: &str, lenient: bool) -> Result<OJson, String> {
    let b = s.as_bytes();
    let mut p = Parser { b, i: 0, lenient };
    p.ws();
    let v = p.value(0)?;
    p.ws();
    if p.i != b.len() {
        return Err(format!("trailing data at byte {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    lenient: bool,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    /// True when the rest of the input is at least one `}` / `]` and whitespace, nothing else.
    fn only_closers_follow(&self) -> bool {
        let rest = &self.b[self.i.min(self.b.len())..];
        rest.iter().any(|c| matches!(c, b'}' | b']'))
            && rest.iter().all(|c| matches!(c, b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r'))
    }
    fn err<T>(&self, m: &str) -> Result<T, String> {
        Err(format!("invalid JSON at byte {}: {m}", self.i))
    }
    fn lit(&mut self, s: &str, v: OJson) -> Result<OJson, String> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(v)
        } else {
            self.err("bad literal")
        }
    }
    fn value(&mut self, depth: usize) -> Result<OJson, String> {
        if depth > 256 {
            return self.err("nesting too deep");
        }
        let Some(&c) = self.b.get(self.i) else { return self.err("unexpected end") };
        match c {
            b'n' => self.lit("null", OJson::Null),
            b't' => self.lit("true", OJson::Bool(true)),
            b'f' => self.lit("false", OJson::Bool(false)),
            b'"' => self.string().map(OJson::Str),
            b'[' => {
                self.i += 1;
                let mut v = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(OJson::Arr(v));
                }
                loop {
                    self.ws();
                    v.push(self.value(depth + 1)?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => {
                            self.i += 1;
                            if self.lenient {
                                self.ws();
                                if self.b.get(self.i) == Some(&b']') {
                                    self.i += 1;
                                    return Ok(OJson::Arr(v));
                                }
                            }
                        }
                        Some(b']') => {
                            self.i += 1;
                            return Ok(OJson::Arr(v));
                        }
                        _ => return self.err("expected , or ]"),
                    }
                }
            }
            b'{' => {
                self.i += 1;
                let mut kv: Vec<(String, OJson)> = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(OJson::Obj(kv));
                }
                loop {
                    self.ws();
                    if self.b.get(self.i) != Some(&b'"') {
                        return self.err("expected key");
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') {
                        return self.err("expected :");
                    }
                    self.i += 1;
                    self.ws();
                    let v = self.value(depth + 1)?;
                    if let Some(slot) = kv.iter_mut().find(|(kk, _)| *kk == k) {
                        slot.1 = v;
                    } else {
                        kv.push((k, v));
                    }
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => {
                            self.i += 1;
                            if self.lenient {
                                self.ws();
                                if self.b.get(self.i) == Some(&b'}') {
                                    self.i += 1;
                                    return Ok(OJson::Obj(kv));
                                }
                            }
                        }
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(OJson::Obj(kv));
                        }
                        _ => return self.err("expected , or }"),
                    }
                }
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => self.err("unexpected character"),
        }
    }
    fn number(&mut self) -> Result<OJson, String> {
        let st = self.i;
        if self.b[self.i] == b'-' {
            self.i += 1;
        }
        let mut is_float = false;
        while self.i < self.b.len() {
            match self.b[self.i] {
                b'0'..=b'9' => {}
                b'.' | b'e' | b'E' | b'+' | b'-' => is_float = true,
                _ => break,
            }
            self.i += 1;
        }
        let lex = std::str::from_utf8(&self.b[st..self.i]).unwrap();
        if is_float {
            lex.parse::<f64>().map(OJson::Float).or_else(|_| self.err("bad number"))
        } else {
            if lex == "-" || (lex.len() > 1 && lex.trim_start_matches('-').starts_with('0') && lex.trim_start_matches('-').len() > 1) {
                return self.err("bad integer");
            }
            let norm = if lex.trim_start_matches('-') == "0" { "0".to_string() } else { lex.to_string() };
            Ok(OJson::Int(norm))
        }
    }
    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() {
            return self.err("short \\u escape");
        }
        let h = std::str::from_utf8(&self.b[self.i..self.i + 4]).map_err(|e| e.to_string())?;
        let v = u32::from_str_radix(h, 16).or_else(|_| self.err("bad \\u escape"))?;
        self.i += 4;
        Ok(v)
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            let st = self.i;
            while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' {
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.b[st..self.i]).map_err(|e| e.to_string())?);
            match self.b.get(self.i) {
                None => return self.err("unterminated string"),
                Some(b'"') => {
                    if self.lenient {
                        let mut k = self.i + 1;
                        while k < self.b.len() && matches!(self.b[k], b' ' | b'\t' | b'\n' | b'\r') {
                            k += 1;
                        }
                        let looks_closing = matches!(
                            self.b.get(k),
                            None | Some(b',') | Some(b'}') | Some(b']') | Some(b':')
                        );
                        if !looks_closing {
                            out.push('"');
                            self.i += 1;
                            continue;
                        }
                    }
                    self.i += 1;
                    return Ok(out);
                }
                _ => {
                    self.i += 1;
                    let Some(&e) = self.b.get(self.i) else { return self.err("bad escape") };
                    self.i += 1;
                    match e {
                        // Lenient: `\"` followed only by closing brackets up to the end of the
                        // input is the string's closing quote, escaped by mistake (otherwise the
                        // string, and the whole call, would never terminate).
                        b'"' if self.lenient && self.only_closers_follow() => return Ok(out),
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => match self.hex4() {
                            Ok(hi) => {
                                let cp = if (0xD800..0xDC00).contains(&hi)
                                    && self.b[self.i..].starts_with(b"\\u")
                                {
                                    self.i += 2;
                                    let lo = self.hex4()?;
                                    if (0xDC00..0xE000).contains(&lo) {
                                        0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                                    } else {
                                        0xFFFD
                                    }
                                } else {
                                    hi
                                };
                                out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                            }
                            Err(_) if self.lenient => {
                                out.push('\\');
                                out.push('u');
                            }
                            Err(err) => return Err(err),
                        },
                        b'\'' if self.lenient => out.push('\''),
                        _ if self.lenient => {
                            out.push('\\');
                            self.i -= 1;
                        }
                        _ => return self.err("bad escape"),
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// chat_template.jinja
// ------------------------------------------------------------------------------------------

const EFFORTS: [(&str, f64); 7] = [
    ("none", 0.0),
    ("minimal", 0.1),
    ("low", 0.2),
    ("medium", 0.7),
    ("high", 0.9),
    ("xhigh", 0.99),
    ("max", 0.99),
];

fn effort_block(effort: Option<&OJson>) -> Result<String, String> {
    let num = match effort {
        None | Some(OJson::Null) => 0.9,
        Some(OJson::Str(s)) => {
            let key = s.trim();
            match EFFORTS.iter().find(|(k, _)| *k == key) {
                Some((_, v)) => *v,
                None => return Err(format!("Unknown reasoning_effort: {s}")),
            }
        }
        Some(v) => v.as_f64().ok_or_else(|| "reasoning_effort must be a number or string".to_string())?,
    };
    if !(0.0..=0.99).contains(&num) {
        return Err("reasoning_effort must be in [0.0, 0.99]".into());
    }
    let n = if num == 0.0 { "0".to_string() } else { py_float(num) };
    Ok(format!("<|message_system|><|content_text|>Thinking effort level: {n}<|end_message|>"))
}

fn tool_declare(tools: &[OJson]) -> Result<String, String> {
    let mut specs = Vec::with_capacity(tools.len());
    for tool in tools {
        let fn_ = tool.get("function").unwrap_or(tool);
        let pick = |k: &str, default: OJson| -> OJson {
            match fn_.get(k) {
                Some(v) if v.truthy() => v.clone(),
                _ => default,
            }
        };
        let name = fn_.get("name").cloned().ok_or("tool is missing a function name")?;
        let ty = match tool.get("type") {
            Some(v) if v.truthy() => v.clone(),
            _ => OJson::Str("function".into()),
        };
        specs.push(OJson::Obj(vec![
            ("description".into(), pick("description", OJson::Str(String::new()))),
            ("name".into(), name),
            ("parameters".into(), pick("parameters", OJson::Obj(Vec::new()))),
            ("type".into(), ty),
        ]));
    }
    Ok(format!(
        "<|message_system|>tool_declare<|content_xml|>{}<|end_message|>",
        py_dumps(&OJson::Arr(specs), true, true)
    ))
}

/// Parses OpenAI-style string `arguments` into objects (the template refuses strings).
fn canonicalize_messages(messages: &[OJson]) -> Result<Vec<OJson>, String> {
    let mut out = messages.to_vec();
    for m in &mut out {
        let OJson::Obj(kv) = m else { continue };
        for (k, v) in kv.iter_mut() {
            if k != "tool_calls" {
                continue;
            }
            let OJson::Arr(calls) = v else { continue };
            for tc in calls.iter_mut() {
                let OJson::Obj(tkv) = tc else { continue };
                for (tk, tv) in tkv.iter_mut() {
                    if tk != "function" {
                        continue;
                    }
                    let OJson::Obj(fkv) = tv else { continue };
                    for (fk, fv) in fkv.iter_mut() {
                        if fk == "arguments" {
                            if let OJson::Str(s) = fv {
                                if !s.trim().is_empty() {
                                    *fv = parse_json(s)
                                        .map_err(|e| format!("tool call arguments are not valid JSON: {e}"))?;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// What the chat request renders to.
pub struct RenderedChat {
    pub prompt: String,
}

/// Renders `messages`/`tools`/`reasoning_effort` exactly as `chat_template.jinja` does.
pub fn render_chat(
    messages: &[OJson],
    tools: Option<&OJson>,
    reasoning_effort: Option<&OJson>,
) -> Result<RenderedChat, String> {
    let messages = canonicalize_messages(messages)?;
    let role_token = |r: &str| match r {
        "user" => Some("<|message_user|>"),
        "assistant" => Some("<|message_model|>"),
        "system" => Some("<|message_system|>"),
        "tool" => Some("<|message_tool|>"),
        _ => None,
    };
    let mut out = String::new();
    if let Some(t) = tools {
        if t.truthy() {
            let OJson::Arr(list) = t else { return Err("tools must be a list".into()) };
            out.push_str(&tool_declare(list)?);
        }
    }
    let mut effort_emitted = false;
    for m in &messages {
        let role_v = m.get("role").ok_or("message without role")?;
        let role = role_v.as_str().unwrap_or("");
        let Some(rt) = role_token(role) else {
            return Err(format!("Unknown message role: {}", role_v.py_str()));
        };
        if !effort_emitted && role != "system" {
            out.push_str(&effort_block(reasoning_effort)?);
            effort_emitted = true;
        }
        let mut header = rt.to_string();
        if role == "tool" {
            if let Some(n) = m.get("name").filter(|n| n.truthy()) {
                header.push_str(&n.py_str());
            } else if let Some(id) = m.get("tool_call_id").filter(|v| v.truthy()) {
                for prev in &messages {
                    if prev.get("role").and_then(OJson::as_str) != Some("assistant") {
                        continue;
                    }
                    let Some(OJson::Arr(calls)) = prev.get("tool_calls").filter(|v| v.truthy()) else {
                        continue;
                    };
                    for tc in calls {
                        let fname = tc.get("function").and_then(|f| f.get("name"));
                        if let (Some(tid), Some(fname)) = (tc.get("id"), fname) {
                            if tid.py_eq(id) {
                                header.push_str(&fname.py_str());
                            }
                        }
                    }
                }
            }
        } else if role == "assistant" {
            if let Some(n) = m.get("name").filter(|n| n.truthy()) {
                header.push_str(&n.py_str());
            }
        }
        let is_asst = role == "assistant";
        let tool_calls = m.get("tool_calls").filter(|v| v.truthy());
        let reasoning = m.get("reasoning_content").filter(|v| v.truthy());
        if is_asst {
            if let Some(r) = reasoning {
                let _ = write!(out, "{header}<|content_thinking|>{}<|end_message|>", r.py_str());
            }
        }
        match m.get("content") {
            Some(OJson::Str(c)) => {
                if !c.is_empty() || !(is_asst && (tool_calls.is_some() || reasoning.is_some())) {
                    let _ = write!(out, "{header}<|content_text|>{c}<|end_message|>");
                }
            }
            Some(c) if c.truthy() => {
                let OJson::Arr(parts) = c else {
                    return Err("message content must be a string or a list of parts".into());
                };
                for part in parts {
                    if let OJson::Str(s) = part {
                        let _ = write!(out, "{header}<|content_text|>{s}<|end_message|>");
                        continue;
                    }
                    let ty = part.get("type");
                    let is_text = match ty {
                        None => true,
                        Some(t) => matches!(t.as_str(), Some("text") | Some("input_text")),
                    };
                    if is_text {
                        let t = part.get("text").and_then(OJson::as_str).unwrap_or("");
                        let _ = write!(out, "{header}<|content_text|>{t}<|end_message|>");
                    } else {
                        match ty.and_then(OJson::as_str) {
                            Some("image" | "input_image" | "image_url") => {
                                return Err("image input is not supported by this server (no vision tower loaded)".into())
                            }
                            Some("audio" | "input_audio" | "audio_url") => {
                                return Err("Inkling 2 does not accept audio input".into())
                            }
                            _ => {
                                return Err(format!(
                                    "Unsupported content part type: {}",
                                    ty.map(|t| t.py_str()).unwrap_or_default()
                                ))
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        if is_asst {
            if let Some(OJson::Arr(calls)) = tool_calls {
                for tc in calls {
                    let fn_ = tc.get("function").ok_or("tool call without function")?;
                    let Some(name) = fn_.get("name").and_then(OJson::as_str) else {
                        return Err("tool call function name must be a string".into());
                    };
                    if let Some(n) = m.get("name").filter(|n| n.truthy()) {
                        if n.as_str() != Some(name) {
                            return Err(format!(
                                "assistant name {} does not match its tool call {name}",
                                n.py_str()
                            ));
                        }
                    }
                    let empty = OJson::Obj(Vec::new());
                    let args = match fn_.get("arguments") {
                        Some(a) if a.truthy() => a,
                        _ => &empty,
                    };
                    if matches!(args, OJson::Str(_)) {
                        return Err("tool call arguments must be a parsed object, not a JSON string; canonicalize upstream".into());
                    }
                    let OJson::Obj(akv) = args else {
                        return Err("tool call arguments must be an object".into());
                    };
                    let mut sorted: Vec<&(String, OJson)> = akv.iter().collect();
                    sorted.sort_by(|a, b| a.0.cmp(&b.0));
                    let _ = write!(
                        out,
                        "<|message_model|>{name}<|content_invoke_tool_json|>{{\"name\":{},\"args\":{{",
                        py_dumps(&OJson::Str(name.to_string()), true, true)
                    );
                    for (i, (k, v)) in sorted.into_iter().enumerate() {
                        if i > 0 {
                            out.push(',');
                        }
                        out.push_str(&py_dumps(&OJson::Str(k.clone()), false, false));
                        out.push(':');
                        out.push_str(&py_dumps(v, true, false));
                    }
                    out.push_str("}}<|end_message|>");
                }
            }
            out.push_str("<|content_model_end_sampling|>");
        }
    }
    if !effort_emitted {
        out.push_str(&effort_block(reasoning_effort)?);
    }
    Ok(RenderedChat { prompt: out })
}

// ------------------------------------------------------------------------------------------
// Answer parsing
// ------------------------------------------------------------------------------------------

pub const TOK_MESSAGE_USER: u32 = 200_000;
pub const TOK_MESSAGE_MODEL: u32 = 200_001;
pub const TOK_MESSAGE_SYSTEM: u32 = 200_002;
pub const TOK_MESSAGE_TOOL: u32 = 200_003;
pub const TOK_CONTENT_TEXT: u32 = 200_004;
pub const TOK_CONTENT_THINKING: u32 = 200_008;
pub const TOK_END_MESSAGE: u32 = 200_010;
pub const TOK_INVOKE_TOOL_JSON: u32 = 200_049;

#[derive(Debug, Clone, PartialEq)]
pub enum AnswerEvent {
    Text(String),
    Thinking(String),
    ToolCall { name: String, arguments: String },
    /// The model started another turn's header (user/system/tool): the answer is over.
    EndOfAnswer,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Body {
    Idle,
    Header,
    Text,
    Thinking,
    ToolJson,
}

pub struct AnswerParser {
    body: Body,
    header: Vec<u8>,
    tool_buf: Vec<u8>,
    text: Utf8Stream,
    thinking: Utf8Stream,
    pub tool_calls: usize,
}

impl AnswerParser {
    pub fn new() -> Self {
        Self {
            body: Body::Idle,
            header: Vec::new(),
            tool_buf: Vec::new(),
            text: Utf8Stream::default(),
            thinking: Utf8Stream::default(),
            tool_calls: 0,
        }
    }

    /// Starts inside a forced tool call (prompt already ends with
    /// `<|message_model|>NAME<|content_invoke_tool_json|>`).
    pub fn new_forced_tool(name: &str) -> Self {
        let mut p = Self::new();
        p.header = name.as_bytes().to_vec();
        p.body = Body::ToolJson;
        p
    }

    fn flush_streams(&mut self, out: &mut Vec<AnswerEvent>) {
        let t = self.text.finish();
        if !t.is_empty() {
            out.push(AnswerEvent::Text(t));
        }
        let t = self.thinking.finish();
        if !t.is_empty() {
            out.push(AnswerEvent::Thinking(t));
        }
    }

    fn finish_tool(&mut self, out: &mut Vec<AnswerEvent>) {
        let raw = String::from_utf8_lossy(&self.tool_buf).into_owned();
        let header = String::from_utf8_lossy(&self.header).into_owned();
        self.tool_buf.clear();
        let parsed = match parse_json(&raw) {
            Ok(v) => Ok(v),
            Err(strict_err) => match parse_json_lenient(&raw) {
                Ok(v) => {
                    eprintln!(
                        "[tool-parse] lenient JSON fallback recovered call (header={header:?}, strict error: {strict_err})"
                    );
                    Ok(v)
                }
                Err(lenient_err) => {
                    let head: String = raw.chars().take(120).collect();
                    let tail: String = raw
                        .chars()
                        .rev()
                        .take(120)
                        .collect::<String>()
                        .chars()
                        .rev()
                        .collect();
                    eprintln!(
                        "[tool-parse] failed (header={header:?}, len={}): strict={strict_err}; lenient={lenient_err} | head={head:?} | tail={tail:?}",
                        raw.len()
                    );
                    Err(())
                }
            },
        };
        match parsed {
            Ok(v)
                if matches!(&v, OJson::Obj(_))
                    && (v.get("args").is_some()
                        || v.get("arguments").is_some()
                        || v.get("name").is_some()
                        || !header.is_empty()) =>
            {
                let name = v
                    .get("name")
                    .and_then(OJson::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| header.clone());
                let args = v
                    .get("args")
                    .or_else(|| v.get("arguments"))
                    .cloned()
                    .unwrap_or(OJson::Obj(Vec::new()));
                self.tool_calls += 1;
                out.push(AnswerEvent::ToolCall { name, arguments: py_dumps(&args, true, false) });
            }
            // Not a parsable call: surface the body as content instead of losing it.
            _ => out.push(AnswerEvent::Text(raw)),
        }
    }

    /// Feeds one generated token.
    pub fn feed(&mut self, id: u32, bytes: &[u8], is_special: bool) -> Vec<AnswerEvent> {
        let mut out = Vec::new();
        if is_special {
            match id {
                TOK_MESSAGE_MODEL => {
                    if self.body == Body::ToolJson && !self.tool_buf.is_empty() {
                        self.finish_tool(&mut out);
                    }
                    self.flush_streams(&mut out);
                    self.body = Body::Header;
                    self.header.clear();
                }
                TOK_CONTENT_THINKING => self.body = Body::Thinking,
                TOK_INVOKE_TOOL_JSON => {
                    self.body = Body::ToolJson;
                    self.tool_buf.clear();
                }
                TOK_END_MESSAGE => {
                    if self.body == Body::ToolJson {
                        self.finish_tool(&mut out);
                    }
                    self.flush_streams(&mut out);
                    self.body = Body::Idle;
                }
                TOK_MESSAGE_USER | TOK_MESSAGE_SYSTEM | TOK_MESSAGE_TOOL => {
                    if self.body == Body::ToolJson && !self.tool_buf.is_empty() {
                        self.finish_tool(&mut out);
                    }
                    self.flush_streams(&mut out);
                    out.push(AnswerEvent::EndOfAnswer);
                }
                // If a tool-name header was emitted before `<|content_text|>` (missing
                // `<|content_invoke_tool_json|>`), buffer as a candidate tool call; non-JSON
                // bodies still fall back cleanly to `AnswerEvent::Text`.
                _ if id == TOK_CONTENT_TEXT && self.body == Body::Header && !self.header.is_empty() => {
                    self.body = Body::ToolJson;
                    self.tool_buf.clear();
                }
                // `<|content_text|>` and every other content kind: the body is content.
                _ if id == TOK_CONTENT_TEXT || self.body == Body::Header => self.body = Body::Text,
                _ => {}
            }
            return out;
        }
        match self.body {
            Body::Header => self.header.extend_from_slice(bytes),
            Body::ToolJson => self.tool_buf.extend_from_slice(bytes),
            Body::Thinking => {
                let s = self.thinking.push(bytes);
                if !s.is_empty() {
                    out.push(AnswerEvent::Thinking(s));
                }
            }
            Body::Text | Body::Idle => {
                let s = self.text.push(bytes);
                if !s.is_empty() {
                    out.push(AnswerEvent::Text(s));
                }
            }
        }
        out
    }

    /// End of generation (EOS, length or stop): flushes partial UTF-8; an unterminated tool
    /// call body is parsed if it is complete JSON, otherwise surfaced as text.
    pub fn finish(&mut self) -> Vec<AnswerEvent> {
        let mut out = Vec::new();
        if self.body == Body::ToolJson && !self.tool_buf.is_empty() {
            self.finish_tool(&mut out);
        }
        self.flush_streams(&mut out);
        self.body = Body::Idle;
        out
    }
}

/// Holds back text that could be the start of a stop string; reports the first full match.
pub struct StopMatcher {
    stops: Vec<String>,
    held: String,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        Self { stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), held: String::new() }
    }

    /// Returns (text safe to emit now, stopped).
    pub fn push(&mut self, text: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (text.to_string(), false);
        }
        self.held.push_str(text);
        if let Some(pos) = self.stops.iter().filter_map(|s| self.held.find(s.as_str())).min() {
            let out = self.held[..pos].to_string();
            self.held.clear();
            return (out, true);
        }
        // Keep the longest suffix that is a proper prefix of some stop string.
        let mut keep = 0usize;
        for s in &self.stops {
            for (i, _) in s.char_indices().skip(1) {
                if self.held.ends_with(&s[..i]) {
                    keep = keep.max(i);
                }
            }
        }
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held.drain(..cut);
        (out, false)
    }

    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_float_matches_python_repr() {
        for (f, s) in [
            (0.9, "0.9"), (0.1, "0.1"), (100.0, "100.0"), (1e16, "1e+16"), (1e-7, "1e-07"),
            (15000000000.0, "15000000000.0"), (2.5, "2.5"), (-0.0, "-0.0"), (1.5e300, "1.5e+300"),
            (0.0001, "0.0001"), (0.00001, "1e-05"), (123456789012345.6, "123456789012345.6"),
            (1234567890123456.7, "1234567890123456.8"), (12345678901234567.0, "1.2345678901234568e+16"),
        ] {
            assert_eq!(py_float(f), s, "{f}");
        }
    }

    #[test]
    fn dumps_and_order() {
        let v = parse_json(r#"{"q":"x <y> & 'z' é\n\u0001","o":{"b":1,"a":{}},"n":[1,2.5,null,true,-0,1E2]}"#).unwrap();
        assert_eq!(
            py_dumps(&v, true, false),
            r#"{"q":"x <y> & 'z' é\n\u0001","o":{"b":1,"a":{}},"n":[1,2.5,null,true,0,100.0]}"#
        );
        assert_eq!(py_dumps(&v, false, true).chars().take(12).collect::<String>(), r#"{"n": [1, 2."#);
    }

    #[test]
    fn parser_routes_thinking_text_and_tool_calls() {
        let mut p = AnswerParser::new();
        let mut ev = Vec::new();
        let toks: Vec<(u32, &[u8], bool)> = vec![
            (TOK_MESSAGE_MODEL, b"", true),
            (TOK_CONTENT_THINKING, b"", true),
            (1, b"plan", false),
            (TOK_END_MESSAGE, b"", true),
            (TOK_MESSAGE_MODEL, b"", true),
            (2, b"get_", false),
            (3, b"weather", false),
            (TOK_INVOKE_TOOL_JSON, b"", true),
            (4, br#"{"name":"get_weather","args":{"city":"Bras"#, false),
            (5, b"\xc3", false),
            (6, b"\xadlia\"}}", false),
            (TOK_END_MESSAGE, b"", true),
        ];
        for (id, b, sp) in toks {
            ev.extend(p.feed(id, b, sp));
        }
        ev.extend(p.finish());
        assert_eq!(
            ev,
            vec![
                AnswerEvent::Thinking("plan".into()),
                AnswerEvent::ToolCall { name: "get_weather".into(), arguments: r#"{"city":"Brasília"}"#.into() },
            ]
        );
    }

    #[test]
    fn lenient_json_closes_a_string_whose_last_quote_was_escaped() {
        let raw = r#"{"name":"run_python","args":{"code":"import os\p='C:/x.yml'\nprint(os.path.exists(p))\nif os.path.exists(p):\n print(open(p).read())\"}}"#;
        assert!(parse_json(raw).is_err());
        let v = parse_json_lenient(raw).expect("recovered");
        let code = v.get("args").and_then(|a| a.get("code")).and_then(OJson::as_str).unwrap();
        assert!(code.ends_with("print(open(p).read())"), "{code:?}");
        assert!(code.contains("\nprint(os.path.exists(p))"));
        // an escaped quote inside a string is still a quote
        let v = parse_json_lenient(r#"{"a":"say \"hi\" now","b":1,}"#).unwrap();
        assert_eq!(v.get("a").and_then(OJson::as_str), Some(r#"say "hi" now"#));
    }

    #[test]
    fn parser_recovers_lenient_tool_call_json() {
        // Strict parse_json still rejects bad escapes and trailing commas.
        assert!(parse_json(r#"{"a":"it\'s"}"#).is_err());
        assert!(parse_json(r#"{"a":"C:\Users\test"}"#).is_err());
        assert!(parse_json(r#"{"a":1,}"#).is_err());

        // Lenient tool-call recovery handles \' escapes, Windows paths (\U, \u non-hex),
        // regex escapes (\s, \d), unescaped HTML attribute quotes, and trailing commas.
        let mut p = AnswerParser::new();
        let mut ev = Vec::new();
        let payload = br#"{"name":"create_artifact","args":{"html":"<div class="hero">it\'s \s+\d+ C:\Users\proj\ufoo</div>","title":"Diagrams",}}"#;
        let toks: Vec<(u32, &[u8], bool)> = vec![
            (TOK_MESSAGE_MODEL, b"", true),
            (1, b"create_artifact", false),
            (TOK_INVOKE_TOOL_JSON, b"", true),
            (2, payload, false),
            (TOK_END_MESSAGE, b"", true),
        ];
        for (id, b, sp) in toks {
            ev.extend(p.feed(id, b, sp));
        }
        ev.extend(p.finish());
        assert_eq!(ev.len(), 1);
        match &ev[0] {
            AnswerEvent::ToolCall { name, arguments } => {
                assert_eq!(name, "create_artifact");
                let parsed = parse_json(arguments).expect("re-serialized arguments must be valid strict JSON");
                assert_eq!(parsed.get("title").and_then(OJson::as_str), Some("Diagrams"));
                assert_eq!(
                    parsed.get("html").and_then(OJson::as_str),
                    Some(r#"<div class="hero">it's \s+\d+ C:\Users\proj\ufoo</div>"#)
                );
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }

        // Also recovers when `<|content_text|>` follows a non-empty tool-name header.
        let mut p2 = AnswerParser::new();
        let mut ev2 = Vec::new();
        for (id, b, sp) in [
            (TOK_MESSAGE_MODEL, &b""[..], true),
            (1, &b"run_cmd"[..], false),
            (TOK_CONTENT_TEXT, &b""[..], true),
            (2, &br#"{"name":"run_cmd","args":{"cmd":"echo \'hi\'"}}"#[..], false),
            (TOK_END_MESSAGE, &b""[..], true),
        ] {
            ev2.extend(p2.feed(id, b, sp));
        }
        assert_eq!(
            ev2,
            vec![AnswerEvent::ToolCall {
                name: "run_cmd".into(),
                arguments: r#"{"cmd":"echo 'hi'"}"#.into(),
            }]
        );
    }

    #[test]
    fn stop_matcher_holds_partial_matches() {
        let mut m = StopMatcher::new(vec!["STOP".into()]);
        assert_eq!(m.push("abc ST"), ("abc ".into(), false));
        assert_eq!(m.push("OP tail"), ("".into(), true));
        let mut m = StopMatcher::new(vec!["xyz".into()]);
        assert_eq!(m.push("hello x"), ("hello ".into(), false));
        assert_eq!(m.push("a"), ("xa".into(), false));
    }
}
