//! OpenAI-Compatible HTTP/1.1 + SSE Server for SmallTide (`smalltide-cli serve`).
//!
//! Serves model **`inkling-2-mini`** on port **`8000`** (configurable via `--port`):
//! - `GET  /v1/models`           OpenAI model listing (`inkling-2-mini`)
//! - `POST /v1/completions`      OpenAI legacy Completion API (streaming SSE + non-streaming)
//! - `POST /v1/chat/completions` OpenAI Chat Completion API (streaming SSE + non-streaming,
//!                               exact Inkling-2-Mini chat template + `reasoning_content` split)
//! - `GET  /v1/tide/stats`       Live TideLM telemetry (hit rates, Stage 1/2 stats, VRAM, tok/s)
//! - `POST /v1/tide/config`      Live rescue knob tuning (`rescue_mode`, `rescue_threshold`)
//! - `GET  /health`              Health check

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use smalltide_backend_hip::HipBackend;
use smalltide_core::{ComputeBackend, RescuePolicy};
use smalltide_io::tokenizer::BpeTokenizer;
use crate::chat_format::{self, AnswerEvent, AnswerParser, OJson, StopMatcher};
use crate::prefix_cache::{PrefixCache, PrefixPlan};
use smalltide_io::tokenizer::Utf8Stream;
use smalltide_model_inkling2::engine::{Inkling2Engine, SamplingParams, SeqSnapshot};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const SERVED_MODEL_NAME: &str = "inkling-2-mini";

const TOK_MESSAGE_USER: u32 = 200_000;
const TOK_EOS: u32 = 200_006;
const TOK_EOT: u32 = 199_999;

static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputChannel {
    Text,
    Thinking,
}

#[derive(Debug)]
enum StreamEvent {
    Delta {
        channel: OutputChannel,
        text: String,
    },
    ToolCall {
        index: usize,
        id: String,
        name: String,
        arguments: String,
    },
    Done {
        finish_reason: &'static str,
        /// Prompt tokens served from the prefix cache (not re-prefilled).
        cached_tokens: usize,
        prompt_tokens: usize,
        completion_tokens: usize,
        prefill_secs: f64,
        decode_secs: f64,
        vram_hit_rate: f64,
    },
    Error(String),
}

/// Logs an engine error for a request; a lost GPU context (HIP launch failure / illegal address:
/// sticky, every later launch fails too) ends the process with a clear message instead of a server
/// that answers every request with an error.
fn engine_error(what: &str, msg: &str) -> String {
    let line = format!("{what}: {msg}");
    eprintln!("[error] {line}");
    if ["hipError_t=719", "hipError_t=700", "hipError_t=999"].iter().any(|c| msg.contains(c)) {
        eprintln!("[fatal] the GPU context is lost ({msg}); the server must be restarted - exiting");
        std::process::exit(2);
    }
    line
}

/// `max_tokens` when a request gives none: `SMALLTIDE_MAX_TOKENS` (default 48K); the lane's
/// context still bounds it.
fn default_max_tokens() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("SMALLTIDE_MAX_TOKENS").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|&v| v > 0).unwrap_or(48 * 1024)
    })
}

enum EngineCommand {
    Generate {
        prompt_text: String,
        is_chat: bool,
        /// Chat only: the prompt already opens this tool's call (`tool_choice` names a function).
        forced_tool: Option<String>,
        /// Stop after the first completed tool call (`parallel_tool_calls: false` or forced tool).
        single_tool_call: bool,
        max_tokens: usize,
        sampling: SamplingParams,
        stop_strings: Vec<String>,
        cache_prompt: bool,
        event_tx: Sender<StreamEvent>,
    },
    GetStats {
        reply_tx: Sender<Value>,
    },
    /// Expert map snapshot as JSON (`with_text`: include the routed tokens' text).
    ExpertMap {
        ask: crate::expert_map::Ask,
        with_text: bool,
        reply_tx: Sender<Value>,
    },
    UpdateConfig {
        rescue_policy: RescuePolicy,
        reply_tx: Sender<Value>,
    },
}

#[derive(Deserialize)]
struct ChatCompletionReq {
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    /// llama.cpp-style: `false` bypasses the prompt prefix cache for this request.
    cache_prompt: Option<bool>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    stop: Option<Value>,
}

#[derive(Deserialize)]
struct CompletionReq {
    #[serde(default)]
    prompt: Value,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    /// llama.cpp-style: `false` bypasses the prompt prefix cache for this request.
    cache_prompt: Option<bool>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    stop: Option<Value>,
}

#[derive(Deserialize)]
struct TideConfigReq {
    rescue_mode: Option<String>,
    rescue_threshold: Option<f32>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: ErrorDetail<'a>,
}

#[derive(Serialize)]
struct ErrorDetail<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    err_type: &'a str,
}

fn parse_stop_strings(v: &Option<Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .filter(|s| !s.is_empty())
            .collect(),
        _ => Vec::new(),
    }
}

/// Renders a chat request (messages, tools, tool_choice, reasoning_effort) with the exact
/// chat template. Returns the prompt and the forced tool name, if `tool_choice` names one.
fn render_chat_request(body: &OJson) -> Result<(String, Option<String>, bool), String> {
    let messages = match body.get("messages") {
        Some(OJson::Arr(m)) if !m.is_empty() => m.clone(),
        _ => return Err("`messages` must be a non-empty list".into()),
    };
    let tools = body.get("tools").filter(|t| !matches!(t, OJson::Null));
    let mut forced = None;
    let mut use_tools = tools;
    match body.get("tool_choice") {
        None | Some(OJson::Null) => {}
        Some(OJson::Str(s)) => match s.as_str() {
            "none" => use_tools = None,
            // "required" cannot be forced without naming a tool; it behaves like "auto".
            "auto" | "required" => {}
            other => return Err(format!("unsupported tool_choice: {other}")),
        },
        Some(tc) => {
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(OJson::as_str)
                .ok_or("tool_choice must be \"none\", \"auto\", \"required\" or {\"type\":\"function\",\"function\":{\"name\":...}}")?;
            let declared = match tools {
                Some(OJson::Arr(list)) => list.iter().any(|t| {
                    t.get("function").unwrap_or(t).get("name").and_then(OJson::as_str) == Some(name)
                }),
                _ => false,
            };
            if !declared {
                return Err(format!("tool_choice names `{name}`, which is not in `tools`"));
            }
            forced = Some(name.to_string());
        }
    }
    let r = chat_format::render_chat(&messages, use_tools, body.get("reasoning_effort"))?;
    let mut prompt = r.prompt;
    if let Some(n) = &forced {
        prompt.push_str(&format!("<|message_model|>{n}<|content_invoke_tool_json|>"));
    }
    let single = forced.is_some() || matches!(body.get("parallel_tool_calls"), Some(OJson::Bool(false)));
    Ok((prompt, forced, single))
}

pub fn run_serve(args: &[String]) {
    if let Some(flag) = args.first() {
        crate::reject_config_flag(flag);
    }
    let s = crate::settings::Settings::from_env();
    let (host, port) = (s.host.clone(), s.port);
    println!("=== SmallTide OpenAI API Server (model: {SERVED_MODEL_NAME}) ===");
    println!("Loading tokenizer, FP8 dense weights, and expert pools from {} ...", s.model_dir.display());
    let crate::settings::Loaded { engine, tok, .. } = crate::settings::build_engine(&s, s.lanes);
    let tok = Arc::new(tok);

    let (cmd_tx, cmd_rx) = mpsc::channel::<EngineCommand>();
    let tok_worker = tok.clone();
    thread::spawn(move || engine_worker_loop(engine, tok_worker, cmd_rx));

    let bind_addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&bind_addr).expect("bind HTTP server");
    println!("Listening on http://{bind_addr}");
    println!("  GET  http://127.0.0.1:{port}/v1/models");
    println!("  POST http://127.0.0.1:{port}/v1/completions");
    println!("  POST http://127.0.0.1:{port}/v1/chat/completions");
    println!("  GET  http://127.0.0.1:{port}/v1/tide/stats");
    println!("  POST http://127.0.0.1:{port}/v1/tide/config");

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let tx = cmd_tx.clone();
                thread::spawn(move || handle_client(stream, tx));
            }
            Err(e) => eprintln!("[HTTP accept error] {e}"),
        }
    }
}

/// One generation in flight on an engine lane.
struct ActiveReq {
    lane: usize,
    /// `prompt[..len - 1]`: key of this prompt's end snapshot (conversation memory).
    prompt_key: Vec<u32>,
    is_chat: bool,
    single_tool_call: bool,
    max_tokens: usize,
    sampling: SamplingParams,
    parser: AnswerParser,
    raw_utf8: Utf8Stream,
    stopper: StopMatcher,
    event_tx: Sender<StreamEvent>,
    req_tag: u64,
    n_tool_calls: usize,
    prompt_len: usize,
    reuse_len: usize,
    pf_secs: f64,
    t_dec: Instant,
    generated: Vec<u32>,
    /// Position the pending `next_tok` will be fed at.
    pos: usize,
    /// Tap row index of `pos - 1` in `engine.taps_buf` (for DSpark speculative drafting).
    tap_row: usize,
    /// Sampled, already streamed to the client, not yet fed to the model.
    next_tok: u32,
    finish_reason: &'static str,
    /// Request log: answer / thinking characters streamed, the last output characters, and the
    /// routing-weight telemetry (total, skipped) at admission.
    log_text: usize,
    log_think: usize,
    log_tail: String,
    log_w0: (f64, f64),
    /// (sync rescues, skipped misses, disk reads) telemetry at admission.
    log_c0: (u64, u64, u64),
}

impl ActiveReq {
    fn log_out(&mut self, t: &str, thinking: bool) {
        if thinking {
            self.log_think += t.chars().count();
        } else {
            self.log_text += t.chars().count();
        }
        self.log_tail.push_str(t);
        let n = self.log_tail.chars().count();
        if n > 80 {
            self.log_tail = self.log_tail.chars().skip(n - 80).collect();
        }
    }

    /// Routes parsed answer events to the client. Returns false to stop generating.
    fn emit(&mut self, evs: Vec<AnswerEvent>) -> bool {
        for ev in evs {
            let sent = match ev {
                AnswerEvent::Text(t) => {
                    let (out, stopped) = self.stopper.push(&t);
                    self.log_out(&out, false);
                    let ok = out.is_empty()
                        || self.event_tx.send(StreamEvent::Delta { channel: OutputChannel::Text, text: out }).is_ok();
                    if stopped {
                        self.finish_reason = "stop";
                        return false;
                    }
                    ok
                }
                AnswerEvent::Thinking(t) => {
                    self.log_out(&t, true);
                    self.event_tx.send(StreamEvent::Delta { channel: OutputChannel::Thinking, text: t }).is_ok()
                }
                AnswerEvent::ToolCall { name, arguments } => {
                    let index = self.n_tool_calls;
                    self.n_tool_calls += 1;
                    let ok = self
                        .event_tx
                        .send(StreamEvent::ToolCall {
                            index,
                            id: format!("call_{:x}_{index}", self.req_tag),
                            name,
                            arguments,
                        })
                        .is_ok();
                    if ok && self.single_tool_call {
                        // A single call was requested: the answer ends here.
                        self.finish_reason = "stop";
                        return false;
                    }
                    ok
                }
                AnswerEvent::EndOfAnswer => {
                    self.finish_reason = "stop";
                    return false;
                }
            };
            if !sent {
                // Client disconnected: abort generation.
                self.finish_reason = "stop";
                return false;
            }
        }
        true
    }

    /// Consumes the freshly sampled `next_tok`: streams it and decides whether generation goes on
    /// (then `next_tok` must be fed at `pos`).
    fn accept(&mut self, tok: &BpeTokenizer, lane_max_seq: usize) -> bool {
        let t = self.next_tok;
        if t == TOK_EOS || t == TOK_EOT {
            self.finish_reason = "stop";
            return false;
        }
        self.generated.push(t);
        let bytes = tok.token_bytes(t);
        let evs = if self.is_chat {
            self.parser.feed(t, bytes, tok.is_special(t))
        } else {
            let s = self.raw_utf8.push(bytes);
            if s.is_empty() { Vec::new() } else { vec![AnswerEvent::Text(s)] }
        };
        if !self.emit(evs) {
            return false;
        }
        self.generated.len() < self.max_tokens && self.pos < lane_max_seq
    }

    /// Flushes held-back output and sends `Done`. Returns the request's decode tok/s.
    fn finish(mut self, vram_hit_rate: f64) -> f64 {
        // Flush held-back UTF-8, partial stop-string prefixes and an unterminated call.
        let tail = if self.is_chat {
            self.parser.finish()
        } else {
            let t = self.raw_utf8.finish();
            if t.is_empty() { Vec::new() } else { vec![AnswerEvent::Text(t)] }
        };
        let tail_ok = self.emit(tail);
        if tail_ok && self.finish_reason != "stop" {
            let held = self.stopper.finish();
            if !held.is_empty() {
                let _ = self.event_tx.send(StreamEvent::Delta { channel: OutputChannel::Text, text: held });
            }
        }
        if self.n_tool_calls > 0 && self.finish_reason == "stop" {
            self.finish_reason = "tool_calls";
        }
        let dec_secs = self.t_dec.elapsed().as_secs_f64();
        let _ = self.event_tx.send(StreamEvent::Done {
            finish_reason: self.finish_reason,
            cached_tokens: self.reuse_len,
            prompt_tokens: self.prompt_len,
            completion_tokens: self.generated.len(),
            prefill_secs: self.pf_secs,
            decode_secs: dec_secs,
            vram_hit_rate,
        });
        if self.generated.is_empty() { 0.0 } else { self.generated.len() as f64 / dec_secs.max(1e-6) }
    }
}

/// Finishes `req` and remembers its routing demand (prompt + answer) as its conversation's
/// expected routing: on its lane (next turn extends it) and on its prompt's end snapshot (next
/// turn restores it, possibly on another lane).
fn finish_req(
    engine: &mut Inkling2Engine<HipBackend>,
    pc: &mut PrefixCache<SeqSnapshot>,
    lane_demand: &mut [Option<Vec<f32>>],
    req: ActiveReq,
) -> f64 {
    let lane = req.lane;
    engine.set_lane_pinned(lane, false);
    let demand = engine.tide.scheduler.cache.take_lane_demand(lane);
    if demand.iter().any(|&d| d > 0.0) {
        pc.set_demand(&req.prompt_key, &demand);
        lane_demand[lane] = Some(demand);
    }
    let hit = engine.telemetry().total_vram_hit_rate();
    let tel = engine.telemetry();
    let (dw, ds) = (tel.routed_weight_total - req.log_w0.0, tel.routed_weight_skipped - req.log_w0.1);
    let (dr, dm, dd) = (
        tel.stage1_sync_rescues - req.log_c0.0,
        tel.stage1_skipped_tail_misses - req.log_c0.1,
        tel.stage2_disk_reads - req.log_c0.2,
    );
    let ntok = req.generated.len().max(1) as f64;
    eprintln!(
        "[req] lane {lane}: finish={} prompt {} (cached {}) -> {} tokens: answer {} chars, thinking {} chars, {} tool call(s) | skipped expert weight {:.1}% | per token: {:.1} rescue loads, {:.1} skipped experts, {:.1} disk reads | ends with {:?}",
        if req.n_tool_calls > 0 && req.finish_reason == "stop" { "tool_calls" } else { req.finish_reason },
        req.prompt_len,
        req.reuse_len,
        req.generated.len(),
        req.log_text,
        req.log_think,
        req.n_tool_calls,
        if dw > 0.0 { 100.0 * ds / dw } else { 0.0 },
        dr as f64 / ntok,
        dm as f64 / ntok,
        dd as f64 / ntok,
        req.log_tail
    );
    let tps = req.finish(hit);
    if let Some(line) = engine.tide.scheduler.finish_request() {
        println!("{line}");
    }
    tps
}

/// Engine worker: owns the engine and serves up to `engine.num_lanes()` requests at once.
///
/// Each round: drain commands (blocking only when idle), admit at most one queued request into
/// a free lane (its prefill runs while the other lanes wait), then advance every active lane by
/// one token. Lanes are independent sequences (own KV caches and short-conv state); the prefix
/// cache's snapshots are shared by all lanes.
fn engine_worker_loop(
    mut engine: Inkling2Engine<HipBackend>,
    tok: Arc<BpeTokenizer>,
    cmd_rx: Receiver<EngineCommand>,
) {
    let n_lanes = engine.num_lanes();
    let env_usize = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(d);
    let mut pc: PrefixCache<SeqSnapshot> = PrefixCache::new(
        n_lanes,
        env_usize("SMALLTIDE_PREFIX_SNAPSHOTS", 12),
        env_usize("SMALLTIDE_PREFIX_TOKENS", 65_536),
    );
    let mut active: Vec<Option<ActiveReq>> = (0..n_lanes).map(|_| None).collect();
    let mut lane_demand: Vec<Option<Vec<f32>>> = vec![None; n_lanes];
    let mut queue: std::collections::VecDeque<EngineCommand> = std::collections::VecDeque::new();
    let mut last_decode_tok_s: f64 = 0.0;
    let mut last_prefill_tok_s: f64 = 0.0;
    let max_lane_ctx = (0..n_lanes).map(|l| engine.lane_max_seq(l)).max().unwrap_or(0);
    let batch_decode = std::env::var("SMALLTIDE_BATCH_DECODE").map(|v| v != "0").unwrap_or(true);
    let fixed_spec_width = std::env::var("SMALLTIDE_SPEC_WIDTH")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|w| w.clamp(2, 8));
    // Tracks which `(lane, committed_pos)` currently matches the single DSpark drafter's KV cache.
    let mut drafter_state: Option<(usize, usize)> = None;
    let mut spec_pred_ema = 3.0f64;
    let mut spec_obs_ema = 2.0f64;
    let mut spec_draft_ema = 3.0f64;
    let mut spec_vbase_ema = 16.0f64;

    loop {
        // 1. Commands: block only when there is nothing to do.
        let idle = queue.is_empty() && active.iter().all(|a| a.is_none());
        let mut cmds = Vec::new();
        if idle {
            match cmd_rx.recv() {
                Ok(c) => cmds.push(c),
                Err(_) => return,
            }
        }
        while let Ok(c) = cmd_rx.try_recv() {
            cmds.push(c);
        }
        for cmd in cmds {
            match cmd {
                EngineCommand::ExpertMap { ask, with_text, reply_tx } => {
                    let busy = !queue.is_empty() || active.iter().any(|a| a.is_some());
                    let m = engine.expert_map(busy, ask.since);
                    let text = |id: u32| tok.decode(&[id]);
                    let v = crate::expert_map::json(&m, SERVED_MODEL_NAME, "inkling2", &ask, with_text.then_some(&text as &dyn Fn(u32) -> String));
                    let _ = reply_tx.send(v);
                }
                EngineCommand::GetStats { reply_tx } => {
                    let tel = engine.telemetry();
                    let (pre, core, coda) = engine.tide.scheduler.cache.count_vram_resident();
                    let ram = engine.tide.scheduler.cache.count_ram_resident();
                    let vram_info = engine.backend.device_info().ok();
                    let pct = |h: u64, m: u64| {
                        let tot = h + m;
                        if tot == 0 { 100.0 } else { (h as f64 / tot as f64) * 100.0 }
                    };
                    let stats = json!({
                        "model": SERVED_MODEL_NAME,
                        "rescue_policy": format!("{:?}", engine.tide.scheduler.tide_cfg.rescue_policy),
                        "last_prefill_tok_s": last_prefill_tok_s,
                        "last_decode_tok_s": last_decode_tok_s,
                        "lanes": {
                            "count": n_lanes,
                            "active": active.iter().filter(|a| a.is_some()).count(),
                            "queued": queue.len(),
                            "ctx": (0..n_lanes).map(|l| engine.lane_max_seq(l)).collect::<Vec<_>>(),
                        },
                        "vram_hit_rate_pct": tel.total_vram_hit_rate() * 100.0,
                        "prelude_hit_rate_pct": pct(tel.vram_hits_prelude, tel.vram_misses_prelude),
                        "loop_core_hit_rate_pct": pct(tel.vram_hits_loop, tel.vram_misses_loop),
                        "coda_hit_rate_pct": pct(tel.vram_hits_coda, tel.vram_misses_coda),
                        "stage1_sync_rescues": tel.stage1_sync_rescues,
                        "stage1_async_promotions": tel.stage1_async_promotions,
                        "stage1_skipped_tail_misses": tel.stage1_skipped_tail_misses,
                        "stage2_disk_reads": tel.stage2_disk_reads,
                        "stage1_bytes_transferred": tel.stage1_bytes_transferred,
                        "routed_weight_total": tel.routed_weight_total,
                        "routed_weight_skipped": tel.routed_weight_skipped,
                        "prefill_weight_total": tel.prefill_weight_total,
                        "prefill_weight_skipped": tel.prefill_weight_skipped,
                        "decode_async_promotions": tel.decode_async_promotions,
                        "prompt_aware_swaps": tel.prompt_aware_swaps,
                        "experts_vram_resident": { "prelude": pre, "loop_core": core, "coda": coda },
                        "experts_ram_warm": ram,
                        "kv_pool": engine.kv_pool_info().map(|(pages, free, per, _)| json!({"pages": pages, "free": free, "positions_per_page": per})),
                        "prefix_cache": {
                            "requests": pc.stats.requests,
                            "extend_hits": pc.stats.extend_hits,
                            "restore_hits": pc.stats.restore_hits,
                            "cross_restores": pc.stats.cross_restores,
                            "evictions": pc.stats.evictions,
                            "snapshot_tokens": pc.snapshot_tokens(),
                            "misses": pc.stats.misses,
                            "reused_tokens": pc.stats.reused_tokens,
                            "prefilled_tokens": pc.stats.prefilled_tokens,
                            "snapshots_taken": pc.stats.snapshots_taken,
                            "snapshot_positions": pc.snapshot_positions(),
                            "live_tokens": pc.live.iter().map(|l| l.len()).collect::<Vec<_>>(),
                            "snapshot_mib": engine.snapshot_bytes() as f64 / (1024.0 * 1024.0),
                        },
                        "kv_max_ctx": max_lane_ctx,
                        "kv_vram_gib": engine.kv_vram_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                        "vram_free_gib": vram_info.as_ref().map(|d| d.free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)),
                        "vram_total_gib": vram_info.as_ref().map(|d| d.total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)),
                    });
                    let _ = reply_tx.send(stats);
                }
                EngineCommand::UpdateConfig { rescue_policy, reply_tx } => {
                    engine.tide.scheduler.tide_cfg.rescue_policy = rescue_policy;
                    let _ = reply_tx.send(json!({
                        "ok": true,
                        "rescue_policy": format!("{:?}", engine.tide.scheduler.tide_cfg.rescue_policy),
                    }));
                }
                g @ EngineCommand::Generate { .. } => queue.push_back(g),
            }
        }

        // Paged KV: lanes whose pages went to another lane no longer hold their sequence on the
        // device (their conversations live on in the prefix cache's snapshots).
        for l in engine.take_lost_lanes() {
            pc.invalidate(l);
        }
        // 2. Admission: at most one queued request per round (its prefill pauses the others).
        if let Some(EngineCommand::Generate { .. }) = queue.front() {
            let free: Vec<usize> = (0..n_lanes).filter(|&l| active[l].is_none()).collect();
            if !free.is_empty() {
                let Some(EngineCommand::Generate {
                    prompt_text,
                    is_chat,
                    forced_tool,
                    single_tool_call,
                    max_tokens,
                    sampling,
                    stop_strings,
                    cache_prompt,
                    event_tx,
                }) = queue.pop_front()
                else {
                    unreachable!()
                };
                let prompt_ids = tok.encode(&prompt_text);
                if prompt_ids.is_empty() {
                    let _ = event_tx.send(StreamEvent::Error("empty prompt".into()));
                } else if prompt_ids.len() >= max_lane_ctx {
                    let _ = event_tx.send(StreamEvent::Error(format!(
                        "prompt length {} exceeds max_ctx {}",
                        prompt_ids.len(),
                        max_lane_ctx
                    )));
                } else {
                    let fits: Vec<usize> = free.iter().copied().filter(|&l| engine.lane_max_seq(l) > prompt_ids.len()).collect();
                    if fits.is_empty() {
                        // Only a busy (larger) lane can hold this prompt: wait for it.
                        queue.push_front(EngineCommand::Generate {
                            prompt_text,
                            is_chat,
                            forced_tool,
                            single_tool_call,
                            max_tokens,
                            sampling,
                            stop_strings,
                            cache_prompt,
                            event_tx,
                        });
                    } else {
                        let (lane, plan) = if cache_prompt {
                            pc.best_lane(&fits, &prompt_ids).unwrap()
                        } else {
                            (fits[0], PrefixPlan::Reset)
                        };
                        // Prompt-aware loading: the conversation's remembered routing (the
                        // snapshot it restores, or this lane's live conversation it extends).
                        let memory: Option<Vec<f32>> = match &plan {
                            PrefixPlan::Restore { snap_idx, .. } => pc.snapshot_demand(*snap_idx).map(|d| d.to_vec()),
                            PrefixPlan::Extend { .. } => lane_demand[lane].clone(),
                            PrefixPlan::Reset => None,
                        };
                        engine.select_lane(lane);
                        engine.set_lane_pinned(lane, true);
                        let _ = engine.tide.scheduler.cache.take_lane_demand(lane);
                        if let Some(d) = &memory {
                            let _ = engine.anticipate(d);
                        }
                        let drafter_coherent_in = engine.drafter.is_some()
                            && match &plan {
                                PrefixPlan::Reset => true,
                                PrefixPlan::Extend { from } => drafter_state.is_some_and(|(dl, dp)| {
                                    dl == lane && (*from == dp || (dp <= 1024 && *from <= dp))
                                }),
                                PrefixPlan::Restore { snap_idx, .. } => pc.snapshot(*snap_idx).drafter_valid,
                            };
                        if let Some(req) = admit(
                            &mut engine, &tok, &mut pc, lane, plan, prompt_ids, is_chat, forced_tool,
                            single_tool_call, max_tokens, sampling, stop_strings, cache_prompt,
                            drafter_coherent_in, event_tx, &mut last_prefill_tok_s,
                        ) {
                            if engine.drafter.is_some() {
                                drafter_state = drafter_coherent_in.then_some((lane, req.pos));
                            }
                            let mut req = req;
                            if req.accept(&tok, engine.lane_max_seq(lane)) {
                                active[lane] = Some(req);
                            } else {
                                last_decode_tok_s = finish_req(&mut engine, &mut pc, &mut lane_demand, req);
                            }
                        } else {
                            drafter_state = None;
                            engine.set_lane_pinned(lane, false);
                        }
                        for l in engine.take_lost_lanes() {
                            pc.invalidate(l);
                        }
                    }
                }
            }
        }

        // 3. Decode: one token for every active lane - one batched pass when several are active.
        // Paged KV: back each lane's next position first; a lane the pool cannot hold any more
        // ends its request (finish_reason "length").
        for l in 0..n_lanes {
            let Some(req) = active[l].as_ref() else { continue };
            match engine.kv_reserve(l, req.pos + 1) {
                Ok(true) => {}
                Ok(false) => {
                    eprintln!("[kv] lane {l}: the KV pool cannot hold position {} - request ends (length)", req.pos + 1);
                    let req = active[l].take().unwrap();
                    last_decode_tok_s = finish_req(&mut engine, &mut pc, &mut lane_demand, req);
                }
                Err(e) => {
                    let msg = engine_error("kv reserve", &e.to_string());
                    let req = active[l].take().unwrap();
                    let _ = req.event_tx.send(StreamEvent::Error(msg));
                    pc.invalidate(l);
                    last_decode_tok_s = finish_req(&mut engine, &mut pc, &mut lane_demand, req);
                }
            }
        }
        for l in engine.take_lost_lanes() {
            pc.invalidate(l);
        }
        let act: Vec<usize> = (0..n_lanes).filter(|&l| active[l].is_some()).collect();
        if act.len() >= 2 && batch_decode {
            drafter_state = None;
            let rows: Vec<(usize, u32, usize)> = act
                .iter()
                .map(|&l| {
                    let r = active[l].as_ref().unwrap();
                    (l, r.next_tok, r.pos)
                })
                .collect();
            let fwd = engine.forward_batch(&rows);
            let mut sampled: Vec<std::result::Result<u32, String>> = Vec::with_capacity(act.len());
            if let Err(e) = &fwd {
                sampled.extend(act.iter().map(|_| Err(format!("decode error: {e}"))));
            } else {
                for (i, &l) in act.iter().enumerate() {
                    let req = active[l].as_mut().unwrap();
                    pc.live[l].push(req.next_tok);
                    req.pos += 1;
                    sampled.push(engine.sample_batch_row(i, l, &req.sampling, &req.generated).map_err(|e| format!("sample error: {e}")));
                }
                let _ = engine.post_batch_tide_step();
            }
            for (&l, res) in act.iter().zip(sampled) {
                let req = active[l].as_mut().unwrap();
                let done = match res {
                    Ok(t) => {
                        req.next_tok = t;
                        !req.accept(&tok, engine.lane_max_seq(l))
                    }
                    Err(msg) => {
                        let _ = req.event_tx.send(StreamEvent::Error(engine_error("decode", &msg)));
                        pc.invalidate(l);
                        true
                    }
                };
                if done {
                    let req = active[l].take().unwrap();
                    last_decode_tok_s = finish_req(&mut engine, &mut pc, &mut lane_demand, req);
                }
            }
            continue;
        }
        // Single-lane speculative decoding (greedy & stochastic rejection sampling) when DSpark is coherent.
        if act.len() == 1 && engine.drafter.is_some() {
            let l = act[0];
            let can_spec = active[l].as_ref().is_some_and(|req| {
                drafter_state == Some((l, req.pos))
                    && req.pos + 2 <= engine.lane_max_seq(l)
            });
            if can_spec {
                let req_pos = active[l].as_ref().unwrap().pos;
                let mut max_w = (engine.lane_max_seq(l) - req_pos).min(8);
                while max_w >= 2 && !matches!(engine.kv_reserve(l, req_pos + max_w), Ok(true)) {
                    max_w -= 1;
                }
                for lost in engine.take_lost_lanes() {
                    pc.invalidate(lost);
                }
                if max_w >= 2 {
                    engine.select_lane(l);
                    let req = active[l].as_mut().unwrap();
                    let n_draft = fixed_spec_width.map(|w| w.min(max_w) - 1).unwrap_or(max_w - 1);
                    let td = Instant::now();
                    let step_res = engine.draft_with_margins(req.next_tok, req.pos, req.tap_row, n_draft);
                    let dt_d = td.elapsed().as_secs_f64() * 1000.0;
                    let mut done = false;
                    match step_res {
                        Err(e) => {
                            let _ = req.event_tx.send(StreamEvent::Error(engine_error("draft error", &e.to_string())));
                            pc.invalidate(l);
                            drafter_state = None;
                            done = true;
                        }
                        Ok(draft_pairs) => {
                            spec_draft_ema = 0.8 * spec_draft_ema + 0.2 * dt_d;
                            let calib = (spec_obs_ema / spec_pred_ema.max(0.5)).clamp(0.25, 1.05);
                            let mut probs = [0.0f64; 7];
                            for (k, &(_, mg)) in draft_pairs.iter().enumerate() {
                                probs[k] = (1.0 / (1.0 + (-(mg as f64)).exp()) * calib).clamp(0.05, 0.98);
                            }
                            let spec_width = fixed_spec_width.map(|w| w.min(max_w)).unwrap_or_else(|| {
                                let (mut best_w, mut best_rate, mut surv, mut exp_toks) = (2usize, 0.0f64, 1.0f64, 1.0f64);
                                for w in 2..=max_w {
                                    surv *= probs[w - 2];
                                    exp_toks += surv;
                                    let cost = spec_draft_ema + spec_vbase_ema + 2.3 * (w - 1) as f64;
                                    let rate = exp_toks / cost;
                                    if rate > best_rate {
                                        best_rate = rate;
                                        best_w = w;
                                    }
                                }
                                best_w
                            });
                            let n = spec_width - 1;
                            let mut block = Vec::with_capacity(spec_width);
                            block.push(req.next_tok);
                            for &(t, _) in &draft_pairs[..n] {
                                block.push(t);
                            }
                            let greedy = req.sampling.temperature <= 0.0
                                && (req.sampling.repetition_penalty - 1.0).abs() < 1e-6;
                            let tv = Instant::now();
                            let vres = engine
                                .verify_for_sampling(&block, req.pos, greedy)
                                .and_then(|_| engine.verify_sample(&block, &req.sampling, &req.generated));
                            match vres {
                                Err(e) => {
                                    let _ = req.event_tx.send(StreamEvent::Error(engine_error("verify error", &e.to_string())));
                                    pc.invalidate(l);
                                    drafter_state = None;
                                    done = true;
                                }
                                Ok((c, next_tok)) => {
                                    if let Err(e) = engine.verify_rewind(spec_width, c + 1) {
                                        let _ = req.event_tx.send(StreamEvent::Error(engine_error("verify rewind error", &e.to_string())));
                                        pc.invalidate(l);
                                        drafter_state = None;
                                        done = true;
                                    } else {
                                        let _ = engine.post_batch_tide_step();
                                        let dt_v = tv.elapsed().as_secs_f64() * 1000.0;
                                        spec_vbase_ema = 0.85 * spec_vbase_ema + 0.15 * (dt_v - 2.3 * n as f64).clamp(10.0, 30.0);
                                        let eval_n = (c + 1).min(n);
                                        let pred_sum: f64 = (0..eval_n)
                                            .map(|k| 1.0 / (1.0 + (-(draft_pairs[k].1 as f64)).exp()))
                                            .sum();
                                        spec_pred_ema = 0.85 * spec_pred_ema + pred_sum;
                                        spec_obs_ema = 0.85 * spec_obs_ema + c as f64;
                                        pc.live[l].extend_from_slice(&block[..=c]);
                                        req.pos += c + 1;
                                        req.tap_row = c;
                                        drafter_state = Some((l, req.pos));
                                        let lane_max = engine.lane_max_seq(l);
                                        for &t in &block[1..=c] {
                                            req.next_tok = t;
                                            if !req.accept(&tok, lane_max) {
                                                done = true;
                                                break;
                                            }
                                        }
                                        if !done {
                                            req.next_tok = next_tok;
                                            if !req.accept(&tok, lane_max) {
                                                done = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if done {
                        let req = active[l].take().unwrap();
                        last_decode_tok_s = finish_req(&mut engine, &mut pc, &mut lane_demand, req);
                    }
                    continue;
                }
            }
        }

        for l in 0..n_lanes {
            let Some(req) = active[l].as_mut() else { continue };
            drafter_state = None;
            engine.select_lane(l);
            let mut done = false;
            if let Err(e) = engine.forward_token(req.next_tok, req.pos, false, true) {
                let _ = req.event_tx.send(StreamEvent::Error(engine_error("decode error", &e.to_string())));
                pc.invalidate(l);
                done = true;
            } else {
                let _ = engine.post_decode_tide_step();
                pc.live[l].push(req.next_tok);
                req.pos += 1;
                match engine.sample_next_token(&req.sampling, &req.generated) {
                    Ok(t) => {
                        req.next_tok = t;
                        if !req.accept(&tok, engine.lane_max_seq(l)) {
                            done = true;
                        }
                    }
                    Err(e) => {
                        let _ = req.event_tx.send(StreamEvent::Error(engine_error("sample error", &e.to_string())));
                        done = true;
                    }
                }
            }
            if done {
                let req = active[l].take().unwrap();
                last_decode_tok_s = finish_req(&mut engine, &mut pc, &mut lane_demand, req);
            }
        }
    }
}

/// Prefills a request's prompt on `lane` (reusing the prefix cache per `plan`) and samples its
/// first token. `None` if it failed (the error was sent to the client).
#[allow(clippy::too_many_arguments)]
fn admit(
    engine: &mut Inkling2Engine<HipBackend>,
    tok: &BpeTokenizer,
    pc: &mut PrefixCache<SeqSnapshot>,
    lane: usize,
    plan: PrefixPlan,
    prompt_ids: Vec<u32>,
    is_chat: bool,
    forced_tool: Option<String>,
    single_tool_call: bool,
    max_tokens: usize,
    sampling: SamplingParams,
    stop_strings: Vec<String>,
    cache_prompt: bool,
    drafter_coherent: bool,
    event_tx: Sender<StreamEvent>,
    last_prefill_tok_s: &mut f64,
) -> Option<ActiveReq> {
    engine.select_lane(lane);
    let t_pf = Instant::now();
    if std::env::var_os("SMALLTIDE_PREFIX_DEBUG").is_some() {
        let cached = &pc.live[lane];
        let lcp = cached.iter().zip(&prompt_ids).take_while(|(a, b)| a == b).count();
        let show = |ids: &[u32]| -> String {
            let end = (lcp + 6).min(ids.len());
            format!("{:?}", ids[lcp.min(end)..end].iter().map(|&t| tok.decode(&[t])).collect::<Vec<_>>())
        };
        eprintln!(
            "[prefix] lane={lane} plan={plan:?} live={} prompt={} lcp={lcp} live_next={} prompt_next={}",
            cached.len(),
            prompt_ids.len(),
            show(cached),
            show(&prompt_ids)
        );
    }
    let rewind = match &plan {
        PrefixPlan::Extend { from } => Ok(*from),
        PrefixPlan::Restore { snap_idx, pos, global } => engine.restore_state(pc.snapshot(*snap_idx), *global).map(|_| *pos),
        PrefixPlan::Reset => engine.reset_sequence().map(|_| 0),
    };
    let reuse_len = match rewind {
        Ok(p) => p,
        Err(e) => {
            pc.invalidate(lane);
            let _ = event_tx.send(StreamEvent::Error(engine_error("prefix restore", &e.to_string())));
            return None;
        }
    };
    pc.rewound_to(lane, &plan, reuse_len);
    pc.touch_prefixes(&prompt_ids);
    eprintln!(
        "[prefix] lane {lane}: {} - reused {reuse_len} of {} prompt tokens",
        match &plan {
            PrefixPlan::Extend { .. } => "same conversation",
            PrefixPlan::Restore { global: true, .. } => "restored a snapshot from another conversation",
            PrefixPlan::Restore { .. } => "restored a snapshot",
            PrefixPlan::Reset => "no cached prefix",
        },
        prompt_ids.len()
    );

    // Snapshot points: the first user-message boundary (shared system/tool prefix) and one
    // token before the prompt end (the last token is always re-processed, so this serves exact
    // repeats and follow-up turns alike). Prefill runs in segments so each snapshot lands
    // exactly on its point.
    let end = prompt_ids.len();
    let mut points: Vec<usize> = Vec::new();
    let shared_point = prompt_ids.iter().position(|&t| t == TOK_MESSAGE_USER);
    if let Some(bd) = shared_point {
        if bd > reuse_len && bd >= 16 {
            points.push(bd);
        }
    }
    if end >= 2 && end - 1 > reuse_len {
        points.push(end - 1);
    }
    points.sort_unstable();
    points.dedup();
    let mut cuts = vec![reuse_len];
    cuts.extend(points.iter().copied());
    cuts.push(end);
    for w in cuts.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a >= b {
            continue;
        }
        if cache_prompt && points.contains(&a) && !pc.has_snapshot_of(&prompt_ids[..a]) {
            let mut slot = pc.take_payload(a);
            match engine.snapshot_state_opt(a, &mut slot, drafter_coherent) {
                Ok(()) => pc.insert_kind(lane, a, slot.unwrap(), Some(a) == shared_point),
                Err(e) => eprintln!("[prefix-cache] snapshot at {a} failed: {e}"),
            }
        }
        if let Err(e) = engine.prefill(&prompt_ids[a..b], a, b == end) {
            pc.invalidate(lane);
            let _ = event_tx.send(StreamEvent::Error(engine_error(&format!("prefill error at pos {a}..{b}"), &e.to_string())));
            return None;
        }
        pc.live[lane].extend_from_slice(&prompt_ids[a..b]);
    }
    let pf_secs = t_pf.elapsed().as_secs_f64();
    let new_pf_toks = end - reuse_len;
    pc.stats.prefilled_tokens += new_pf_toks as u64;
    if new_pf_toks > 0 {
        *last_prefill_tok_s = new_pf_toks as f64 / pf_secs.max(1e-6);
    }
    let _ = engine.rebalance_after_prefill(128);

    let next_tok = match engine.sample_next_token(&sampling, &[]) {
        Ok(t) => t,
        Err(e) => {
            let _ = event_tx.send(StreamEvent::Error(engine_error("sample error", &e.to_string())));
            return None;
        }
    };
    Some(ActiveReq {
        lane,
        prompt_key: prompt_ids[..end.saturating_sub(1)].to_vec(),
        is_chat,
        single_tool_call,
        max_tokens,
        sampling,
        parser: match &forced_tool {
            Some(n) => AnswerParser::new_forced_tool(n),
            None => AnswerParser::new(),
        },
        raw_utf8: Utf8Stream::default(),
        stopper: StopMatcher::new(stop_strings),
        event_tx,
        req_tag: REQ_COUNTER.load(Ordering::Relaxed),
        n_tool_calls: 0,
        prompt_len: end,
        reuse_len,
        pf_secs,
        t_dec: Instant::now(),
        generated: Vec::with_capacity(max_tokens.min(4096)),
        pos: end,
        tap_row: engine.last_tap_rows.max(1) - 1,
        next_tok,
        finish_reason: "length",
        log_text: 0,
        log_think: 0,
        log_tail: String::new(),
        log_w0: (engine.telemetry().routed_weight_total, engine.telemetry().routed_weight_skipped),
        log_c0: {
            let t = engine.telemetry();
            (t.stage1_sync_rescues, t.stage1_skipped_tail_misses, t.stage2_disk_reads)
        },
    })
}

fn write_json_response(stream: &mut TcpStream, status: &str, body: &Value) {
    let payload = body.to_string();
    let resp = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: *\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
         Connection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.flush();
}

fn write_sse_headers(stream: &mut TcpStream) -> std::io::Result<()> {
    let hdr = "HTTP/1.1 200 OK\r\n\
               Content-Type: text/event-stream\r\n\
               Cache-Control: no-cache\r\n\
               Access-Control-Allow-Origin: *\r\n\
               Access-Control-Allow-Headers: *\r\n\
               Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
               Connection: close\r\n\r\n";
    stream.write_all(hdr.as_bytes())?;
    stream.flush()
}

fn handle_client(mut stream: TcpStream, cmd_tx: Sender<EngineCommand>) {
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    });

    let mut request_line = String::new();
    if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
        return;
    }
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return;
    }
    let method = parts[0];
    let path = parts[1].split('?').next().unwrap_or("/");
    let query = parts[1].split_once('?').map(|(_, q)| q).unwrap_or("");

    let mut content_length = 0usize;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }

    if method == "OPTIONS" {
        let resp = "HTTP/1.1 204 No Content\r\n\
                    Access-Control-Allow-Origin: *\r\n\
                    Access-Control-Allow-Headers: *\r\n\
                    Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
                    Content-Length: 0\r\n\
                    Connection: close\r\n\r\n";
        let _ = stream.write_all(resp.as_bytes());
        return;
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 && reader.read_exact(&mut body).is_err() {
        return;
    }

    match (method, path) {
        ("GET", "/health") => {
            write_json_response(&mut stream, "200 OK", &json!({"status": "ok", "model": SERVED_MODEL_NAME}));
        }
        ("GET", "/v1/models") => {
            let resp = json!({
                "object": "list",
                "data": [{
                    "id": SERVED_MODEL_NAME,
                    "object": "model",
                    "created": unix_secs(),
                    "owned_by": "smalltide",
                }]
            });
            write_json_response(&mut stream, "200 OK", &resp);
        }
        ("GET", "/expert-map") => {
            let page = crate::expert_map::PAGE;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
                page.len()
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.write_all(page.as_bytes());
            let _ = stream.flush();
        }
        ("GET", "/v1/expert-map") => {
            // Token text only to this machine: the server has no API keys and may listen on the LAN.
            let local = stream.peer_addr().map(|a| a.ip().is_loopback()).unwrap_or(false);
            let ask = crate::expert_map::Ask::parse(query);
            let (tx, rx) = mpsc::channel();
            static LAST: std::sync::Mutex<Option<Value>> = std::sync::Mutex::new(None);
            if cmd_tx.send(EngineCommand::ExpertMap { ask, with_text: local, reply_tx: tx }).is_err() {
                write_json_response(&mut stream, "503 Service Unavailable", &json!({"error": "the engine is not running"}));
                return;
            }
            match rx.recv_timeout(std::time::Duration::from_millis(1500)) {
                Ok(v) => {
                    let mut kept = v.clone();
                    if let Some(rs) = kept["routes"].as_array_mut() {
                        for r in rs {
                            r["token"] = Value::Null;
                        }
                    }
                    kept["tokens_shown"] = false.into();
                    *LAST.lock().unwrap() = Some(kept);
                    write_json_response(&mut stream, "200 OK", &v);
                }
                Err(_) => match LAST.lock().unwrap().clone() {
                    // The engine is inside a long step (a prefill): the last snapshot, marked old.
                    Some(mut v) => {
                        v["stale"] = true.into();
                        v["busy"] = true.into();
                        v["routes"] = json!([]);
                        v["events"] = json!([]);
                        v["route_seq"] = ask.since.into();
                        v["event_seq"] = ask.events_since.into();
                        write_json_response(&mut stream, "200 OK", &v);
                    }
                    None => write_json_response(&mut stream, "503 Service Unavailable", &json!({"error": "the engine is busy: no snapshot yet"})),
                },
            }
        }
        ("GET", "/v1/tide/stats") => {
            let (tx, rx) = mpsc::channel();
            if cmd_tx.send(EngineCommand::GetStats { reply_tx: tx }).is_ok() {
                if let Ok(stats) = rx.recv() {
                    write_json_response(&mut stream, "200 OK", &stats);
                    return;
                }
            }
            write_json_response(&mut stream, "500 Internal Server Error", &json!({"error": "engine unavailable"}));
        }
        ("POST", "/v1/tide/config") => {
            let req: TideConfigReq = match serde_json::from_slice(&body) {
                Ok(r) => r,
                Err(e) => {
                    write_json_response(
                        &mut stream,
                        "400 Bad Request",
                        &serde_json::to_value(ErrorBody {
                            error: ErrorDetail { message: &e.to_string(), err_type: "invalid_request_error" },
                        }).unwrap(),
                    );
                    return;
                }
            };
            let policy = match req.rescue_mode.as_deref().unwrap_or("threshold") {
                "exact" => RescuePolicy::Exact,
                "disabled" | "none" => RescuePolicy::Disabled,
                _ => RescuePolicy::Threshold(req.rescue_threshold.unwrap_or(0.25)),
            };
            let (tx, rx) = mpsc::channel();
            if cmd_tx
                .send(EngineCommand::UpdateConfig {
                    rescue_policy: policy,
                    reply_tx: tx,
                })
                .is_ok()
            {
                if let Ok(reply) = rx.recv() {
                    write_json_response(&mut stream, "200 OK", &reply);
                    return;
                }
            }
            write_json_response(&mut stream, "500 Internal Server Error", &json!({"error": "engine unavailable"}));
        }
        ("POST", "/v1/chat/completions") => {
            // Dev: SMALLTIDE_CAPTURE_DIR keeps every chat request body (replaying a failing one).
            if let Some(dir) = std::env::var_os("SMALLTIDE_CAPTURE_DIR") {
                let f = std::path::Path::new(&dir).join(format!("chat_{}.json", std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)));
                let _ = std::fs::write(&f, &body);
            }
            let req: ChatCompletionReq = match serde_json::from_slice(&body) {
                Ok(r) => r,
                Err(e) => {
                    write_json_response(
                        &mut stream,
                        "400 Bad Request",
                        &json!({"error": {"message": e.to_string(), "type": "invalid_request_error"}}),
                    );
                    return;
                }
            };
            let rendered = std::str::from_utf8(&body)
                .map_err(|e| e.to_string())
                .and_then(chat_format::parse_json)
                .and_then(|b| render_chat_request(&b));
            let (prompt_text, forced_tool, single_tool_call) = match rendered {
                Ok(r) => r,
                Err(e) => {
                    write_json_response(
                        &mut stream,
                        "400 Bad Request",
                        &json!({"error": {"message": e, "type": "invalid_request_error"}}),
                    );
                    return;
                }
            };
            // No limit given: like OpenAI, generate until the answer ends or the lane's context is
            // full (a thinking model can spend hundreds of tokens before its answer starts).
            let max_tokens = req.max_completion_tokens.or(req.max_tokens).unwrap_or(default_max_tokens());
            let sampling = SamplingParams {
                temperature: req.temperature.unwrap_or(0.6),
                top_p: req.top_p.unwrap_or(0.95),
                top_k: req.top_k.unwrap_or(40),
                repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
                seed: unix_secs() ^ REQ_COUNTER.load(Ordering::Relaxed),
            };
            let stop_strings = parse_stop_strings(&req.stop);
            let is_stream = req.stream.unwrap_or(false);
            let req_id = format!("chatcmpl-st-{}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed));
            let created = unix_secs();

            let (ev_tx, ev_rx) = mpsc::channel();
            if cmd_tx
                .send(EngineCommand::Generate {
                    prompt_text,
                    is_chat: true,
                    forced_tool,
                    single_tool_call,
                    max_tokens,
                    sampling,
                    stop_strings,
                    cache_prompt: req.cache_prompt.unwrap_or(true),
                    event_tx: ev_tx,
                })
                .is_err()
            {
                write_json_response(&mut stream, "500 Internal Server Error", &json!({"error": "engine down"}));
                return;
            }

            if is_stream {
                if write_sse_headers(&mut stream).is_err() {
                    return;
                }
                let role_chunk = json!({
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": SERVED_MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "delta": { "role": "assistant" },
                        "finish_reason": Value::Null
                    }]
                });
                let _ = write!(stream, "data: {}\n\n", role_chunk);
                let _ = stream.flush();

                while let Ok(ev) = ev_rx.recv() {
                    match ev {
                        StreamEvent::Delta { channel, text } => {
                            let delta = match channel {
                                OutputChannel::Thinking => json!({ "reasoning_content": text }),
                                OutputChannel::Text => json!({ "content": text }),
                            };
                            let chunk = json!({
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": SERVED_MODEL_NAME,
                                "choices": [{
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": Value::Null
                                }]
                            });
                            if write!(stream, "data: {}\n\n", chunk).is_err() || stream.flush().is_err() {
                                return;
                            }
                        }
                        StreamEvent::ToolCall { index, id, name, arguments } => {
                            let chunk = json!({
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": SERVED_MODEL_NAME,
                                "choices": [{
                                    "index": 0,
                                    "delta": { "tool_calls": [{
                                        "index": index,
                                        "id": id,
                                        "type": "function",
                                        "function": { "name": name, "arguments": arguments }
                                    }]},
                                    "finish_reason": Value::Null
                                }]
                            });
                            if write!(stream, "data: {}\n\n", chunk).is_err() || stream.flush().is_err() {
                                return;
                            }
                        }
                        StreamEvent::Done {
                            finish_reason,
                            prompt_tokens,
                            completion_tokens,
                            cached_tokens,
                            ..
                        } => {
                            let end_chunk = json!({
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": SERVED_MODEL_NAME,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason
                                }],
                                "usage": {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens,
                                    "prompt_tokens_details": { "cached_tokens": cached_tokens }
                                }
                            });
                            let _ = write!(stream, "data: {}\n\ndata: [DONE]\n\n", end_chunk);
                            let _ = stream.flush();
                            break;
                        }
                        StreamEvent::Error(msg) => {
                            let _ = write!(stream, "data: {}\n\ndata: [DONE]\n\n", json!({"error": msg}));
                            let _ = stream.flush();
                            break;
                        }
                    }
                }
            } else {
                let mut content = String::new();
                let mut reasoning = String::new();
                let mut tool_calls: Vec<Value> = Vec::new();
                let mut finish_reason = "length";
                let (mut pt, mut ct, mut pf_s, mut dec_s, mut hit) = (0usize, 0usize, 0.0, 0.0, 1.0);
                let mut cached = 0usize;
                while let Ok(ev) = ev_rx.recv() {
                    match ev {
                        StreamEvent::Delta { channel, text } => match channel {
                            OutputChannel::Thinking => reasoning.push_str(&text),
                            OutputChannel::Text => content.push_str(&text),
                        },
                        StreamEvent::ToolCall { id, name, arguments, .. } => tool_calls.push(json!({
                            "id": id,
                            "type": "function",
                            "function": { "name": name, "arguments": arguments }
                        })),
                        StreamEvent::Done {
                            finish_reason: fr,
                            prompt_tokens,
                            completion_tokens,
                            cached_tokens,
                            prefill_secs,
                            decode_secs,
                            vram_hit_rate,
                        } => {
                            finish_reason = fr;
                            cached = cached_tokens;
                            pt = prompt_tokens;
                            ct = completion_tokens;
                            pf_s = prefill_secs;
                            dec_s = decode_secs;
                            hit = vram_hit_rate;
                            break;
                        }
                        StreamEvent::Error(msg) => {
                            write_json_response(&mut stream, "500 Internal Server Error", &json!({"error": {"message": msg}}));
                            return;
                        }
                    }
                }
                let mut message = json!({
                    "role": "assistant",
                    "content": if content.is_empty() && !tool_calls.is_empty() { Value::Null } else { Value::String(content) },
                });
                if !reasoning.is_empty() {
                    message["reasoning_content"] = Value::String(reasoning);
                }
                if !tool_calls.is_empty() {
                    message["tool_calls"] = Value::Array(tool_calls);
                }
                let resp = json!({
                    "id": req_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": SERVED_MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "total_tokens": pt + ct,
                        "prompt_tokens_details": { "cached_tokens": cached }
                    },
                    "timings": {
                        "prompt_ms": pf_s * 1000.0,
                        "prompt_per_second": (pt - cached) as f64 / pf_s.max(1e-6),
                        "predicted_per_second": ct as f64 / dec_s.max(1e-6),
                        "vram_hit_rate_pct": hit * 100.0
                    }
                });
                write_json_response(&mut stream, "200 OK", &resp);
            }
        }
        ("POST", "/v1/completions") => {
            let req: CompletionReq = match serde_json::from_slice(&body) {
                Ok(r) => r,
                Err(e) => {
                    write_json_response(
                        &mut stream,
                        "400 Bad Request",
                        &json!({"error": {"message": e.to_string(), "type": "invalid_request_error"}}),
                    );
                    return;
                }
            };
            let prompt_text = match &req.prompt {
                Value::String(s) => s.clone(),
                Value::Array(arr) => arr
                    .first()
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                _ => String::new(),
            };
            let max_tokens = req.max_tokens.unwrap_or(default_max_tokens());
            let sampling = SamplingParams {
                temperature: req.temperature.unwrap_or(0.6),
                top_p: req.top_p.unwrap_or(0.95),
                top_k: req.top_k.unwrap_or(40),
                repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
                seed: unix_secs() ^ REQ_COUNTER.load(Ordering::Relaxed),
            };
            let stop_strings = parse_stop_strings(&req.stop);
            let is_stream = req.stream.unwrap_or(false);
            let req_id = format!("cmpl-st-{}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed));
            let created = unix_secs();

            let (ev_tx, ev_rx) = mpsc::channel();
            if cmd_tx
                .send(EngineCommand::Generate {
                    prompt_text,
                    is_chat: false,
                    forced_tool: None,
                    single_tool_call: false,
                    max_tokens,
                    sampling,
                    stop_strings,
                    cache_prompt: req.cache_prompt.unwrap_or(true),
                    event_tx: ev_tx,
                })
                .is_err()
            {
                write_json_response(&mut stream, "500 Internal Server Error", &json!({"error": "engine down"}));
                return;
            }

            if is_stream {
                if write_sse_headers(&mut stream).is_err() {
                    return;
                }
                while let Ok(ev) = ev_rx.recv() {
                    match ev {
                        StreamEvent::ToolCall { .. } => {}
                        StreamEvent::Delta { text, .. } => {
                            let chunk = json!({
                                "id": req_id,
                                "object": "text_completion",
                                "created": created,
                                "model": SERVED_MODEL_NAME,
                                "choices": [{
                                    "index": 0,
                                    "text": text,
                                    "finish_reason": Value::Null
                                }]
                            });
                            if write!(stream, "data: {}\n\n", chunk).is_err() || stream.flush().is_err() {
                                return;
                            }
                        }
                        StreamEvent::Done {
                            finish_reason,
                            prompt_tokens,
                            completion_tokens,
                            cached_tokens,
                            ..
                        } => {
                            let end_chunk = json!({
                                "id": req_id,
                                "object": "text_completion",
                                "created": created,
                                "model": SERVED_MODEL_NAME,
                                "choices": [{
                                    "index": 0,
                                    "text": "",
                                    "finish_reason": finish_reason
                                }],
                                "usage": {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens,
                                    "prompt_tokens_details": { "cached_tokens": cached_tokens }
                                }
                            });
                            let _ = write!(stream, "data: {}\n\ndata: [DONE]\n\n", end_chunk);
                            let _ = stream.flush();
                            break;
                        }
                        StreamEvent::Error(msg) => {
                            let _ = write!(stream, "data: {}\n\ndata: [DONE]\n\n", json!({"error": msg}));
                            let _ = stream.flush();
                            break;
                        }
                    }
                }
            } else {
                let mut text = String::new();
                let mut finish_reason = "length";
                let (mut pt, mut ct, mut pf_s, mut dec_s, mut hit) = (0usize, 0usize, 0.0, 0.0, 1.0);
                let mut cached = 0usize;
                while let Ok(ev) = ev_rx.recv() {
                    match ev {
                        StreamEvent::Delta { text: piece, .. } => text.push_str(&piece),
                        StreamEvent::ToolCall { .. } => {}
                        StreamEvent::Done {
                            finish_reason: fr,
                            prompt_tokens,
                            completion_tokens,
                            cached_tokens,
                            prefill_secs,
                            decode_secs,
                            vram_hit_rate,
                        } => {
                            finish_reason = fr;
                            cached = cached_tokens;
                            pt = prompt_tokens;
                            ct = completion_tokens;
                            pf_s = prefill_secs;
                            dec_s = decode_secs;
                            hit = vram_hit_rate;
                            break;
                        }
                        StreamEvent::Error(msg) => {
                            write_json_response(&mut stream, "500 Internal Server Error", &json!({"error": {"message": msg}}));
                            return;
                        }
                    }
                }
                let resp = json!({
                    "id": req_id,
                    "object": "text_completion",
                    "created": created,
                    "model": SERVED_MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "text": text,
                        "finish_reason": finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "total_tokens": pt + ct,
                        "prompt_tokens_details": { "cached_tokens": cached }
                    },
                    "timings": {
                        "prompt_ms": pf_s * 1000.0,
                        "prompt_per_second": (pt - cached) as f64 / pf_s.max(1e-6),
                        "predicted_per_second": ct as f64 / dec_s.max(1e-6),
                        "vram_hit_rate_pct": hit * 100.0
                    }
                });
                write_json_response(&mut stream, "200 OK", &resp);
            }
        }
        _ => {
            write_json_response(
                &mut stream,
                "404 Not Found",
                &json!({"error": {"message": format!("unknown route {method} {path}"), "type": "invalid_request_error"}}),
            );
        }
    }
}
