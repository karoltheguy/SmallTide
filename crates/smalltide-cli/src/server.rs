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
use smalltide_core::{ComputeBackend, DType, Inkling2Config, RescuePolicy, TideConfig};
use smalltide_io::dense_loader::Inkling2DenseHost;
use smalltide_io::stide_pack::{ExpertSlabLayout, StidePackReader};
use smalltide_io::tokenizer::BpeTokenizer;
use crate::chat_format::{self, AnswerEvent, AnswerParser, OJson, StopMatcher};
use crate::prefix_cache::{PrefixCache, PrefixPlan};
use smalltide_io::tokenizer::Utf8Stream;
use smalltide_model_inkling2::engine::{Inkling2Engine, SamplingParams, SeqSnapshot};
use smalltide_model_inkling2::weights::Inkling2DenseGpu;
use smalltide_tide::TideExecutor;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
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
        event_tx: Sender<StreamEvent>,
    },
    GetStats {
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
    let mut model_dir = r"models\Inkling-2-Mini".to_string();
    let mut host = "0.0.0.0".to_string();
    let mut port: u16 = 8000;
    let mut max_ctx: usize = 32_768;
    let mut rescue_mode = "threshold".to_string();
    let mut rescue_threshold: f32 = 0.25;
    let mut nvfp4_slots_override: Option<usize> = None;
    let mut mxfp8_slots_override: Option<usize> = None;
    let mut ram_gb: f64 = 10.0;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => { model_dir = args[i + 1].clone(); i += 2; }
            "--host" if i + 1 < args.len() => { host = args[i + 1].clone(); i += 2; }
            "--port" if i + 1 < args.len() => { port = args[i + 1].parse().unwrap_or(8000); i += 2; }
            "--max-ctx" if i + 1 < args.len() => { max_ctx = args[i + 1].parse().unwrap_or(32_768); i += 2; }
            "--rescue-mode" if i + 1 < args.len() => { rescue_mode = args[i + 1].clone(); i += 2; }
            "--rescue-threshold" if i + 1 < args.len() => { rescue_threshold = args[i + 1].parse().unwrap_or(0.25); i += 2; }
            "--nvfp4-slots" if i + 1 < args.len() => { nvfp4_slots_override = args[i + 1].parse().ok(); i += 2; }
            "--mxfp8-slots" if i + 1 < args.len() => { mxfp8_slots_override = args[i + 1].parse().ok(); i += 2; }
            "--ram-gb" if i + 1 < args.len() => { ram_gb = args[i + 1].parse().unwrap_or(10.0); i += 2; }
            other => { eprintln!("Unknown serve flag: {other}"); i += 1; }
        }
    }

    let cfg = Inkling2Config::default();
    let dir = Path::new(&model_dir);
    let t_start = Instant::now();
    println!("=== SmallTide OpenAI API Server (model: {SERVED_MODEL_NAME}) ===");
    println!("Loading tokenizer, FP8 dense weights, and TideLM 2-stage expert pools from {} ...", dir.display());

    let tok = Arc::new(BpeTokenizer::from_file(&dir.join("tokenizer.json")).expect("load tokenizer.json"));
    let dense_host = Inkling2DenseHost::load_from_safetensors(
        &cfg,
        &dir.join("model-00001-of-00001.safetensors"),
    )
    .expect("load dense weights");
    let pack = StidePackReader::open(&dir.join("experts.stide")).expect("open experts.stide");
    let backend = Arc::new(HipBackend::new(0).expect("init HipBackend"));

    let dense_gpu = Inkling2DenseGpu::upload(backend.clone(), dense_host).expect("upload dense GPU");
    let info_after_dense = backend.device_info().expect("device_info");

    let nvfp4_slab = ExpertSlabLayout::for_inkling2(&cfg, DType::NVFP4).aligned_slab_bytes;
    let mxfp8_slab = ExpertSlabLayout::for_inkling2(&cfg, DType::MXFP8).aligned_slab_bytes;
    let est_kv_bytes = 40 * (2 * 8 * cfg.local_window_size * 128 + 2 * 8 * cfg.local_window_size * 4)
        + 8 * (2 * 8 * max_ctx * 128 + 2 * 8 * max_ctx * 4)
        + 64 * 1024 * 1024;
    let headroom_bytes = 900 * 1024 * 1024usize;
    let scratch_ring_bytes = 8 * (nvfp4_slab + mxfp8_slab);
    let avail_expert_vram = info_after_dense
        .free_vram_bytes
        .saturating_sub(est_kv_bytes + headroom_bytes + scratch_ring_bytes);

    let total_nvfp4 = cfg.mxfp8_start_layer * cfg.n_routed_experts;
    let total_mxfp8 = (cfg.num_physical_layers - cfg.mxfp8_start_layer) * cfg.n_routed_experts;
    let (nvfp4_slots, mxfp8_slots) = match (nvfp4_slots_override, mxfp8_slots_override) {
        (Some(n4), Some(m8)) => (n4.min(total_nvfp4), m8.min(total_mxfp8)),
        (Some(n4), None) => {
            let n4 = n4.min(total_nvfp4);
            let rem = avail_expert_vram.saturating_sub(n4 * nvfp4_slab);
            (n4, (rem / mxfp8_slab).min(total_mxfp8).max(16))
        }
        (None, Some(m8)) => {
            let m8 = m8.min(total_mxfp8);
            let rem = avail_expert_vram.saturating_sub(m8 * mxfp8_slab);
            ((rem / nvfp4_slab).min(total_nvfp4).max(1536), m8)
        }
        (None, None) => {
            let min_mxfp8 = 128usize;
            let max_n4_budget = avail_expert_vram.saturating_sub(min_mxfp8 * mxfp8_slab);
            let n4 = (max_n4_budget / nvfp4_slab).min(total_nvfp4).max(1536);
            let rem = avail_expert_vram.saturating_sub(n4 * nvfp4_slab);
            let m8 = (rem / mxfp8_slab).min(total_mxfp8).max(16);
            (n4, m8)
        }
    };

    let rescue_policy = match rescue_mode.as_str() {
        "exact" => RescuePolicy::Exact,
        "disabled" | "none" => RescuePolicy::Disabled,
        _ => RescuePolicy::Threshold(rescue_threshold),
    };
    let mut tide_cfg = TideConfig::default();
    tide_cfg.pinned_staging_mib = 256;
    tide_cfg.rescue_policy = rescue_policy;

    let ram_budget_bytes = (ram_gb * 1024.0 * 1024.0 * 1024.0) as usize;
    let mut tide = TideExecutor::new(
        backend.clone(),
        cfg.clone(),
        tide_cfg,
        nvfp4_slots,
        mxfp8_slots,
        ram_budget_bytes,
        Some(pack),
    )
    .expect("create TideExecutor");

    let preload_ops = tide.scheduler.plan_pre_prefill(None);
    tide.execute_transfers(&preload_ops, true, false)
        .expect("Mode 1 pre-prefill");
    backend.sync_stream(tide.copy_stream).expect("sync prefill DMA");

    let engine = Inkling2Engine::new(cfg, dense_gpu, tide, max_ctx).expect("create Inkling2Engine");
    let info_ready = backend.device_info().expect("device_info ready");
    let (pre_r, core_r, coda_r) = engine.tide.scheduler.cache.count_vram_resident();
    println!(
        "[Ready in {:.2}s] VRAM free: {:.2}/{:.2} GiB | KV({} ctx): {:.2} GiB | Experts VRAM: pre={}, core={}/1536, coda={} | Rescue: {:?}",
        t_start.elapsed().as_secs_f64(),
        info_ready.free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        info_ready.total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        max_ctx,
        engine.kv_vram_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        pre_r,
        core_r,
        coda_r,
        rescue_policy,
    );

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

fn engine_worker_loop(
    mut engine: Inkling2Engine<HipBackend>,
    tok: Arc<BpeTokenizer>,
    cmd_rx: Receiver<EngineCommand>,
) {
    // Prompt prefix cache: live-sequence extension + rewind to pinned-RAM state snapshots.
    let mut pc: PrefixCache<SeqSnapshot> = PrefixCache::new(4);
    let mut last_decode_tok_s: f64 = 0.0;
    let mut last_prefill_tok_s: f64 = 0.0;

    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
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
                    "vram_hit_rate_pct": tel.total_vram_hit_rate() * 100.0,
                    "prelude_hit_rate_pct": pct(tel.vram_hits_prelude, tel.vram_misses_prelude),
                    "loop_core_hit_rate_pct": pct(tel.vram_hits_loop, tel.vram_misses_loop),
                    "coda_hit_rate_pct": pct(tel.vram_hits_coda, tel.vram_misses_coda),
                    "stage1_sync_rescues": tel.stage1_sync_rescues,
                    "stage1_async_promotions": tel.stage1_async_promotions,
                    "stage1_skipped_tail_misses": tel.stage1_skipped_tail_misses,
                    "stage2_disk_reads": tel.stage2_disk_reads,
                    "experts_vram_resident": { "prelude": pre, "loop_core": core, "coda": coda },
                    "experts_ram_warm": ram,
                    "prefix_cache": {
                        "requests": pc.stats.requests,
                        "extend_hits": pc.stats.extend_hits,
                        "restore_hits": pc.stats.restore_hits,
                        "misses": pc.stats.misses,
                        "reused_tokens": pc.stats.reused_tokens,
                        "prefilled_tokens": pc.stats.prefilled_tokens,
                        "snapshots_taken": pc.stats.snapshots_taken,
                        "snapshot_positions": pc.snapshot_positions(),
                        "live_tokens": pc.cached.len(),
                        "snapshot_mib": engine.snapshot_bytes() as f64 / (1024.0 * 1024.0),
                    },
                    "kv_max_ctx": engine.global_max_seq,
                    "kv_vram_gib": engine.kv_vram_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                    "vram_free_gib": vram_info.as_ref().map(|d| d.free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)),
                    "vram_total_gib": vram_info.as_ref().map(|d| d.total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)),
                });
                let _ = reply_tx.send(stats);
            }
            EngineCommand::UpdateConfig {
                rescue_policy,
                reply_tx,
            } => {
                engine.tide.scheduler.tide_cfg.rescue_policy = rescue_policy;
                let _ = reply_tx.send(json!({
                    "ok": true,
                    "rescue_policy": format!("{:?}", engine.tide.scheduler.tide_cfg.rescue_policy),
                }));
            }
            EngineCommand::Generate {
                prompt_text,
                is_chat,
                forced_tool,
                single_tool_call,
                max_tokens,
                sampling,
                stop_strings,
                event_tx,
            } => {
                let prompt_ids = tok.encode(&prompt_text);
                if prompt_ids.is_empty() {
                    let _ = event_tx.send(StreamEvent::Error("empty prompt".into()));
                    continue;
                }
                if prompt_ids.len() >= engine.global_max_seq {
                    let _ = event_tx.send(StreamEvent::Error(format!(
                        "prompt length {} exceeds max_ctx {}",
                        prompt_ids.len(),
                        engine.global_max_seq
                    )));
                    continue;
                }

                let t_pf = Instant::now();
                let plan = pc.plan(&prompt_ids);
                if std::env::var_os("SMALLTIDE_PREFIX_DEBUG").is_some() {
                    let lcp = pc.cached.iter().zip(&prompt_ids).take_while(|(a, b)| a == b).count();
                    let show = |ids: &[u32]| -> String {
                        let end = (lcp + 6).min(ids.len());
                        format!("{:?}", ids[lcp.min(end)..end].iter().map(|&t| tok.decode(&[t])).collect::<Vec<_>>())
                    };
                    eprintln!(
                        "[prefix] plan={plan:?} live={} prompt={} lcp={lcp} live_next={} prompt_next={}",
                        pc.cached.len(),
                        prompt_ids.len(),
                        show(&pc.cached),
                        show(&prompt_ids)
                    );
                }
                let rewind = match &plan {
                    PrefixPlan::Extend { from } => Ok(*from),
                    PrefixPlan::Restore { snap_idx, pos } => {
                        engine.restore_state(pc.snapshot(*snap_idx)).map(|_| *pos)
                    }
                    PrefixPlan::Reset => engine.reset_sequence().map(|_| 0),
                };
                let reuse_len = match rewind {
                    Ok(p) => p,
                    Err(e) => {
                        pc.invalidate();
                        let _ = event_tx.send(StreamEvent::Error(format!("prefix restore: {e}")));
                        continue;
                    }
                };
                pc.rewound_to(&plan, reuse_len);

                // Snapshot points: first user-message boundary (shared system/tool prefix) and
                // the prompt end (multi-turn divergence right after the previous prompt).
                let boundary = prompt_ids
                    .iter()
                    .position(|&t| t == TOK_MESSAGE_USER)
                    .filter(|&b| b > reuse_len && b >= 16);
                let mut pf_err = None;
                for idx in reuse_len..prompt_ids.len() {
                    if Some(idx) == boundary && !pc.has_snapshot_at(idx) {
                        let mut slot = pc.take_payload();
                        match engine.snapshot_state(idx, &mut slot) {
                            Ok(()) => pc.insert(idx, slot.unwrap()),
                            Err(e) => eprintln!("[prefix-cache] snapshot at {idx} failed: {e}"),
                        }
                    }
                    let is_last = idx + 1 == prompt_ids.len();
                    if let Err(e) = engine.forward_token(prompt_ids[idx], idx, true, is_last) {
                        pf_err = Some(format!("prefill error at pos {idx}: {e}"));
                        break;
                    }
                    pc.cached.push(prompt_ids[idx]);
                }
                if let Some(err) = pf_err {
                    pc.invalidate();
                    let _ = event_tx.send(StreamEvent::Error(err));
                    continue;
                }
                let end = prompt_ids.len();
                if !pc.has_snapshot_at(end) {
                    let mut slot = pc.take_payload();
                    match engine.snapshot_state(end, &mut slot) {
                        Ok(()) => pc.insert(end, slot.unwrap()),
                        Err(e) => eprintln!("[prefix-cache] snapshot at {end} failed: {e}"),
                    }
                }
                let pf_secs = t_pf.elapsed().as_secs_f64();
                let new_pf_toks = prompt_ids.len() - reuse_len;
                pc.stats.prefilled_tokens += new_pf_toks as u64;
                if new_pf_toks > 0 {
                    last_prefill_tok_s = new_pf_toks as f64 / pf_secs.max(1e-6);
                }
                let _ = engine.rebalance_after_prefill(128);

                let t_dec = Instant::now();
                let mut generated: Vec<u32> = Vec::with_capacity(max_tokens.min(4096));
                let mut pos = prompt_ids.len();
                let mut finish_reason: &'static str = "length";
                let mut parser = match &forced_tool {
                    Some(n) => AnswerParser::new_forced_tool(n),
                    None => AnswerParser::new(),
                };
                let mut raw_utf8 = Utf8Stream::default();
                let mut stopper = StopMatcher::new(stop_strings);
                let req_tag = REQ_COUNTER.load(Ordering::Relaxed);
                let mut n_tool_calls = 0usize;

                // Routes parsed answer events to the client. Returns false to stop generating.
                let mut emit = |evs: Vec<AnswerEvent>, finish: &mut &'static str| -> bool {
                    for ev in evs {
                        let sent = match ev {
                            AnswerEvent::Text(t) => {
                                let (out, stopped) = stopper.push(&t);
                                let ok = out.is_empty()
                                    || event_tx.send(StreamEvent::Delta { channel: OutputChannel::Text, text: out }).is_ok();
                                if stopped {
                                    *finish = "stop";
                                    return false;
                                }
                                ok
                            }
                            AnswerEvent::Thinking(t) => event_tx
                                .send(StreamEvent::Delta { channel: OutputChannel::Thinking, text: t })
                                .is_ok(),
                            AnswerEvent::ToolCall { name, arguments } => {
                                let index = n_tool_calls;
                                n_tool_calls += 1;
                                let ok = event_tx
                                    .send(StreamEvent::ToolCall {
                                        index,
                                        id: format!("call_{req_tag:x}_{index}"),
                                        name,
                                        arguments,
                                    })
                                    .is_ok();
                                if ok && single_tool_call {
                                    // A single call was requested: the answer ends here.
                                    *finish = "stop";
                                    return false;
                                }
                                ok
                            }
                            AnswerEvent::EndOfAnswer => {
                                *finish = "stop";
                                return false;
                            }
                        };
                        if !sent {
                            // Client disconnected: abort generation.
                            *finish = "stop";
                            return false;
                        }
                    }
                    true
                };

                let mut next_tok = match engine.sample_next_token(&sampling, &generated) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = event_tx.send(StreamEvent::Error(format!("sample error: {e}")));
                        continue;
                    }
                };

                for step in 0..max_tokens {
                    if next_tok == TOK_EOS || next_tok == TOK_EOT {
                        finish_reason = "stop";
                        break;
                    }
                    generated.push(next_tok);
                    let bytes = tok.token_bytes(next_tok);
                    let evs = if is_chat {
                        parser.feed(next_tok, bytes, tok.is_special(next_tok))
                    } else {
                        let t = raw_utf8.push(bytes);
                        if t.is_empty() { Vec::new() } else { vec![AnswerEvent::Text(t)] }
                    };
                    if !emit(evs, &mut finish_reason) {
                        break;
                    }

                    if step + 1 >= max_tokens || pos >= engine.global_max_seq {
                        break;
                    }
                    if let Err(e) = engine.forward_token(next_tok, pos, false, true) {
                        let _ = event_tx.send(StreamEvent::Error(format!("decode error: {e}")));
                        pc.invalidate();
                        break;
                    }
                    let _ = engine.post_decode_tide_step();
                    pc.cached.push(next_tok);
                    pos += 1;
                    next_tok = match engine.sample_next_token(&sampling, &generated) {
                        Ok(t) => t,
                        Err(e) => {
                            let _ = event_tx.send(StreamEvent::Error(format!("sample error: {e}")));
                            break;
                        }
                    };
                }
                // Flush held-back UTF-8, partial stop-string prefixes and an unterminated call.
                let tail = if is_chat {
                    parser.finish()
                } else {
                    let t = raw_utf8.finish();
                    if t.is_empty() { Vec::new() } else { vec![AnswerEvent::Text(t)] }
                };
                let mut fr = finish_reason;
                let tail_ok = emit(tail, &mut fr);
                drop(emit);
                if tail_ok && fr != "stop" {
                    let held = stopper.finish();
                    if !held.is_empty() {
                        let _ = event_tx.send(StreamEvent::Delta { channel: OutputChannel::Text, text: held });
                    }
                }
                finish_reason = fr;
                if n_tool_calls > 0 && finish_reason == "stop" {
                    finish_reason = "tool_calls";
                }

                let dec_secs = t_dec.elapsed().as_secs_f64();
                if !generated.is_empty() {
                    last_decode_tok_s = generated.len() as f64 / dec_secs.max(1e-6);
                }
                let hit_rate = engine.telemetry().total_vram_hit_rate();
                let _ = event_tx.send(StreamEvent::Done {
                    finish_reason,
                    cached_tokens: reuse_len,
                    prompt_tokens: prompt_ids.len(),
                    completion_tokens: generated.len(),
                    prefill_secs: pf_secs,
                    decode_secs: dec_secs,
                    vram_hit_rate: hit_rate,
                });
            }
        }
    }
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
            let max_tokens = req.max_completion_tokens.or(req.max_tokens).unwrap_or(512);
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
            let max_tokens = req.max_tokens.unwrap_or(256);
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
