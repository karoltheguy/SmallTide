//! GET /expert-map (the page) and GET /v1/expert-map (its data): which routed experts are resident
//! now and how the decoded tokens route through them. Ported from TideLM
//! (tidelm-server/src/expert_map.rs + expert_map.html). The engine thread answers between its
//! steps with a copy of host data; nothing is asked of the engine unless a page polls.

use smalltide_tide::expert_map::{ExpertMap, RescueSnapshot, RESCUE_HIST_BIN_PCT};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn rescue_json(r: &RescueSnapshot) -> serde_json::Value {
    let zones: Vec<serde_json::Value> = r
        .zones
        .iter()
        .map(|z| {
            serde_json::json!({
                "name": z.name,
                "layers": z.layers_label,
                "gate": z.gate,
                "base_gate": z.base_gate,
                "base_budget": z.base_budget,
                "avg_budget": (z.avg_budget * 100.0).round() / 100.0,
                "avg_used": (z.avg_used * 100.0).round() / 100.0,
                "p95_used": z.p95_used,
                "max_used": z.max_used,
                "util_pct": (z.util_pct * 10.0).round() / 10.0,
                "capped_pct": (z.capped_pct * 10.0).round() / 10.0,
                "zero_pct": (z.zero_pct * 10.0).round() / 10.0,
                "boot_capped_pct": (z.boot_capped_pct * 10.0).round() / 10.0,
                "boot_rescued": z.boot_rescued,
                "avg_capped_layers": (z.avg_capped_layers * 100.0).round() / 100.0,
                "avg_blocked_picks": (z.avg_blocked_picks * 100.0).round() / 100.0,
                "avg_below_gate_picks": (z.avg_below_gate_picks * 100.0).round() / 100.0,
                "avg_blocked_weight": (z.avg_blocked_weight * 1000.0).round() / 1000.0,
                "avg_below_weight": (z.avg_below_weight * 1000.0).round() / 1000.0,
                "rescued_share_avg": (z.rescued_share_avg * 1000.0).round() / 1000.0,
                "rescued_share_min": (z.rescued_share_min * 1000.0).round() / 1000.0,
                "miss_hist": z.miss_hist,
                "status": z.status,
            })
        })
        .collect();
    serde_json::json!({
        "policy": r.policy,
        "auto_gate": r.auto_gate,
        "auto_target_pct": r.auto_target_pct,
        "auto_gate_min": r.auto_gate_min,
        "auto_gate_max": r.auto_gate_max,
        "window_steps": r.window_steps,
        "total_steps": r.total_steps,
        "base_budget_total": r.base_budget_total,
        "last_verify_rows": r.last_verify_rows,
        "hist_bin_width_pct": RESCUE_HIST_BIN_PCT,
        "zones": zones,
    })
}

pub const PAGE: &str = include_str!("expert_map.html");

/// When this server process first answered the page (ms since the Unix epoch). The route and
/// event sequence numbers start again with the process, so a page left open across a restart
/// must start again from 0 too: it sees this value change.
fn boot_id() -> u64 {
    static BOOT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BOOT.get_or_init(|| SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0))
}

/// What a page asks for, from the query: `since` the last route it holds and `events_since` the last event (only
/// newer ones are sent), `demand=0` (leave out usage rows), `routes=0` (leave out route payloads while advancing
/// `route_seq`), `compact=1` (base64-pack route picks when `experts <= 256`), and `states_hash=<hex>` (omit
/// `states` when unchanged).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ask {
    pub since: u64,
    pub events_since: u64,
    pub demand: bool,
    pub routes: bool,
    pub compact: bool,
    pub states_hash: u64,
}

impl Ask {
    pub fn parse(query: &str) -> Ask {
        let mut a = Ask { demand: true, routes: true, ..Ask::default() };
        for (k, v) in query.split('&').filter_map(|kv| kv.split_once('=')) {
            match k {
                "since" => a.since = v.parse().unwrap_or(0),
                "events_since" => a.events_since = v.parse().unwrap_or(0),
                "demand" => a.demand = v != "0",
                "routes" => a.routes = v != "0",
                "compact" => a.compact = v == "1",
                "states_hash" => a.states_hash = u64::from_str_radix(v, 16).unwrap_or(0),
                _ => {}
            }
        }
        a
    }
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    if h == 0 { 1 } else { h }
}

fn base64_encode(data: &[u8]) -> String {
    const TAB: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let v = (b0 << 16) | (b1 << 8) | b2;
        out.push(TAB[((v >> 18) & 63) as usize] as char);
        out.push(TAB[((v >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 { TAB[((v >> 6) & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { TAB[(v & 63) as usize] as char } else { '=' });
    }
    out
}

/// The snapshot as the page reads it. `boot`: changes when the server restarts (see `boot_id`). `states`: a string a layer, a digit an expert (0 not loaded,
/// 1 resident, 2 in the hot-read ring, 3 being read; null when `ask.states_hash` matches `states_hash`); `demand`: the same shape in base 36, an
/// expert's share of its layer's routing history (0 none, z the layer's most used; null when not asked for,
/// `has_demand` says whether the engine keeps one). `events`: the engine's recent residency events after
/// `events_since`, oldest first, each [seq, age ms, row, expert, kind, share in thousandths, victim row,
/// victim expert] (-1, -1 without a victim; `event_kinds` names the kinds) - what brought each expert in: a
/// missed pick (its share of the token's routing weight), an inline rescue, an adaptation, a preload.
/// `routes`: the decoded tokens' routing after `since`, oldest first, each {seq, age_ms, lane, pos, token (its
/// text, to a keyed request only: it is the conversation's; else null), k, experts [row * k + pick], share
/// (thousandths, the same order), missed (a digit a pick: 1 when it was not resident and not rescued, so the pass skipped it)}
/// — or when `compact=1` and `experts <= 256`, `p`: base64 of 2 bytes per pick (`[expert_u8, (missed<<7)|share_0_127]`).
pub fn json(m: &ExpertMap, model: &str, engine: &str, ask: &Ask, token: Option<&dyn Fn(u32) -> String>) -> serde_json::Value {
    let c = m.counts();
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    let row = |l: u16| m.layers.binary_search(&(l as usize)).map(|r| r as i64).unwrap_or(-1);
    let events: Vec<serde_json::Value> = m.events.iter().filter(|e| e.seq > ask.events_since && row(e.layer) >= 0).map(|e| {
        let (vr, vx) = e.victim.map(|(l, x)| (row(l), x as i64)).unwrap_or((-1, -1));
        serde_json::json!([e.seq, ((now - e.at) * 1000.0).max(0.0).round() as i64, row(e.layer), e.expert, e.kind,
                           (e.share * 1000.0).round() as i64, vr, vx])
    }).collect();
    let pack_routes = ask.compact && m.experts <= 256;
    let routes: Vec<serde_json::Value> = if !ask.routes {
        Vec::new()
    } else {
        m.routes.iter().filter(|r| r.seq > ask.since && r.picks.len() == m.layers.len() * r.topk as usize).map(|r| {
            let age_ms = ((now - r.at) * 1000.0).max(0.0).round() as i64;
            if pack_routes {
                let mut raw = Vec::with_capacity(r.picks.len() * 2);
                for &(ex, sh, miss) in &r.picks {
                    let s7 = (sh.clamp(0.0, 1.0) * 127.0).round() as u8;
                    raw.push(ex as u8);
                    raw.push((if miss { 0x80 } else { 0 }) | (s7 & 0x7f));
                }
                serde_json::json!({
                    "seq": r.seq, "age_ms": age_ms, "lane": r.lane, "pos": r.pos,
                    "token": token.map(|f| f(r.token)), "k": r.topk,
                    "p": base64_encode(&raw),
                })
            } else {
                serde_json::json!({
                    "seq": r.seq, "age_ms": age_ms, "lane": r.lane, "pos": r.pos,
                    "token": token.map(|f| f(r.token)), "k": r.topk,
                    "experts": r.picks.iter().map(|p| p.0).collect::<Vec<_>>(),
                    "share": r.picks.iter().map(|p| (p.1 * 1000.0).round() as i64).collect::<Vec<_>>(),
                    "missed": r.picks.iter().map(|p| if p.2 { '1' } else { '0' }).collect::<String>(),
                })
            }
        }).collect()
    };
    let resident_per_layer: Vec<usize> =
        m.state.chunks(m.experts.max(1)).map(|r| r.iter().filter(|&&s| s == smalltide_tide::expert_map::RESIDENT).count()).collect();
    let counters: serde_json::Map<String, serde_json::Value> = m.counters.iter().map(|&(k, v)| (k.to_string(), v.into())).collect();
    let states_hash = fnv1a_64(&m.state);
    let states_val = if ask.states_hash != 0 && ask.states_hash == states_hash {
        serde_json::Value::Null
    } else {
        serde_json::json!(m.state_rows())
    };
    serde_json::json!({
        "version": 3,
        "boot": boot_id(),
        "model": model,
        "engine": engine,
        "sparks": m.world,
        "busy": m.busy,
        "stale": false,
        "sampled_at": now,
        "layers": m.layers,
        "experts_per_layer": m.experts,
        "physical_layers": m.physical_layers(),
        "total_experts": m.physical_layers() * m.experts,
        "state_names": ["not_loaded", "resident", "hot", "loading"],
        "counts": {"not_loaded": c[0], "resident": c[1], "hot": c[2], "loading": c[3]},
        "resident_per_layer": resident_per_layer,
        "arena_slots": m.slots,
        "spare_slots": m.spare,
        "counters": counters,
        "states": states_val,
        "states_hash": format!("{states_hash:016x}"),
        "has_demand": m.demand.is_some(),
        "demand": if ask.demand { serde_json::json!(m.demand_rows()) } else { serde_json::Value::Null },
        "events": events,
        "event_seq": m.events.last().map(|e| e.seq).unwrap_or(0).max(ask.events_since),
        "event_kinds": smalltide_tide::expert_map::EV_NAMES,
        "routes": routes,
        "route_seq": m.routes.last().map(|r| r.seq).unwrap_or(0).max(ask.since),
        "tokens_shown": token.is_some(),
        "rescue": m.rescue.as_ref().map(rescue_json),
    })
}


#[cfg(test)]
mod tests {
    use super::{base64_encode, json, Ask};
    use smalltide_tide::expert_map::{ExpertMap, MapRoute};

    #[test]
    fn the_query_asks() {
        assert_eq!(Ask::parse(""), Ask { since: 0, events_since: 0, demand: true, routes: true, compact: false, states_hash: 0 });
        assert_eq!(
            Ask::parse("since=41&events_since=7&demand=0&routes=0&compact=1&states_hash=0123abcd"),
            Ask { since: 41, events_since: 7, demand: false, routes: false, compact: true, states_hash: 0x0123abcd }
        );
        assert_eq!(Ask::parse("since=x&demand=1&other"), Ask { since: 0, events_since: 0, demand: true, routes: true, compact: false, states_hash: 0 });
    }

    #[test]
    fn compact_routes_and_unchanged_states_shrink_payload() {
        let m = ExpertMap {
            world: 1,
            busy: true,
            layers: vec![0, 1],
            physical: vec![0, 1],
            experts: 128,
            slots: 64,
            spare: 4,
            state: vec![1; 256],
            demand: None,
            counters: vec![],
            events: vec![],
            routes: vec![MapRoute {
                seq: 1,
                at: 0.0,
                lane: 0,
                pos: 5,
                token: 42,
                topk: 2,
                picks: vec![(3, 0.5, false), (127, 1.0, true), (0, 0.0, false), (64, 0.25, true)],
            }],
            rescue: None,
        };
        let full = json(&m, "m", "e", &Ask::parse("compact=1"), None);
        assert!(full["states"].is_array());
        let sh = full["states_hash"].as_str().unwrap().to_string();
        assert_eq!(sh.len(), 16);
        let p = full["routes"][0]["p"].as_str().unwrap();
        let expected = base64_encode(&[3, 64, 127, 0x80 | 127, 0, 0, 64, 0x80 | 32]);
        assert_eq!(p, expected);

        let unchanged = json(&m, "m", "e", &Ask::parse(&format!("compact=1&routes=0&states_hash={sh}")), None);
        assert!(unchanged["states"].is_null());
        assert_eq!(unchanged["routes"].as_array().unwrap().len(), 0);
        assert_eq!(unchanged["route_seq"].as_u64().unwrap(), 1);
    }
}
