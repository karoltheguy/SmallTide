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
        "window_steps": r.window_steps,
        "total_steps": r.total_steps,
        "base_budget_total": r.base_budget_total,
        "last_verify_rows": r.last_verify_rows,
        "hist_bin_width_pct": RESCUE_HIST_BIN_PCT,
        "zones": zones,
    })
}

pub const PAGE: &str = include_str!("expert_map.html");

/// What a page asks for, from the query: `since` the last route it holds and `events_since` the last event (only
/// newer ones are sent: a page polls five times a second), `demand=0` (the usage rows left out, a fast poll's).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ask {
    pub since: u64,
    pub events_since: u64,
    pub demand: bool,
}

impl Ask {
    pub fn parse(query: &str) -> Ask {
        let mut a = Ask { demand: true, ..Ask::default() };
        for (k, v) in query.split('&').filter_map(|kv| kv.split_once('=')) {
            match k {
                "since" => a.since = v.parse().unwrap_or(0),
                "events_since" => a.events_since = v.parse().unwrap_or(0),
                "demand" => a.demand = v != "0",
                _ => {}
            }
        }
        a
    }
}

/// The snapshot as the page reads it. `states`: a string a layer, a digit an expert (0 not loaded,
/// 1 resident, 2 in the hot-read ring, 3 being read); `demand`: the same shape in base 36, an
/// expert's share of its layer's routing history (0 none, z the layer's most used; null when not asked for,
/// `has_demand` says whether the engine keeps one). `events`: the engine's recent residency events after
/// `events_since`, oldest first, each [seq, age ms, row, expert, kind, share in thousandths, victim row,
/// victim expert] (-1, -1 without a victim; `event_kinds` names the kinds) - what brought each expert in: a
/// missed pick (its share of the token's routing weight), an inline rescue, an adaptation, a preload.
/// `routes`: the decoded tokens' routing after `since`, oldest first, each {seq, age_ms, lane, pos, token (its
/// text, to a keyed request only: it is the conversation's; else null), k, experts [row * k + pick], share
/// (thousandths, the same order), missed (a digit a pick: 1 when it was not resident as the pass ran)}.
pub fn json(m: &ExpertMap, model: &str, engine: &str, ask: &Ask, token: Option<&dyn Fn(u32) -> String>) -> serde_json::Value {
    let c = m.counts();
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    let row = |l: u16| m.layers.binary_search(&(l as usize)).map(|r| r as i64).unwrap_or(-1);
    let events: Vec<serde_json::Value> = m.events.iter().filter(|e| e.seq > ask.events_since && row(e.layer) >= 0).map(|e| {
        let (vr, vx) = e.victim.map(|(l, x)| (row(l), x as i64)).unwrap_or((-1, -1));
        serde_json::json!([e.seq, ((now - e.at) * 1000.0).max(0.0).round() as i64, row(e.layer), e.expert, e.kind,
                           (e.share * 1000.0).round() as i64, vr, vx])
    }).collect();
    let routes: Vec<serde_json::Value> = m.routes.iter().filter(|r| r.seq > ask.since && r.picks.len() == m.layers.len() * r.topk as usize).map(|r| {
        serde_json::json!({
            "seq": r.seq, "age_ms": ((now - r.at) * 1000.0).max(0.0).round() as i64, "lane": r.lane, "pos": r.pos,
            "token": token.map(|f| f(r.token)), "k": r.topk,
            "experts": r.picks.iter().map(|p| p.0).collect::<Vec<_>>(),
            "share": r.picks.iter().map(|p| (p.1 * 1000.0).round() as i64).collect::<Vec<_>>(),
            "missed": r.picks.iter().map(|p| if p.2 { '1' } else { '0' }).collect::<String>(),
        })
    }).collect();
    let resident_per_layer: Vec<usize> =
        m.state.chunks(m.experts.max(1)).map(|r| r.iter().filter(|&&s| s == smalltide_tide::expert_map::RESIDENT).count()).collect();
    let counters: serde_json::Map<String, serde_json::Value> = m.counters.iter().map(|&(k, v)| (k.to_string(), v.into())).collect();
    serde_json::json!({
        "version": 3,
        "model": model,
        "engine": engine,
        "sparks": m.world,
        "busy": m.busy,
        "stale": false,
        "sampled_at": now,
        "layers": m.layers,
        "experts_per_layer": m.experts,
        "total_experts": m.state.len(),
        "state_names": ["not_loaded", "resident", "hot", "loading"],
        "counts": {"not_loaded": c[0], "resident": c[1], "hot": c[2], "loading": c[3]},
        "resident_per_layer": resident_per_layer,
        "arena_slots": m.slots,
        "spare_slots": m.spare,
        "counters": counters,
        "states": m.state_rows(),
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
    use super::Ask;

    #[test]
    fn the_query_asks() {
        assert_eq!(Ask::parse(""), Ask { since: 0, events_since: 0, demand: true });
        assert_eq!(Ask::parse("since=41&events_since=7&demand=0"), Ask { since: 41, events_since: 7, demand: false });
        assert_eq!(Ask::parse("since=x&demand=1&other"), Ask { since: 0, events_since: 0, demand: true });
    }
}
