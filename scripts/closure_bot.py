"""Closure Bot (SHADOW MODE) — the first manager bot for mk-forecast.

Charter (see BOTS.md): own the bounce-back. When a headliner goes down,
crowds displace and predictions wobble; when it reopens, models trained
on normal operations misprice the settle. This bot watches for closure
and reopen events at Magic Kingdom and records what it WOULD have
predicted instead of the shipped forecast:

  D1 (displacement): a ride with a posted wait >= 40 min has been down
     >= 15 min -> same-land neighbors get +5 min on their +30/+60/+90
     shadow predictions (bounded, expires with the event).
  D2 (reopen settle): a ride that was down >= 15 min reopens -> for its
     +30, the shadow prediction is simply the first posted wait at
     reopen (persistence-from-reopen) instead of the shipped number.

Every decision is appended to data/bot_decisions.jsonl with the shipped
prediction it challenges; once the target time passes, the decision is
graded against the posted wait that actually happened, and the running
record lands in data/bot_scorecard.json. In shadow mode NOTHING here
changes a shipped prediction. The bot earns live authority only if its
graded record beats the incumbent — the same rule as every feature.

Fail-safe by design: this script always exits 0. Any error is noted in
the scorecard's debug field and the forecast pipeline is unaffected.
"""
import json, os, sys
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
WAITS = os.path.join(ROOT, "data", "waits.csv")
PREDLOG = os.path.join(ROOT, "data", "pred_log.csv")
STATE = os.path.join(ROOT, "data", "bot_state.json")
LEDGER = os.path.join(ROOT, "data", "bot_decisions.jsonl")
SCORECARD = os.path.join(ROOT, "data", "bot_scorecard.json")
ET = "America/New_York"

HEADLINER_MIN_WAIT = 40     # pre-closure posted wait that counts as a headliner
MIN_DOWN_MIN = 15           # a spell must last this long to matter
D1_BUMP = 5                 # minutes added to neighbors in shadow
D1_HORIZONS = (30, 60, 90)
D2_HORIZONS = (30,)
SHADOW_CAP = 95

# Land assignment by name keyword (Magic Kingdom). Rides matching no
# keyword get no land and never receive or trigger neighbor bumps.
LAND_KEYWORDS = [
    ("TRON", "Tomorrowland"), ("Space Mountain", "Tomorrowland"),
    ("Buzz Lightyear", "Tomorrowland"), ("Astro Orbiter", "Tomorrowland"),
    ("PeopleMover", "Tomorrowland"), ("Speedway", "Tomorrowland"),
    ("Laugh Floor", "Tomorrowland"), ("Carousel of Progress", "Tomorrowland"),
    ("Mine Train", "Fantasyland"), ("Peter Pan", "Fantasyland"),
    ("small world", "Fantasyland"), ("Winnie the Pooh", "Fantasyland"),
    ("Mad Tea Party", "Fantasyland"), ("Dumbo", "Fantasyland"),
    ("Barnstormer", "Fantasyland"), ("Little Mermaid", "Fantasyland"),
    ("Regal Carrousel", "Fantasyland"), ("Enchanted Tales", "Fantasyland"),
    ("PhilharMagic", "Fantasyland"),
    ("Tiana", "Frontierland"), ("Big Thunder", "Frontierland"),
    ("Country Bear", "Frontierland"),
    ("Haunted Mansion", "Liberty Square"), ("Hall of Presidents", "Liberty Square"),
    ("Riverboat", "Liberty Square"),
    ("Pirates of the Caribbean", "Adventureland"), ("Jungle Cruise", "Adventureland"),
    ("Magic Carpets", "Adventureland"), ("Tiki Room", "Adventureland"),
    ("Swiss Family", "Adventureland"),
    ("Main Street Vehicles", "Main Street"), ("Railroad - Main Street", "Main Street"),
    ("Railroad - Frontierland", "Frontierland"),
    ("Railroad - Fantasyland", "Fantasyland"),
]


def land_of(name):
    for kw, land in LAND_KEYWORDS:
        if kw.lower() in name.lower():
            return land
    return None


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def grid_today():
    """Today's (ET) waits on the pipeline's 5-min grid, long format."""
    df = pd.read_csv(WAITS, usecols=["ts_utc", "name", "status", "wait"])
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, format="mixed")
    df["ts"] = df["ts_utc"].dt.ceil("5min").dt.tz_convert(ET)
    today = pd.Timestamp.now(tz=ET).normalize()
    df = df[df["ts"].dt.tz_localize(None).dt.normalize() ==
            today.tz_localize(None)]
    return df.sort_values("ts_utc").drop_duplicates(["name", "ts"], keep="last")


def latest_predictions():
    """Freshest run in pred_log: {(name, horizon): (made_at, target, pred)}."""
    out = {}
    if not os.path.exists(PREDLOG):
        return out
    log = pd.read_csv(PREDLOG)
    if not len(log):
        return out
    log["made_at"] = pd.to_datetime(log["made_at"], utc=True, format="mixed")
    log["target"] = pd.to_datetime(log["target"], utc=True, format="mixed")
    fresh = log["made_at"].max()
    for _, r in log[log["made_at"] == fresh].iterrows():
        out[(r["name"], int(r["horizon"]))] = (fresh, r["target"], float(r["pred"]))
    return out


def main():
    now_utc = pd.Timestamp.now(tz="UTC")
    state = load_json(STATE, {"rides": {}, "seq": 0})
    rides_state = state.setdefault("rides", {})
    debug = []
    new_decisions, graded_now = [], []

    g = grid_today()
    if not len(g):
        raise RuntimeError("no samples for today yet")
    preds = latest_predictions()

    # ---- per-ride current status + today's down spells -------------------
    for name, grp in g.groupby("name"):
        grp = grp.sort_values("ts")
        last = grp.iloc[-1]
        st = rides_state.setdefault(name, {"down_since": None, "pre_wait": None,
                                           "d1_fired": False, "was_down": False})
        is_down = str(last["status"]).upper() in ("DOWN", "CLOSED")
        operated_today = (grp["status"] == "OPERATING").any()
        if is_down and operated_today and not st.get("was_down"):
            # spell start: first down sample after the last operating one
            down_rows = grp[grp["status"].str.upper().isin(["DOWN", "CLOSED"])]
            op_rows = grp[grp["status"] == "OPERATING"]
            start = down_rows[down_rows["ts"] > op_rows["ts"].max()]
            st["down_since"] = str(start["ts"].iloc[0]) if len(start) else str(last["ts"])
            pw = op_rows.iloc[-1]["wait"]
            st["pre_wait"] = None if pd.isna(pw) else float(pw)
            st["d1_fired"] = False
            st["was_down"] = True
        elif not is_down:
            if st.get("was_down") and st.get("down_since"):
                down_min = (pd.Timestamp(last["ts"]) -
                            pd.Timestamp(st["down_since"])).total_seconds() / 60
                # ---- D2: reopen settle ----------------------------------
                if down_min >= MIN_DOWN_MIN and pd.notna(last["wait"]):
                    for h in D2_HORIZONS:
                        p = preds.get((name, h))
                        if p:
                            state["seq"] += 1
                            new_decisions.append({
                                "id": f"cb-{state['seq']}", "type": "D2",
                                "decided_at": now_utc.isoformat(),
                                "event_ride": name, "ride": name, "horizon": h,
                                "made_at": p[0].isoformat(), "target": p[1].isoformat(),
                                "base": p[2], "shadow": float(last["wait"]),
                                "reason": f"reopened after {int(down_min)} min down; "
                                          f"shadow = first posted wait at reopen"})
            st["was_down"] = False
            st["down_since"] = None
            st["pre_wait"] = None
        # ---- D1: displacement -------------------------------------------
        if st.get("was_down") and st.get("down_since") and not st.get("d1_fired"):
            age = (pd.Timestamp(last["ts"]) -
                   pd.Timestamp(st["down_since"])).total_seconds() / 60
            if (age >= MIN_DOWN_MIN and (st.get("pre_wait") or 0) >= HEADLINER_MIN_WAIT
                    and land_of(name)):
                for (n2, h), p in sorted(preds.items()):
                    if h not in D1_HORIZONS or n2 == name or land_of(n2) != land_of(name):
                        continue
                    n2_last = g[g["name"] == n2]
                    if not len(n2_last) or str(n2_last.iloc[-1]["status"]).upper() != "OPERATING":
                        continue
                    state["seq"] += 1
                    new_decisions.append({
                        "id": f"cb-{state['seq']}", "type": "D1",
                        "decided_at": now_utc.isoformat(),
                        "event_ride": name, "ride": n2, "horizon": h,
                        "made_at": p[0].isoformat(), "target": p[1].isoformat(),
                        "base": p[2], "shadow": min(SHADOW_CAP, p[2] + D1_BUMP),
                        "reason": f"{name} (posted {int(st['pre_wait'])}) down "
                                  f"{int(age)} min; same-land displacement +{D1_BUMP}"})
                if any(d["event_ride"] == name and d["type"] == "D1" for d in new_decisions):
                    st["d1_fired"] = True

    # ---- persist new decisions -------------------------------------------
    open_decisions = load_json(os.path.join(ROOT, "data", "bot_open.json"), [])
    if new_decisions:
        with open(LEDGER, "a") as f:
            for d in new_decisions:
                f.write(json.dumps(d) + "\n")
        open_decisions.extend(new_decisions)

    # ---- grade matured decisions ------------------------------------------
    still_open, graded = [], load_json(os.path.join(ROOT, "data", "bot_graded.json"), [])
    for d in open_decisions:
        target = pd.Timestamp(d["target"])
        if target.tzinfo is None:
            target = target.tz_localize("UTC")
        if target.tz_convert(ET) > pd.Timestamp.now(tz=ET):
            still_open.append(d)
            continue
        rows = g[(g["name"] == d["ride"]) &
                 (g["ts"] >= target.tz_convert(ET) - pd.Timedelta(minutes=3)) &
                 (g["ts"] <= target.tz_convert(ET) + pd.Timedelta(minutes=3))]
        rows = rows[pd.notna(rows["wait"])]
        if len(rows) and (pd.Timestamp.now(tz=ET) - target.tz_convert(ET)) < pd.Timedelta(hours=3):
            actual = float(rows.iloc[0]["wait"])
            eb, es = abs(d["base"] - actual), abs(d["shadow"] - actual)
            d.update({"actual": actual, "err_base": eb, "err_shadow": es,
                      "outcome": "win" if es < eb else "loss" if es > eb else "tie",
                      "graded_at": now_utc.isoformat()})
            graded.append(d)
            graded_now.append(d)
        elif (pd.Timestamp.now(tz=ET) - target.tz_convert(ET)) < pd.Timedelta(hours=3):
            still_open.append(d)   # actual not on grid yet; retry next runs
        # else: ungradeable (no posted wait near target) -> dropped, counted below
    with open(os.path.join(ROOT, "data", "bot_open.json"), "w") as f:
        json.dump(still_open, f)
    with open(os.path.join(ROOT, "data", "bot_graded.json"), "w") as f:
        json.dump(graded[-2000:], f)
    with open(STATE, "w") as f:
        json.dump(state, f)

    # ---- scorecard ----------------------------------------------------------
    def agg(ds):
        if not ds:
            return {"n": 0}
        return {"n": len(ds),
                "wins": sum(d["outcome"] == "win" for d in ds),
                "losses": sum(d["outcome"] == "loss" for d in ds),
                "ties": sum(d["outcome"] == "tie" for d in ds),
                "mae_base": round(sum(d["err_base"] for d in ds) / len(ds), 2),
                "mae_shadow": round(sum(d["err_shadow"] for d in ds) / len(ds), 2)}
    card = {"bot": "closure_bot", "mode": "shadow",
            "updated": now_utc.isoformat(),
            "charter": "Own the bounce-back: displacement while a headliner is "
                       "down (D1) and the reopen settle (D2).",
            "active_spells": {n: s["down_since"] for n, s in rides_state.items()
                              if s.get("was_down")},
            "decisions_open": len(still_open),
            "graded_all": agg(graded),
            "by_type": {t: agg([d for d in graded if d["type"] == t])
                        for t in ("D1", "D2")},
            "new_decisions_this_run": len(new_decisions),
            "graded_this_run": len(graded_now),
            "last_decisions": [{k: d[k] for k in ("id", "type", "event_ride", "ride",
                                "horizon", "base", "shadow", "reason")}
                               for d in new_decisions[-5:]],
            "debug": "; ".join(debug)}
    with open(SCORECARD, "w") as f:
        json.dump(card, f, indent=1)
    print(f"closure_bot: {len(new_decisions)} new decisions, "
          f"{len(graded_now)} graded, {len(still_open)} open, "
          f"all-time {card['graded_all']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:   # shadow bot must never break the pipeline
        try:
            card = load_json(SCORECARD, {"bot": "closure_bot", "mode": "shadow"})
            card["updated"] = datetime.now(timezone.utc).isoformat()
            card["debug"] = f"EXC {type(ex).__name__}: {ex}"
            with open(SCORECARD, "w") as f:
                json.dump(card, f, indent=1)
        except Exception:
            pass
        print(f"closure_bot EXC (swallowed): {type(ex).__name__}: {ex}")
    sys.exit(0)
