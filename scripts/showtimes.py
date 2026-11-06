"""Showtime tracker: record parade / fireworks / show times from the
ThemeParks.wiki /live payloads that collect.py already fetches, and log
how they change during the day.

Why: a moved or cancelled nighttime show dumps thousands of guests back
into ride lines (and a parade pulls them out) — signal the wait model
cannot see today. The API publishes showtimes same-day only and never
flags a cancellation, so history exists only if we snapshot it
ourselves: this module diffs consecutive polls and writes

  data/showtimes.csv     one row per showtime version as first seen
                         (first_seen_utc, park, show, date, start/end ET)
  data/show_changes.csv  append-only events: added / moved /
                         removed_presumed_cancelled / status flips
  data/showtimes_state.json  the last known schedule per park (diff base)

Rules that keep the log honest:
  * The first poll that sees a date's times is that date's BASELINE —
    the published plan. Baselines write showtimes.csv rows but no
    change events (otherwise every midnight would flood the log with
    the whole next day "added").
  * A Performance Time that vanishes while its start is still ahead
    (or within 30 min after) is logged removed_presumed_cancelled —
    presumed, because the API never says the word. One that vanishes
    later is a performed show rolling off: no event.
  * If a show flips to CLOSED and all today's times vanish in the same
    poll, one status event is logged instead of a pile of removals.
  * Showtimes are kept only when their ET date is today or tomorrow.
    Dormant party entities carry stale (e.g. December 2025) times with
    status OPERATING — the date filter is what keeps those phantoms
    out of the record.

Stdlib only, and collect.py calls track() inside try/except: this
tracker can never break wait collection.
"""
import csv, json, os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

SHOWTIMES_HDR = ["first_seen_utc", "park", "show_id", "name", "show_date",
                 "type", "start_et", "end_et", "status", "src_updated_utc"]
CHANGES_HDR = ["detected_utc", "park", "name", "show_date", "change",
               "old_start_et", "new_start_et", "old_status", "new_status"]


def _hhmm(dt):
    return dt.strftime("%H:%M")


def _view(payload, now_et):
    """Extract {show_id: {name, status, src, dates:{date:{start:rec}}}}
    from one /live payload, keeping only today/tomorrow ET showtimes."""
    keep = {(now_et + timedelta(days=d)).strftime("%Y-%m-%d") for d in (0, 1)}
    out = {}
    for e in payload.get("liveData", []):
        if e.get("entityType") != "SHOW":
            continue
        rec = {"name": e.get("name", ""), "status": e.get("status", ""),
               "src": e.get("lastUpdated", ""), "dates": {}}
        for st in e.get("showtimes") or []:
            try:
                s = datetime.fromisoformat(st["startTime"]).astimezone(ET)
                en = datetime.fromisoformat(st["endTime"]).astimezone(ET)
            except Exception:
                continue
            d = s.strftime("%Y-%m-%d")
            if d not in keep:
                continue  # stale leftovers (e.g. last season's party shows)
            rec["dates"].setdefault(d, {})[_hhmm(s)] = {
                "end": _hhmm(en), "type": st.get("type", ""), "src": rec["src"]}
        out[e["id"]] = rec
    return out


def _append(path, header, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def track(park, payload, ts_utc, data_dir=DATA_DIR):
    """Diff one park's /live payload against the stored schedule.
    Returns the number of change events logged."""
    now_utc = datetime.strptime(ts_utc, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)
    now_et = now_utc.astimezone(ET)
    today = now_et.strftime("%Y-%m-%d")
    state_path = os.path.join(data_dir, "showtimes_state.json")
    state = {}
    if os.path.exists(state_path):
        try:
            state = json.load(open(state_path))
        except Exception:
            state = {}
    pstate = state.setdefault(park, {"baseline_dates": [], "shows": {}})
    old = pstate["shows"]
    new = _view(payload, now_et)
    baselined = set(pstate["baseline_dates"])

    st_rows, ch_rows = [], []

    def note_showtime(sid, rec, date, start, srec):
        st_rows.append([ts_utc, park, sid, rec["name"], date,
                        srec["type"], start, srec["end"], rec["status"],
                        srec["src"]])

    for sid, rec in new.items():
        orec = old.get(sid)
        for date, times in rec["dates"].items():
            otimes = (orec or {}).get("dates", {}).get(date, {})
            is_base = date not in baselined
            for start, srec in times.items():
                if start not in otimes:
                    note_showtime(sid, rec, date, start, srec)
            if is_base:
                continue  # published plan: rows only, no events
            removed = sorted(set(otimes) - set(times))
            added = sorted(set(times) - set(otimes))
            # Pair removals with additions on the same show+date as moves
            # (closest start times first); leftovers are true in/out.
            pairs = []
            for r in removed[:]:
                if not added:
                    break
                a = min(added, key=lambda x: abs(
                    datetime.strptime(x, "%H:%M")
                    - datetime.strptime(r, "%H:%M")))
                pairs.append((r, a))
                removed.remove(r)
                added.remove(a)
            closed_flip = rec["status"] == "CLOSED" and orec \
                and orec.get("status") != "CLOSED"
            for r, a in pairs:
                ch_rows.append([ts_utc, park, rec["name"], date, "moved",
                                r, a, "", ""])
            for a in added:
                ch_rows.append([ts_utc, park, rec["name"], date, "added",
                                "", a, "", ""])
            if not closed_flip:
                for r in removed:
                    start_dt = datetime.strptime(
                        f"{date} {r}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
                    if date == today and now_et <= start_dt + timedelta(minutes=30):
                        ch_rows.append([ts_utc, park, rec["name"], date,
                                        "removed_presumed_cancelled",
                                        r, "", "", ""])
        # Status flips on shows that have (or had) times today
        if orec and orec.get("status") != rec["status"] and (
                today in rec["dates"] or today in orec.get("dates", {})):
            ch_rows.append([ts_utc, park, rec["name"], today, "status",
                            "", "", orec.get("status", ""), rec["status"]])

    # Shows that vanished from the payload entirely: treat like a close
    # if they had times today still ahead.
    for sid, orec in old.items():
        if sid in new:
            continue
        for start in sorted(orec.get("dates", {}).get(today, {})):
            start_dt = datetime.strptime(
                f"{today} {start}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
            if now_et <= start_dt + timedelta(minutes=30):
                ch_rows.append([ts_utc, park, orec["name"], today,
                                "removed_presumed_cancelled", start, "",
                                orec.get("status", ""), ""])

    # Persist: new state (prune dates older than yesterday), CSV appends
    cutoff = (now_et - timedelta(days=1)).strftime("%Y-%m-%d")
    for rec in new.values():
        rec["dates"] = {d: t for d, t in rec["dates"].items() if d >= cutoff}
    pstate["shows"] = new
    pstate["baseline_dates"] = sorted(
        {d for rec in new.values() for d in rec["dates"]} | baselined)
    pstate["baseline_dates"] = [d for d in pstate["baseline_dates"]
                                if d >= cutoff]
    os.makedirs(data_dir, exist_ok=True)
    if st_rows:
        _append(os.path.join(data_dir, "showtimes.csv"), SHOWTIMES_HDR, st_rows)
    if ch_rows:
        _append(os.path.join(data_dir, "show_changes.csv"), CHANGES_HDR, ch_rows)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=1)
    return len(ch_rows)
