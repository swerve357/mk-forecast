"""Per-park forecast engine — the production method, one park at a time.

Magic Kingdom's forecast.py stays untouched. This engine gives another
park its own forecast section inside docs/data.json (under "parks"), so
the dashboard can switch parks while each park's models train ONLY on
that park's data and face the ship gate on that park's own holdout:
same recipe (per-ride dynamics, chronological 80/20 holdout with
horizon embargo, delta target + round to 5), separate kitchen.

Features are the cross-park intersection: ride dynamics, park
aggregates computed within the park, calendar, that park's hours /
Early Entry / ticketed-event flag / Lightning Lane Multi Pass price
from its own schedule file, and the resort-wide weather file. (MK's
extra production features — curated party nights, ride-specific LL
prices — stay MK-only until a park's engine proves it needs them.)

Cadence: designed to be called after every MK forecast run via a
fail-safe hook, but each park only retrains when DUE (every ~30 min,
tracked in data/park_engine_state.json) so the 5-minute collection
loop never waits on it. --force ignores the cadence. A park engine
failure can never affect MK: it writes only its own "parks" section,
its own prediction log (data/pred_log_<park>.csv), and exits 0.
"""
import json, os, sys
from datetime import datetime, timezone

import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forecast as F

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "data.json")
STATE = os.path.join(ROOT, "data", "park_engine_state.json")
DUE_MINUTES = 25

PARKS = {
    "EPCOT": {"waits": "data/waits_epcot.csv",
              "sched": "data/park_schedule_epcot.csv",
              "log": "data/pred_log_epcot.csv",
              "label": "EPCOT"},
    "HS": {"waits": "data/waits_hollywood_studios.csv",
           "sched": "data/park_schedule_hollywood_studios.csv",
           "log": "data/pred_log_hollywood_studios.csv",
           "label": "Hollywood Studios"},
    "AK": {"waits": "data/waits_animal_kingdom.csv",
           "sched": "data/park_schedule_animal_kingdom.csv",
           "log": "data/pred_log_animal_kingdom.csv",
           "label": "Animal Kingdom"},
}

X_COLS = [
    "wait", "lag_5", "lag_15", "lag_30", "lag_60", "lag_120", "lag_yday",
    "diff_15", "diff_30", "roll_mean_30", "roll_mean_60", "roll_std_60",
    "down_mins_60", "reopened_30",
    "park_mean", "park_p90", "park_frac_busy",
    "hour_sin", "hour_cos", "dow", "is_weekend",
    "mins_since_open", "mins_to_close", "operating_mins", "early_entry",
    "is_early_close", "has_ticketed_event", "ll_multipass",
    "is_holiday", "is_school_break", "days_until_christmas",
    "temp_f", "feels_f", "humidity", "precip_in", "precip_prob",
    "is_raining", "is_storm", "wind_mph", "gust_mph",
    "ride_code",
]


def load_park_waits(path):
    df = pd.read_csv(path)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, format="mixed")
    df["ts"] = df["ts_utc"].dt.ceil("5min").dt.tz_convert("America/New_York")
    return df.sort_values("ts_utc").drop_duplicates(["ride_id", "ts"], keep="last")


def add_context(feat, sched_path):
    cal = F.calendar_frame(feat.index)
    for c in cal.columns:
        feat[c] = cal[c].to_numpy()
    by_ts = feat.groupby(level=0)["wait"]
    feat["park_mean"] = by_ts.transform("mean")
    feat["park_p90"] = by_ts.transform(lambda s: s.quantile(0.9))
    feat["park_frac_busy"] = by_ts.transform(lambda s: (s >= 40).mean())
    feat["date_str"] = feat.index.strftime("%Y-%m-%d")
    feat["hour_key"] = feat.index.strftime("%Y-%m-%dT%H:00")

    def mins(x):
        try:
            h, m = str(x).split(":")[:2]
            return int(h) * 60 + int(m)
        except Exception:
            return np.nan
    s = pd.read_csv(sched_path, dtype={"date": str})
    s["_open"] = s["open"].map(mins)
    s["_close"] = s["close"].map(mins)
    idx = s.set_index("date")
    day_min = feat.index.hour * 60 + feat.index.minute
    feat["mins_since_open"] = day_min - feat["date_str"].map(idx["_open"])
    feat["mins_to_close"] = feat["date_str"].map(idx["_close"]) - day_min
    feat["operating_mins"] = feat["date_str"].map(idx["operating_mins"])
    feat["early_entry"] = feat["date_str"].map(
        idx["early_entry"].fillna("").ne("").astype(int))
    feat["is_early_close"] = (feat["date_str"].map(idx["_close"]) <= 18 * 60).astype(float)
    feat["has_ticketed_event"] = feat["date_str"].map(idx["has_ticketed_event"].fillna(0))
    feat["ll_multipass"] = feat["date_str"].map(idx["ll_multipass_usd"])

    wmap = {"temperature_2m": "temp_f", "apparent_temperature": "feels_f",
            "relative_humidity_2m": "humidity", "precipitation": "precip_in",
            "precipitation_probability": "precip_prob", "wind_speed_10m": "wind_mph",
            "wind_gusts_10m": "gust_mph", "weather_code": "_wcode"}
    if os.path.exists(F.WEATHER):
        w = pd.read_csv(F.WEATHER, dtype={"hour_local": str})
        widx = w.set_index("hour_local")
        for src, dst in wmap.items():
            feat[dst] = pd.to_numeric(feat["hour_key"].map(widx[src]), errors="coerce")
        feat["is_raining"] = (feat["precip_in"].fillna(0) > 0.01).astype(float)
        feat["is_storm"] = (feat["_wcode"].fillna(0) >= 95).astype(float)
    for c in X_COLS:
        if c not in feat.columns:
            feat[c] = np.nan
    return feat


def run_park(park):
    cfg = PARKS[park]
    raw = load_park_waits(os.path.join(ROOT, cfg["waits"]))
    feat = add_context(F.per_ride_frames(raw), os.path.join(ROOT, cfg["sched"]))
    latest_ts = feat.index.max()
    now = feat[feat.index == latest_ts].dropna(subset=["wait"])

    metrics, models, intervals, ride_scores = {}, {}, {}, {}
    for h in F.HORIZONS:
        d = feat.dropna(subset=["wait", f"y{h}"]).sort_index()
        if len(d):
            d = d[d.index >= d.index.max() - pd.Timedelta(days=F.TRAIN_WINDOW_DAYS)]
        valid_times = np.sort(d.index.unique())
        cut = valid_times[int(len(valid_times) * 0.8)] if len(valid_times) >= 10 else None
        entry = {"available_pairs": int(len(d)), "train_rows": 0, "test_rows": 0,
                 "model": "persistence (still collecting)"}
        if cut is not None and len(d) >= F.MIN_TRAIN_ROWS:
            tr, te = d[d.index + pd.Timedelta(minutes=h) < cut], d[d.index >= cut]
            if len(tr) >= F.MIN_TRAIN_ROWS * 0.5 and len(te) >= 50:
                cols = [c for c in X_COLS if tr[c].notna().any()
                        and (c == "ride_code" or tr[c].nunique(dropna=True) > 1)]
                cat = [cols.index("ride_code")] if "ride_code" in cols else None
                m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.07,
                                                  categorical_features=cat)
                m.fit(tr[cols], tr[f"y{h}"] - tr["wait"])

                def ship_pred(frame, model):
                    p = frame["wait"].values + model.predict(frame[cols])
                    return np.clip(np.round(p / 5.0) * 5, 0, None)
                s = F.score(ship_pred(te, m), te[f"y{h}"].values, te["wait"].values)
                entry.update(s, train_rows=int(len(tr)), test_rows=int(len(te)),
                             evaluation="chronological holdout with horizon embargo")
                shipped_pred = (ship_pred(te, m) if s["mae"] < s["baseline_mae"]
                                else te["wait"].values)
                tmp = pd.DataFrame({"name": te["name"].values,
                                    "err": np.abs(np.asarray(shipped_pred, float)
                                                  - te[f"y{h}"].values)})
                ride_scores[h] = {
                    n: {"mae": round(float(g["err"].mean()), 1),
                        "within10": round(float((g["err"] <= 10).mean() * 100)),
                        "within5": round(float((g["err"] <= 5).mean() * 100)),
                        "pairs": int(len(g))}
                    for n, g in tmp.groupby("name") if len(g) >= 30}
                try:
                    sample = te.sample(min(2000, len(te)), random_state=0)
                    pi = permutation_importance(m, sample[cols],
                                                sample[f"y{h}"] - sample["wait"],
                                                n_repeats=3, random_state=0,
                                                scoring="neg_mean_absolute_error")
                    order = np.argsort(pi.importances_mean)[::-1]
                    entry["top_features"] = [cols[i] for i in order
                                             if pi.importances_mean[i] > 0][:6]
                except Exception as e:
                    print(f"{park} importance {h}: {e}")
                if s["mae"] < s["baseline_mae"]:
                    m.fit(d[cols], d[f"y{h}"] - d["wait"])
                    models[h] = (m, cols)
                    entry["model"] = "gradient boosting (delta+round5)"
                    if len(d) >= F.MIN_QUANTILE_ROWS:
                        try:
                            lo = HistGradientBoostingRegressor(
                                loss="quantile", quantile=0.1, max_iter=150,
                                learning_rate=0.07, categorical_features=cat)
                            hi = HistGradientBoostingRegressor(
                                loss="quantile", quantile=0.9, max_iter=150,
                                learning_rate=0.07, categorical_features=cat)
                            lo.fit(d[cols], d[f"y{h}"] - d["wait"])
                            hi.fit(d[cols], d[f"y{h}"] - d["wait"])
                            intervals[h] = (lo, hi, cols)
                        except Exception as e:
                            print(f"{park} quantile {h}: {e}")
                else:
                    entry["candidate_mae"] = s["mae"]
                    entry.update(F.score(te["wait"], te[f"y{h}"], te["wait"]))
                    entry["model"] = "persistence (model did not beat baseline)"
        metrics[str(h)] = entry

    rides = []
    for _, r in now.reset_index().iterrows():
        row = {"name": r["name"], "now": int(round(r["wait"])),
               "trend30": int(round(r["diff_30"])) if pd.notna(r.get("diff_30")) else 0}
        def _n(v):
            return int(round(v)) if pd.notna(v) else None
        # Live inputs behind this ride's calls — powers the dashboard's
        # "why this call" panel (the model's actual evidence, not a story).
        row["why"] = {"lag60": _n(r.get("lag_60")), "lag120": _n(r.get("lag_120")),
                      "lag_yday": _n(r.get("lag_yday")), "down60": _n(r.get("down_mins_60")),
                      "reopened": bool(r.get("reopened_30")) if pd.notna(r.get("reopened_30")) else False,
                      "park_mean": _n(r.get("park_mean")),
                      "park_busy": int(round(float(r["park_frac_busy"]) * 100)) if pd.notna(r.get("park_frac_busy")) else None,
                      "mins_open": _n(r.get("mins_since_open")), "mins_close": _n(r.get("mins_to_close"))}
        for h in F.HORIZONS:
            if h in models:
                mdl, mcols = models[h]
                p = float(r["wait"]) + float(mdl.predict(pd.DataFrame([r])[mcols])[0])
                p = max(0.0, round(p / 5.0) * 5)
            else:
                p = float(r["wait"])
            row[f"f{h}"] = int(p)
            if h in intervals:
                lo, hi, icols = intervals[h]
                base = float(r["wait"])
                row[f"f{h}_lo"] = int(max(0.0, round((base + float(lo.predict(pd.DataFrame([r])[icols])[0])) / 5.0) * 5))
                row[f"f{h}_hi"] = int(max(0.0, round((base + float(hi.predict(pd.DataFrame([r])[icols])[0])) / 5.0) * 5))
            sc = ride_scores.get(h, {}).get(r["name"])
            if sc:
                row[f"score{h}_mae"] = sc["mae"]
                row[f"score{h}_in5"] = sc["within5"]
                row[f"score{h}_in10"] = sc["within10"]
                row[f"score{h}_pairs"] = sc["pairs"]
            # The shipped change-from-now per horizon. Between retrains,
            # refresh_park() re-issues calls as current wait + this delta,
            # so the park logs calls every cycle like MK does.
            row[f"d{h}"] = row[f"f{h}"] - row["now"]
            if f"f{h}_lo" in row:
                row[f"d{h}_lo"] = row[f"f{h}_lo"] - row[f"f{h}"]
                row[f"d{h}_hi"] = row[f"f{h}_hi"] - row[f"f{h}"]
        row["people"] = None
        rides.append(row)
    rides.sort(key=lambda x: -x["now"])

    # ---- prediction log + lookback (same design as MK) ----
    lts = pd.Timestamp(latest_ts)
    lts = lts.tz_localize("UTC") if lts.tzinfo is None else lts.tz_convert("UTC")
    log = pd.DataFrame(columns=["made_at", "name", "horizon", "target", "pred"])
    try:
        new_rows = [{"made_at": lts.isoformat(), "name": row["name"], "horizon": h,
                     "target": (lts + pd.Timedelta(minutes=h)).isoformat(),
                     "pred": row[f"f{h}"]}
                    for row in rides for h in F.HORIZONS]
        log = _append_log(cfg, new_rows, lts)
    except Exception as ex:
        print(f"{park} predlog: {ex}")
    lookback = {}
    today_stats = {}
    try:
        f2 = feat[["name", "wait"]].reset_index()
        act = f2.pivot_table(index=f2.columns[0], columns="name",
                             values="wait", aggfunc="last")
        act.index = pd.to_datetime(act.index, utc=True)
        win = act.index[(act.index >= lts - pd.Timedelta(hours=4)) & (act.index <= lts)]
        # Rows on the 15-min grid PLUS every real call target in the
        # window: a call made for 10:20 must be visible even though the
        # grid's rows sit at :15/:30/:45.
        extra = log.loc[(log["target"] >= lts - pd.Timedelta(hours=4))
                        & (log["target"] <= lts), "target"].unique()
        slots = sorted(set(win[::3].tolist())
                       | set(pd.to_datetime(extra, utc=True).tolist()))
        pred_idx = log.set_index(["name", "horizon", "target"])["pred"].to_dict()
        stat = raw.groupby(["ts", "name"])["status"].last().unstack()
        stat.index = pd.to_datetime(stat.index, utc=True)
        for row in rides:
            name = row["name"]
            if name not in act.columns:
                continue
            entries = []
            for T in slots:
                a = act.at[T, name] if T in act.index else np.nan
                if pd.isna(a):
                    # Closed/down at this slot: mark it (dashboard shows *)
                    st = stat.at[T, name] if (name in stat.columns and T in stat.index) else None
                    if st in ("CLOSED", "DOWN", "REFURBISHMENT"):
                        e = {"t": T.tz_convert("America/New_York").strftime("%H:%M"),
                             "closed": 1}
                        for h in F.HORIZONS:
                            p = pred_idx.get((name, h, T))
                            if p is not None and pd.notna(p):
                                e[f"p{h}"] = int(p)
                        entries.append(e)
                    continue
                e = {"t": T.tz_convert("America/New_York").strftime("%H:%M"),
                     "actual": int(round(a))}
                for h in F.HORIZONS:
                    p = pred_idx.get((name, h, T))
                    if p is not None and pd.notna(p):
                        e[f"p{h}"] = int(p)
                entries.append(e)
            if entries:
                lookback[name] = entries

        # ---- today's running prediction rate (same design as MK) ----
        et = "America/New_York"
        day0 = lts.tz_convert(et).normalize().tz_convert("UTC")
        due = log[(log["target"] >= day0) & (log["target"] <= lts)]
        acts = act.stack().rename("actual").reset_index()
        acts.columns = ["target", "name", "actual"]
        mm = due.merge(acts, on=["name", "target"], how="inner").dropna(subset=["actual"])
        if len(mm):
            err = (mm["pred"] - mm["actual"]).abs()
            today_stats.update({
                "date": str(lts.tz_convert(et).date()), "graded": int(len(mm)),
                "exact": int(round(float((err == 0).mean() * 100))),
                "in5": int(round(float((err <= 5).mean() * 100))),
                "in10": int(round(float((err <= 10).mean() * 100))),
                "in15": int(round(float((err <= 15).mean() * 100)))})
            for hh, g in mm.groupby("horizon"):
                e2 = (g["pred"] - g["actual"]).abs()
                today_stats.setdefault("by_horizon", {})[str(int(hh))] = {
                    "graded": int(len(g)),
                    "exact": int(round(float((e2 == 0).mean() * 100))),
                    "in5": int(round(float((e2 <= 5).mean() * 100))),
                    "in10": int(round(float((e2 <= 10).mean() * 100))),
                    "in15": int(round(float((e2 <= 15).mean() * 100)))}
    except Exception as ex:
        print(f"{park} lookback: {ex}")

    # ---- context snapshot (same keys as MK's ctx) ----
    ctx = {"date": str(latest_ts.date())}
    sched_path = os.path.join(ROOT, cfg["sched"])
    if os.path.exists(sched_path):
        s = pd.read_csv(sched_path, dtype={"date": str})
        today = s[s["date"] == ctx["date"]]
        if len(today):
            t = today.iloc[0]
            ctx.update({"open": t["open"], "close": t["close"],
                        "early_entry": t["early_entry"],
                        "event": t["ticketed_event_desc"] if pd.notna(t.get("ticketed_event_desc")) else "",
                        "ll_multipass_usd": None if pd.isna(t.get("ll_multipass_usd")) else float(t["ll_multipass_usd"])})
    ctx["is_party_night"] = bool(now["has_ticketed_event"].max()) if len(now) else False

    section = {"updated": datetime.now(timezone.utc).isoformat(),
               "last_sample": raw["ts_utc"].max().isoformat(),
               "samples": int(len(raw)), "history_samples": 0,
               "rides": rides, "metrics": metrics, "context": ctx,
               "demo": False, "lookback": lookback, "today": today_stats,
               "park_label": cfg["label"]}

    # Store the section on disk (in data/, which the workflow commits):
    # forecast.py rewrites docs/data.json wholesale on every MK run, so
    # attach() re-merges the stored sections on every engine invocation
    # and the park views survive between this park's 30-min retrains.
    json.dump(section, open(section_path(park), "w"), indent=1)
    shipped = [h for h in F.HORIZONS if metrics[str(h)]["model"].startswith("gradient")]
    print(f"{park}: wrote {len(rides)} rides; shipped models: {shipped or 'none yet'}")
    return section


def _append_log(cfg, new_rows, lts):
    """Append prediction rows to a park's log, dedupe on
    (made_at, name, horizon), prune to 30h, return the merged log."""
    log_path = os.path.join(ROOT, cfg["log"])
    log = pd.DataFrame(new_rows)
    if os.path.exists(log_path):
        log = pd.concat([pd.read_csv(log_path), log], ignore_index=True)
    log["made_at"] = pd.to_datetime(log["made_at"], utc=True, format="mixed")
    log["target"] = pd.to_datetime(log["target"], utc=True, format="mixed")
    cutoff = lts - pd.Timedelta(hours=30)   # a full day + margin for the daily rate
    log = (log.drop_duplicates(subset=["made_at", "name", "horizon"], keep="last")
              .loc[lambda x: x["made_at"] >= cutoff]
              .sort_values(["made_at", "name", "horizon"]))
    log.to_csv(log_path, index=False)
    return log


def refresh_park(park):
    """Between retrains (this runs every forecast cycle): re-issue the
    standing model's calls from the latest posted waits — prediction =
    current wait + the delta shipped at the last retrain, rounded to 5 —
    append them to the park's prediction log, and refresh the stored
    section's displayed numbers to match. MK logs calls every cycle
    because it retrains every cycle; the parks retrain every ~25 min,
    so without this their calls would exist only at retrain moments and
    most lookback squares would have nothing to grade."""
    cfg = PARKS[park]
    sp = section_path(park)
    if not os.path.exists(sp):
        return
    section = json.load(open(sp))
    rides = section.get("rides") or []
    if not rides or "d30" not in rides[0]:
        return  # section predates delta storage; its next retrain adds them
    raw = load_park_waits(os.path.join(ROOT, cfg["waits"]))
    if not len(raw):
        return
    gts = raw["ts"].max()
    snap = raw[raw["ts"] == gts]
    cur = {r["name"]: float(r["wait"]) for _, r in snap.iterrows()
           if pd.notna(r["wait"])}
    if not cur:
        return  # park is posting no waits right now (closed / overnight)
    lts = pd.Timestamp(gts).tz_convert("UTC")
    prev = raw[raw["ts"] == gts - pd.Timedelta(minutes=30)]
    prev_map = {r["name"]: float(r["wait"]) for _, r in prev.iterrows()
                if pd.notna(r["wait"])}
    new_rows = []
    for row in rides:
        w = cur.get(row["name"])
        if w is None:
            continue
        row["now"] = int(round(w))
        if row["name"] in prev_map:
            row["trend30"] = int(round(w - prev_map[row["name"]]))
        for h in F.HORIZONS:
            d = row.get(f"d{h}")
            if d is None:
                continue
            p = int(max(0.0, round((w + d) / 5.0) * 5))
            row[f"f{h}"] = p
            if f"d{h}_lo" in row:
                row[f"f{h}_lo"] = int(max(0.0, round((p + row[f"d{h}_lo"]) / 5.0) * 5))
                row[f"f{h}_hi"] = int(max(0.0, round((p + row[f"d{h}_hi"]) / 5.0) * 5))
            new_rows.append({"made_at": lts.isoformat(), "name": row["name"],
                             "horizon": h,
                             "target": (lts + pd.Timedelta(minutes=h)).isoformat(),
                             "pred": p})
    if not new_rows:
        return
    _append_log(cfg, new_rows, lts)
    rides.sort(key=lambda x: -x["now"])
    section["rides"] = rides
    section["updated"] = datetime.now(timezone.utc).isoformat()
    section["last_sample"] = raw["ts_utc"].max().isoformat()
    json.dump(section, open(sp, "w"), indent=1)
    print(f"{park}: refreshed calls for {len(new_rows) // len(F.HORIZONS)} rides "
          f"from standing deltas")


def section_path(park):
    return os.path.join(ROOT, "data", f"park_section_{park}.json")


def attach():
    """Merge every stored park section into docs/data.json (cheap)."""
    if not os.path.exists(OUT):
        return
    try:
        data = json.load(open(OUT))
    except Exception:
        return
    parks = data.get("parks") or {}
    for park in PARKS:
        sp = section_path(park)
        if os.path.exists(sp):
            try:
                parks[park] = json.load(open(sp))
            except Exception:
                pass
    if parks:
        data["parks"] = parks
        json.dump(data, open(OUT, "w"), indent=1)


def main():
    force = "--force" in sys.argv
    wanted = [a for a in sys.argv[1:] if a in PARKS] or list(PARKS)
    state = {}
    if os.path.exists(STATE):
        try:
            state = json.load(open(STATE))
        except Exception:
            state = {}
    now_utc = datetime.now(timezone.utc)
    retrained = set()
    for park in wanted:
        last = state.get(park)
        due = True
        if not force and last:
            try:
                age = (now_utc - datetime.fromisoformat(last)).total_seconds() / 60
                due = age >= DUE_MINUTES
            except Exception:
                due = True
        if not due:
            continue
        try:
            run_park(park)
            retrained.add(park)
            state[park] = now_utc.isoformat()
            # Save progress per park: if the calling run is cut short,
            # finished parks stay done and only the rest remain due.
            try:
                json.dump(state, open(STATE, "w"), indent=1)
            except Exception:
                pass
        except Exception as ex:   # one park's engine never blocks the others
            print(f"{park} engine EXC (swallowed): {type(ex).__name__}: {ex}")
    # Parks that did not retrain this cycle still issue fresh calls from
    # their standing model's deltas, so every park logs every cycle.
    for park in wanted:
        if park in retrained:
            continue
        try:
            refresh_park(park)
        except Exception as ex:   # a refresh never blocks the others
            print(f"{park} refresh EXC (swallowed): {type(ex).__name__}: {ex}")
    attach()


if __name__ == "__main__":
    main()
    sys.exit(0)
