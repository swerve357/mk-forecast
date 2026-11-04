"""Magic Kingdom wait forecast v2 - proof-of-concept learning engine.

Trains one gradient-boosting model per horizon (30/60/90/120 min) on
data/waits.csv plus the context factors the model is allowed to learn from:

  ride dynamics   current wait, lags 5m..24h, 15/30-min trend, rolling
                  mean/std, downtime in the last hour, just-reopened flag
  park-wide       mean / p90 wait across all rides, share of rides busy
  park hours      open/close that day, minutes since open / to close,
                  operating day length, Early Entry, early close
  special events  curated data/events.csv (Christmas parties, holidays)
                  + ticketed-event flags from the schedule feed
  Disney demand   that day's Lightning Lane prices from the schedule feed
                  (Multi / Premier / TRON / Mine Train) and sold-out count
  weather         Open-Meteo hourly: temp, feels-like, humidity, rain,
                  rain probability, storm code, wind/gusts (data/weather_hourly.csv)
  calendar        hour (sin/cos), day of week, weekend, US holidays,
                  school-break windows, days until Christmas

Validation is a strict time split: train on the first 80% of *time*,
score on the last 20%, and always compare against the persistence
baseline ('wait will be the same as now'). A horizon only ships its
model when it beats that baseline on unseen time; otherwise the
dashboard falls back to persistence and says so.

Error is reported three ways because posted waits are rounded to 5 min:
MAE in minutes, % within 10 minutes, and MAPE computed only where the
true wait is >= 20 min (percentages on a 5- or 10-min wait are noise).

With enough rows it also trains 10th/90th percentile models so the
dashboard can show a range, and runs permutation importance so we can
see which factors the engine actually learned from.
"""
import json, os
from datetime import date, datetime, timedelta, timezone
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV = os.path.join(ROOT, "data", "waits.csv")
SCHED = os.path.join(ROOT, "data", "park_schedule.csv")
EVENTS = os.path.join(ROOT, "data", "events.csv")
WEATHER = os.path.join(ROOT, "data", "weather_hourly.csv")
OUT = os.path.join(ROOT, "docs", "data.json")
CAP = os.path.join(ROOT, "capacity.json")
HORIZONS = {30: 6, 60: 12, 90: 18, 120: 24}   # minutes -> 5-minute steps
DEFAULT_CAPACITY = None
MIN_TRAIN_ROWS = 800          # pooled ride-rows before a model may ship
MIN_QUANTILE_ROWS = 3000      # pooled rows before interval models ship
TRAIN_WINDOW_DAYS = 60        # train on the most recent 60 days only:
                              # bounds the 5-min retrain cost as data grows
                              # and lets the engine adapt to the season

X_COLS = [
    "wait", "lag_5", "lag_15", "lag_30", "lag_60", "lag_120", "lag_yday",
    "diff_15", "diff_30", "roll_mean_30", "roll_mean_60", "roll_std_60",
    "down_mins_60", "reopened_30",
    "park_mean", "park_p90", "park_frac_busy",
    "hour_sin", "hour_cos", "dow", "is_weekend",
    "mins_since_open", "mins_to_close", "operating_mins", "early_entry",
    "is_early_close", "has_ticketed_event",
    "ll_multipass", "ll_premier", "ll_tron", "ll_mine_train", "ll_soldout",
    "is_party_night", "is_holiday", "is_school_break",
    "mins_to_party_start", "is_party_entry_window", "is_party_hours",
    "days_until_christmas",
    "temp_f", "feels_f", "humidity", "precip_in", "precip_prob",
    "is_raining", "is_storm", "wind_mph", "gust_mph",
    "ride_code",
]

# ---------- calendar helpers ----------

def nth_weekday(year, month, weekday, n):
    d = date(year, month, 1)
    return d + timedelta(days=((weekday - d.weekday()) % 7) + 7 * (n - 1))

def last_weekday(year, month, weekday):
    d = (date(year, month + 1, 1) - timedelta(days=1)) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)

def us_holidays(year):
    fixed = [date(year, 1, 1), date(year, 6, 19), date(year, 7, 4),
             date(year, 11, 11), date(year, 12, 25)]
    floating = [nth_weekday(year, 1, 0, 3), nth_weekday(year, 2, 0, 3),
                last_weekday(year, 5, 0), nth_weekday(year, 9, 0, 1),
                nth_weekday(year, 10, 0, 2), nth_weekday(year, 11, 3, 4)]
    return set(fixed + floating)

def thanksgiving(year):
    return nth_weekday(year, 11, 3, 4)

def calendar_frame(ts_index):
    dates = ts_index.date
    out = pd.DataFrame(index=ts_index)
    hour = ts_index.hour + ts_index.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow"] = ts_index.dayofweek
    out["is_weekend"] = (ts_index.dayofweek >= 5).astype(int)
    hol, brk, dtc = [], [], []
    cache = {}
    for d in dates:
        if d.year not in cache:
            cache[d.year] = (us_holidays(d.year), thanksgiving(d.year))
        hols, tg = cache[d.year]
        hol.append(1 if d in hols else 0)
        xmas_break = (d.month == 12 and d.day >= 19) or (d.month == 1 and d.day <= 3)
        tg_break = tg - timedelta(days=1) <= d <= tg + timedelta(days=3)
        brk.append(1 if (xmas_break or tg_break) else 0)
        dtc.append((date(d.year, 12, 25) - d).days)
    out["is_holiday"] = hol
    out["is_school_break"] = brk
    out["days_until_christmas"] = dtc
    return out

# ---------- loading ----------

def load_waits():
    frames = [pd.read_csv(CSV)]
    history = os.path.join(ROOT, "data", "history.csv.gz")
    if os.path.exists(history):
        frames.insert(0, pd.read_csv(history))
    df = pd.concat(frames, ignore_index=True)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, format="mixed")
    df["ts"] = df["ts_utc"].dt.ceil("5min").dt.tz_convert("America/New_York")
    return df.sort_values("ts_utc").drop_duplicates(["ride_id", "ts"], keep="last")

def per_ride_frames(df):
    parts = []
    for (rid, name), g in df.groupby(["ride_id", "name"]):
        g = g.set_index("ts").sort_index()
        wait = g["wait"].where(g["status"] == "OPERATING").resample("5min").last()
        down = (g["status"] == "DOWN").astype(float).resample("5min").max()
        full = wait.index.union(down.index)
        wait = wait.reindex(full)  # Never fill targets or features from future observations.
        down = down.reindex(full)  # Missing observations are not proof of operation.
        f = pd.DataFrame({"wait": wait})
        f["lag_5"] = wait.shift(1)
        f["lag_15"] = wait.shift(3)
        f["lag_30"] = wait.shift(6)
        f["lag_60"] = wait.shift(12)
        f["lag_120"] = wait.shift(24)
        f["lag_yday"] = wait.shift(288)       # same time yesterday
        f["diff_15"] = wait - f["lag_15"]
        f["diff_30"] = wait - f["lag_30"]
        f["roll_mean_30"] = wait.rolling(6).mean()
        f["roll_mean_60"] = wait.rolling(12).mean()
        f["roll_std_60"] = wait.rolling(12).std()
        f["down_mins_60"] = down.rolling(12).sum() * 5
        just_up = ((down.shift(1) == 1) & (down == 0) & wait.notna()).astype(float)
        f["reopened_30"] = just_up.rolling(6).max()
        f["ride_id"], f["name"] = rid, name
        for h, s in HORIZONS.items():
            f[f"y{h}"] = wait.shift(-s)
        parts.append(f)
    out = pd.concat(parts)
    out["ride_code"] = out["ride_id"].astype("category").cat.codes
    return out

def add_context(feat):
    # NOTE: timestamps repeat across rides, so a plain index join would
    # cartesian-duplicate rows. Assign calendar columns positionally.
    cal = calendar_frame(feat.index)
    for c in cal.columns:
        feat[c] = cal[c].to_numpy()
    # park-wide pressure, same timestamp across rides
    by_ts = feat.groupby(level=0)["wait"]
    feat["park_mean"] = by_ts.transform("mean")
    feat["park_p90"] = by_ts.transform(lambda s: s.quantile(0.9))
    feat["park_frac_busy"] = by_ts.transform(lambda s: (s >= 40).mean())
    feat["date_str"] = feat.index.strftime("%Y-%m-%d")
    feat["hour_key"] = feat.index.strftime("%Y-%m-%dT%H:00")

    # park schedule (hours + Lightning Lane demand prices)
    sched_cols = ["mins_since_open", "mins_to_close", "operating_mins",
                  "early_entry", "is_early_close", "has_ticketed_event",
                  "ll_multipass", "ll_premier", "ll_tron", "ll_mine_train",
                  "ll_soldout", "_open_min", "_close_min", "_event_start_min"]
    if os.path.exists(SCHED):
        s = pd.read_csv(SCHED, dtype={"date": str})
        def mins(x):
            try:
                h, m = str(x).split(":")[:2]
                return int(h) * 60 + int(m)
            except Exception:
                return np.nan
        s["_open"] = s["open"].map(mins)
        s["_close"] = s["close"].map(mins)
        s["_ev"] = s["ticketed_event_start"].map(mins)
        day_min = feat.index.hour * 60 + feat.index.minute
        idx = s.set_index("date")
        feat["_open_min"] = feat["date_str"].map(idx["_open"])
        feat["_close_min"] = feat["date_str"].map(idx["_close"])
        feat["mins_since_open"] = day_min - feat["_open_min"]
        feat["mins_to_close"] = feat["_close_min"] - day_min
        feat["operating_mins"] = feat["date_str"].map(idx["operating_mins"])
        feat["early_entry"] = feat["date_str"].map(
            idx["early_entry"].fillna("").ne("").astype(int))
        feat["is_early_close"] = (feat["_close_min"] <= 18 * 60).astype(float)
        feat["has_ticketed_event"] = feat["date_str"].map(
            idx["has_ticketed_event"].fillna(0))
        for src, dst in [("ll_multipass_usd", "ll_multipass"), ("ll_premier_usd", "ll_premier"),
                         ("ll_tron_usd", "ll_tron"), ("ll_mine_train_usd", "ll_mine_train"),
                         ("ll_soldout_count", "ll_soldout")]:
            feat[dst] = feat["date_str"].map(idx[src])
        feat["_event_start_min"] = feat["date_str"].map(idx["_ev"])
    else:
        for c in sched_cols:
            feat[c] = np.nan

    # curated special events (Christmas parties, holidays)
    feat["is_party_night"] = 0
    feat["_party_start_min"] = np.nan
    if os.path.exists(EVENTS):
        ev = pd.read_csv(EVENTS, dtype={"date": str})
        party = ev[ev["kind"] == "christmas_party"]
        feat["is_party_night"] = feat["date_str"].isin(set(party["date"])).astype(int)
        ps = party.set_index("date")["party_start"].dropna()
        def mins2(x):
            try:
                h, m = str(x).split(":")[:2]
                return int(h) * 60 + int(m)
            except Exception:
                return np.nan
        feat["_party_start_min"] = feat["date_str"].map(ps.map(mins2))
    # schedule's ticketed event also proves a party/event night
    feat["is_party_night"] = np.maximum(feat["is_party_night"],
                                        feat["has_ticketed_event"].fillna(0))
    start = feat["_party_start_min"].fillna(feat["_event_start_min"])
    day_min = feat.index.hour * 60 + feat.index.minute
    feat["mins_to_party_start"] = np.where(feat["is_party_night"] == 1,
                                           start - day_min, np.nan)
    feat["is_party_entry_window"] = ((feat["is_party_night"] == 1) &
                                     (feat["mins_to_party_start"] <= 180) &
                                     (feat["mins_to_party_start"] > 0)).astype(int)
    feat["is_party_hours"] = ((feat["is_party_night"] == 1) &
                              (feat["mins_to_party_start"] <= 0)).astype(int)

    # weather, joined on the local hour
    wmap = {"temperature_2m": "temp_f", "apparent_temperature": "feels_f",
            "relative_humidity_2m": "humidity", "precipitation": "precip_in",
            "precipitation_probability": "precip_prob", "wind_speed_10m": "wind_mph",
            "wind_gusts_10m": "gust_mph", "weather_code": "_wcode"}
    if os.path.exists(WEATHER):
        w = pd.read_csv(WEATHER, dtype={"hour_local": str})
        widx = w.set_index("hour_local")
        for src, dst in wmap.items():
            feat[dst] = pd.to_numeric(feat["hour_key"].map(widx[src]), errors="coerce")
        feat["is_raining"] = (feat["precip_in"].fillna(0) > 0.01).astype(float)
        feat["is_storm"] = (feat["_wcode"].fillna(0) >= 95).astype(float)
    else:
        for dst in list(wmap.values()) + ["is_raining", "is_storm"]:
            feat[dst] = np.nan
    for c in X_COLS:
        if c not in feat.columns:
            feat[c] = np.nan
    return feat

# ---------- metrics ----------

def score(pred, true, base_pred):
    pred, true, base_pred = np.asarray(pred, float), np.asarray(true, float), np.asarray(base_pred, float)
    err = np.abs(pred - true)
    berr = np.abs(base_pred - true)
    ge20 = true >= 20
    return {
        "mae": round(float(err.mean()), 1),
        "rmse": round(float(np.sqrt(((pred - true) ** 2).mean())), 1),
        "within10": round(float((err <= 10).mean() * 100)),
        "within5": round(float((err <= 5).mean() * 100)),
        "mape_ge20": round(float((err[ge20] / true[ge20]).mean() * 100), 1) if ge20.any() else None,
        "bias": round(float((pred - true).mean()), 1),
        "baseline_mae": round(float(berr.mean()), 1),
        "baseline_mape_ge20": round(float((berr[ge20] / true[ge20]).mean() * 100), 1) if ge20.any() else None,
        "improvement_pct": round(float((1 - err.mean() / berr.mean()) * 100), 1) if berr.mean() else None,
    }

# ---------- main ----------

def main():
    if not os.path.exists(CSV):
        print("no data yet"); return
    raw = load_waits()
    if raw.empty:
        print("no rows yet"); return
    feat = add_context(per_ride_frames(raw))
    cap = json.load(open(CAP)) if os.path.exists(CAP) else {}
    latest_ts = feat.index.max()
    now = feat[feat.index == latest_ts].dropna(subset=["wait"])
    cat_idx = [X_COLS.index("ride_code")]
    uniq_times = np.sort(feat.index.unique())
    cut = uniq_times[int(len(uniq_times) * 0.8)] if len(uniq_times) >= 10 else None

    metrics, models, intervals = {}, {}, {}
    ride_scores = {}   # per-ride holdout scores for the dashboard (all horizons)
    for h in HORIZONS:
        d = feat.dropna(subset=["wait", f"y{h}"]).sort_index()
        if len(d):
            d = d[d.index >= d.index.max() - pd.Timedelta(days=TRAIN_WINDOW_DAYS)]
        valid_times = np.sort(d.index.unique())
        cut = valid_times[int(len(valid_times) * 0.8)] if len(valid_times) >= 10 else None
        entry = {"available_pairs": int(len(d)), "train_rows": 0, "test_rows": 0, "model": "persistence (still collecting)"}
        if cut is not None and len(d) >= MIN_TRAIN_ROWS:
            tr, te = d[d.index + pd.Timedelta(minutes=h) < cut], d[d.index >= cut]
            if len(tr) >= MIN_TRAIN_ROWS * 0.5 and len(te) >= 50:
                # Drop features that are all-missing or constant in the
                # training window (e.g. weather before its first successful
                # collection): they carry no signal and some versions of
                # the binner choke on a single distinct value.
                cols = [c for c in X_COLS if tr[c].notna().any()
                        and (c == "ride_code" or tr[c].nunique(dropna=True) > 1)]
                cat = [cols.index("ride_code")] if "ride_code" in cols else None
                m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.07,
                                                  categorical_features=cat)
                # Delta target + round-to-5 (adopted 2026-11-03 after A/B:
                # the model learns the CHANGE from the current wait, and
                # predictions snap to Disney's 5-minute posted steps. The
                # ship gate below scores exactly this shipped form.)
                m.fit(tr[cols], tr[f"y{h}"] - tr["wait"])
                def ship_pred(frame, model):
                    p = frame["wait"].values + model.predict(frame[cols])
                    return np.clip(np.round(p / 5.0) * 5, 0, None)
                s = score(ship_pred(te, m), te[f"y{h}"].values, te["wait"].values)
                entry.update(s, train_rows=int(len(tr)), test_rows=int(len(te)), evaluation="chronological holdout with horizon embargo")
                # Score the predictor that will actually ship (the model if
                # it won, else persistence), per ride, for the dashboard's
                # per-horizon score columns.
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
                    pi = permutation_importance(m, sample[cols], sample[f"y{h}"] - sample["wait"],
                                                n_repeats=3, random_state=0,
                                                scoring="neg_mean_absolute_error")
                    order = np.argsort(pi.importances_mean)[::-1]
                    entry["top_features"] = [cols[i] for i in order
                                             if pi.importances_mean[i] > 0][:6]
                except Exception as e:
                    print(f"importance {h}: {e}")
                if s["mae"] < s["baseline_mae"]:   # only ship a model that wins
                    m.fit(d[cols], d[f"y{h}"] - d["wait"])
                    models[h] = (m, cols)
                    entry["model"] = "gradient boosting (delta+round5)"
                    if len(d) >= MIN_QUANTILE_ROWS:
                        try:
                            lo = HistGradientBoostingRegressor(
                                loss="quantile", quantile=0.1, max_iter=150,
                                learning_rate=0.07, categorical_features=cat)
                            hi = HistGradientBoostingRegressor(
                                loss="quantile", quantile=0.9, max_iter=150,
                                learning_rate=0.07, categorical_features=cat)
                            lo.fit(d[cols], d[f"y{h}"] - d["wait"]); hi.fit(d[cols], d[f"y{h}"] - d["wait"])
                            intervals[h] = (lo, hi, cols)
                        except Exception as e:
                            print(f"quantile {h}: {e}")
                else:
                    entry["candidate_mae"] = s["mae"]
                    entry.update(score(te["wait"], te[f"y{h}"], te["wait"]))
                    entry["model"] = "persistence (model did not beat baseline)"
        elif len(d):
            entry["available_pairs"] = int(len(d))  # Too few observations for a holdout claim.
        metrics[str(h)] = entry

    rides = []
    for _, r in now.reset_index().iterrows():
        row = {"name": r["name"], "now": int(round(r["wait"])),
               "trend30": int(round(r["diff_30"])) if pd.notna(r.get("diff_30")) else 0}
        for h in HORIZONS:
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
                lo_p = max(0.0, round((base + float(lo.predict(pd.DataFrame([r])[icols])[0])) / 5.0) * 5)
                hi_p = max(0.0, round((base + float(hi.predict(pd.DataFrame([r])[icols])[0])) / 5.0) * 5)
                row[f"f{h}_lo"] = int(lo_p)
                row[f"f{h}_hi"] = int(max(lo_p, hi_p))
        for h in HORIZONS:
            sc = ride_scores.get(h, {}).get(r["name"])
            if sc:
                row[f"score{h}_mae"] = sc["mae"]
                row[f"score{h}_in10"] = sc["within10"]
                row[f"score{h}_in5"] = sc["within5"]
                row[f"score{h}_pairs"] = sc["pairs"]
        profile = cap.get(r["name"])
        row["people"] = None
        if isinstance(profile, dict):
            throughput = profile["hourly_capacity_range"]
            share = profile["standby_share_range"]
            row["people_lo"] = round(throughput[0] * share[0] * r["wait"] / 60)
            row["people_hi"] = round(throughput[1] * share[1] * r["wait"] / 60)
            row["people_basis"] = "Uncalibrated operating scenario; not measured attendance"
        elif isinstance(profile, (int, float)):
            row["people"] = round(profile * r["wait"] / 60)
        rides.append(row)
    rides.sort(key=lambda x: -x["now"])

    # context snapshot for the dashboard header (latest day)
    ctx = {"date": str(latest_ts.date())}
    if os.path.exists(SCHED):
        s = pd.read_csv(SCHED, dtype={"date": str})
        today = s[s["date"] == ctx["date"]]
        if len(today):
            t = today.iloc[0]
            ctx.update({"open": t["open"], "close": t["close"],
                        "early_entry": t["early_entry"],
                        "ticketed_event": t["ticketed_event_desc"] if pd.notna(t["ticketed_event_desc"]) else "",
                        "ll_multipass_usd": None if pd.isna(t["ll_multipass_usd"]) else float(t["ll_multipass_usd"])})
    if os.path.exists(EVENTS):
        ev = pd.read_csv(EVENTS, dtype={"date": str})
        e = ev[ev["date"] == ctx["date"]]
        if len(e):
            ctx["event"] = e.iloc[0]["event"]
    ctx["is_party_night"] = bool(now["is_party_night"].max()) if len(now) else False

    out = {"updated": datetime.now(timezone.utc).isoformat(),
           "last_sample": raw["ts_utc"].max().isoformat(), "samples": int(len(raw)),
           "history_samples": int(raw.get("source", pd.Series(dtype=str)).eq("themeparks.wiki history replay").sum()),
           "rides": rides, "metrics": metrics, "context": ctx, "demo": False}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {len(rides)} rides; shipped models: {list(models) or 'none yet'}; "
          f"intervals: {list(intervals) or 'none'}")

if __name__ == "__main__":
    main()

