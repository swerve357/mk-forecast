"""Collect the context factors the model learns from, alongside wait times.

Two sources, both keyless:
1. ThemeParks.wiki schedule for Magic Kingdom -> data/park_schedule.csv
   One row per day: operating open/close, Early Entry, ticketed events
   (e.g. Mickey's Very Merry Christmas Party shows up as a 'Special
   Ticketed Event' + an early 18:00 close for day guests), and the day's
   Lightning Lane prices. LL prices are set by Disney per day and encode
   Disney's own demand expectation, so they are a strong crowd signal.
2. Open-Meteo hourly forecast for the Magic Kingdom coordinates ->
   data/weather_hourly.csv (keyless). Weather is refreshed, not appended
   forever: one row per local hour, newest fetch wins.

Failure policy: a failed source logs and leaves its previous file alone.
A 429 is respected (Retry-After / no hammering) and never fails the run,
so the 5-minute wait collector is never blocked by context collection.
"""
import csv, json, os, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime

ROOT = os.path.join(os.path.dirname(__file__), "..")
PARK_ID = "75ea578a-adc8-4116-a54d-dccb60765ef9"
LAT, LON = 28.4177, -81.5812  # Magic Kingdom
UA = {"User-Agent": "mk-forecast/0.2 poc"}


def fetch_json(url, tries=2):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get("Retry-After") or 0)
                print(f"429 from {url.split('/')[2]}; Retry-After={wait or 'n/a'}")
                return None  # respect the limit; do not retry inside a cron tick
            if i == tries - 1:
                print(f"HTTP {e.code} from {url}")
                return None
            time.sleep(2)
        except Exception as e:  # network hiccup etc.
            if i == tries - 1:
                print(f"fetch failed {url}: {e}")
                return None
            time.sleep(2)
    return None


def hhmm(iso):
    return iso[11:16] if iso and len(iso) >= 16 else ""


def collect_schedule():
    data = fetch_json(f"https://api.themeparks.wiki/v1/entity/{PARK_ID}/schedule")
    if not data:
        return
    by_date = {}
    for s in data.get("schedule", []):
        d = by_date.setdefault(s["date"], {
            "date": s["date"], "open": "", "close": "", "operating_mins": "",
            "early_entry": "", "has_ticketed_event": 0, "ticketed_event_desc": "",
            "ticketed_event_start": "", "ticketed_event_end": "",
            "ll_multipass_usd": "", "ll_premier_usd": "",
            "ll_tron_usd": "", "ll_mine_train_usd": "", "ll_soldout_count": 0})
        if s["type"] == "OPERATING":
            d["open"], d["close"] = hhmm(s.get("openingTime")), hhmm(s.get("closingTime"))
            try:
                o = datetime.fromisoformat(s["openingTime"])
                c = datetime.fromisoformat(s["closingTime"])
                d["operating_mins"] = int((c - o).total_seconds() // 60)
            except Exception:
                pass
            for p in s.get("purchases", []) or []:
                price = ((p.get("price") or {}).get("amount") or 0) / 100
                name = p.get("name", "")
                if "Multi Pass" in name:
                    d["ll_multipass_usd"] = price
                elif "Premier Pass" in name:
                    d["ll_premier_usd"] = price
                elif "TRON" in name:
                    d["ll_tron_usd"] = price
                elif "Seven Dwarfs" in name:
                    d["ll_mine_train_usd"] = price
                if not p.get("available", True):
                    d["ll_soldout_count"] += 1
        elif s["type"] == "TICKETED_EVENT":
            desc = s.get("description", "")
            if desc == "Early Entry":
                d["early_entry"] = hhmm(s.get("openingTime"))
            else:
                d["has_ticketed_event"] = 1
                d["ticketed_event_desc"] = desc
                d["ticketed_event_start"] = hhmm(s.get("openingTime"))
                d["ticketed_event_end"] = hhmm(s.get("closingTime"))
    rows = [by_date[d] for d in sorted(by_date)]
    out = os.path.join(ROOT, "data", "park_schedule.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"schedule: wrote {len(rows)} days ({rows[0]['date']}..{rows[-1]['date']})")


# Data-only context for the other three WDW parks (same schedule endpoint).
# Magic Kingdom's file keeps its ride-specific Lightning Lane columns
# (ll_tron_usd, ll_mine_train_usd) because forecast.py reads them; the other
# parks use a generic schema that captures up to three single-ride LL prices
# by name, whatever rides Disney prices that day.
OTHER_PARKS = [
    ("EPCOT", "47f90d2c-e191-4239-a466-5892ef59a88b", "park_schedule_epcot.csv"),
    ("Hollywood Studios", "288747d1-8b4f-4a64-867e-ea7c9b27bad8", "park_schedule_hollywood_studios.csv"),
    ("Animal Kingdom", "1c84a229-8862-4648-9c71-378ddd2c7693", "park_schedule_animal_kingdom.csv"),
]

GENERIC_FIELDS = ["date", "open", "close", "operating_mins", "early_entry",
                  "has_ticketed_event", "ticketed_event_desc",
                  "ticketed_event_start", "ticketed_event_end",
                  "ll_multipass_usd", "ll_premier_usd",
                  "ll_single_1_name", "ll_single_1_usd",
                  "ll_single_2_name", "ll_single_2_usd",
                  "ll_single_3_name", "ll_single_3_usd",
                  "ll_soldout_count"]


def collect_schedule_generic(park_name, park_id, out_name):
    data = fetch_json(f"https://api.themeparks.wiki/v1/entity/{park_id}/schedule")
    if not data:
        print(f"{park_name}: schedule unavailable this run, keeping previous file")
        return
    by_date = {}
    for s in data.get("schedule", []):
        d = by_date.setdefault(s["date"], {k: "" for k in GENERIC_FIELDS})
        d["date"] = s["date"]
        d["has_ticketed_event"] = 0
        d["ll_soldout_count"] = 0
        if s["type"] == "OPERATING":
            d["open"], d["close"] = hhmm(s.get("openingTime")), hhmm(s.get("closingTime"))
            try:
                o = datetime.fromisoformat(s["openingTime"])
                c = datetime.fromisoformat(s["closingTime"])
                d["operating_mins"] = int((c - o).total_seconds() // 60)
            except Exception:
                pass
            singles = []
            for p in s.get("purchases", []) or []:
                price = ((p.get("price") or {}).get("amount") or 0) / 100
                name = p.get("name", "")
                if "Multi Pass" in name:
                    d["ll_multipass_usd"] = price
                elif "Premier Pass" in name:
                    d["ll_premier_usd"] = price
                elif name:
                    singles.append((name, price))
                if not p.get("available", True):
                    d["ll_soldout_count"] += 1
            for i, (name, price) in enumerate(sorted(singles)[:3], 1):
                d[f"ll_single_{i}_name"] = name
                d[f"ll_single_{i}_usd"] = price
        elif s["type"] == "TICKETED_EVENT":
            desc = s.get("description", "")
            if desc == "Early Entry":
                d["early_entry"] = hhmm(s.get("openingTime"))
            else:
                d["has_ticketed_event"] = 1
                d["ticketed_event_desc"] = desc
                d["ticketed_event_start"] = hhmm(s.get("openingTime"))
                d["ticketed_event_end"] = hhmm(s.get("closingTime"))
    rows = [by_date[d] for d in sorted(by_date)]
    out = os.path.join(ROOT, "data", out_name)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GENERIC_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{park_name} schedule: wrote {len(rows)} days -> {out_name}")


def collect_weather():
    params = {
        "latitude": LAT, "longitude": LON, "timezone": "America/New_York",
        "past_days": 2, "forecast_days": 3, "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph", "precipitation_unit": "inch",
        "hourly": ",".join(["temperature_2m", "apparent_temperature",
                            "relative_humidity_2m", "precipitation",
                            "precipitation_probability", "weather_code",
                            "wind_speed_10m", "wind_gusts_10m"]),
    }
    data = fetch_json("https://api.open-meteo.com/v1/forecast?" +
                      urllib.parse.urlencode(params))
    if not data or "hourly" not in data:
        print("weather: unavailable this run, keeping previous file")
        return
    h = data["hourly"]
    keys = ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
            "precipitation", "precipitation_probability", "weather_code",
            "wind_speed_10m", "wind_gusts_10m"]
    out = os.path.join(ROOT, "data", "weather_hourly.csv")
    merged = {}
    if os.path.exists(out):  # keep older hours, newest fetch wins per hour
        with open(out) as f:
            for row in csv.DictReader(f):
                merged[row["hour_local"]] = row
    fetched = datetime.now().astimezone().isoformat(timespec="seconds")
    for i, t in enumerate(h["time"]):
        row = {"hour_local": t, "fetched_at": fetched}
        for k in keys:
            row[k] = h[k][i]
        merged[t] = row
    rows = [merged[t] for t in sorted(merged)]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["hour_local", "fetched_at"] + keys)
        w.writeheader()
        w.writerows(rows)
    print(f"weather: {len(rows)} hourly rows stored")


if __name__ == "__main__":
    collect_schedule()
    for park_name, park_id, out_name in OTHER_PARKS:
        try:
            collect_schedule_generic(park_name, park_id, out_name)
        except Exception as e:
            print(f"{park_name}: schedule failed ({e}); keeping previous file")
    collect_weather()
