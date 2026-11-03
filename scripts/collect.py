"""Poll ThemeParks.wiki for live waits at all four Walt Disney World parks
and append one CSV per park.

Magic Kingdom stays in data/waits.csv (the forecast pipeline reads it).
The other three parks are collected data-only for now, into
data/waits_epcot.csv, data/waits_hollywood_studios.csv and
data/waits_animal_kingdom.csv, in the identical row format, so their
history is already banked when forecasting expands beyond Magic Kingdom.

Failure policy: one park's outage never loses the others — each park is
collected independently, and the run only fails if every park fails.
"""
import csv, json, os, urllib.request
from datetime import datetime, timezone

# Park entity IDs on ThemeParks.wiki (verify at https://api.themeparks.wiki/v1/destinations)
PARKS = [
    ("Magic Kingdom", "75ea578a-adc8-4116-a54d-dccb60765ef9", "waits.csv"),
    ("EPCOT", "47f90d2c-e191-4239-a466-5892ef59a88b", "waits_epcot.csv"),
    ("Hollywood Studios", "288747d1-8b4f-4a64-867e-ea7c9b27bad8", "waits_hollywood_studios.csv"),
    ("Animal Kingdom", "1c84a229-8862-4648-9c71-378ddd2c7693", "waits_animal_kingdom.csv"),
]
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def collect_park(name, park_id, filename):
    url = f"https://api.themeparks.wiki/v1/entity/{park_id}/live"
    req = urllib.request.Request(url, headers={"User-Agent": "mk-forecast/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for e in data.get("liveData", []):
        if e.get("entityType") != "ATTRACTION":
            continue
        wait = ((e.get("queue") or {}).get("STANDBY") or {}).get("waitTime")
        rows.append([ts, e["id"], e["name"], e.get("status", ""), "" if wait is None else wait])
    out = os.path.join(DATA_DIR, filename)
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts_utc", "ride_id", "name", "status", "wait"])
        w.writerows(rows)
    print(f"{name}: {ts} wrote {len(rows)} rides -> {filename}")
    return len(rows)


def main():
    ok = 0
    for name, park_id, filename in PARKS:
        try:
            collect_park(name, park_id, filename)
            ok += 1
        except Exception as e:
            print(f"{name}: FAILED ({e})")
    if ok == 0:
        raise SystemExit("all parks failed to collect")
    print(f"collected {ok}/{len(PARKS)} parks")


if __name__ == "__main__":
    main()
