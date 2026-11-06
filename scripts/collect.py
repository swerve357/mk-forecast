"""Poll ThemeParks.wiki for live waits at all four Walt Disney World parks
and append one CSV per park.

Collection strategy (Dan's call, 2026-11-06): ONE call to the Walt
Disney World resort entity's /live feed returns every park's entities
at a single instant — all four parks banked together each cycle, so a
cycle that runs long can never lose one park's data while keeping
another's. The payload is sliced by parkId into the same per-park CSVs
as always. If the resort call fails — or a park's slice comes back
empty — that park falls back to its own /live call, preserving the
original per-park failure isolation.

Magic Kingdom stays in data/waits.csv (the forecast pipeline reads it);
the other parks write data/waits_epcot.csv, data/waits_hollywood_studios.csv
and data/waits_animal_kingdom.csv in the identical row format.

Failure policy: one park's outage never loses the others — the run only
fails if every park fails.
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
# The resort entity's /live feed spans all six WDW parks (we slice out
# our four; water parks and untagged resort entities are ignored).
RESORT_ID = "e957da41-3552-4cf6-b636-5babc5cbc4e5"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ts_utc is OUR pull DTG (exact, to the second, one stamp per poll).
# src_updated_utc is the source's own per-ride lastUpdated from the API
# payload (millisecond precision): when ThemeParks.wiki last refreshed
# that ride's posted value. Two collectors polling at different moments
# record different ts_utc but the SAME src_updated_utc for the same
# posted value, which is what makes cross-source confirmation possible.
HEADER = ["ts_utc", "ride_id", "name", "status", "wait", "src_updated_utc"]


def ensure_header(out):
    """Create the file with HEADER, or one-time migrate an older file by
    appending the new column (empty for legacy rows). Returns True if the
    file was just created and still needs its header written on append."""
    if not os.path.exists(out):
        return True
    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    if rows and rows[0] == HEADER:
        return False
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in rows[1:] if rows else []:
            w.writerow((r + [""] * len(HEADER))[:len(HEADER)])
    return False


def fetch_live(entity_id):
    url = f"https://api.themeparks.wiki/v1/entity/{entity_id}/live"
    req = urllib.request.Request(url, headers={"User-Agent": "mk-forecast/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def write_park(name, data, filename, ts):
    """Append one park's ATTRACTION rows from a /live payload (the whole
    park payload, or this park's slice of the resort payload), then run
    the showtime tracker on the same entities."""
    rows = []
    for e in data.get("liveData", []):
        if e.get("entityType") != "ATTRACTION":
            continue
        wait = ((e.get("queue") or {}).get("STANDBY") or {}).get("waitTime")
        rows.append([ts, e["id"], e["name"], e.get("status", ""),
                     "" if wait is None else wait, e.get("lastUpdated", "")])
    out = os.path.join(DATA_DIR, filename)
    new = ensure_header(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        w.writerows(rows)
    print(f"{name}: {ts} wrote {len(rows)} rides -> {filename}")
    # Showtime tracker: same payload, zero extra API calls. Fail-safe —
    # a tracker problem must never cost us wait data.
    try:
        import showtimes as _shows
        n = _shows.track(name, data, ts)
        print(f"{name}: showtime tracker ok ({n} change events)")
    except Exception as e:
        print(f"{name}: showtime tracker skipped ({e})")
    return len(rows)


def collect_park_fallback(name, park_id, filename, ts):
    """The original per-park call, used when the resort feed fails or a
    park's slice comes back empty."""
    data = fetch_live(park_id)
    return write_park(name, data, filename, ts)


def main():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    slices = {}
    try:
        resort = fetch_live(RESORT_ID)
        for e in resort.get("liveData", []):
            slices.setdefault(e.get("parkId"), []).append(e)
        print(f"resort call ok: {len(resort.get('liveData', []))} entities")
    except Exception as e:
        print(f"resort call FAILED ({e}) — falling back to per-park calls")
    ok = 0
    for name, park_id, filename in PARKS:
        try:
            ents = slices.get(park_id) or []
            if len(ents) < 10:
                raise ValueError(f"resort slice too small ({len(ents)})")
            write_park(name, {"liveData": ents}, filename, ts)
            ok += 1
        except Exception as e:
            print(f"{name}: resort slice unusable ({e}); per-park fallback")
            try:
                collect_park_fallback(name, park_id, filename, ts)
                ok += 1
            except Exception as e2:
                print(f"{name}: FAILED ({e2})")
    if ok == 0:
        raise SystemExit("all parks failed to collect")
    print(f"collected {ok}/{len(PARKS)} parks")


if __name__ == "__main__":
    main()
