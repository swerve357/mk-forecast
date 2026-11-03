"""Poll ThemeParks.wiki for Magic Kingdom live waits and append to data/waits.csv."""
import csv, json, os, urllib.request
from datetime import datetime, timezone

# Magic Kingdom Park entity ID on ThemeParks.wiki (verify at https://api.themeparks.wiki/v1/destinations)
PARK_ID = "75ea578a-adc8-4116-a54d-dccb60765ef9"
URL = f"https://api.themeparks.wiki/v1/entity/{PARK_ID}/live"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "waits.csv")

def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "mk-forecast/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for e in data.get("liveData", []):
        if e.get("entityType") != "ATTRACTION":
            continue
        wait = ((e.get("queue") or {}).get("STANDBY") or {}).get("waitTime")
        rows.append([ts, e["id"], e["name"], e.get("status", ""), "" if wait is None else wait])
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts_utc", "ride_id", "name", "status", "wait"])
        w.writerows(rows)
    print(f"{ts}: wrote {len(rows)} rides")

if __name__ == "__main__":
    main()
