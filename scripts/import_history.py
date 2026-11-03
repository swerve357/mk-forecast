"""Convert downloaded ThemeParks.wiki daily history JSON to separate replay CSV.
Past states only, at most 30 minutes old. Never infer unchanged state indefinitely.
Live snapshots remain in waits.csv and take precedence on duplicate ride/time bins.
Usage: python scripts/import_history.py history-*.json
"""
import csv, json, sys, gzip
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

FIELDS = ['ts_utc', 'ride_id', 'name', 'status', 'wait', 'source', 'state_observed_at']

def parse(x):
    return datetime.fromisoformat(x.replace('Z', '+00:00'))

def replay(data):
    zone = ZoneInfo('America/New_York')
    start = datetime.fromisoformat(data['range']['from']).replace(tzinfo=zone).astimezone(timezone.utc)
    end = (datetime.fromisoformat(data['range']['to']) + timedelta(days=1)).replace(tzinfo=zone).astimezone(timezone.utc)
    for ent in data['entities']:
        if ent.get('entityType') != 'ATTRACTION':
            continue
        events = list(ent.get('history', []))
        opening = ent.get('opening')
        if opening and opening.get('observedAt'):
            events.append(dict(opening, time=opening['observedAt']))
        events.sort(key=lambda e: parse(e['time']))
        i, state = 0, None
        t = start
        while t < end:
            while i < len(events) and parse(events[i]['time']) <= t:
                state = events[i]; i += 1
            if state and timedelta(0) <= t - parse(state['time']) <= timedelta(minutes=30):
                wait = ((state.get('queue') or {}).get('STANDBY') or {}).get('waitTime')
                yield [t.isoformat(), ent['id'], ent['name'], state.get('status', ''),
                       '' if wait is None else wait, 'themeparks.wiki history replay', state['time']]
            t += timedelta(minutes=5)

if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'data/history.csv.gz'
    rows = {}
    if out.exists():
        with gzip.open(out, 'rt') as f:
            for r in csv.reader(f):
                if r[0] != 'ts_utc': rows[(r[0], r[1])] = r
    for path in sys.argv[1:]:
        for r in replay(json.loads(Path(path).read_text())):
            rows[(r[0], r[1])] = r
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, 'wt', newline='') as f:
        w = csv.writer(f); w.writerow(FIELDS); w.writerows(rows[k] for k in sorted(rows))
    print(f'{len(rows)} historical replay rows -> {out}')
