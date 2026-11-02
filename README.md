# Magic Kingdom wait forecast

Predicts each ride's posted wait **30 / 60 / 90 / 120 minutes ahead** and
grades itself honestly against the "same as now" baseline. See
[`POC.md`](POC.md) for the factors, the accuracy gates, and the decision
rule for whether Disney's raw line data is worth plugging in later.

## Run it
1. Push this folder to a new GitHub repo.
2. Settings → Pages → deploy from branch, folder `/docs`.
3. Settings → Actions → General → allow read and write permissions.
4. Actions → collect-and-forecast → Run workflow (it then runs about every 5 minutes).

## How it works
- `scripts/collect.py` — appends posted waits from ThemeParks.wiki to `data/waits.csv` every 5 min.
- `scripts/collect_context.py` — refreshes the factors: park hours, Early Entry,
  ticketed events (Christmas parties), and that day's Lightning Lane prices
  from the same schedule feed → `data/park_schedule.csv`; hourly weather
  from Open-Meteo (keyless) → `data/weather_hourly.csv`. Context failures
  never block wait collection.
- `data/events.csv` — curated special events (all 25 Mickey's Very Merry
  Christmas Party nights 2026, Thanksgiving, Christmas, New Year).
- `scripts/forecast.py` — builds ride/park/weather/event features, trains one
  gradient-boosting model per horizon on the first 80% of time, scores on
  the unseen last 20%, ships a model only if it beats persistence, and
  writes `docs/data.json` (forecasts, ranges when data allows, accuracy,
  and the top factors per horizon).
- `docs/index.html` — the dashboard reads `docs/data.json`.

Accuracy on the dashboard: MAE in minutes, % within 10 min, and MAPE on
20+ min waits only. Edit `capacity.json` (`{"Ride name": guests_per_hour}`)
to tune people-in-line.

Local test: `python -m venv .venv && .venv/bin/pip install pandas scikit-learn`
then run the three scripts in order.
