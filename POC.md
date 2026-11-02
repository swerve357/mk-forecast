# Proof of concept: can posted data predict Magic Kingdom waits?

## The question
Using only public data (posted waits + context), can a model predict each
ride's posted wait 30 / 60 / 90 / 120 minutes ahead with a low enough error
that it would be worth plugging in Disney's actual raw line data later
(true wait times, throughput, downtime logs, Lightning Lane returns)?

## Factors the engine can learn from (v2)
| Group | Features | Source |
|---|---|---|
| Ride dynamics | wait now; lags 5m, 15m, 30m, 1h, 2h, same time yesterday; 15/30-min trend; 30/60-min rolling mean; 60-min volatility; downtime minutes in last hour; just-reopened flag | ThemeParks.wiki live feed, every 5 min |
| Park-wide pressure | mean wait, 90th-percentile wait, share of rides at 40+ min (same timestamp) | derived from the live feed |
| Park hours | open/close, minutes since open, minutes to close, operating day length, Early Entry, early close (<= 18:00) | ThemeParks.wiki schedule feed |
| Special events | Christmas party flag, minutes to party start, party entry window (16:00-19:00), party hours, holiday flag, school-break windows, days until Christmas | `data/events.csv` (25 Mickey's Very Merry Christmas Party nights, Nov 8 - Dec 22 2026) + schedule ticketed events |
| Disney's own demand signal | that day's Lightning Lane Multi Pass / Premier / TRON / Mine Train prices and sold-out count — Disney prices these by expected demand | schedule feed `purchases` |
| Weather | temp, feels-like, humidity, rain amount, rain probability, thunderstorm code, wind, gusts — current hour | Open-Meteo (keyless), Magic Kingdom coordinates |
| Calendar | hour of day (sin/cos), day of week, weekend | derived |

The model is one HistGradientBoosting regressor per horizon. It retrains
on every collection run and reports permutation importance per horizon,
so we can see which factors it is actually using — not which ones we
assume matter. Weather sensitivity is learned per ride (ride id is a
categorical feature), so outdoor rides can react to rain differently
from indoor rides without us hand-coding it.

## How accuracy is judged (no cheating)
- Strict time split: train on the first 80% of time, score only on the
  final 20% the model never saw.
- Every score is shown next to the persistence baseline ("the wait in
  X minutes = the wait now"). A model only ships if it beats that
  baseline on unseen time; otherwise the dashboard honestly falls back.
- Three error views, because posted waits are rounded to 5 minutes:
  MAE (minutes), % of predictions within 10 minutes, and MAPE computed
  only where the true wait was >= 20 minutes.

## Proof-of-concept gates
Data gate (before judging anything):
- [ ] >= 21 days of continuous 5-minute collection
- [ ] Coverage of both party and non-party nights, at least one rainy day,
      and Thanksgiving week or Christmas week
- [ ] >= 100k ride-samples in `data/waits.csv`

Accuracy gates (on the unseen final 20% of time, per horizon):
- [ ] Beats persistence MAE by >= 15% at +30 and +60, >= 10% at +90 and +120
- [ ] MAE <= 10 min at +30, <= 13 min at +60, <= 16 min at +90, <= 18 min at +120
- [ ] MAPE (true waits >= 20 min) <= 25% at +30/+60, <= 35% at +90/+120
- [ ] >= 70% of predictions within 10 minutes at +30

## Decision rule
- Gates pass -> posted-data POC is real; Disney raw data (actual vs posted
  waits, per-ride throughput, LL return flow, downtime causes) is the
  obvious next lever and should mainly cut the remaining error from
  breakdowns and posted-wait inflation.
- Model wins at +30/+60 but not +90/+120 -> the signal horizon is ~1 hour
  with public data; raw demand data (reservations, LL inventory burn) is
  what a 2-hour forecast would need.
- Model cannot beat persistence anywhere -> public posted data is too
  coarse; do not invest further without raw data.

## Known limits of the public-data POC
- Posted waits are Disney's displayed number (rounded, sometimes padded),
  not a stopwatch wait. We predict the posted number.
- Weather rows are the latest forecast/observation for that hour; a
  stricter setup would archive the forecast exactly as issued.
- Lightning Lane prices are a daily signal, not intraday.
- Sudden breakdowns, parade/fireworks surges, and unannounced closures
  will always create an error floor — the downtime/reopen features only
  partially capture them.
