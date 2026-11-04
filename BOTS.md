# Manager Bots

The forecast pipeline is the engine room: it produces a prediction for
every ride every 5 minutes, deterministically. Manager bots are the
layer above it — each owns a territory and a number, watches the park,
and proposes changes. The mission they serve: **call the posted wait
2 hours out with 2% error.**

## The rules every bot lives under

1. **Shadow first.** A new bot only records what it *would* have
   predicted (in `data/bot_decisions.jsonl`). It changes nothing.
2. **Authority is earned, per rule, on graded evidence** — the same
   ship-gate discipline as model features. A bot's rule goes live only
   when its shadow record beats the shipped predictions it challenged.
3. **Bounded and expiring.** Live corrections (when earned) are capped
   and carry an expiry. If a bot dies, its decisions lapse and the
   system falls back to the plain pipeline.
4. **Fully audited.** Every decision is timestamped with its reason;
   every grade is kept (`data/bot_graded.json`); the running record is
   `data/bot_scorecard.json`.
5. **Fail-safe.** A bot can never break the forecast run.

Bots are plain scripts (statistics and rules) running inside the
existing 5-minute pipeline — no AI deliberation per cycle. Judgment
enters at design time and at periodic reviews of the scorecards.

## Roster

### Closure Bot — `scripts/closure_bot.py` (active, shadow)
**Charter: own the bounce-back.** Closures and reopens are where
predictions wobble worst and recover slowest.

- **D1 — displacement.** A ride posted at 40+ min has been down 15+
  min → same-land neighbors get +5 min on their shadow +30/+60/+90.
- **D2 — reopen settle.** A ride down 15+ min reopens → its shadow
  +30 is the first posted wait at reopen (persistence-from-reopen)
  instead of the shipped number.

**Record so far:**
- D1 dry-run on Oct 30 – Nov 4 data (89 graded neighbor predictions,
  graded against persistence): **the bump loses** — 21 wins vs 68
  losses, MAE 9.3 vs 6.6. By 15 minutes into a closure, neighbors'
  waits were on average already *falling* (−2.7 min mean move): any
  displacement is priced into the posted numbers before the trigger,
  or doesn't exist at this scale. D1 stays in shadow — party nights
  and busier days are a different regime, and the live scorecard is
  the arbiter — but the prior is now against it.
- D2 has no backtest (it challenges the shipped model specifically);
  it grades live from Nov 4.

### Future bots (not yet built)
- **Land bots** — own per-land error budgets; run their land's A/Bs.
- **Weather bot** — rain/heat reaction rules once rain data exists.
- **Data-health bot** — collection stalls and source anomalies.
- **Park manager** — owns the headline 2% at +120; arbitrates
  conflicts between bots; promotes/demotes rules on the evidence.
