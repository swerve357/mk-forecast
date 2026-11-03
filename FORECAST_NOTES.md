# Forecast data and operating assumptions

Historical seed: ThemeParks.wiki Magic Kingdom daily history, October 30–November 1, 2026. Powered by ThemeParks.wiki (https://themeparks.wiki/).

`data/history.csv.gz` contains 11,713 conservative replay records, not 11,713 independent live polls. Each record includes source and state timestamp. Import with `python scripts/import_history.py history-YYYY-MM-DD.json ...`. The importer uses only preceding states, carries them at most 30 minutes, excludes stale opening states and handles local-day UTC/DST boundaries. This conservative rule can underrepresent long unchanged waits. Live observations override duplicate ride/time bins. Original live observations are preserved in waits.csv.

Missing samples remain missing; closed/down rides do not receive interpolated waits. Forecast labels are paired at the requested elapsed horizon. Training targets must fall strictly before the holdout starts. Each horizon's last 20% of eligible timestamps is used for model selection, so reported accuracy remains preliminary and is not independent prospective validation. Quantile ranges are model estimates, not calibrated coverage guarantees. More days and forward evaluation are needed before operational reliance.

Capacity ranges come from the user's supplied planning table, not verified measurements. For attractions with Lightning Lane, 20–60% standby allocation is an explicit sensitivity assumption, not an observed share. The resulting queue ranges describe operating scenarios, not confidence intervals. They omit posted-wait inflation, walking/preshow components and real-time capacity loss. Unknown attraction capacities show Not configured. Queue scenarios are not total park attendance.

# Collection and Actions diagnosis

Observed November 2, 2026: run 41563570900 was created at 21:02:15 Eastern, its job started at 21:07:39 and completed at 21:08:05. All three observed collector runs were manually dispatched; no scheduled collector run had yet been observed. No run was queued at inspection. Run 41557887513 failed because setup-python's pip cache could not find requirements.txt; later runs succeeded after that file was added.

The workflow offsets its five-minute cron from minute zero, uses requirements.txt consistently, starts on relevant main-branch code changes, and limits job runtime to ten minutes. This does not guarantee punctual GitHub scheduling or bound queue time. GitHub documents scheduler delays under load: https://docs.github.com/en/actions/how-tos/troubleshoot-workflows . The dashboard warns when the newest observed sample is older than 15 minutes. Forecast horizons are anchored to that sample, not browser time. For reliably timed polling, use a dedicated collector/scheduler rather than relying solely on Actions cron.

Validation: `python -m unittest discover -s tests`; Python compile checks; inline JavaScript syntax check; full forecast execution on the seeded history and available live records. Browser appearance was not visually verified in this change.
