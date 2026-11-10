# Audit and station backtest — 2026-11-10

## Existing Kaggle result

Saved notebook: https://www.kaggle.com/code/williams784/notebooke1fbf5ed50/output?scriptVersionId=367932227
The saved research-output/metrics.json was read in the cloud browser. Initial training used commit b4e144cb86bb2478e047a95245693bfc1297906b; later pilot code used 9694674c1080173330f23420698dae11db4e4040.

6 cities, 180 days (2026-05-07–2026-11-02), 900 training rows and 180 test rows. Training ends October 3; test October 4–November 2. No failed cities in metrics. Both experimental joblib files exist in saved output; existence is not a deserialization/integrity test.

| Target | Raw MAE °C | Full LightGBM MAE °C | Relative improvement |
|---|---:|---:|---:|
| Tmax | 0.9367 | 1.0176 | -8.64% |
| Tmin | 1.6317 | 1.0016 | 38.62% |

Labels are ERA5, not independent stations. Previous Runs represents hourly lead-aligned forecasts, not a common prior-evening issue. Later pilot policy selection inspected this test window, so it is exploratory. Pilot MAE must not be presented as a new independent result. The existing production endpoint and research model feature/lead/label definitions are not interchangeable. The prospective scorer still measures ERA5 and does not implement a fixed local 20:00 cutoff or verify artifact workflow identity; the daily forecast remains operationally separate from correction.

## New completed experiment

Executed on the assistant's free cloud CPU runtime, **not a new Kaggle run**. No paid service or GPU used. Source is research/train_station.py, with exact source SHA256 in metrics. Execution occurred on parent c20e32469763c944c974dac9127d9060ff475d86 with uncommitted research changes (honestly recorded as git_dirty=true); this commit publishes that source, tests and results.

Station: NOAA ISD 54511099999, Beijing Capital Airport, 40.080 N 116.585 E, elevation 35.4 m, Asia/Shanghai. This is an airport-coordinate experiment, not Beijing city-center accuracy.

Forecast: Open-Meteo Single Runs `ecmwf_ifs`, explicit 00 UTC run, GMT hourly temperature. Select the latest 00 UTC run at least 12 hours before the prior-local-day 20:00 cutoff. For this station that is previous-day 00 UTC. Historical provider publication timestamps are unavailable: **12-hour availability is an assumption, not verified historical delivery**. Model grid coordinates are recorded in paired_data.csv. API hourly values may involve provider interpolation; they are not asserted to be native hourly model output.

Observation: NOAA Global Hourly service JSON, FM-15 reports, TMP flags 1/5 only, tenths °C converted to °C. Reject missing and conflicting duplicate temperatures. Require observations in every UTC hour bin of the local day (including 23/25-hour DST support). Do not interpolate. Labels are daily extrema of available subhourly samples, not certified continuous daily extrema. Station observations are independent labels from reanalysis; upstream forecast assimilation is not claimed to exclude this station.

Requested May 1–July 29 2025 (90 days); 83 complete days accepted, seven excluded for missing hourly coverage (7.78%). No API failures in the completed collection. Exact exclusions are in metrics.json.

Predeclared calendar split: train May 1–June 29 (55 accepted days); validation June 30–July 14 (14); test July 15–29 (14). Missing July 14 and 29 mean last observed validation/test dates are July 13/28. All splits fixed before scoring. Train only on training data, use validation to choose correction scaling from 0/0.5/1, freeze and evaluate test once. No refit on validation, no test tuning.

| Target | Test raw MAE | Test full LightGBM MAE | Validation-selected MAE | Selected weight |
|---|---:|---:|---:|---:|
| Tmax | 1.614286 | 2.576524 | 1.614286 | 0 |
| Tmin | 1.071429 | 1.665606 | 1.071429 | 0 |

Selected policy improves neither target (0%). Full correction deteriorates both. RMSE, signed bias (prediction minus observation), fractions within 1/2°C and sample counts are in metrics.json. Both trained weights are retained for reproducibility, not deployment. New models are reloaded and predictions checked before publication.

Raw baseline bias changes sign: train Tmax +0.3782/Tmin +1.2545°C; validation -1.2500/-0.8214°C; test -1.4714/-0.7857°C. Seasonal/regime shift is a plausible failure mechanism, not a proven causal conclusion. One airport and 14 test days cannot establish global or seasonal skill. The original ERA5 result and this station result differ in year, location, lead and label, and cannot be compared as a controlled A/B test.

## Reproduction and preserved artifacts

Install requirements-train.txt. Run:

```sh
python -m research.train_station --start 2025-05-01 --days 90 --validation-days 15 --test-days 15 --output new-station-output
python -m pytest -q
```

`artifacts.tar.gz` preserves raw JSON requests/responses with retrieval timestamps and response hashes, paired daily data, test predictions, two model bundles, environment package versions, collection metadata and metrics. Extract to inspect or seed a new output/cache before rerunning; a completed output refuses overwrite. Models are Python joblib files: load only this trusted repository's artifacts. No credentials are in the archive. Source and raw cache permit reproduction even if the provider later changes historical data. 7 tests pass; one third-party pytest deprecation warning.

## Next experiment

Do not upgrade production. Pre-register rolling seasonal validation and a new untouched final interval; compare raw, training-only recent bias, and the existing LightGBM under identical station/run conditions. Add stations only after metadata and observation coverage checks. Resolve model native temporal resolution and historical availability before claiming operational fixed-cutoff validity. Independently assess actual daily extrema where available. Do not retune on this July test and advertise the result as independent.

Official data specifications:
- https://open-meteo.com/en/docs/single-runs-api
- https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf
- https://www.ncei.noaa.gov/access/services/data/v1
