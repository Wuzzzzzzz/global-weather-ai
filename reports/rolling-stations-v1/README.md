# Preregistered 20-city rolling station backtest

Status at registration: observation collection/coverage checks complete; forecast collection and model comparison pending. No new MAE or accuracy claim yet.

## Fixed protocol

- Data interval: 2024-04-01 through 2025-05-31 (426 calendar days).
- First 60 days are warmup; rolling evaluation 2024-05-31 through 2025-05-31 (366 calendar days). Refit every 30 calendar days, maximum 365-day trailing training window. Final fold may be shorter.
- Exactly 20 stations selected by the order and coverage criteria in configs/rolling-stations-v1.json, before examining forecast error. 23 candidates were screened. All first 20 qualified, so no replacements were needed.
- Minimum per-station coverage: 300 complete days, 365-day span, 220 complete days in evaluation interval. Actual selected observation coverage: 354–409 complete days, 423–426-day spans. Missing days are excluded, never imputed. Counts are in coverage.json. Some stations have fewer than 365 complete days despite the 14-month interval; do not describe this as 426 complete days per station.
- Cities: Beijing, Hong Kong, Taipei, Tokyo, Seoul, Singapore, Bangkok, Sydney, Auckland, London, Paris, Frankfurt, Madrid, Amsterdam, New York, Los Angeles, Chicago, Toronto, Dubai and Sao Paulo. Each city's representative is the identified airport station, not the city center.
- Model: ECMWF via Open-Meteo Single Runs `ecmwf_ifs`, one explicit initialization per target day/city. Cutoff fixed at local previous-day 20:00. Use the latest 00 UTC run initialized at least 12h earlier. Preserve run and cutoff UTC, local date, station/grid coordinates. This is a common local issuance rule, not a common UTC lead across all cities. No mixed-run Previous Runs or ERA5 labels.
- NOAA Global Hourly FM-15 temperature, TMP quality flags 1/5, reject conflicting duplicates and missing values. Require every hour-bin of each local day (including DST). Labels are sampled-observation extrema, not certified continuous extremes. Forecast uses hourly API temperatures, which may involve provider interpolation.
- At every refit, training labels end before the earliest cutoff across all stations, minus a seven-day observation availability assumption and two extra boundary days. No contemporaneous or future labels are features. Training and bias estimates freeze for the entire fold.
- Historical actual publication/observation availability and QC vintages are not reconstructed; the 12h/7d delays are explicit conservative assumptions. This is a fixed-run simulated issuance backtest, not a verified operational archive replay.

## Three methods on identical rows

1. Raw ECMWF daily extrema.
2. Per-station mean residual over latest 30 accepted training days within 90 calendar days, minimum 15 samples; otherwise fall back to raw and log the fallback.
3. Shared LightGBM residual models for Tmax/Tmin; 150 trees, learning rate .04, 15 leaves, minimum 40 samples/leaf, seed 42, 2 CPU threads. Existing forecast, forecast range, coordinates and seasonal sin/cos predictors. No hyperparameter search or selection against rolling test scores.

Preserve overall, per-station and per-fold MAE, RMSE, signed Bias, within-1°C/2°C proportions, sample counts and relative MAE changes. No model automatically enters production. Positive aggregate improvement alone is insufficient evidence of cross-region robustness. These rolling windows are exploratory evaluation after this protocol is frozen; later tuning requires a new holdout.

## Execution and persistence

```sh
pip install -r requirements-train.txt -r requirements.txt
python -m pytest -q
python -u -m research.rolling_station_backtest --output rolling-output
```

Public repository standard GitHub-hosted ubuntu-24.04 CPU only; no secrets, paid endpoints, GPU, billing changes or automatic purchases. Workflow runs on registration-config push or manual dispatch, not on a recurring schedule. Production daily workflow remains unchanged.

Conservative weighted quotas: 400/minute, 4500/hour, 9000/day; each coordinate and retry charged separately. Persist ledger and compressed response cache. Respect 429 by stopping and preserving outputs, not circumventing limits. Expected collection requires over an hour due to free limits. Requests cover accepted observation days only, approximately 7,600 location-calls, plus retries. NOAA collection uses monthly batches.

Upload cache, provenance, coverage, paired data, logs, every fold's model bundles, metrics, report and SHA256 manifest as one GitHub Actions artifact (90-day retention), even on failure. To resume an interrupted run, download and extract its artifact into rolling-output and rerun the same command/protocol. Completed outputs refuse overwrite. Preserve a downloaded archive for longer retention; no permanent artifact retention is claimed.

## Sources

- NOAA station catalog: https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv
- NOAA schema/quality flags: https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf
- NOAA service: https://www.ncei.noaa.gov/access/services/data/v1
- Fixed runs: https://open-meteo.com/en/docs/single-runs-api
- Quotas: https://open-meteo.com/en/pricing
- Multiple coordinates multiply call cost: https://github.com/open-meteo/open-meteo/issues/1295
