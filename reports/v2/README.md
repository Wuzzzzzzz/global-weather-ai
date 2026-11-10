# Global Weather AI V2 — protocol and V1 audit

V1 source run: https://github.com/Wuzzzzzzz/global-weather-ai/actions/runs/42784203935
Commit c0b170175b89ffc01473c154cc7c9908808ce680; artifact 13753540718.
Downloaded archive SHA256 matches GitHub: 57b3b464135b17b19a9a9e287402f88274b9a76bd22dcd2443157e5026c0f3eb. All manifest entries match; all six MAEs independently recomputed.

## V1 findings

See v1-audit/overall.csv, cities.csv, folds.csv and audit.json for every method, city, window, MAE/RMSE/bias and within-1/2°C fractions.

| Method | Tmax MAE | Tmin MAE |
|---|---:|---:|
| ECMWF | 1.488374 | 1.277239 |
| 30-sample bias | 1.210322 | 1.065866 |
| LightGBM | 1.150842 | 1.037191 |

LightGBM improves Tmax in 19/20 cities, Tmin in 17/20. London Tmax deteriorates 9.21%; Tmin deteriorates Seoul 4.05%, Auckland 4.50%, London 4.73%. All 13 windows improve overall, but the last window is only six calendar days, so these are not 13 independent seasons.

Seven-day moving date-block bootstrap (2,000 replicates, all cities resampled together): LightGBM MAE reduction versus ECMWF Tmax 0.338°C [0.300,0.378], Tmin 0.240°C [0.209,0.274]. Versus bias: Tmax 0.059°C [0.041,0.079], Tmin 0.029°C [0.011,0.047]. Conditional evidence only; not a guarantee of regional generalization.

Test missingness is 10.93% relative to 20×366 calendar city-days. Labels are quality-controlled NOAA FM-15 sampled airport extrema, not continuous daily thermometer extrema. Complete-hour screening may introduce selection bias. Same stations appear in training and test: unseen-city/region generalization NOT verified. No duplicate city/day and all recorded initializations plus 12h precede cutoff. Actual historical provider publication and observation QC revision availability remain unverified. Protocol has a conservative label embargo; archived revised labels are not vintage-reconstructed.

## AIFS feasibility: actual probes

Open-Meteo 2026-06-01T00 IFS/AIFS calls succeed for temperature, cloud, humidity, wind and radiation (some initial radiation hours null). Same-date NOAA queries for all 20 stations return an empty list. Open-Meteo 2025-06-01T00 AIFS returns HTTP 400, run not available. Raw probe evidence is preserved here. Do not infer that NOAA lacks all 2026 data from this one sampled interval.

Alternative **dynamical.org** archive is accessible anonymously without paid cloud provisioning. An exact 2025-06-01T00 AIFS run was successfully read at 40.0N,116.5E: first nine six-hourly temperatures in °C are 21.125,25.125,23.25,19.125,21.5,30.25,25.625,18.5,23.5. Adapter uses exact initialization, nearest spatial grid, explicit unit and coverage checks, bounded point/chunk reads and per-run local caches. No global 966GB array is loaded.

AIFS native output is 6-hourly. We do **not** call its four samples native daily Tmax/Tmin. Construct an estimated hourly candidate by adding linearly interpolated six-hour AIFS-minus-IFS anomalies to the IFS hourly curve, then take target-local-day extrema. This retains IFS diurnal shape and is an IFS+AIFS candidate, NOT a standalone AIFS daily extreme forecast. Compare 50/50 daily-extreme blending and validation-selected weights 0,.25,.5,.75,1 against raw IFS on identical common rows. This evaluates the reconstruction method together with AIFS information, not isolated model skill.

## Frozen V2 experiment (before new holdout scores)

20 existing airport coordinates, 2025-06-01–2025-08-20. Initial train through June 30; validation July 10–20; final refit through July 20; untouched final test August 1–20. No prior displayed experiment has evaluated this August window. July Beijing observations were previously examined, so validation is development data, not an independent score. Final test is short and summer-only; no production promotion.

Candidates: raw IFS; causal recent bias; basic LightGBM; equal IFS/AIFS blend; validation-selected blend; enriched LightGBM (cloud cover, relative humidity, wind, shortwave radiation, recent residuals/counts, AIFS reconstructed daily estimates). Train basic/enriched on identical rows; only 7 vs 15 leaves selected by validation. Refit through July 20, then evaluate fixed models on August once. No test-driven parameter or station selection. Initial expected output is pending until real run completes.

All meteorological predictors come from the same pre-cutoff IFS run. Recent bias uses at most 30 accepted days in a 90-day window, strictly before target minus 10 days; fewer than 15 defaults to zero with count feature. Seed history uses V1 rows ending May 31. No future observations are prediction features. Weather missingness excludes the whole row for all methods. At least 80% AIFS common support is required; missing runs or wrong units fail closed. All 20 stations must appear in the final test. No aircraft data.

## Reproduce

```sh
pip install -r requirements-v2.txt -r requirements.txt
python -m pytest -q
python -m research.audit_v1 --input v1-results --output audit-output
python -m research.v2_features --history v1-results/paired_data.csv --output v2-output
```

New independent free public GitHub Actions standard CPU job; no paid services, GPU, credentials for weather data or production changes. Original workflows and weights preserved. Outputs include raw responses, native AIFS point series, fixed protocol, daily predictions, model bundles, package versions and SHA256 manifest. Workflow artifact retention is 90 days.

Sources/attribution: NOAA ISD; ECMWF IFS via Open-Meteo; ECMWF AIFS Single forecasts processed by dynamical.org (CC BY 4.0 and ECMWF terms).
- https://open-meteo.com/en/docs/single-runs-api
- https://www.ecmwf.int/en/forecasts/datasets/open-data
- https://dynamical.org/catalog/ecmwf-aifs-single-forecast/
