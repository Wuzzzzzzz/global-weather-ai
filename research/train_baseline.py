"""Train an experimental 24-hour-ahead temperature bias corrector.

Forecast: Open-Meteo Previous Runs hourly forecast at fixed 24-hour lead.
Label: ERA5 reanalysis (NOT independent thermometer/station observations).
The hourly lead-aligned samples are reduced to daily extrema in LOCAL time.
This is NOT a fixed evening-issued next-day forecast experiment.
"""
import argparse
import csv
import json
import math
import time
from datetime import date, timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error

FORECAST_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
FEATURES = ["forecast", "latitude", "longitude", "day_sin", "day_cos",
            "forecast_range", "is_max"]
FORECAST_VAR = "temperature_2m_previous_day1"


def get_json(session, url, params, retries=3):
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=80)
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(payload.get("reason", "Weather API error"))
            return payload
        except (requests.RequestException, ValueError) as exc:
            if attempt + 1 == retries:
                raise RuntimeError(f"{url}: {exc}") from exc
            time.sleep(2 ** attempt)


def daily_extrema(payload, variable):
    """Group local ISO hourly labels (timezone=auto) to daily maxima/minima."""
    hourly = payload["hourly"]
    times, values = hourly["time"], hourly[variable]
    if len(times) != len(values):
        raise ValueError("time/value lengths disagree")
    grouped = {}
    for stamp, value in zip(times, values):
        if value is None:
            continue
        key = stamp[:10]
        grouped.setdefault(key, []).append(float(value))
    # Prefer complete days; require >= 20 valid hourly samples.
    return {d: (max(v), min(v)) for d, v in grouped.items() if len(v) >= 20}


def fetch_city(session, city, start, end):
    params = {
        "latitude": float(city["latitude"]),
        "longitude": float(city["longitude"]),
        "start_date": start, "end_date": end,
        "timezone": "auto",
        "hourly": FORECAST_VAR,
        "models": "ecmwf_ifs",
    }
    forecast = get_json(session, FORECAST_API, params)
    actual = get_json(session, ARCHIVE_API, {
        "latitude": params["latitude"], "longitude": params["longitude"],
        "start_date": start, "end_date": end, "timezone": "auto",
        "hourly": "temperature_2m", "models": "era5"
    })
    preds = daily_extrema(forecast, FORECAST_VAR)
    truth = daily_extrema(actual, "temperature_2m")
    rows = []
    for day in sorted(set(preds) & set(truth)):
        pmax, pmin = preds[day]
        amax, amin = truth[day]
        dt = date.fromisoformat(day)
        rows.append({
            "city_id": city["city_id"], "date": day,
            "latitude": params["latitude"], "longitude": params["longitude"],
            "forecast_max": pmax, "forecast_min": pmin,
            "actual_max": amax, "actual_min": amin,
            "day_sin": math.sin(2 * math.pi * dt.timetuple().tm_yday / 365.25),
            "day_cos": math.cos(2 * math.pi * dt.timetuple().tm_yday / 365.25),
        })
    return rows


def transform(frame, kind):
    result = pd.DataFrame({
        "forecast": frame[f"forecast_{kind}"].astype(float),
        "latitude": frame["latitude"].astype(float),
        "longitude": frame["longitude"].astype(float),
        "day_sin": frame["day_sin"].astype(float),
        "day_cos": frame["day_cos"].astype(float),
        "forecast_range": (frame["forecast_max"] - frame["forecast_min"]).astype(float),
        "is_max": 1 if kind == "max" else 0,
    })
    return result[FEATURES]


def evaluate(frame, model, kind):
    base = frame[f"forecast_{kind}"].to_numpy(dtype=float)
    actual = frame[f"actual_{kind}"].to_numpy(dtype=float)
    corrected = base + model.predict(transform(frame, kind))
    # The pilot keeps the raw daily high and corrects the daily low.
    use_correction = kind == "min"
    pilot = corrected if use_correction else base
    return {
        "baseline_mae_c": round(float(mean_absolute_error(actual, base)), 4),
        "corrected_mae_c": round(float(mean_absolute_error(actual, corrected)), 4),
        "pilot_method": "lightgbm" if use_correction else "raw",
        "pilot_mae_c": round(float(mean_absolute_error(actual, pilot)), 4),
        "n": len(frame),
    }


def fit_and_score(frame, output_dir, min_train_days=45, test_days=30):
    """Chronological, shared cutoff by date: never random split across time."""
    dates = sorted(frame["date"].unique())
    if len(dates) < min_train_days + test_days:
        raise ValueError(f"Need >= {min_train_days + test_days} distinct dates, got {len(dates)}")
    cutoff = dates[-test_days]
    train = frame[frame.date < cutoff].copy()
    test = frame[frame.date >= cutoff].copy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = {"cutoff": cutoff, "train_rows": len(train), "test_rows": len(test),
              "label": "ERA5 reanalysis, not independent weather station measurements",
              "lead": "fixed 24-hour lead for each valid hour, not a fixed daily issue time"}
    for kind in ("max", "min"):
        target = train[f"actual_{kind}"] - train[f"forecast_{kind}"]
        model = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.04,
                                   num_leaves=15, min_child_samples=40,
                                   random_state=42, verbosity=-1, n_jobs=2)
        model.fit(transform(train, kind), target)
        joblib.dump(model, output_dir / f"temp_{kind}_experimental.joblib")
        scores[kind] = evaluate(test, model, kind)
    (output_dir / "metrics.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities-file", default="data/cities.csv")
    ap.add_argument("--cities", type=int, default=6, help="Limit initial requests")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument("--output", default="research-output")
    args = ap.parse_args()
    end = date.today() - timedelta(days=8)  # ERA5 latency buffer
    start = end - timedelta(days=args.days - 1)
    with open(args.cities_file, newline="", encoding="utf-8") as f:
        cities = list(csv.DictReader(f))[:args.cities]
    if not cities:
        raise SystemExit("No cities")
    session = requests.Session()
    rows, failures = [], {}
    for city in cities:
        try:
            part = fetch_city(session, city, start.isoformat(), end.isoformat())
            if not part:
                raise ValueError("No overlapping hourly forecast and reanalysis dates")
            rows.extend(part)
            print(f"{city['city_id']}: {len(part)} days", flush=True)
        except (RuntimeError, ValueError, KeyError) as exc:
            failures[city["city_id"]] = str(exc)
            print(f"{city['city_id']}: ERROR {exc}", flush=True)
        time.sleep(0.5)
    if not rows:
        raise SystemExit(f"No training data collected. Errors: {failures}")
    frame = pd.DataFrame(rows).sort_values(["date", "city_id"]).reset_index(drop=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "training_data.csv", index=False)
    scores = fit_and_score(frame, out, test_days=args.test_days)
    scores["cities_succeeded"] = len(set(frame.city_id))
    scores["cities_failed"] = failures
    (out / "metrics.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
