"""Next-local-day city Tmax/Tmin baseline from Open-Meteo ECMWF data."""
import argparse
import csv
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import requests

URL = "https://api.open-meteo.com/v1/ecmwf"
FIELDS = ["run_utc", "city_id", "name", "country", "latitude", "longitude",
          "timezone", "local_forecast_date", "tmax_c", "tmin_c", "source", "status", "error"]


def target_for_timezone(data, now_utc):
    """Select tomorrow according to the returned city's timezone."""
    local_today = now_utc.astimezone(ZoneInfo(data["timezone"])).date()
    tomorrow = str(local_today + timedelta(days=1))
    dates = data["daily"]["time"]
    if tomorrow not in dates:
        raise ValueError(f"Missing target {tomorrow} in daily forecast")
    idx = dates.index(tomorrow)
    return (tomorrow, data["daily"]["temperature_2m_max"][idx],
            data["daily"]["temperature_2m_min"][idx])


def fetch_city(session, city, now_utc, timeout=30):
    params = {"latitude": float(city["latitude"]), "longitude": float(city["longitude"]),
              "daily": "temperature_2m_max,temperature_2m_min",
              "timezone": "auto", "forecast_days": 3}
    response = session.get(URL, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise ValueError(data.get("reason", "Provider error"))
    date, hi, lo = target_for_timezone(data, now_utc)
    if hi is None or lo is None:
        raise ValueError("Null temperature data")
    hi, lo = float(hi), float(lo)
    if hi < lo:
        raise ValueError("Tmax lower than Tmin")
    return {"timezone": data["timezone"], "local_forecast_date": date,
            "tmax_c": hi, "tmin_c": lo, "status": "ok", "error": ""}


def collect(csv_path, now_utc, session=None, pause_seconds=0.3):
    session = session or requests.Session()
    with open(csv_path, encoding="utf-8", newline="") as f:
        cities = list(csv.DictReader(f))
    if not cities:
        raise ValueError("No cities provided")
    rows = []
    for city in cities:
        row = {key: city.get(key, "") for key in
               ("city_id", "name", "country", "latitude", "longitude")}
        row.update(run_utc=now_utc.isoformat(), timezone="", local_forecast_date="",
                   tmax_c="", tmin_c="", source="Open-Meteo ECMWF API",
                   status="error", error="")
        try:
            row.update(fetch_city(session, city, now_utc))
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            row["error"] = str(exc)[:300]
            logging.warning("%s: %s", city["city_id"], exc)
        rows.append(row)
        if pause_seconds:
            time.sleep(pause_seconds)
    return rows


def save(rows, directory, now_utc):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"forecast_{now_utc.strftime('%Y-%m-%dT%H-%MZ')}.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (directory / "latest.csv").write_bytes(path.read_bytes())
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", default="data/cities.csv")
    parser.add_argument("--output", default="output")
    parser.add_argument("--pause", type=float, default=0.3)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    rows = collect(args.cities, now, pause_seconds=args.pause)
    path = save(rows, args.output, now)
    successes = sum(r["status"] == "ok" for r in rows)
    print(json.dumps({"file": str(path), "success": successes, "total": len(rows)}))
    if successes == 0:
        raise SystemExit("All forecasts failed; inspect CSV errors")


if __name__ == "__main__":
    main()
