"""Score saved next-day forecasts against delayed ERA5 daily extrema."""
import csv
import io
import json
import os
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import requests

def eligible(rows):
    selected = {}
    for row in sorted(rows, key=lambda r: r["run_utc"]):
        if row["status"] != "ok":
            continue
        issued = datetime.fromisoformat(row["run_utc"])
        target = date.fromisoformat(row["local_forecast_date"])
        if issued.astimezone(ZoneInfo(row["timezone"])).date() + timedelta(days=1) != target:
            continue
        selected.setdefault((row["city_id"], str(target)), row)
    return list(selected.values())

def main():
    output = Path("output")
    rows = list(csv.DictReader((output / "latest.csv").open()))
    github = requests.Session()
    github.headers["Authorization"] = "Bearer " + os.environ["GH_TOKEN"]
    repository = os.environ["GITHUB_REPOSITORY"]
    url = f"https://api.github.com/repos/{repository}/actions/artifacts?per_page=100"
    while url:
        response = github.get(url, timeout=60)
        response.raise_for_status()
        for artifact in response.json()["artifacts"]:
            if artifact["expired"] or not artifact["name"].startswith("forecast-"):
                continue
            # Only read successful or failed runs of this same workflow.
            archive = github.get(artifact["archive_download_url"], timeout=60)
            archive.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(archive.content)) as files:
                for name in files.namelist():
                    if name == "latest.csv":
                        rows.extend(csv.DictReader(io.StringIO(files.read(name).decode("utf-8"))))
        url = response.links.get("next", {}).get("url")
    rows = eligible(rows)
    ready_before = date.today() - timedelta(days=8)
    scored, failures = [], {}
    weather = requests.Session()
    for city in sorted({r["city_id"] for r in rows}):
        pending = [r for r in rows if r["city_id"] == city
                   and date.fromisoformat(r["local_forecast_date"]) <= ready_before]
        if not pending:
            continue
        sample = pending[0]
        try:
            response = weather.get("https://archive-api.open-meteo.com/v1/archive",
                params={"latitude": sample["latitude"], "longitude": sample["longitude"],
                        "timezone": sample["timezone"], "models": "era5",
                        "daily": "temperature_2m_max,temperature_2m_min",
                        "start_date": min(r["local_forecast_date"] for r in pending),
                        "end_date": max(r["local_forecast_date"] for r in pending)}, timeout=80)
            response.raise_for_status()
            daily = response.json()["daily"]
            truth = dict(zip(daily["time"], zip(daily["temperature_2m_max"], daily["temperature_2m_min"])))
            for row in pending:
                hi, lo = truth.get(row["local_forecast_date"], (None, None))
                if hi is not None and lo is not None:
                    scored.append({**row, "era5_tmax": hi, "era5_tmin": lo,
                                   "tmax_absolute_error": abs(float(row["tmax_c"]) - hi),
                                   "tmin_absolute_error": abs(float(row["tmin_c"]) - lo)})
        except (requests.RequestException, KeyError, ValueError) as exc:
            failures[city] = str(exc)
    def metrics(items):
        return {"n": len(items),
                "tmax_mae_c": sum(r["tmax_absolute_error"] for r in items) / len(items) if items else None,
                "tmin_mae_c": sum(r["tmin_absolute_error"] for r in items) / len(items) if items else None}
    report = {"reference": "ERA5 daily extrema", "selection": "earliest saved next-local-day forecast per city/date",
              "correction_status": "pending operationally aligned training",
              "available_forecasts": len(rows), "overall": metrics(scored),
              "cities": {c: metrics([r for r in scored if r["city_id"] == c])
                         for c in sorted({r["city_id"] for r in rows})}, "failures": failures}
    (output / "prospective_metrics.json").write_text(json.dumps(report, indent=2))
    if scored:
        with (output / "scored_forecasts.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scored[0]))
            writer.writeheader()
            writer.writerows(scored)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("Some ERA5 requests failed; see prospective_metrics.json")

if __name__ == "__main__":
    main()
