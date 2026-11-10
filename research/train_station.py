"""Bounded station backtest. Research only: sampled extrema, assumed latency.

python -m research.train_station --output station-output-2025
No production model is replaced. NOAA TMP QC 1/5 only; no interpolation.
"""
import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from research.train_baseline import transform, FEATURES

UTC = timezone.utc
STATION = {"station_id": "54511099999", "city_id": "beijing_capital_airport",
           "latitude": 40.080, "longitude": 116.585, "elevation_m": 35.4,
           "timezone": "Asia/Shanghai"}
NOAA = "https://www.ncei.noaa.gov/access/services/data/v1"
SINGLE = "https://single-runs-api.open-meteo.com/v1/forecast"


def window(day, tz):
    zone = ZoneInfo(tz)
    start = datetime.combine(day, dtime(), zone).astimezone(UTC)
    end = datetime.combine(day + timedelta(days=1), dtime(), zone).astimezone(UTC)
    cutoff = datetime.combine(day - timedelta(days=1), dtime(20), zone).astimezone(UTC)
    return start, end, cutoff


def run_for(cutoff):
    # Latest 00 UTC run at least 12h before cutoff. Historical publication
    # timestamps are unavailable: this is an assumption, NOT verified availability.
    latest = cutoff - timedelta(hours=12)
    return latest.replace(hour=0, minute=0, second=0, microsecond=0)


def cached_json(session, url, params, cache):
    key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
    path = cache / (key + '.json')
    if path.exists():
        return json.loads(path.read_text())["payload"]
    for attempt in range(3):
        try:
            r = session.get(url, params=params, timeout=60)
            r.raise_for_status()
            if 'json' not in r.headers.get('content-type', ''):
                raise ValueError('Expected JSON; rejected non-JSON response')
            payload = r.json()
            if isinstance(payload, dict) and payload.get('error'):
                raise ValueError(str(payload))
            path.write_text(json.dumps({"url": r.url, "fetched_at": datetime.now(UTC).isoformat(),
                "sha256": hashlib.sha256(r.content).hexdigest(), "payload": payload}))
            time.sleep(.25)
            return payload
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def observations(records, station):
    if not isinstance(records, list):
        raise ValueError('NOAA response must be a list')
    grouped = {}
    for row in records:
        if not {'DATE', 'TMP', 'STATION', 'REPORT_TYPE'} <= row.keys():
            raise ValueError('NOAA observation schema missing fields')
        if row['STATION'] != station or row['REPORT_TYPE'] != 'FM-15':
            continue
        raw, flag = row['TMP'].split(',')
        if flag not in {'1', '5'} or raw == '+9999':
            continue
        value = int(raw) / 10
        if not -90 <= value <= 60:
            continue
        stamp = datetime.fromisoformat(row['DATE']).replace(tzinfo=UTC)
        grouped.setdefault(stamp, set()).add(value)
    # Conflicting reports at the exact same time are discarded, not averaged.
    return {t: next(iter(v)) for t, v in grouped.items() if len(v) == 1}


def observed_day(obs, start, end):
    selected = {t: v for t, v in obs.items() if start <= t < end}
    hours = int((end - start).total_seconds() / 3600)
    bins = {int((t-start).total_seconds()//3600) for t in selected}
    if bins != set(range(hours)):
        raise ValueError(f'Incomplete observed day: {len(bins)}/{hours} hourly bins')
    return max(selected.values()), min(selected.values()), len(selected)


def forecast_day(payload, start, end):
    if payload.get('utc_offset_seconds') != 0 or payload.get('hourly_units', {}).get('temperature_2m') != '°C':
        raise ValueError('Unexpected timezone or temperature units')
    h = payload['hourly']
    if len(h['time']) != len(h['temperature_2m']):
        raise ValueError('Forecast length mismatch')
    values = {}
    for stamp, value in zip(h['time'], h['temperature_2m']):
        t = datetime.fromisoformat(stamp).replace(tzinfo=UTC)
        if start <= t < end and value is not None and math.isfinite(value):
            if t in values:
                raise ValueError('Duplicate forecast timestamp')
            values[t] = float(value)
    expected = {start + timedelta(hours=i) for i in range(int((end-start).total_seconds()/3600))}
    if set(values) != expected:
        raise ValueError('Incomplete forecast day')
    return max(values.values()), min(values.values())


def metrics(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    return {'mae_c': float(np.mean(abs(error))), 'rmse_c': float(np.sqrt(np.mean(error**2))),
            'bias_c': float(np.mean(error)), 'within_1c': float(np.mean(abs(error)<=1)),
            'within_2c': float(np.mean(abs(error)<=2)), 'n': len(error)}


def fit(frame, start, days, val_days, test_days, out):
    val_start = (start + timedelta(days=days-val_days-test_days)).isoformat()
    test_start = (start + timedelta(days=days-test_days)).isoformat()
    train = frame[frame.date < val_start]
    val = frame[(frame.date >= val_start) & (frame.date < test_start)]
    test = frame[frame.date >= test_start]
    if len(train)<30 or len(val)<10 or len(test)<10:
        raise ValueError('Insufficient complete days in predeclared calendar split')
    report = {'split': {k: {'start': str(v.date.min()), 'end': str(v.date.max()), 'n': len(v)}
              for k,v in [('train',train),('validation',val),('test',test)]},
              'validation_boundary': val_start, 'test_boundary': test_start, 'targets': {}}
    predictions = test[['date','city_id','actual_max','actual_min','forecast_max','forecast_min']].copy()
    for kind in ['max','min']:
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=.03, num_leaves=7,
                                 min_child_samples=15, random_state=42, n_jobs=2, verbosity=-1)
        model.fit(transform(train,kind),train[f'actual_{kind}']-train[f'forecast_{kind}'])
        base_v = val[f'forecast_{kind}'].to_numpy()
        delta_v = model.predict(transform(val,kind))
        scores = {str(w): metrics(val[f'actual_{kind}'],base_v+w*delta_v)['mae_c'] for w in (0,.5,1)}
        weight = float(min(scores,key=scores.get))  # validation only; test never consulted
        base = test[f'forecast_{kind}'].to_numpy()
        full = base + model.predict(transform(test,kind))
        chosen = base + weight*(full-base)
        baseline = metrics(test[f'actual_{kind}'],base)
        corrected = metrics(test[f'actual_{kind}'],chosen)
        report['targets'][kind] = {'validation_mae_by_weight':scores, 'selected_weight':weight,
            'baseline':baseline, 'full_lightgbm':metrics(test[f'actual_{kind}'],full),
            'selected':corrected, 'improvement_pct':100*(baseline['mae_c']-corrected['mae_c'])/baseline['mae_c'] if baseline['mae_c'] else None}
        predictions[f'corrected_{kind}'] = chosen
        joblib.dump({'model':model,'weight':weight,'features':FEATURES,'station':STATION,
                     'status':'research_only'},out/f'T{kind}_corrector.joblib')
    predictions.to_csv(out/'test_predictions.csv',index=False)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start',default='2025-05-01')
    ap.add_argument('--days',type=int,default=90)
    ap.add_argument('--validation-days',type=int,default=15)
    ap.add_argument('--test-days',type=int,default=15)
    ap.add_argument('--output',required=True)
    args = ap.parse_args()
    if not 60 <= args.days <= 180 or min(args.validation_days,args.test_days)<10 or args.days-args.validation_days-args.test_days<30:
        raise SystemExit('Require 60–180 days, >=30 train and >=10 validation/test days')
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():
        raise SystemExit('Refusing to overwrite completed experiment; choose new output')
    cache = out/'cache'; cache.mkdir(exist_ok=True)
    start = date.fromisoformat(args.start)
    session = requests.Session()
    obs = {}
    for offset in range(0,args.days,30):
        a,_,_ = window(start+timedelta(days=offset),STATION['timezone'])
        _,b,_ = window(start+timedelta(days=min(offset+30,args.days)-1),STATION['timezone'])
        records = cached_json(session,NOAA,{'dataset':'global-hourly','stations':STATION['station_id'],
            'startDate':a.date().isoformat(),'endDate':b.date().isoformat(),'dataTypes':'TMP','format':'json'},cache)
        obs.update(observations(records,STATION['station_id']))
    rows, failures = [], {}
    for offset in range(args.days):
        day = start+timedelta(days=offset)
        a,b,cutoff = window(day,STATION['timezone']); run = run_for(cutoff)
        try:
            actual_max,actual_min,n = observed_day(obs,a,b)
            payload = cached_json(session,SINGLE,{'latitude':STATION['latitude'],'longitude':STATION['longitude'],
                'run':run.strftime('%Y-%m-%dT%H:%M'),'hourly':'temperature_2m','models':'ecmwf_ifs',
                'timezone':'GMT','forecast_days':4},cache)
            pmax,pmin = forecast_day(payload,a,b)
            rows.append({'city_id':STATION['city_id'],'date':day.isoformat(),
                'latitude':STATION['latitude'],'longitude':STATION['longitude'],
                'forecast_max':pmax,'forecast_min':pmin,'actual_max':actual_max,'actual_min':actual_min,
                'day_sin':math.sin(2*math.pi*day.timetuple().tm_yday/365.25),
                'day_cos':math.cos(2*math.pi*day.timetuple().tm_yday/365.25),
                'run_utc':run.isoformat(),'cutoff_utc':cutoff.isoformat(),'observations':n,
                'grid_latitude':payload['latitude'],'grid_longitude':payload['longitude']})
        except (ValueError,KeyError,requests.RequestException) as exc:
            failures[day.isoformat()] = str(exc)
        print(day,'OK' if day.isoformat() not in failures else failures[day.isoformat()],flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'paired_data.csv',index=False)
    metadata = {'parameters':vars(args),'station':STATION,'failures':failures,'requested_days':args.days,
                'accepted_days':len(rows),'missing_fraction':1-len(rows)/args.days,
                'label':'NOAA ISD FM-15 TMP QC 1/5 sampled extrema, not continuous daily extrema',
                'forecast':'Open-Meteo Single Runs ecmwf_ifs hourly at station coordinates',
                'availability':'12h latency assumed; historical publication timestamps NOT verified',
                'deployment':'research only; production unchanged',
                'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'collection.json').write_text(json.dumps(metadata,indent=2))
    report = fit(frame,start,args.days,args.validation_days,args.test_days,out)
    report.update(metadata)
    report['artifacts_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    subprocess.run(['python','-m','pip','freeze'],stdout=(out/'environment.txt').open('w'),check=True)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
