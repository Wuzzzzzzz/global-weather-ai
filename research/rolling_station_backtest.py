"""Free-CPU, resumable 20-station fixed-run rolling comparison. No deployment."""
import argparse
import gzip
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from research.train_baseline import transform, FEATURES
from research.train_station import window, run_for, observations, observed_day, metrics, NOAA, SINGLE

UTC = timezone.utc


def write_json(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False))
    temp.replace(path)


class Cache:
    def __init__(self, out):
        self.root = out/'cache'; self.root.mkdir(parents=True,exist_ok=True)
        self.ledger = out/'request_ledger.json'
        self.events = json.loads(self.ledger.read_text()) if self.ledger.exists() else []
        self.session = requests.Session()

    def reserve(self, cost):
        # Count each location separately, including unsuccessful attempts.
        for _ in range(3000):
            now = time.time()
            self.events = [e for e in self.events if e[0]>now-86400]
            waits=[]
            for seconds,limit in [(60,400),(3600,4500),(86400,9000)]:
                recent=[e for e in self.events if e[0]>now-seconds]
                if sum(e[1] for e in recent)+cost>limit:
                    waits.append(min(e[0] for e in recent)+seconds-now+1)
            if not waits:
                self.events.append([now,cost]); write_json(self.ledger,self.events); return
            delay=min(30,max(waits))
            print(f'Free API budget pause: {max(waits):.0f}s remaining',flush=True)
            time.sleep(delay)
        raise RuntimeError('Budget wait exhausted; resume from cache later')

    def get(self,url,params,cost=0):
        key=hashlib.sha256(json.dumps([url,params],sort_keys=True).encode()).hexdigest()
        path=self.root/(key+'.json.gz')
        if path.exists():
            with gzip.open(path,'rt') as f: return json.load(f)['payload']
        for attempt in range(3):
            if cost: self.reserve(cost)
            try:
                r=self.session.get(url,params=params,timeout=120)
                if r.status_code==429:
                    raise RuntimeError('Provider rate limit: stop, retain cache, resume later')
                r.raise_for_status()
                if 'json' not in r.headers.get('content-type',''):
                    raise ValueError('Non-JSON payload rejected')
                payload=r.json()
                if isinstance(payload,dict) and payload.get('error'): raise ValueError(str(payload))
                record={'url':r.url,'fetched_at':datetime.now(UTC).isoformat(),
                        'response_sha256':hashlib.sha256(r.content).hexdigest(),'payload':payload}
                tmp=path.with_suffix('.tmp')
                with gzip.open(tmp,'wt') as f: json.dump(record,f)
                tmp.replace(path)
                time.sleep(.5)
                return payload
            except (requests.RequestException,ValueError):
                if attempt==2: raise
                time.sleep(2**attempt)


def collect_observations(config,cache,out):
    start=date.fromisoformat(config['start']); end=date.fromisoformat(config['end'])
    by_station=defaultdict(list)
    # Each request covers 31 days and all candidates. UTC padding for local days.
    cursor=start-timedelta(days=1)
    while cursor<=end+timedelta(days=1):
        stop=min(cursor+timedelta(days=30),end+timedelta(days=1))
        records=cache.get(NOAA,{'dataset':'global-hourly','stations':','.join(s['station_id'] for s in config['stations']),
            'startDate':cursor.isoformat(),'endDate':stop.isoformat(),'dataTypes':'TMP','format':'json'})
        if not isinstance(records,list): raise ValueError('NOAA must return list')
        for r in records:
            if not {'STATION','DATE','TMP','REPORT_TYPE'}<=r.keys(): raise ValueError('NOAA schema mismatch')
            by_station[r['STATION']].append(r)
        print('NOAA',cursor,stop,len(records),flush=True)
        cursor=stop+timedelta(days=1)
    rows=[]; quality={}; eligible=[]
    eval_start=start+timedelta(days=config['warmup_days'])
    for station in config['stations']:
        sid=station['station_id']; obs=observations(by_station[sid],sid)
        accepted=[]; rejected={}
        for offset in range((end-start).days+1):
            day=start+timedelta(days=offset); a,b,_=window(day,station['timezone'])
            try:
                hi,lo,n=observed_day(obs,a,b)
                row={'station_id':sid,'city_id':station['city_id'],'date':day.isoformat(),
                     'actual_max':hi,'actual_min':lo,'observations':n}
                rows.append(row); accepted.append(day)
            except ValueError as e: rejected[day.isoformat()]=str(e)
        span=(max(accepted)-min(accepted)).days+1 if accepted else 0
        ntest=sum(d>=eval_start for d in accepted)
        quality[sid]={'city':station['city'],'complete_days':len(accepted),'span_days':span,
                      'test_days':ntest,'rejected_days':rejected}
        if len(accepted)>=config['min_complete_days'] and span>=config['min_span_days'] and ntest>=config['min_test_days']:
            eligible.append(station)
        print('Station coverage',station['city'],len(accepted),span,ntest,flush=True)
    write_json(out/'observation_quality.json',quality)
    pd.DataFrame(rows).to_csv(out/'observations.csv',index=False)
    selected=eligible[:config['station_count']]
    write_json(out/'selected_stations.json',selected)
    if len(selected)<config['station_count']:
        raise RuntimeError(f'Only {len(selected)} stations meet preregistered coverage; do not claim 20-city completion')
    return pd.DataFrame(rows),selected


def forecast_extrema(payload,start,end):
    if payload.get('utc_offset_seconds')!=0 or payload.get('hourly_units',{}).get('temperature_2m')!='°C':
        raise ValueError('Forecast units/timezone mismatch')
    h=payload['hourly']; values={}
    if len(h['time'])!=len(h['temperature_2m']): raise ValueError('Time/value length mismatch')
    for raw,v in zip(h['time'],h['temperature_2m']):
        t=datetime.fromisoformat(raw).replace(tzinfo=UTC)
        if start<=t<end and v is not None and math.isfinite(v):
            if t in values: raise ValueError('Duplicate forecast timestamp')
            values[t]=float(v)
    first=start.replace(minute=0,second=0,microsecond=0)
    if first<start: first+=timedelta(hours=1)
    expected=set(); t=first
    while t<end: expected.add(t); t+=timedelta(hours=1)
    if set(values)!=expected: raise ValueError('Incomplete hourly forecast')
    return max(values.values()),min(values.values())


def collect_pairs(config,cache,out,obs,stations):
    start=date.fromisoformat(config['start']); end=date.fromisoformat(config['end'])
    observed={(r.station_id,r.date):r for r in obs.itertuples(index=False)}
    rows=[]; failures={}
    for offset in range((end-start).days+1):
        day=start+timedelta(days=offset); groups=defaultdict(list)
        for s in stations:
            if (s['station_id'],day.isoformat()) not in observed: continue
            a,b,cutoff=window(day,s['timezone']); run=run_for(cutoff)
            groups[run].append((s,a,b,cutoff))
        for run,items in groups.items():
            params={'latitude':','.join(str(v[0]['latitude']) for v in items),
                'longitude':','.join(str(v[0]['longitude']) for v in items),'run':run.strftime('%Y-%m-%dT%H:%M'),
                'hourly':'temperature_2m','models':config['forecast_model'],'timezone':'GMT','forecast_days':4}
            payload=cache.get(SINGLE,params,cost=len(items))
            payloads=payload if isinstance(payload,list) else [payload]
            if len(payloads)!=len(items): raise ValueError('Location count mismatch')
            for i,(p,(s,a,b,cutoff)) in enumerate(zip(payloads,items)):
                # Open-Meteo location_id indexes the coordinate list; check when present.
                if 'location_id' in p and p['location_id']!=i: raise ValueError('Location order mismatch')
                if abs(p['latitude']-s['latitude'])>.5 or abs(p['longitude']-s['longitude'])>.5:
                    raise ValueError('Unexpected model grid location')
                key=s['station_id']+':'+day.isoformat()
                try:
                    hi,lo=forecast_extrema(p,a,b)
                    r=observed[(s['station_id'],day.isoformat())]._asdict()
                    r.update(latitude=s['latitude'],longitude=s['longitude'],elevation_m=s['elevation_m'],
                        forecast_max=hi,forecast_min=lo,run_utc=run.isoformat(),cutoff_utc=cutoff.isoformat(),
                        grid_latitude=p['latitude'],grid_longitude=p['longitude'],
                        day_sin=math.sin(2*math.pi*day.timetuple().tm_yday/365.25),
                        day_cos=math.cos(2*math.pi*day.timetuple().tm_yday/365.25))
                    rows.append(r)
                except (KeyError,ValueError) as e: failures[key]=str(e)
        if offset%10==0 or offset==(end-start).days:
            pd.DataFrame(rows).to_csv(out/'paired_data.csv',index=False)
            write_json(out/'forecast_failures.json',failures)
            print('Forecast progress',offset+1,(end-start).days+1,'paired',len(rows),flush=True)
    frame=pd.DataFrame(rows).sort_values(['date','city_id']).reset_index(drop=True)
    for s in stations:
        part=frame[frame.station_id==s['station_id']]
        span=(date.fromisoformat(part.date.max())-date.fromisoformat(part.date.min())).days+1 if len(part) else 0
        test_n=sum(part.date>=(start+timedelta(days=config['warmup_days'])).isoformat())
        if len(part)<config['min_complete_days'] or span<config['min_span_days'] or test_n<config['min_test_days']:
            raise RuntimeError('Paired station coverage below threshold: '+s['city'])
    return frame


def training_cutoff(fold_start,stations,embargo):
    # Use earliest global issuance cutoff across cities. Require label day to
    # have ended everywhere, plus seven whole days. Delayed archive revisions
    # are not represented by this assumed operational latency.
    earliest=min(window(fold_start,s['timezone'])[2] for s in stations)
    return (earliest-timedelta(days=embargo+2)).date()


def score_group(frame):
    result={}
    for kind in ['max','min']:
        result[kind]={method:metrics(frame['actual_'+kind],frame[method+'_'+kind])
                      for method in ['ecmwf','bias','lightgbm']}
        base=result[kind]['ecmwf']['mae_c']
        for method in ['bias','lightgbm']:
            result[kind][method]['improvement_pct']=100*(base-result[kind][method]['mae_c'])/base if base else None
    return result


def rolling_evaluate(frame,config,stations,out):
    start=date.fromisoformat(config['start']); end=date.fromisoformat(config['end'])
    cursor=start+timedelta(days=config['warmup_days']); folds=[]; predictions=[]
    modeldir=out/'models'; modeldir.mkdir(exist_ok=True)
    while cursor<=end:
        stop=min(cursor+timedelta(days=config['fold_days']-1),end)
        cutoff=training_cutoff(cursor,stations,config['label_embargo_days'])
        lower=cutoff-timedelta(days=config['train_window_days'])
        train=frame[(frame.date<cutoff.isoformat())&(frame.date>=lower.isoformat())]
        test=frame[(frame.date>=cursor.isoformat())&(frame.date<=stop.isoformat())].copy()
        if len(train)<300 or test.empty: raise ValueError('Insufficient fold data')
        test['fold_start']=cursor.isoformat(); test['training_cutoff']=cutoff.isoformat()
        fold={'start':cursor.isoformat(),'end':stop.isoformat(),'training_before':cutoff.isoformat(),
              'train_rows':len(train),'test_rows':len(test),'bias_fallbacks':{}}
        for kind in ['max','min']:
            residual=train['actual_'+kind]-train['forecast_'+kind]
            model=lgb.LGBMRegressor(n_estimators=150,learning_rate=.04,num_leaves=15,
                min_child_samples=40,random_state=42,n_jobs=2,verbosity=-1)
            model.fit(transform(train,kind),residual)
            test['ecmwf_'+kind]=test['forecast_'+kind]
            test['lightgbm_'+kind]=test['forecast_'+kind]+model.predict(transform(test,kind))
            biases={}; fallbacks=[]
            for s in stations:
                part=train[(train.station_id==s['station_id']) &
                           (train.date>=(cutoff-timedelta(days=config['bias_max_age_days'])).isoformat())]
                part=part.sort_values('date').tail(config['bias_recent_samples'])
                if len(part)<config['bias_min_samples']:
                    biases[s['station_id']]=0.; fallbacks.append(s['station_id'])
                else: biases[s['station_id']]=float((part['actual_'+kind]-part['forecast_'+kind]).mean())
            test['bias_'+kind]=test['forecast_'+kind]+test.station_id.map(biases)
            fold['bias_fallbacks'][kind]=fallbacks
            joblib.dump({'model':model,'features':FEATURES,'training_before':cutoff.isoformat(),
                'biases':biases,'protocol':config,'deployment':'research_only'},modeldir/f'{cursor}-{kind}.joblib')
        fold['metrics']=score_group(test); folds.append(fold); predictions.append(test)
        print('Scored fold',cursor,len(test),flush=True)
        cursor=stop+timedelta(days=1)
    prediction=pd.concat(predictions,ignore_index=True)
    prediction.to_csv(out/'test_predictions.csv',index=False)
    per_city={s['city_id']:score_group(prediction[prediction.station_id==s['station_id']]) for s in stations}
    report={'status':'completed','protocol':config,'selected_stations':stations,'paired_rows':len(frame),
        'test_rows':len(prediction),'test_start':prediction.date.min(),'test_end':prediction.date.max(),
        'overall':score_group(prediction),'per_city':per_city,'folds':folds,
        'limitations':['Station sample extrema, not continuous daily extremes',
            '12h model publication latency and 7d observation latency assumed, not historically verified',
            'NOAA historical QC revisions may postdate simulated issue; no archive vintage reconstruction',
            'Airport coordinates, not city centers; regional representation is unbalanced',
            'No production deployment; no tuning or model selection using these test scores']}
    write_json(out/'metrics.json',report)
    lines=['# Rolling station comparison','',f"{len(stations)} stations; {config['start']}–{config['end']}; {len(prediction)} test city-days.",'',
        '| Target | Method | MAE °C | RMSE °C | Bias °C | Improvement |','|---|---|---:|---:|---:|---:|']
    for k,methods in report['overall'].items():
        for method,m in methods.items():
            lines.append(f"| T{k} | {method} | {m['mae_c']:.4f} | {m['rmse_c']:.4f} | {m['bias_c']:.4f} | {m.get('improvement_pct',0):.2f}% |")
    lines+=['','## Limitations','']+['- '+x for x in report['limitations']]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/rolling-stations-v1.json')
    ap.add_argument('--output',default='rolling-output'); ap.add_argument('--observations-only',action='store_true')
    args=ap.parse_args(); config=json.loads(Path(args.config).read_text()); out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists(): raise SystemExit('Completed output exists; refuse overwrite')
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=config:
        raise SystemExit('Config differs from cached protocol')
    write_json(out/'protocol.json',config)
    write_json(out/'provenance.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_at':datetime.now(UTC).isoformat()})
    with (out/'environment.txt').open('w') as f: subprocess.run(['python','-m','pip','freeze'],stdout=f,check=True)
    try:
        cache=Cache(out); obs,stations=collect_observations(config,cache,out)
        if args.observations_only: return
        paired=collect_pairs(config,cache,out,obs,stations)
        rolling_evaluate(paired,config,stations,out)
        manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        write_json(out/'manifest.json',manifest)
    except Exception as e:
        write_json(out/'failure.json',{'error':str(e),'at':datetime.now(UTC).isoformat()})
        raise


if __name__=='__main__': main()
