"""Independent IFS feature experiment; no production writes.
AIFS is excluded unless matching native-run station data can be verified.
"""
import argparse,json,math,hashlib,subprocess
from pathlib import Path
from datetime import date,timedelta,datetime,timezone
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from research.rolling_station_backtest import Cache,collect_observations,forecast_extrema,write_json,SINGLE
from research.train_station import window,run_for,metrics
from research.train_baseline import transform

WEATHER=['cloud_cover','relative_humidity_2m','wind_speed_10m','shortwave_radiation']


def aggregate_weather(payload,start,end):
    h=payload['hourly'];times=[datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in h['time']]
    result={}
    expected={'cloud_cover':'%','relative_humidity_2m':'%','wind_speed_10m':'km/h','shortwave_radiation':'W/m²'}
    for var in WEATHER:
        if payload['hourly_units'].get(var)!=expected[var]:raise ValueError('Unexpected unit: '+var)
        if len(h[var])!=len(times):raise ValueError('Weather length mismatch')
        vals=[v for t,v in zip(times,h[var]) if start<=t<end]
        if not vals or any(v is None or not math.isfinite(v) for v in vals):raise ValueError('Missing target-day weather: '+var)
        result[var+'_mean']=float(np.mean(vals))
    return result


def collect(cfg,out):
    cache=Cache(out);obs,stations=collect_observations(cfg,cache,out)
    mapping={(r.station_id,r.date):r._asdict() for r in obs.itertuples(index=False)}
    rows=[];failures={};start=date.fromisoformat(cfg['start']);end=date.fromisoformat(cfg['end'])
    for n in range((end-start).days+1):
        d=start+timedelta(days=n);groups=defaultdict(list)
        for s in stations:
            if (s['station_id'],str(d)) not in mapping:continue
            a,b,c=window(d,s['timezone']);groups[run_for(c)].append((s,a,b,c))
        for run,items in groups.items():
            p={'latitude':','.join(str(x[0]['latitude']) for x in items),'longitude':','.join(str(x[0]['longitude']) for x in items),
               'run':run.strftime('%Y-%m-%dT%H:%M'),'hourly':','.join(['temperature_2m']+WEATHER),'models':'ecmwf_ifs','timezone':'GMT','forecast_days':4}
            payload=cache.get(SINGLE,p,len(items));payload=payload if isinstance(payload,list) else [payload]
            if len(payload)!=len(items):raise ValueError('Location mismatch')
            for i,(v,(s,a,b,c)) in enumerate(zip(payload,items)):
                if v.get('location_id',i)!=i:raise ValueError('Location order mismatch')
                if abs(v['latitude']-s['latitude'])>.5 or abs(v['longitude']-s['longitude'])>.5:raise ValueError('Grid mismatch')
                try:
                    hi,lo=forecast_extrema(v,a,b);weather=aggregate_weather(v,a,b)
                    row=mapping[(s['station_id'],str(d))].copy()
                    row.update(forecast_max=hi,forecast_min=lo,latitude=s['latitude'],longitude=s['longitude'],
                      day_sin=math.sin(2*math.pi*d.timetuple().tm_yday/365.25),day_cos=math.cos(2*math.pi*d.timetuple().tm_yday/365.25),
                      run_utc=run.isoformat(),cutoff_utc=c.isoformat(),ifs_hourly=json.dumps(v['hourly']),**weather)
                    rows.append(row)
                except (ValueError,KeyError) as e:failures[s['city_id']+':'+str(d)]=str(e)
        pd.DataFrame(rows).to_csv(out/'paired_data.csv',index=False);write_json(out/'failures.json',failures)
        if n%10==0:print('V2 IFS features',n+1,'/',(end-start).days+1,'rows',len(rows),flush=True)
    return pd.DataFrame(rows).sort_values(['date','city_id'])


def recent_features(frame,history):
    """Every row uses residual labels strictly older than target minus 10 days.
    90-day age cap, most recent 30 accepted samples, fallback zero if <15.
    This conservatively exceeds the previous 7-day assumed label latency.
    """
    result=frame.copy();history=history.sort_values('date')
    for k in ['max','min']:
        means=[];counts=[]
        for r in frame.itertuples(index=False):
            cutoff=date.fromisoformat(r.date)-timedelta(days=10)
            part=history[(history.city_id==r.city_id)&(history.date<str(cutoff))&
                         (history.date>=str(cutoff-timedelta(days=90)))].tail(30)
            means.append(float((part['actual_'+k]-part['forecast_'+k]).mean()) if len(part)>=15 else 0.)
            counts.append(len(part))
        result['recent_'+k]=means;result['recent_count_'+k]=counts
    return result


def matrix(frame,kind,variant):
    x=transform(frame,kind)
    if variant=='enriched':
        for c in [v+'_mean' for v in WEATHER]+['recent_max','recent_min','recent_count_max','recent_count_min']+(['aifs_shape_max','aifs_shape_min'] if 'aifs_shape_max' in frame else []):
            x[c]=frame[c].to_numpy()
    return x


def evaluate(frame,cfg,out):
    train=frame[frame.date<=cfg['train_end']]
    val=frame[(frame.date>=cfg['validation_start'])&(frame.date<=cfg['validation_end'])]
    final_train=frame[frame.date<=cfg['validation_end']]
    test=frame[(frame.date>=cfg['test_start'])&(frame.date<=cfg['end'])].copy()
    if min(len(train),len(val),len(test))<100 or test.city_id.nunique()!=20:raise ValueError('Insufficient common holdout coverage')
    report={'status':'completed','protocol':cfg,'train_rows':len(train),'validation_rows':len(val),'refit_rows':len(final_train),'test_rows':len(test),'targets':{},'per_city':{}}
    models=out/'models';models.mkdir(exist_ok=True)
    for k in ['max','min']:
        base=test['forecast_'+k];test['raw_'+k]=base;test['bias_'+k]=base+test['recent_'+k]
        report['targets'][k]={'raw':metrics(test['actual_'+k],base),'bias':metrics(test['actual_'+k],test['bias_'+k])}
        if 'aifs_shape_'+k in frame:
            weights={w:metrics(val['actual_'+k],val['forecast_'+k]+w*(val['aifs_shape_'+k]-val['forecast_'+k]))['mae_c'] for w in [0.,.25,.5,.75,1.]}
            weight=min(weights,key=weights.get)
            test['blend_'+k]=base+weight*(test['aifs_shape_'+k]-base)
            test['equal_blend_'+k]=.5*(base+test['aifs_shape_'+k])
            for method in ['equal_blend','blend']:
                report['targets'][k][method]=metrics(test['actual_'+k],test[method+'_'+k])
            report['targets'][k]['blend'].update(selected_aifs_weight=weight,validation_mae=weights)
        for variant in ['basic','enriched']:
            choices={}
            for leaves in [7,15]:
                model=lgb.LGBMRegressor(n_estimators=150,learning_rate=.04,num_leaves=leaves,min_child_samples=40,random_state=42,n_jobs=2,verbosity=-1)
                model.fit(matrix(train,k,variant),train['actual_'+k]-train['forecast_'+k])
                pred=val['forecast_'+k]+model.predict(matrix(val,k,variant))
                choices[leaves]=metrics(val['actual_'+k],pred)['mae_c']
            leaves=min(choices,key=choices.get)
            model=lgb.LGBMRegressor(n_estimators=150,learning_rate=.04,num_leaves=leaves,min_child_samples=40,random_state=42,n_jobs=2,verbosity=-1)
            model.fit(matrix(final_train,k,variant),final_train['actual_'+k]-final_train['forecast_'+k])
            test[variant+'_'+k]=base+model.predict(matrix(test,k,variant))
            report['targets'][k][variant]={**metrics(test['actual_'+k],test[variant+'_'+k]),'validation_mae':choices,'selected_leaves':leaves}
            joblib.dump({'model':model,'features':list(matrix(test,k,variant).columns),'protocol':cfg,'status':'research_only'},models/f'{variant}_{k}.joblib')
        for method,v in report['targets'][k].items():v['improvement_pct']=100*(1-v['mae_c']/report['targets'][k]['raw']['mae_c'])
    for city,part in test.groupby('city_id'):
        report['per_city'][city]={k:{m:metrics(part['actual_'+k],part[m+'_'+k]) for m in ['raw','bias','basic','enriched']+(['equal_blend','blend'] if 'aifs_shape_max' in frame else [])} for k in ['max','min']}
    report['deployment']='not eligible: only 20-day summer holdout; no geographic holdout; availability assumptions unresolved'
    report['aifs']='IFS hourly shape plus interpolated native-six-hour AIFS-minus-IFS anomaly; NOT standalone native AIFS daily extremes' if 'aifs_shape_max' in frame else 'not included'
    write_json(out/'metrics.json',report);test.to_csv(out/'test_predictions.csv',index=False)
    print(json.dumps(report['targets'],indent=2))
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/v2-ifs-features.json');ap.add_argument('--history',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():raise SystemExit('Refuse overwrite completed experiment')
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=cfg:raise SystemExit('Protocol changed')
    write_json(out/'protocol.json',cfg)
    write_json(out/'provenance.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'started_at':datetime.now(timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    with (out/'environment.txt').open('w') as f:subprocess.run(['python','-m','pip','freeze'],stdout=f,check=True)
    frame=collect(cfg,out)
    if cfg.get('include_aifs'):
        from research.v2_aifs import add_aifs
        frame=add_aifs(frame,cfg,out)
    history=pd.concat([pd.read_csv(a.history),frame],ignore_index=True)
    if history.duplicated(['city_id','date']).any():raise ValueError('Overlapping history')
    frame=recent_features(frame,history);frame.to_csv(out/'features.csv',index=False)
    evaluate(frame,cfg,out)
    write_json(out/'manifest.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'})

if __name__=='__main__':main()
