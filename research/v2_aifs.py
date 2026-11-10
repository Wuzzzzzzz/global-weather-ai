"""Native AIFS archive adapter. Bounded point reads; never load global dataset."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.rolling_station_backtest import write_json


def reconstruct(ifs_time,ifs_temp,native_time,native_temp,target_time):
    it=np.asarray(ifs_time,dtype=float);iv=np.asarray(ifs_temp,dtype=float)
    nt=np.asarray(native_time,dtype=float);nv=np.asarray(native_temp,dtype=float);tt=np.asarray(target_time,dtype=float)
    valid=(nt>=it.min())&(nt<=it.max());nt=nt[valid];nv=nv[valid]
    if len(nt)<2 or nt.min()>tt.min() or nt.max()<tt.max():raise ValueError('AIFS does not bracket target hours')
    if not np.isfinite(nv).all() or not np.isfinite(iv).all():raise ValueError('Missing AIFS/IFS temperature')
    if not (np.diff(nt)>0).all():raise ValueError('Native times not ordered')
    delta=nv-np.interp(nt,it,iv)
    return np.interp(tt,it,iv)+np.interp(tt,nt,delta)


def add_aifs(frame,cfg,out):
    import dynamical_catalog
    import xarray as xr
    from research.train_station import window
    from datetime import date
    cache=out/'aifs-cache';cache.mkdir(exist_ok=True)
    ds=None;results=[];failures={}
    stations={s['city_id']:s for s in cfg['stations']}
    for run,group in frame.groupby('run_utc',sort=True):
        stamp=pd.Timestamp(run);name=stamp.strftime('%Y%m%dT%H%M');path=cache/(name+'.json')
        if path.exists():payload=json.loads(path.read_text())
        else:
            if ds is None:ds=dynamical_catalog.open('ecmwf-aifs-single-forecast',chunks=None)
            cities=sorted(group.city_id.unique())
            # Exact init_time: never select nearest run or silently substitute.
            v=ds.temperature_2m.sel(init_time=stamp.tz_localize(None).to_datetime64()).sel(
              latitude=xr.DataArray([stations[c]['latitude'] for c in cities],dims='station'),
              longitude=xr.DataArray([stations[c]['longitude'] for c in cities],dims='station'),method='nearest')
            v=v.sel(lead_time=slice(np.timedelta64(0,'h'),np.timedelta64(96,'h'))).load()
            if v.attrs.get('units')!='degree_Celsius':raise ValueError('AIFS units mismatch')
            native=((v.valid_time.values.astype('datetime64[s]')).astype('int64')).tolist()
            payload={'source':'dynamical.org ECMWF AIFS Single','init_time':run,'fetched_at':pd.Timestamp.now(tz='UTC').isoformat(),
                     'native_time_unix':native,'cities':{c:{'temperature':v.isel(station=i).values.tolist(),
                     'grid_latitude':float(v.latitude.isel(station=i)),'grid_longitude':float(v.longitude.isel(station=i))} for i,c in enumerate(cities)}}
            # NaN indicates missing archive runs; preserve null explicitly.
            for value in payload['cities'].values():value['temperature']=[float(x) if np.isfinite(x) else None for x in value['temperature']]
            write_json(path,payload)
        for row in group.to_dict('records'):
            try:
                s=stations[row['city_id']];a,b,_=window(date.fromisoformat(row['date']),s['timezone'])
                h=json.loads(row['ifs_hourly']);it=pd.to_datetime(h['time'],utc=True).astype('int64').to_numpy()/1e9
                target=it[(it>=a.timestamp())&(it<b.timestamp())]
                native=payload['cities'][row['city_id']]
                if abs(native['grid_latitude']-s['latitude'])>.3 or abs(native['grid_longitude']-s['longitude'])>.3:raise ValueError('AIFS grid mismatch')
                vals=reconstruct(it,h['temperature_2m'],payload['native_time_unix'],native['temperature'],target)
                row['aifs_shape_max']=float(max(vals));row['aifs_shape_min']=float(min(vals));results.append(row)
            except ValueError as e:failures[row['city_id']+':'+row['date']]=str(e)
        pd.DataFrame(results).to_csv(out/'aifs_paired_partial.csv',index=False)
        write_json(out/'aifs_failures.json',failures)
        print('AIFS run',run,'matched',len(results),'excluded',len(failures),flush=True)
    result=pd.DataFrame(results)
    if len(result)<.8*len(frame):raise ValueError('AIFS common support below 80%; no fair-comparison claim')
    return result.sort_values(['date','city_id'])
