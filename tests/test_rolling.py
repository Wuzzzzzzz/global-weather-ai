from datetime import date,timedelta
import json
import pandas as pd
import pytest
from research.rolling_station_backtest import training_cutoff, forecast_extrema, rolling_evaluate, Cache
from research.train_station import window


def test_global_cutoff_excludes_future_labels():
    stations=[{'timezone':'Pacific/Auckland'},{'timezone':'America/Los_Angeles'}]
    fold=date(2025,3,10)
    cutoff=training_cutoff(fold,stations,7)
    for s in stations:
        _,end,_=window(cutoff-timedelta(days=1),s['timezone'])
        issue=window(fold,s['timezone'])[2]
        assert end+timedelta(days=7)<issue


def test_half_hour_timezone_forecast():
    a,b,_=window(date(2025,1,1),'Asia/Kolkata')
    first=a.replace(minute=0)+timedelta(hours=1)
    p={'utc_offset_seconds':0,'hourly_units':{'temperature_2m':'°C'},'hourly':{
        'time':[(first+timedelta(hours=i)).replace(tzinfo=None).isoformat() for i in range(24)],
        'temperature_2m':list(range(24))}}
    assert forecast_extrema(p,a,b)==(23,0)


def test_budget_counts_location_cost(tmp_path):
    cache=Cache(tmp_path); cache.reserve(20); cache.reserve(3)
    assert sum(r[1] for r in json.loads((tmp_path/'request_ledger.json').read_text()))==23


def test_future_labels_cannot_change_first_fold_predictions(tmp_path):
    cfg={'start':'2024-04-01','end':'2024-06-14','warmup_days':60,'fold_days':15,
         'label_embargo_days':7,'train_window_days':365,'bias_max_age_days':90,
         'bias_recent_samples':30,'bias_min_samples':15}
    stations=[{'station_id':str(i),'city_id':str(i),'timezone':'Asia/Shanghai'} for i in range(8)]
    rows=[]
    for n in range(75):
        d=date(2024,4,1)+timedelta(days=n)
        for s in stations:
            rows.append(dict(station_id=s['station_id'],city_id=s['city_id'],date=d.isoformat(),
               forecast_max=20+n*.02,forecast_min=10+n*.02,actual_max=21+n*.02,actual_min=9+n*.02,
               latitude=30,longitude=110,day_sin=.1,day_cos=.9))
    frame=pd.DataFrame(rows); a=tmp_path/'a'; b=tmp_path/'b'; a.mkdir();b.mkdir()
    rolling_evaluate(frame,cfg,stations,a)
    frame.loc[frame.date>='2024-06-01',['actual_max','actual_min']]+=50
    rolling_evaluate(frame,cfg,stations,b)
    x=pd.read_csv(a/'test_predictions.csv');y=pd.read_csv(b/'test_predictions.csv')
    for name in ['ecmwf_max','bias_max','lightgbm_max','ecmwf_min','bias_min','lightgbm_min']:
        pd.testing.assert_series_equal(x[name],y[name])
