"""Recompute v1 scores and date-block bootstrap uncertainty, without refitting."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 root=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 m=json.loads((root/'metrics.json').read_text());p=pd.read_csv(root/'test_predictions.csv');rows=[]
 for k in ['max','min']:
  for method in ['ecmwf','bias','lightgbm']:
   err=p[method+'_'+k]-p['actual_'+k]
   assert abs(abs(err).mean()-m['overall'][k][method]['mae_c'])<1e-10
   rows.append({'target':k,'method':method,**m['overall'][k][method]})
 pd.DataFrame(rows).to_csv(out/'overall.csv',index=False)
 cities=[];folds=[]
 for s in m['selected_stations']:
  for k in ['max','min']:
   for method,v in m['per_city'][s['city_id']][k].items():cities.append({'city':s['city'],'target':k,'method':method,**v})
 for f in m['folds']:
  for k in ['max','min']:
   for method,v in f['metrics'][k].items():folds.append({'start':f['start'],'end':f['end'],'target':k,'method':method,**v})
 pd.DataFrame(cities).to_csv(out/'cities.csv',index=False);pd.DataFrame(folds).to_csv(out/'folds.csv',index=False)
 intervals=[];rng=np.random.default_rng(42)
 for k in ['max','min']:
  for ref in ['ecmwf','bias']:
   q=pd.DataFrame({'date':p.date,'gain':abs(p[ref+'_'+k]-p['actual_'+k])-abs(p['lightgbm_'+k]-p['actual_'+k])})
   # Seven-day moving blocks; retain all cities of a sampled date together.
   daily=q.groupby('date').gain.agg(['sum','count']).reindex(pd.date_range(p.date.min(),p.date.max()).strftime('%Y-%m-%d'),fill_value=0)
   samples=[];n=len(daily)
   for _ in range(2000):
    idx=np.concatenate([(s+np.arange(7))%n for s in rng.integers(0,n,size=(n+6)//7)])[:n]
    v=daily.iloc[idx];samples.append(v['sum'].sum()/v['count'].sum())
   intervals.append({'target':k,'reference':ref,'mae_reduction_c':float(q.gain.mean()),'block7_bootstrap_95pct':np.quantile(samples,[.025,.975]).tolist()})
 assert not p.duplicated(['city_id','date']).any()
 assert (pd.to_datetime(p.run_utc,utc=True)+pd.Timedelta(hours=12)<=pd.to_datetime(p.cutoff_utc,utc=True)).all()
 assert (p.date>=p.fold_start).all()
 result={'bootstrap':intervals,'n':len(p),'missing_fraction_calendar':1-len(p)/(20*366),
 'checks':['six MAEs recomputed','no duplicate city/date','all run+12h <= cutoff','all target dates >= fold start'],
 'limitations':['same 20 stations in training and test: geographic holdout NOT tested','13 folds include one short terminal fold; not 13 independent seasons','historical publication/QC vintages not verified','sampled airport temperature extrema; missing days excluded','bootstrap conditional on these cities and model choices; not global guarantee']}
 (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
