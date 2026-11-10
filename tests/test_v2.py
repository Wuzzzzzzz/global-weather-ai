from datetime import date,timedelta
import pandas as pd
from research.v2_features import recent_features,matrix


def test_recent_bias_excludes_recent_and_future_labels():
 rows=[]
 for n in range(60):
  rows.append({'city_id':'x','date':str(date(2025,6,1)+timedelta(days=n)),
               'forecast_max':20.,'forecast_min':10.,'actual_max':21.,'actual_min':9.})
 history=pd.DataFrame(rows); target=pd.DataFrame([{'city_id':'x','date':'2025-07-25'}])
 a=recent_features(target,history)
 history.loc[history.date>='2025-07-15',['actual_max','actual_min']]=999
 b=recent_features(target,history)
 assert a.recent_max.iloc[0]==b.recent_max.iloc[0]==1
 assert a.recent_min.iloc[0]==b.recent_min.iloc[0]==-1
 assert a.recent_count_max.iloc[0]==30


def test_missing_history_falls_back_to_raw():
 history=pd.DataFrame(columns=['city_id','date','forecast_max','forecast_min','actual_max','actual_min'])
 x=recent_features(pd.DataFrame([{'city_id':'x','date':'2025-07-25'}]),history)
 assert x.recent_max.iloc[0]==0 and x.recent_count_max.iloc[0]==0


def test_aifs_anomaly_preserves_hourly_ifs_shape():
 import numpy as np
 from research.v2_aifs import reconstruct
 hours=np.arange(25);ifs=20+5*np.sin(hours*np.pi/12)
 anchors=np.arange(0,25,6);aifs=ifs[anchors]+2
 result=reconstruct(hours,ifs,anchors,aifs,hours)
 np.testing.assert_allclose(result,ifs+2)
 import pytest
 with pytest.raises(ValueError):reconstruct(hours,ifs,anchors,aifs,np.array([25]))
