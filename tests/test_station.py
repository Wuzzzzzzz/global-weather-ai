from datetime import date, timedelta
import pytest
from research.train_station import window, run_for, observations, observed_day, forecast_day


def test_dst_and_cutoff():
    for day, hours in [(date(2025,3,9),23),(date(2025,11,2),25)]:
        a,b,c = window(day,'America/New_York')
        assert (b-a).total_seconds() == hours*3600
        assert run_for(c)+timedelta(hours=12)<=c<a


def test_qc_and_conflicts():
    def row(t,v):
        return {'DATE':t,'TMP':v,'STATION':'123','REPORT_TYPE':'FM-15'}
    rows = [row('2025-01-01T00:00:00','+0100,1'),row('2025-01-01T00:00:00','+0120,1'),
            row('2025-01-01T01:00:00','+9999,1'),row('2025-01-01T02:00:00','+0110,3'),
            row('2025-01-01T03:00:00','-0010,5')]
    assert list(observations(rows,'123').values()) == [-1.0]
    with pytest.raises(ValueError):
        observations({'html':'error'},'123')


def test_missing_hours_and_day_boundary():
    a,b,_ = window(date(2025,7,1),'Asia/Shanghai')
    obs = {a+timedelta(hours=i):float(i) for i in range(24)}
    obs[b] = 999
    assert observed_day(obs,a,b) == (23,0,24)
    del obs[a+timedelta(hours=12)]
    with pytest.raises(ValueError):
        observed_day(obs,a,b)
    payload = {'utc_offset_seconds':0,'hourly_units':{'temperature_2m':'°C'},
               'hourly':{'time':[(a+timedelta(hours=i)).replace(tzinfo=None).isoformat() for i in range(24)],
                         'temperature_2m':list(range(24))}}
    assert forecast_day(payload,a,b) == (23,0)
    payload['hourly']['temperature_2m'][12] = None
    with pytest.raises(ValueError):
        forecast_day(payload,a,b)
