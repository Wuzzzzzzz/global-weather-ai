from datetime import date, timedelta
import numpy as np
import pandas as pd
from research.train_baseline import daily_extrema, fit_and_score


def test_extrema_and_missing_hours():
    x = {"hourly": {"time": [f"2026-01-01T{h:02d}:00" for h in range(24)],
                    "temperature_2m_previous_day1": list(range(24))}}
    assert daily_extrema(x, "temperature_2m_previous_day1")["2026-01-01"] == (23., 0.)


def test_chronological_train_scoring(tmp_path):
    rows = []
    start = date(2026, 1, 1)
    for i in range(80):
        d = (start + timedelta(days=i)).isoformat()
        for c in range(3):
            p = 18 + np.sin(i / 10) + c
            rows.append({"date": d, "city_id": str(c), "latitude": c * 10.,
                         "longitude": 20. + c, "day_sin": np.sin(i),
                         "day_cos": np.cos(i), "forecast_max": p + 8,
                         "forecast_min": p, "actual_max": p + 8.5,
                         "actual_min": p - 0.3})
    result = fit_and_score(pd.DataFrame(rows), tmp_path, test_days=20)
    assert result["train_rows"] == 180
    assert result["test_rows"] == 60
    assert (tmp_path / "metrics.json").exists()
