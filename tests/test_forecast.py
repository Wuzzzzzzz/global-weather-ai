from datetime import datetime, timezone
from src.forecast import collect, save, target_for_timezone


def test_local_tomorrow_across_date_line():
    now = datetime(2026, 11, 10, 15, tzinfo=timezone.utc)
    daily = {"time": ["2026-11-10", "2026-11-11", "2026-11-12"],
             "temperature_2m_max": [20, 22, 21],
             "temperature_2m_min": [12, 13, 11]}
    assert target_for_timezone({"timezone": "Asia/Tokyo", "daily": daily}, now) == ("2026-11-12", 21, 11)
    assert target_for_timezone({"timezone": "America/Los_Angeles", "daily": daily}, now) == ("2026-11-11", 22, 13)


class FakeResponse:
    def raise_for_status(self):
        pass
    def json(self):
        return {"timezone": "Asia/Shanghai", "daily": {
            "time": ["2026-11-10", "2026-11-11", "2026-11-12"],
            "temperature_2m_max": [21, 24, 23],
            "temperature_2m_min": [12, 14, 13]}}


class FakeSession:
    def get(self, url, params, timeout):
        assert "temperature_2m_max" in params["daily"]
        return FakeResponse()


def test_collect_and_save(tmp_path):
    path = tmp_path / "cities.csv"
    path.write_text("city_id,name,country,latitude,longitude\nGZ,Guangzhou,CN,23.1,113.3\n")
    now = datetime(2026, 11, 10, 1, tzinfo=timezone.utc)
    rows = collect(path, now, FakeSession(), pause_seconds=0)
    assert rows[0]["status"] == "ok"
    assert rows[0]["tmax_c"] == 24.0
    output = save(rows, tmp_path / "output", now)
    assert output.exists()
    assert (tmp_path / "output/latest.csv").exists()
