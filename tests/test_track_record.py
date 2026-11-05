"""Track record and site history (backend/track_record.py): every number the
dashboard shows is computed here, from the committed chronological runs, and must
equal the reported numbers (docs/CORRECTED_NUMBERS.md, docs/ERROR_BAND.md)."""
import pytest
from fastapi.testclient import TestClient

import config
from backend import track_record as tr
from forecasting.metrics import alert_parts


def test_holdout_summary_matches_the_reported_numbers():
    s = tr.window_summary("holdout")
    assert s["n"] == 409
    assert s["r2"] == pytest.approx([0.5162, 0.3081, 0.6538], abs=5e-4)    # chronological JSON
    assert s["mae"] == pytest.approx([7.4407, 6.1999, 8.8675], abs=5e-4)
    assert s["models"]["ensemble"]["r2"] == pytest.approx(s["r2"][0])
    assert s["models"]["naive"]["r2"] == pytest.approx(0.225, abs=5e-4)   # CORRECTED_NUMBERS
    assert s["onsets"] == {"n": 27, "caught": 19, "quiet": 292, "quiet_alarmed": 87}
    assert s["alarms"] == {"n": 106, "followed_by_jump": 19, "false_below_5": 21,
                           "false_5_to_10": 18, "false_near_miss": 48}
    assert s["band"]["coverage"] == pytest.approx(0.802, abs=5e-4)
    assert s["band"]["coverage_at_onsets"] == pytest.approx(10 / 27)
    assert (s["band"]["inside"], s["band"]["n_banded"], s["band"]["onsets_inside"]) == (328, 409, 10)
    assert s["band"]["median_width"] == pytest.approx(15.08, abs=0.01)
    assert s["levels"] == {"Quiet": {"weeks": 213, "followed_by_jump": 8},
                           "Watch": {"weeks": 68, "followed_by_jump": 9},
                           "Warning": {"weeks": 38, "followed_by_jump": 10}}


def test_validation_and_per_site_reliability():
    v = tr.window_summary("validation")
    assert v["onsets"]["caught"] == 27 and v["band"]["coverage"] == pytest.approx(0.819, abs=5e-4)
    assert v["levels"] == {"Quiet": {"weeks": 420, "followed_by_jump": 7},
                           "Watch": {"weeks": 63, "followed_by_jump": 13},
                           "Warning": {"weeks": 37, "followed_by_jump": 14}}
    assert (v["alarms"]["false_below_5"], v["alarms"]["false_5_to_10"], v["alarms"]["false_near_miss"]) == (23, 13, 37)
    g = tr.window_summary("holdout", "Gold Beach")
    assert g["n"] == 40 and g["band"]["coverage"] == pytest.approx(0.375)
    assert g["r2"][0] == pytest.approx(-0.208, abs=5e-4)
    assert tr.window_summary("holdout", "Cannon Beach")["r2"] is None   # 2 rows, no JSON entry


def test_recomputed_alert_equals_the_engines_spike_alert():
    for window in ("holdout", "validation"):
        rows = tr.window_rows(window)
        for r in rows.itertuples():
            parts = alert_parts(r.spike_probability, r.predicted_da, r.naive_prediction,
                                prob_cutoff=config.SPIKE_ALERT_PROB_THRESHOLD,
                                regression_cutoff=config.SPIKE_REGRESSION_ALERT_THRESHOLD,
                                warning_cutoff=config.ALERT_WARNING_PROB_CUTOFF)
            assert parts["alert"] == bool(r.spike_alert), (window, r.site, r.date)


@pytest.fixture
def client():
    from backend import api
    with TestClient(api.app) as c:
        yield c


def test_track_record_endpoint_returns_rows_and_summaries(client):
    body = client.get("/api/track-record", params={"window": "holdout"}).json()
    assert body["window"] == {"name": "holdout", "start": "2022-01-01", "end": "2024-01-01"}
    assert len(body["rows"]) == 409 and body["summary"]["onsets"]["caught"] == 19
    assert body["per_site"]["Gold Beach"]["band"]["coverage"] == pytest.approx(0.375)
    one = client.get("/api/track-record", params={"window": "holdout", "site": "gold-beach"}).json()
    assert {r["site"] for r in one["rows"]} == {"Gold Beach"} and one["summary"]["n"] == 40


def test_track_record_rejects_unknown_window(client):
    assert client.get("/api/track-record", params={"window": "2015"}).status_code == 422


def test_history_resolves_aliases_and_404s_unknown_sites(client):
    rows = client.get("/api/sites/newport/history", params={"since": "2022-01-01", "until": "2022-12-31"}).json()["rows"]
    assert rows and {r["site"] for r in rows} == {"Newport"}
    assert all("2022-01-01" <= r["date"] <= "2022-12-31" for r in rows)
    assert {"band_lo", "band_hi", "spike_probability", "alert", "level"} <= set(rows[0])
    assert client.get("/api/sites/Atlantis/history").status_code == 404
    assert client.get("/api/track-record", params={"site": "Atlantis"}).status_code == 404


def test_history_with_since_after_until_is_empty(client):
    body = client.get("/api/sites/Newport/history", params={"since": "2023-01-01", "until": "2022-01-01"}).json()
    assert body["rows"] == []



def test_history_rejects_malformed_dates(client):
    assert client.get("/api/sites/Newport/history", params={"since": "garbage"}).status_code == 422
    assert client.get("/api/sites/Newport/history", params={"until": "2022-13-45"}).status_code == 422
