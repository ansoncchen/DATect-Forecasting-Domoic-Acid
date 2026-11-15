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
    # the alert is the hybrid alert or the bloom Watch (3 holdout rows flagged at Long Beach and
    # Twin Harbors; no sudden rise from below 5 there, so 19 of 27 stands and 3 quiet weeks gain a Watch)
    assert s["onsets"] == {"n": 27, "caught": 19, "quiet": 292, "quiet_alarmed": 90}
    assert s["alarms"] == {"n": 109, "followed_by_jump": 19, "false_below_5": 23,
                           "false_5_to_10": 19, "false_near_miss": 48}
    assert s["band"]["coverage"] == pytest.approx(0.802, abs=5e-4)
    assert s["band"]["coverage_at_onsets"] == pytest.approx(10 / 27)
    assert (s["band"]["inside"], s["band"]["n_banded"], s["band"]["onsets_inside"]) == (328, 409, 10)
    assert s["band"]["median_width"] == pytest.approx(15.08, abs=0.01)
    assert s["levels"] == {"Quiet": {"weeks": 210, "followed_by_jump": 8},
                           "Watch": {"weeks": 71, "followed_by_jump": 9},
                           "Warning": {"weeks": 38, "followed_by_jump": 10}}


def test_validation_and_per_site_reliability():
    v = tr.window_summary("validation")
    # the bloom Watch adds one caught jump (Long Beach, 2020-10-22) and 5 quiet-week alarms
    assert v["onsets"]["caught"] == 28 and v["onsets"]["quiet_alarmed"] == 78
    assert v["band"]["coverage"] == pytest.approx(0.819, abs=5e-4)
    assert v["levels"] == {"Quiet": {"weeks": 414, "followed_by_jump": 6},
                           "Watch": {"weeks": 69, "followed_by_jump": 14},
                           "Warning": {"weeks": 37, "followed_by_jump": 14}}
    assert (v["alarms"]["n"], v["alarms"]["followed_by_jump"]) == (106, 28)
    assert (v["alarms"]["false_below_5"], v["alarms"]["false_5_to_10"], v["alarms"]["false_near_miss"]) == (25, 16, 37)
    g = tr.window_summary("holdout", "Gold Beach")
    assert g["n"] == 40 and g["band"]["coverage"] == pytest.approx(0.375)
    assert g["r2"][0] == pytest.approx(-0.208, abs=5e-4)
    assert tr.window_summary("holdout", "Cannon Beach")["r2"] is None   # 2 rows, no JSON entry


def test_recomputed_alert_equals_the_engines_spike_alert():
    """The engine's ``spike_alert`` is the hybrid alert; the bloom Watch is added at serve time."""
    for window in ("holdout", "validation"):
        rows = tr.window_rows(window)
        for r in rows.itertuples():
            parts = alert_parts(r.spike_probability, r.predicted_da, r.naive_prediction,
                                prob_cutoff=config.SPIKE_ALERT_PROB_THRESHOLD,
                                regression_cutoff=config.SPIKE_REGRESSION_ALERT_THRESHOLD,
                                warning_cutoff=config.ALERT_WARNING_PROB_CUTOFF)
            assert (parts["by_classifier"] or parts["by_regression"]) == bool(r.spike_alert), (window, r.site, r.date)


@pytest.fixture
def client():
    from backend import api
    with TestClient(api.app) as c:
        yield c


def test_track_record_endpoint_returns_rows_and_summaries(client):
    body = client.get("/api/track-record", params={"window": "holdout"}).json()
    assert body["window"] == {"name": "holdout", "start": "2022-01-01", "end": "2024-01-01"}
    assert len(body["rows"]) == 409 and body["summary"]["onsets"]["caught"] == 19
    # rows carry the alert as shown (hybrid or bloom); it agrees with the summary's alarm count
    below = [r for r in body["rows"] if r["naive_prediction"] < 20]
    assert sum(r["alert"] for r in below) == body["summary"]["alarms"]["n"] == 109
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


def test_r2_is_undefined_on_a_constant_or_tiny_target():
    """Cannon Beach's 11 validation measurements are all zero: R2 means nothing there."""
    cannon = tr.window_rows("validation", "Cannon Beach")
    assert len(cannon) == 11 and cannon["actual_da"].nunique() == 1
    s = tr.window_summary("validation", "Cannon Beach")
    assert s["r2"] is None and s["log_r2"] is None and s["models"] is None
    assert s["mae"] is not None and s["n"] == 11
    assert tr.window_summary("holdout", "Cannon Beach")["models"] is None   # 2 rows
    assert tr.window_summary("holdout", "Gold Beach")["models"]["naive"]["r2"] is not None


def test_washington_beaches_on_the_holdout():
    """Per-beach counts. Long Beach and Twin Harbors match the port spec (section 5); the
    northern beaches have the hybrid alert only."""
    expected = {  # quiet weeks alarmed, alarms (followed), Watch weeks
        "Kalaloch": (7, (7, 0), 6), "Quinault": (10, (15, 5), 6), "Copalis": (18, (18, 0), 13),
        "Twin Harbors": (10, (10, 0), 9), "Long Beach": (7, (9, 2), 7),
    }
    for site, (quiet_alarmed, (alarms, followed), watch) in expected.items():
        s = tr.window_summary("holdout", site)
        assert s["onsets"]["quiet_alarmed"] == quiet_alarmed, site
        assert (s["alarms"]["n"], s["alarms"]["followed_by_jump"]) == (alarms, followed), site
        assert s["levels"]["Watch"]["weeks"] == watch, site


def test_history_rows_say_when_the_bloom_watch_was_on(client):
    rows = client.get("/api/sites/Long Beach/history", params={"since": "2020-10-01", "until": "2020-10-31"}).json()["rows"]
    day = next(r for r in rows if r["date"] == "2020-10-22")
    assert day["by_bloom"] is True and day["alert"] is True and day["level"] == "Watch"


def test_bloom_record_before_the_holdout():
    """The record the dashboard quotes for the bloom Watch, from the committed all-years run
    (its Long Beach rows start in November 2007)."""
    b = tr.bloom_record()
    assert (b["start"], b["end"]) == ("2007-11-14", "2022-01-01")
    assert (b["on"], b["followed_by_sudden_rise"], b["sudden_rises"], b["caught"]) == (37, 6, 6, 6)
    assert b["caught_by_site"] == {"Long Beach": 4, "Twin Harbors": 2}
    # counts known 1 and 2 days after sampling (onset research, round 9: 34/5 and 29/4)
    assert b["late"] == [{"days": 1, "on": 34, "caught": 5}, {"days": 2, "on": 29, "caught": 4}]
