"""Endpoints the dashboard never calls are gone; the ones it uses still answer."""
from fastapi.testclient import TestClient

from backend.api import app


def test_endpoints_the_dashboard_never_calls_are_gone():
    with TestClient(app) as client:
        body = {"date": "2019-06-03", "site": "Newport", "task": "regression", "model": "ensemble"}
        assert client.post("/api/forecast", json=body).status_code in (404, 405)
        assert client.post("/api/retrospective", json={"task": "regression", "model": "ensemble"}).status_code in (404, 405)
        assert client.get("/api/models").status_code == 404
        assert client.get("/api/config").status_code == 200
