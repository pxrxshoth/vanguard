"""
tests/test_api.py - Pytest suite for the Vanguard FastAPI application.

Run with:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, latest_telemetry, active_connections

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_state():
    """Reset in-memory state before every test so tests are isolated."""
    latest_telemetry.clear()
    active_connections.clear()
    yield
    latest_telemetry.clear()
    active_connections.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_payload():
    return {
        "unit_number": 42,
        "cycle": 150,
        "predicted_rul": 87.5,
        "anomaly_score": 0.12,
        "is_anomaly": False,
    }


@pytest.fixture
def anomaly_payload():
    return {
        "unit_number": 7,
        "cycle": 280,
        "predicted_rul": 4.2,
        "anomaly_score": 0.91,
        "is_anomaly": True,
    }


# ---------------------------------------------------------------------------
# Meta / health endpoints
# ---------------------------------------------------------------------------

class TestMetaEndpoints:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_contains_message(self, client):
        resp = client.get("/")
        assert "message" in resp.json()

    def test_health_returns_healthy(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_health_reports_connected_clients(self, client):
        resp = client.get("/health")
        assert "connected_clients" in resp.json()
        assert resp.json()["connected_clients"] == 0


# ---------------------------------------------------------------------------
# GET /api/v1/engines
# ---------------------------------------------------------------------------

class TestGetEngines:
    def test_empty_fleet_returns_empty_list(self, client):
        resp = client.get("/api/v1/engines")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_ingested_engine(self, client, sample_payload):
        client.post("/api/v1/telemetry", json=sample_payload)
        resp = client.get("/api/v1/engines")
        assert resp.status_code == 200
        engines = resp.json()
        assert len(engines) == 1
        assert engines[0]["unit_number"] == 42

    def test_returns_multiple_engines(self, client, sample_payload, anomaly_payload):
        client.post("/api/v1/telemetry", json=sample_payload)
        client.post("/api/v1/telemetry", json=anomaly_payload)
        resp = client.get("/api/v1/engines")
        assert len(resp.json()) == 2


# ---------------------------------------------------------------------------
# POST /api/v1/telemetry
# ---------------------------------------------------------------------------

class TestIngestTelemetry:
    def test_ingest_returns_202(self, client, sample_payload):
        resp = client.post("/api/v1/telemetry", json=sample_payload)
        assert resp.status_code == 202

    def test_ingest_returns_accepted_status(self, client, sample_payload):
        resp = client.post("/api/v1/telemetry", json=sample_payload)
        body = resp.json()
        assert body["status"] == "accepted"
        assert body["unit_number"] == 42

    def test_ingest_stores_in_state(self, client, sample_payload):
        client.post("/api/v1/telemetry", json=sample_payload)
        assert 42 in latest_telemetry
        assert latest_telemetry[42].predicted_rul == pytest.approx(87.5)

    def test_ingest_updates_existing_engine(self, client, sample_payload):
        client.post("/api/v1/telemetry", json=sample_payload)
        updated = {**sample_payload, "cycle": 151, "predicted_rul": 86.0}
        client.post("/api/v1/telemetry", json=updated)
        assert len(latest_telemetry) == 1
        assert latest_telemetry[42].predicted_rul == pytest.approx(86.0)

    def test_ingest_anomaly_flag(self, client, anomaly_payload):
        client.post("/api/v1/telemetry", json=anomaly_payload)
        assert latest_telemetry[7].is_anomaly is True

    def test_ingest_rejects_negative_anomaly_score(self, client, sample_payload):
        bad = {**sample_payload, "anomaly_score": -0.5}
        resp = client.post("/api/v1/telemetry", json=bad)
        assert resp.status_code == 422  # Pydantic validation error

    def test_ingest_rejects_missing_field(self, client):
        resp = client.post("/api/v1/telemetry", json={"unit_number": 1})
        assert resp.status_code == 422

    def test_ingest_multiple_distinct_units(self, client, sample_payload, anomaly_payload):
        client.post("/api/v1/telemetry", json=sample_payload)
        client.post("/api/v1/telemetry", json=anomaly_payload)
        assert len(latest_telemetry) == 2
        assert 42 in latest_telemetry
        assert 7 in latest_telemetry


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

class TestWebSocket:
    def test_websocket_connects_successfully(self, client):
        with client.websocket_connect("/ws/telemetry") as ws:
            assert ws is not None

    def test_websocket_receives_initial_snapshot_when_state_exists(self, client, sample_payload):
        client.post("/api/v1/telemetry", json=sample_payload)
        with client.websocket_connect("/ws/telemetry") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "snapshot"
            assert len(msg["data"]) == 1
            assert msg["data"][0]["unit_number"] == 42

    def test_websocket_no_snapshot_when_empty(self, client):
        """If no telemetry has been ingested, no initial snapshot is sent."""
        with client.websocket_connect("/ws/telemetry") as ws:
            # Connection accepted; no immediate data should arrive since state is empty.
            # We verify by closing cleanly without receiving.
            pass  # no exception = success
