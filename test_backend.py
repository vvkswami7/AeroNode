"""
Integration tests for the AeroNode backend API.

Run with:  pytest test_backend.py -v
Coverage:
  - GET  /api/state  — structure, defaults, post-telemetry population
  - POST /api/telemetry — success path, persistence, validation rejection
"""

import pytest
from fastapi.testclient import TestClient

from backend import app, venue_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_venue_state():
    """Clear the global venue_state before and after every test for isolation."""
    venue_state.clear()
    yield
    venue_state.clear()


@pytest.fixture(scope="module")
def client():
    """Shared TestClient that triggers the FastAPI lifespan once per module."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /api/state
# ---------------------------------------------------------------------------

class TestGetState:

    def test_returns_http_200(self, client):
        """GET /api/state should always return HTTP 200."""
        response = client.get("/api/state")
        assert response.status_code == 200

    def test_response_contains_required_keys(self, client):
        """Response body must contain both 'nodes' and 'active_command' keys."""
        data = client.get("/api/state").json()
        assert "nodes" in data
        assert "active_command" in data

    def test_nodes_empty_before_any_telemetry(self, client):
        """Nodes dict must be empty when no telemetry has been posted."""
        data = client.get("/api/state").json()
        assert data["nodes"] == {}

    def test_active_command_is_null_by_default(self, client):
        """active_command must be null when no AI surge has been triggered."""
        data = client.get("/api/state").json()
        assert data["active_command"] is None


# ---------------------------------------------------------------------------
# POST /api/telemetry
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "node_id": "gate_1",
    "node_type": "gate",
    "acoustic_density": 45,
    "rf_attenuation": 0.30,
    "timestamp": "2024-01-01T00:00:00+00:00",
}


class TestPostTelemetry:

    def test_returns_http_200_on_valid_payload(self, client):
        """POST /api/telemetry with a valid payload must return HTTP 200."""
        response = client.post("/api/telemetry", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_returns_success_status_body(self, client):
        """Response body must be exactly {'status': 'success'}."""
        response = client.post("/api/telemetry", json=VALID_PAYLOAD)
        assert response.json() == {"status": "success"}

    def test_telemetry_is_persisted_in_state(self, client):
        """Node data POSTed must be reflected in GET /api/state."""
        client.post("/api/telemetry", json=VALID_PAYLOAD)
        state = client.get("/api/state").json()
        assert "gate_1" in state["nodes"]
        assert state["nodes"]["gate_1"]["acoustic_density"] == 45
        assert state["nodes"]["gate_1"]["rf_attenuation"] == pytest.approx(0.30)

    def test_multiple_distinct_nodes_are_all_stored(self, client):
        """Multiple unique nodes POSTed must all appear in the state."""
        for i in range(1, 4):
            payload = {**VALID_PAYLOAD, "node_id": f"food_stall_{i}"}
            client.post("/api/telemetry", json=payload)
        state = client.get("/api/state").json()
        for i in range(1, 4):
            assert f"food_stall_{i}" in state["nodes"]

    def test_missing_required_fields_returns_422(self, client):
        """POST with missing required fields must return HTTP 422 Unprocessable Entity."""
        response = client.post("/api/telemetry", json={"node_id": "gate_1"})
        assert response.status_code == 422

    def test_invalid_node_id_characters_returns_422(self, client):
        """POST with a node_id containing illegal characters must return HTTP 422."""
        bad_payload = {**VALID_PAYLOAD, "node_id": "123!bad-id"}
        response = client.post("/api/telemetry", json=bad_payload)
        assert response.status_code == 422

    def test_wrong_data_type_for_acoustic_density_returns_422(self, client):
        """POST with acoustic_density as a string must return HTTP 422."""
        bad_payload = {**VALID_PAYLOAD, "acoustic_density": "high"}
        response = client.post("/api/telemetry", json=bad_payload)
        assert response.status_code == 422

    def test_node_data_can_be_updated(self, client):
        """Re-POSTing the same node_id must overwrite the previous reading."""
        client.post("/api/telemetry", json=VALID_PAYLOAD)
        updated = {**VALID_PAYLOAD, "acoustic_density": 88}
        client.post("/api/telemetry", json=updated)
        state = client.get("/api/state").json()
        assert state["nodes"]["gate_1"]["acoustic_density"] == 88
