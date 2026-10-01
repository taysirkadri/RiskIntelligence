import pytest
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from credit_risk.api import app

def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}
