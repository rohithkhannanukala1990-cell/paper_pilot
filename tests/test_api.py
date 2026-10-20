"""Offline FastAPI tests using FakePilot via dependency_overrides."""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from api import app, get_pilot
from tests.conftest import FakePilot


@pytest.fixture
def client() -> TestClient:
    pilot = FakePilot()
    app.dependency_overrides[get_pilot] = lambda: pilot
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_upload_rejects_non_pdf(client: TestClient) -> None:
    response = client.post(
        "/papers",
        files={"file": ("notes.txt", BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400


def test_ask_rejects_short_question(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 422


def test_compare_rejects_one_paper(client: TestClient) -> None:
    response = client.post(
        "/compare",
        json={"paper_ids": ["p1"], "aspects": ["method"]},
    )
    assert response.status_code == 422


def test_ask_maps_llm_error_to_503(client: TestClient) -> None:
    response = client.post(
        "/ask",
        json={"question": "FAIL_LLM please"},
    )
    assert response.status_code == 503
    assert "detail" in response.json()
