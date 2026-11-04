"""Tests de la API REST (FastAPI) con TestClient."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402

client = TestClient(app)

AI_TEXT = (
    "La inteligencia artificial ha transformado numerosos sectores. En primer lugar, "
    "ha optimizado procesos manuales. Además, permite analizar grandes volúmenes de "
    "datos. Por lo tanto, es fundamental establecer marcos regulatorios adecuados. "
    "En conclusión, constituye una herramienta poderosa."
)


class TestHealth:
    def test_health_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert isinstance(body["neural_available"], bool)

    def test_root_redirige_a_docs(self):
        response = client.get("/", follow_redirects=True)
        assert response.status_code == 200


class TestAnalyze:
    def test_analisis_correcto(self):
        response = client.post("/v1/analyze", json={"text": AI_TEXT})
        assert response.status_code == 200
        body = response.json()
        assert 0 <= body["score"] <= 100
        assert body["verdict"]
        assert body["engine"]
        assert body["language"] == "es"
        assert body["word_count"] > 10
        assert len(body["features"]) >= 7
        assert len(body["sentences"]) >= 3

    def test_texto_vacio_422(self):
        response = client.post("/v1/analyze", json={"text": ""})
        assert response.status_code == 422

    def test_solo_espacios_422(self):
        response = client.post("/v1/analyze", json={"text": "     "})
        assert response.status_code == 422

    def test_texto_cortisimo_422(self):
        response = client.post("/v1/analyze", json={"text": "hola"})
        assert response.status_code == 422

    def test_sin_frases(self):
        response = client.post(
            "/v1/analyze", json={"text": AI_TEXT, "include_sentences": False}
        )
        assert response.status_code == 200
        assert response.json()["sentences"] == []

    def test_modo_invalido_422(self):
        response = client.post("/v1/analyze", json={"text": AI_TEXT, "mode": "x"})
        assert response.status_code == 422


class TestBatch:
    def test_batch_correcto(self):
        response = client.post(
            "/v1/analyze/batch",
            json={"texts": [AI_TEXT, "Ayer compré pan ¡baratísimo! Y de paso saludé "
                                    "al panadero del barrio, que ya me conoce."]},
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 2
        scores = [r["score"] for r in results]
        assert scores[0] > scores[1]  # estilo IA > estilo humano

    def test_batch_con_vacio_422(self):
        response = client.post("/v1/analyze/batch", json={"texts": [AI_TEXT, "  "]})
        assert response.status_code == 422
