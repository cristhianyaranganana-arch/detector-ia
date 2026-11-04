"""Tests de integración del Detector (motor heurístico)."""

import json
from pathlib import Path

import pytest

from detector_ia import DetectionResult, Detector

SAMPLES_PATH = Path(__file__).resolve().parents[1] / "data" / "samples.json"
SAMPLES = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))["samples"]

AI_TEXT = SAMPLES[0]["text"]
HUMAN_TEXT = next(s["text"] for s in SAMPLES if s["id"] == "es-human-1")


class TestDetectorBasico:
    def test_score_en_rango(self):
        result = Detector(mode="heuristic").analyze(AI_TEXT)
        assert 0 <= result.score <= 100

    def test_veredicto_valido(self):
        result = Detector(mode="heuristic").analyze(AI_TEXT)
        validos = {
            "Muy probablemente humano",
            "Probablemente humano",
            "Incierto / texto mixto",
            "Probablemente generado por IA",
            "Muy probablemente generado por IA",
        }
        assert result.verdict in validos

    def test_texto_vacio_lanza_error(self):
        with pytest.raises(ValueError):
            Detector(mode="heuristic").analyze("   ")

    def test_texto_cortisimo_lanza_error(self):
        with pytest.raises(ValueError):
            Detector(mode="heuristic").analyze("hola")

    def test_modo_invalido(self):
        with pytest.raises(ValueError):
            Detector(mode="cuántico")

    def test_resultado_serializable(self):
        result = Detector(mode="heuristic").analyze(AI_TEXT)
        data = result.to_dict()
        json.dumps(data, ensure_ascii=False)  # no debe lanzar
        assert isinstance(data["score"], int)

    def test_confianza_texto_corto_es_baja(self):
        result = Detector(mode="heuristic").analyze(
            "El gato duerme mucho hoy en casa. Los perros ladran."
        )
        assert result.confidence == "baja"

    def test_es_instancia_de_resultado(self):
        result = Detector(mode="heuristic").analyze(AI_TEXT)
        assert isinstance(result, DetectionResult)
        assert result.language == "es"
        assert result.word_count > 20
        assert result.sentence_count >= 3
        assert len(result.sentences) == result.sentence_count
        assert len(result.features) == 7


class TestDiscriminacion:
    """El detector debe puntuar más alto el texto estilo IA que el humano."""

    def test_ia_mayor_que_humano(self):
        detector = Detector(mode="heuristic")
        ai = detector.analyze(AI_TEXT)
        human = detector.analyze(HUMAN_TEXT)
        assert ai.score > human.score
        assert ai.score >= 60
        assert human.score <= 40

    def test_separacion_sobre_muestras(self):
        detector = Detector(mode="heuristic")
        scores = {
            s["id"]: detector.analyze(s["text"], include_sentences=False).score
            for s in SAMPLES
        }
        ai_scores = [v for k, v in scores.items() if "-ai-" in k]
        human_scores = [v for k, v in scores.items() if "-human-" in k]
        # Las muestras ilustrativas deben separarse con margen amplio.
        assert min(ai_scores) > max(human_scores)
        # Exactitud >= 11/12 con umbral 50.
        correctos = sum(1 for k, v in scores.items() if (v >= 50) == ("-ai-" in k))
        assert correctos >= 11


class TestAnalisisPorFrase:
    def test_frases_con_puntuacion_y_flags(self):
        result = Detector(mode="heuristic").analyze(AI_TEXT)
        assert all(0 <= s.score <= 100 for s in result.sentences)
        assert any("conectivo formal" in s.flags for s in result.sentences)

    def test_sin_frases_si_se_pide(self):
        result = Detector(mode="heuristic").analyze(AI_TEXT, include_sentences=False)
        assert result.sentences == []
