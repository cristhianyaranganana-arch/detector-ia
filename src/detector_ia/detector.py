"""Orquestador principal: combina los motores y produce el veredicto final."""

import statistics
from dataclasses import asdict, dataclass, field

from detector_ia.config import (
    COLLOQUIAL_EN,
    COLLOQUIAL_ES,
    CONNECTIVES_EN,
    CONNECTIVES_ES,
    FORMAL_STARTERS_EN,
    FORMAL_STARTERS_ES,
    DetectorConfig,
    Feature,
)
from detector_ia.heuristic_engine import HeuristicEngine
from detector_ia.neural_engine import NeuralEngine
from detector_ia.preprocessing import split_sentences, tokenize

# (umbral exclusivo, veredicto) — el score va de 0 a 100.
_VERDICTS: list[tuple[int, str]] = [
    (20, "Muy probablemente humano"),
    (40, "Probablemente humano"),
    (60, "Incierto / texto mixto"),
    (80, "Probablemente generado por IA"),
    (101, "Muy probablemente generado por IA"),
]


@dataclass
class SentenceResult:
    """Análisis individual de una frase."""

    text: str
    score: int  # 0-100
    flags: list[str] = field(default_factory=list)


@dataclass
class DetectionResult:
    """Resultado completo del análisis de un texto."""

    score: int  # 0 = humano … 100 = IA
    verdict: str
    confidence: str  # baja | media | alta
    engine: str
    language: str
    word_count: int
    sentence_count: int
    features: list[Feature]
    sentences: list[SentenceResult]

    def to_dict(self) -> dict:
        """Versión serializable a JSON."""
        return asdict(self)


class Detector:
    """Punto de entrada principal del detector.

    Modes:
        - "auto":      motor neuronal si está disponible (híbrido); si no, heurístico.
        - "heuristic": solo estadístico, sin dependencias.
        - "neural":    solo perplejidad GPT-2 (requiere extras).
    """

    MODES = ("auto", "heuristic", "neural")

    def __init__(
        self,
        mode: str = "auto",
        model_name: str | None = None,
        config: DetectorConfig | None = None,
    ):
        if mode not in self.MODES:
            raise ValueError(f"Modo inválido: {mode!r}. Usa uno de {self.MODES}")
        self.mode = mode
        self.cfg = config or DetectorConfig()
        self._heuristic = HeuristicEngine(self.cfg)
        self._neural = NeuralEngine(model_name=model_name) if mode in ("auto", "neural") else None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def analyze(
        self,
        text: str,
        include_sentences: bool = True,
        language: str = "auto",
    ) -> DetectionResult:
        """Analiza un texto y devuelve puntuación, veredicto y señales."""
        if not text or not text.strip():
            raise ValueError("El texto está vacío.")
        words = tokenize(text)
        if len(words) < 5:
            raise ValueError("Texto demasiado corto: escribe al menos 5 palabras.")

        # Decidir motor.
        use_neural = self.mode == "neural" or (
            self.mode == "auto" and NeuralEngine.available()
        )
        if self.mode == "neural" and not NeuralEngine.available():
            raise RuntimeError(
                "El motor neuronal requiere extras: pip install \"detector-ia[neural]\" "
                "(o pip install torch transformers)"
            )

        # Motor heurístico (siempre disponible; también alimenta al híbrido).
        h_score, features, lang = self._heuristic.analyze(text, language)

        if use_neural and self._neural is not None:
            n_score, n_features = self._neural.analyze(text, lang)
            features = features + n_features
            if self.mode == "neural":
                final, engine = n_score, "neuronal (GPT-2)"
            else:
                final = 0.45 * h_score + 0.55 * n_score
                engine = "híbrido (heurístico + neuronal)"
        else:
            final, engine = h_score, "heurístico"

        score = round(max(0.0, min(1.0, final)) * 100)
        sentences = self._sentence_results(text, lang) if include_sentences else []
        sentence_count = len([s for s in split_sentences(text) if s.strip()])

        return DetectionResult(
            score=score,
            verdict=self._verdict(score),
            confidence=self._confidence(score, len(words)),
            engine=engine,
            language=lang,
            word_count=len(words),
            sentence_count=sentence_count,
            features=features,
            sentences=sentences,
        )

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    @staticmethod
    def _verdict(score: int) -> str:
        for threshold, name in _VERDICTS:
            if score < threshold:
                return name
        return _VERDICTS[-1][1]

    @staticmethod
    def _confidence(score: int, n_words: int) -> str:
        if n_words < 40:
            return "baja"
        distance = abs(score - 50)
        if distance >= 25:
            return "alta"
        if distance >= 10:
            return "media"
        return "baja"

    def _sentence_results(self, text: str, lang: str) -> list[SentenceResult]:
        """Puntuación por frase para la visualización del análisis."""
        sentences = [s for s in split_sentences(text) if s.strip()]
        lengths = [len(tokenize(s)) for s in sentences]
        median = statistics.median(lengths) if lengths else 0

        connectives = CONNECTIVES_ES if lang == "es" else CONNECTIVES_EN
        colloquials = COLLOQUIAL_ES if lang == "es" else COLLOQUIAL_EN
        starters = FORMAL_STARTERS_ES if lang == "es" else FORMAL_STARTERS_EN

        results = []
        for sentence, length in zip(sentences, lengths):
            score = 0.0
            flags = []
            lowered = sentence.lower()

            if any(conn in lowered for conn in connectives):
                score += 0.35
                flags.append("conectivo formal")
            if len(sentences) >= 3 and median and abs(length - median) / median < 0.25:
                score += 0.30
                flags.append("longitud uniforme")
            tokens = tokenize(sentence)
            if tokens and tokens[0] in starters:
                score += 0.15
                flags.append("inicio formal")
            has_human_punct = any(ch in sentence for ch in "!?¡¿…")
            has_colloquial = any(marker in lowered for marker in colloquials)
            if not has_human_punct and not has_colloquial:
                score += 0.20
                flags.append("registro formal/limpio")

            results.append(SentenceResult(
                text=sentence,
                score=round(min(1.0, score) * 100),
                flags=flags,
            ))
        return results
