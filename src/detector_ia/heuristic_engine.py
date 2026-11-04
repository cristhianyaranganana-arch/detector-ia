"""Motor heurístico: rasgos estadísticos y estilométricos del texto.

No requiere dependencias externas. Combina siete señales ponderadas en una
puntuación de "IA-idad" entre 0 (humano) y 1 (IA).
"""

from detector_ia.config import DetectorConfig, Feature
from detector_ia.metrics import (
    avg_word_length,
    burstiness,
    colloquial_density,
    connective_density,
    punctuation_humanity,
    repeated_starters,
    trigram_repetition,
)
from detector_ia.preprocessing import detect_language, split_sentences, tokenize


def _clamp01(x: float) -> float:
    """Acota un valor al rango [0, 1]."""
    return max(0.0, min(1.0, x))


class HeuristicEngine:
    """Analiza el texto con métricas lingüísticas clásicas (estilo GPTZero)."""

    def __init__(self, config: DetectorConfig | None = None):
        self.cfg = config or DetectorConfig()

    def analyze(self, text: str, lang: str = "auto") -> tuple[float, list[Feature], str]:
        """Devuelve (puntuación 0-1, señales, idioma detectado)."""
        if lang == "auto":
            lang = detect_language(text)

        sentences = [s for s in split_sentences(text) if s.strip()]
        words = tokenize(text)
        n_sents, n_words = len(sentences), len(words)
        w = self.cfg.weights

        features: list[Feature] = []

        # 1. Burstiness: frases de longitud uniforme → IA.
        cv = burstiness(sentences)
        if n_sents >= 3:
            b_score = _clamp01((self.cfg.burstiness_center - cv) / self.cfg.burstiness_span)
        else:
            b_score = 0.5  # neutral: demasiadas pocas frases
        features.append(Feature(
            name="Uniformidad de las frases (burstiness)",
            value=round(cv, 3),
            score=b_score,
            weight=w["burstiness"],
            description="Valores bajos = redacción homogénea, típica de la IA",
        ))

        # 2. Conectores formales por frase.
        c_den = connective_density(sentences, lang)
        c_score = _clamp01(c_den / self.cfg.connective_max) if n_sents else 0.5
        features.append(Feature(
            name="Conectores discursivos formales",
            value=round(c_den, 3),
            score=c_score,
            weight=w["connectives"],
            description="'en primer lugar', 'sin embargo', 'furthermore'… propios de la IA",
        ))

        # 3. Marcadores coloquiales por 100 palabras (ausencia → IA).
        q_den = colloquial_density(text, words, lang)
        if n_words >= 15:
            q_score = _clamp01(1 - q_den / self.cfg.colloquial_max)
        else:
            q_score = 0.5
        features.append(Feature(
            name="Marcadores coloquiales",
            value=round(q_den, 2),
            score=q_score,
            weight=w["colloquial"],
            description="'o sea', 'jaja', 'lol'… delatan redacción humana",
        ))

        # 4. Puntuación expresiva (ausencia → IA).
        p_den = punctuation_humanity(text, n_sents)
        p_score = _clamp01(1 - p_den / self.cfg.punct_max) if n_sents else 0.5
        features.append(Feature(
            name="Puntuación expresiva",
            value=round(p_den, 3),
            score=p_score,
            weight=w["punctuation"],
            description="¡ ¿ … — ( ) abundan más en texto humano",
        ))

        # 5. Arranques de frase repetidos.
        r_start = repeated_starters(sentences)
        s_score = _clamp01(r_start * self.cfg.starter_gain) if n_sents >= 3 else 0.5
        features.append(Feature(
            name="Arranques de frase repetidos",
            value=round(r_start, 3),
            score=s_score,
            weight=w["starters"],
            description="La IA reutiliza inicios de frase ('El…', 'This…')",
        ))

        # 6. Longitud media de palabra (registro formal).
        awl = avg_word_length(words)
        if lang == "en":
            a_score = _clamp01((awl - self.cfg.awl_center_en) / self.cfg.awl_span_en)
        else:
            a_score = _clamp01((awl - self.cfg.awl_center_es) / self.cfg.awl_span_es)
        features.append(Feature(
            name="Longitud media de palabra",
            value=round(awl, 2),
            score=a_score,
            weight=w["word_length"],
            description="Vocabulario formal/largo es más frecuente en la IA",
        ))

        # 7. Trigramas repetidos (estructuras recicladas).
        tri = trigram_repetition(words)
        t_score = _clamp01(tri * self.cfg.trigram_gain) if n_words >= 25 else 0.5
        features.append(Feature(
            name="Trigramas repetidos",
            value=round(tri, 4),
            score=t_score,
            weight=w["trigrams"],
            description="Repetición de estructuras de 3 palabras",
        ))

        total = sum(f.score * f.weight for f in features)
        return _clamp01(total), features, lang
