"""Métricas estadísticas y estilométricas para detectar redacción IA.

Todas las funciones son puras y sin dependencias externas.
"""

import math

from detector_ia.config import (
    COLLOQUIAL_EN,
    COLLOQUIAL_ES,
    CONNECTIVES_EN,
    CONNECTIVES_ES,
    PUNCTUATION_HUMAN,
)
from detector_ia.preprocessing import tokenize


def burstiness(sentences: list[str]) -> float:
    """Coeficiente de variación de la longitud de las frases.

    La redacción humana alterna frases largas y cortas (valor alto);
    la IA produce frases de longitud homogénea (valor bajo).
    """
    lengths = [len(tokenize(s)) for s in sentences]
    if len(lengths) < 2:
        return 0.0
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    return math.sqrt(variance) / mean


def connective_density(sentences: list[str], lang: str = "es") -> float:
    """Conectores discursivos formales por frase.

    Los LLM abusan de 'en primer lugar', 'furthermore', 'por lo tanto', etc.
    """
    connectives = CONNECTIVES_ES if lang == "es" else CONNECTIVES_EN
    if not sentences:
        return 0.0
    hits = sum(
        sentence.lower().count(conn)
        for sentence in sentences
        for conn in connectives
    )
    return hits / len(sentences)


def repeated_starters(sentences: list[str]) -> float:
    """Proporción de frases que repiten una palabra inicial ya usada.

    0.0 = todas las frases empiezan con palabras distintas (humano);
    0.6+ = arranques repetitivos (típico de la IA).
    """
    if len(sentences) < 3:
        return 0.0
    starters = []
    for sentence in sentences:
        tokens = tokenize(sentence)
        starters.append(tokens[0] if tokens else "")
    return 1 - len(set(starters)) / len(starters)


def punctuation_humanity(text: str, n_sentences: int) -> float:
    """Marcas de puntuación informal por frase (¡ ¿ … — ! ? paréntesis).

    Los textos humanos tienden a usar más puntuación expresiva.
    """
    if n_sentences == 0:
        return 0.0
    text = text.replace("...", "…")
    hits = sum(text.count(ch) for ch in PUNCTUATION_HUMAN)
    hits += text.count("(")
    return hits / n_sentences


def colloquial_density(text: str, words: list[str], lang: str = "es") -> float:
    """Marcadores coloquiales por cada 100 palabras ('o sea', 'lol', 'jaja')."""
    markers = COLLOQUIAL_ES if lang == "es" else COLLOQUIAL_EN
    if not words:
        return 0.0
    lowered = text.lower()
    hits = sum(lowered.count(marker) for marker in markers)
    return hits / len(words) * 100


def avg_word_length(words: list[str]) -> float:
    """Longitud media de las palabras alfabéticas (registro formal → más larga)."""
    alpha = [w for w in words if any(c.isalpha() for c in w)]
    if not alpha:
        return 0.0
    return sum(len(w) for w in alpha) / len(alpha)


def type_token_ratio(words: list[str]) -> float:
    """Riqueza léxica: tipos distintos / tokens totales."""
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def trigram_repetition(words: list[str]) -> float:
    """Proporción de trigramas repetidos (estructuras recicladas por la IA)."""
    if len(words) < 6:
        return 0.0
    trigrams = list(zip(words, words[1:], words[2:]))
    if not trigrams:
        return 0.0
    return 1 - len(set(trigrams)) / len(trigrams)
