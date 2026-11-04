"""Preprocesado de texto: limpieza, segmentación en frases y tokenización.

Funciona solo con la librería estándar (sin nltk ni dependencias externas).
"""

import re

from detector_ia.config import STOPWORDS_EN, STOPWORDS_ES

# Abreviaturas cuyo punto final NO debe interpretarse como fin de frase.
_ABBREVIATIONS = [
    "dr.", "dra.", "sr.", "sra.", "etc.", "vs.", "ej.", "aprox.",
    "inc.", "s.a.", "pág.", "fig.",
]
_ABBR_RE = re.compile("|".join(re.escape(a) for a in _ABBREVIATIONS), re.IGNORECASE)

# Tokens: letras unicode (acentos, ñ, è, ç…), números, guiones y apóstrofes.
# [^\W_] equivale a \w sin el guion bajo → letras y dígitos unicode.
_WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*")


def clean_text(text: str) -> str:
    """Normaliza espacios y caracteres de control preservando la puntuación."""
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\x00", " ")).strip()


def split_sentences(text: str) -> list[str]:
    """Divide el texto en frases usando puntuación terminal.

    Protege abreviaturas comunes en español para evitar cortes falsos.
    """
    text = clean_text(text)
    if not text:
        return []

    # Proteger abreviaturas (sin distinguir mayúsculas) quitando su punto:
    # "Dr." → "«ABDr»". Se restaura al final, conservando el caso original.
    text = _ABBR_RE.sub(lambda m: f"«AB{m.group(0)[:-1]}»", text)

    parts = re.split(r"(?<=[.!?…])\s+", text)

    sentences = []
    for part in parts:
        part = re.sub(r"«AB([^»]*)»", r"\1.", part.strip())
        if part:
            sentences.append(part)
    return sentences


def tokenize(text: str) -> list[str]:
    """Tokeniza a palabras en minúsculas (orden de aparición)."""
    return _WORD_RE.findall(text.lower())


def detect_language(text: str) -> str:
    """Detecta 'es' o 'en' contando palabras funcionales. Por defecto: 'es'."""
    words = set(tokenize(text))
    es_hits = len(words & set(STOPWORDS_ES))
    en_hits = len(words & set(STOPWORDS_EN))
    if en_hits > es_hits:
        return "en"
    return "es"
