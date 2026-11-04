"""Configuración, constantes lingüísticas y umbrales del detector.

Todos los umbrales estadísticos están centralizados aquí para poder
calibrarlos sin tocar la lógica de los motores.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Conectores discursivos formales: la redacción IA los abusa.
# ---------------------------------------------------------------------------
CONNECTIVES_ES = [
    "en primer lugar",
    "en segundo lugar",
    "por otra parte",
    "sin embargo",
    "no obstante",
    "además",
    "asimismo",
    "por lo tanto",
    "por consiguiente",
    "en consecuencia",
    "en conclusión",
    "en resumen",
    "en definitiva",
    "cabe destacar",
    "es importante",
    "es fundamental",
    "es crucial",
    "de hecho",
    "en otras palabras",
    "por último",
]

CONNECTIVES_EN = [
    "in the first place",
    "firstly",
    "secondly",
    "furthermore",
    "moreover",
    "however",
    "therefore",
    "consequently",
    "in addition",
    "additionally",
    "in conclusion",
    "in summary",
    "to summarize",
    "overall",
    "it is important",
    "in essence",
    "ultimately",
    "as a result",
    "on the other hand",
    "lastly",
]

# ---------------------------------------------------------------------------
# Marcadores coloquiales: señal de redacción humana (registro informal).
# ---------------------------------------------------------------------------
COLLOQUIAL_ES = [
    "o sea",
    "en plan",
    "jaja",
    "jeje",
    "jajaja",
    "xd",
    "tío",
    "tía",
    "wey",
    "neta",
    "órale",
    "chido",
    "chévere",
    "bacán",
    "guay",
    "flipar",
    "ya pues",
    "total que",
    "en serio",
    "qué va",
    "vaya vaya",
    "para la próxima",
]

COLLOQUIAL_EN = [
    "gonna",
    "wanna",
    "kinda",
    "sorta",
    "yeah",
    "yep",
    "nope",
    "lol",
    "haha",
    "btw",
    "omg",
    "dude",
    "guys",
    "you know",
    "i mean",
    "so anyway",
    "for real",
    "big time",
    "ok so",
]

# ---------------------------------------------------------------------------
# Palabras funcionales para detección de idioma (es / en).
# ---------------------------------------------------------------------------
STOPWORDS_ES = [
    "el", "la", "los", "las", "de", "que", "y", "en", "un", "una",
    "por", "con", "para", "como", "más", "pero", "se", "su", "al", "del",
]

STOPWORDS_EN = [
    "the", "of", "and", "to", "in", "is", "that", "for", "it", "with",
    "as", "was", "on", "are", "but", "his", "her", "from", "this", "have",
]

# Inicios de frase frecuentes en redacción formal/IA (para el análisis por frase).
FORMAL_STARTERS_ES = [
    "además", "asimismo", "por", "en", "este", "esta", "estos", "estas",
    "también", "sin", "dentro", "cabe", "es", "cuando", "el", "la",
]

FORMAL_STARTERS_EN = [
    "the", "this", "these", "those", "it", "there", "in", "on", "furthermore",
    "moreover", "however", "therefore", "overall", "thus", "consequently",
    "additionally", "firstly",
]

# Puntuación que delata redacción humana informal.
PUNCTUATION_HUMAN = "!?¡¿…—–"


@dataclass
class Feature:
    """Resultado de una señal individual del análisis."""

    name: str
    value: float | None
    score: float  # 0.0 = muy humano … 1.0 = muy IA
    weight: float
    description: str


@dataclass
class DetectorConfig:
    """Umbrales y pesos de calibración del motor heurístico."""

    # Burstiness (coef. de variación de la longitud de las frases)
    burstiness_center: float = 0.55  # por debajo → IA
    burstiness_span: float = 0.40

    # Conectores formales por frase
    connective_max: float = 0.60

    # Marcadores coloquiales por cada 100 palabras
    colloquial_max: float = 2.0

    # Marcas de puntuación "humana" por frase
    punct_max: float = 0.60

    # Longitud media de palabra (difiere por idioma)
    awl_center_es: float = 5.2
    awl_span_es: float = 1.4
    awl_center_en: float = 4.6
    awl_span_en: float = 1.2

    # Ganancias de las señales débiles
    starter_gain: float = 1.5
    trigram_gain: float = 2.5

    # Pesos de cada señal en la puntuación final (suman 1.0)
    weights: dict = field(
        default_factory=lambda: {
            "burstiness": 0.30,
            "connectives": 0.18,
            "colloquial": 0.15,
            "punctuation": 0.13,
            "starters": 0.09,
            "word_length": 0.08,
            "trigrams": 0.07,
        }
    )

    # Motor neuronal (perplejidad GPT-2)
    neural_model_es: str = "datificate/gpt2-small-spanish"
    neural_model_en: str = "gpt2"
    ppl_center: float = 30.0  # perplejidad por debajo de esto → IA
    ppl_width: float = 18.0
