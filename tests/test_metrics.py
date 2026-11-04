"""Tests de las métricas estadísticas y estilométricas."""

from detector_ia.metrics import (
    avg_word_length,
    burstiness,
    colloquial_density,
    connective_density,
    punctuation_humanity,
    repeated_starters,
    trigram_repetition,
    type_token_ratio,
)
from detector_ia.preprocessing import tokenize

UNIFORME = [
    "la casa roja tiene ventanas",
    "el perro grande corre rápido",
    "mi hermana compró pan hoy",
    "los niños juegan en el parque",
    "el coche azul es muy rápido",
]

VARIADO = [
    "no",
    "el perro grande del vecino corría cada mañana por el parque antes del desayuno",
    "quizás",
    "ayer mi hermana compró pan",
    "los niños, cansadísimos ya, se fueron a sus casas cuando empezó a llover",
]


class TestBurstiness:
    def test_texto_uniforme_da_valor_bajo(self):
        assert burstiness(UNIFORME) < 0.2

    def test_texto_variado_da_valor_alto(self):
        assert burstiness(VARIADO) > 0.4

    def test_una_sola_frase(self):
        assert burstiness(["solo una frase"]) == 0.0


class TestConnectiveDensity:
    def test_texto_con_conectores(self):
        texto = ("En primer lugar, hay que analizar los datos. Sin embargo, "
                 "existen limitaciones. En conclusión, el estudio es válido.")
        sentences = [texto]  # una "frase" con 3 conectores → densidad 3.0
        assert connective_density(sentences, "es") == 3.0

    def test_texto_sin_conectores(self):
        sentences = ["El gato duerme mucho.", "Los perros ladran fuerte."]
        assert connective_density(sentences, "es") == 0.0

    def test_ingles(self):
        sentences = ["Furthermore, the results are clear. However, more data is needed."]
        assert connective_density(sentences, "en") > 0


class TestRepeatedStarters:
    def test_arranques_repetidos(self):
        sentences = ["El gato duerme.", "El perro corre.", "El niño juega.", "El sol brilla."]
        assert repeated_starters(sentences) == 0.75  # 3 de 4 repiten "el"

    def test_arranques_distintos(self):
        sentences = ["El gato duerme.", "Los perros corren.", "Mi casa es grande."]
        assert repeated_starters(sentences) == 0.0

    def test_menos_de_tres_frases(self):
        assert repeated_starters(["El gato.", "El perro."]) == 0.0


class TestPunctuationHumanity:
    def test_puntuacion_expresiva(self):
        text = "¡Qué sorpresa! ¿En serio? No me lo creo... (increíble)"
        # 5 signos + 1 paréntesis = 6 en 4 frases → 1.5 por frase
        assert punctuation_humanity(text, 4) == 1.5

    def test_texto_formal(self):
        assert punctuation_humanity("La reunión fue productiva. Se aprobó el plan.", 2) == 0.0


class TestColloquialDensity:
    def test_texto_coloquial(self):
        text = "o sea, fue un desastre, jaja, en serio total que no volvemos"
        words = tokenize(text)
        density = colloquial_density(text, words, "es")
        assert density > 0

    def test_texto_formal_es_cero(self):
        text = "El análisis de los datos reveló tendencias significativas del mercado."
        words = tokenize(text)
        assert colloquial_density(text, words, "es") == 0.0


class TestAvgWordLength:
    def test_rango_razonable(self):
        words = tokenize("El análisis estadístico de los datos reveló conclusiones")
        assert 3.0 < avg_word_length(words) < 8.0

    def test_ignora_numeros(self):
        assert avg_word_length(["abc", "1234"]) == 3.0


class TestTypeTokenRatio:
    def test_sin_repeticiones(self):
        assert type_token_ratio(["a", "b", "c"]) == 1.0

    def test_con_repeticiones(self):
        assert type_token_ratio(["a", "a", "a"]) < 1.0


class TestTrigramRepetition:
    def test_texto_corto(self):
        assert trigram_repetition(["uno", "dos"]) == 0.0

    def test_texto_repetitivo(self):
        words = tokenize("el gato duerme el gato duerme el gato duerme")[:12]
        assert trigram_repetition(words) > 0.3
