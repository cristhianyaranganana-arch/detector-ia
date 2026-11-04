"""Tests del preprocesado: limpieza, segmentación y detección de idioma."""

from detector_ia.preprocessing import clean_text, detect_language, split_sentences, tokenize


class TestSplitSentences:
    def test_tres_frases(self):
        sentences = split_sentences("Hola. ¿Cómo estás? ¡Bien!")
        assert len(sentences) == 3

    def test_abreviatura_no_corta(self):
        sentences = split_sentences("El Dr. Pérez vino ayer. Luego se fue a casa.")
        assert len(sentences) == 2

    def test_texto_vacio(self):
        assert split_sentences("   ") == []

    def test_sin_puntuacion_final(self):
        assert split_sentences("una sola frase sin punto") == ["una sola frase sin punto"]

    def test_multiples_espacios(self):
        sentences = split_sentences("Hola.    Mundo.")
        assert sentences == ["Hola.", "Mundo."]


class TestTokenize:
    def test_minusculas_y_acentos(self):
        # Las letras unicode (è, ç, ñ…) se conservan dentro del token.
        assert tokenize("El Café était") == ["el", "café", "était"]

    def test_guiones(self):
        assert tokenize("teórico-práctico") == ["teórico-práctico"]

    def test_numeros(self):
        assert tokenize("hay 42 gatos") == ["hay", "42", "gatos"]


class TestCleanText:
    def test_normaliza_espacios(self):
        assert clean_text("hola   \n\t mundo") == "hola mundo"


class TestDetectLanguage:
    def test_espanol(self):
        assert detect_language("El perro corre rápido por el parque de la ciudad") == "es"

    def test_ingles(self):
        assert detect_language("The dog runs fast in the park of the city") == "en"

    def test_por_defecto_espanol(self):
        # Sin palabras funcionales claras, el default documentado es 'es'.
        assert detect_language("xyzzy quux blorb") == "es"
