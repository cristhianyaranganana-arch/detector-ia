"""Demo web interactiva de Detector IA (Streamlit).

Ejecutar:
    streamlit run demo/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Permite ejecutar la demo sin instalar el paquete (usa el código de src/).
try:
    from detector_ia import Detector
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from detector_ia import Detector

st.set_page_config(page_title="Detector IA", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    mode = st.selectbox(
        "Motor de análisis",
        ["auto", "heuristic"],
        help=("auto: usa el motor neuronal si `torch` y `transformers` están "
              "instalados; heuristic: solo estadístico (sin dependencias)."),
    )
    show_sentences = st.toggle("Mostrar análisis por frase", value=True)
    st.divider()
    st.caption(
        "**Motores disponibles**\n\n"
        "- Heurístico: burstiness, conectores, coloquialismos…\n"
        "- Neuronal: perplejidad GPT-2 (requiere `pip install "
        "\"detector-ia[neural]\"`)"
    )

# ---------------------------------------------------------------------------
# Página principal
# ---------------------------------------------------------------------------
st.title("🤖 Detector IA")
st.markdown(
    "Analiza un texto y mide la intensidad de las señales estilométricas "
    "asociadas a la redacción generada por IA (ChatGPT, Gemini, Claude…)."
)

example = st.session_state.get("example", "")
text = st.text_area(
    "Texto a analizar",
    value=example,
    height=250,
    placeholder="Pega aquí el texto que quieres analizar (mínimo 5 palabras)…",
)

with st.expander("💡 O prueba con un ejemplo"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ejemplo con estilo IA"):
            st.session_state["example"] = (
                "La inteligencia artificial ha transformado numerosos sectores de la "
                "sociedad en la última década. En primer lugar, ha optimizado procesos "
                "que antes requerían horas de trabajo manual. Además, permite analizar "
                "grandes volúmenes de datos en cuestión de segundos. Por otra parte, su "
                "implementación plantea desafíos éticos importantes. Por lo tanto, es "
                "fundamental establecer marcos regulatorios adecuados. En conclusión, "
                "la inteligencia artificial constituye una herramienta poderosa."
            )
            st.rerun()
    with col2:
        if st.button("Ejemplo con estilo humano"):
            st.session_state["example"] = (
                "Ayer fui al mercado con hambre y compré de todo, en serio. Mi abuela "
                "siempre dice «compra poco pero compra bien»... como si fuera tan "
                "fácil. Total que gasté cuarenta euros ¡en fruta! ¿Quién me avisa que "
                "las cerezas de junio están carísimas? Nadie. Para la próxima hago "
                "lista (y no me desvío ni un metro del presupuesto, jaja)."
            )
            st.rerun()

if not st.button("🔍 Analizar", type="primary"):
    st.info("👆 Escribe o pega un texto y pulsa **Analizar**.")
    st.stop()

if not text.strip():
    st.error("El texto está vacío.")
    st.stop()

detector = Detector(mode=mode)
try:
    result = detector.analyze(text, include_sentences=show_sentences)
except ValueError as exc:
    st.error(f"No se pudo analizar: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Resultado principal
# ---------------------------------------------------------------------------
if result.score >= 60:
    color = "🔴"
elif result.score >= 40:
    color = "🟡"
else:
    color = "🟢"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Intensidad IA", f"{result.score}/100")
col2.metric("Veredicto", f"{color} {result.verdict.split(' / ')[0]}")
col3.metric("Confianza", result.confidence.capitalize())
col4.metric("Motor", result.engine.split(" (")[0].capitalize())

st.progress(result.score / 100, text=f"Índice de intensidad de señales IA: "
                                     f"**{result.score}%**")

st.caption(
    f"Idioma: **{result.language}** · {result.word_count} palabras · "
    f"{result.sentence_count} frases · Motor: {result.engine}"
)

# ---------------------------------------------------------------------------
# Señales analizadas
# ---------------------------------------------------------------------------
st.subheader("📊 Señales analizadas")
st.caption("Cada barra es la intensidad de la señal hacia IA (0 = humano, 1 = IA).")

for feature in result.features:
    value_str = "—" if feature.value is None else f"{feature.value:g}"
    st.progress(
        feature.score,
        text=f"**{feature.name}** — valor: {value_str} · peso: "
             f"{feature.weight:.0%} · {feature.description}",
    )

# ---------------------------------------------------------------------------
# Análisis por frase
# ---------------------------------------------------------------------------
if show_sentences and result.sentences:
    st.subheader("📝 Análisis por frase")
    for sentence in result.sentences:
        if sentence.score >= 60:
            emoji = "🔴"
        elif sentence.score >= 40:
            emoji = "🟡"
        else:
            emoji = "🟢"
        flags = ", ".join(sentence.flags) if sentence.flags else "sin señales destacables"
        st.markdown(
            f"{emoji} `{sentence.score:>3}` {sentence.text}  \n"
            f"<sub>Señales: {flags}</sub>",
            unsafe_allow_html=True,
        )

st.warning(
    "⚠️ **Aviso**: ningún detector es 100% fiable. Usa el resultado como un "
    "indicio, nunca como prueba concluyente en contextos académicos, "
    "laborales o legales."
)
