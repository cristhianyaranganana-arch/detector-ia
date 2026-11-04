# 🤖 Detector IA

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Licencia](https://img.shields.io/badge/licencia-MIT-green)
![Motores](https://img.shields.io/badge/motores-heur%C3%ADstico%20%2B%20neuronal-8A2BE2)
![Dependencias núcleo](https://img.shields.io/badge/n%C3%BAcleo-0%20dependencias-success)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)

**Detector IA** analiza un texto y mide el **índice de intensidad de señales IA (0-100)** asociadas a la redacción generada por IA (ChatGPT, Gemini, Claude, Llama…). Funciona en **español e inglés**, sin dependencias obligatorias, con **API REST**, **demo web** y **CLI**.

> ⚠️ **Aviso honesto**: ningún detector de IA es 100% fiable. Usa los resultados como un *indicio*, nunca como prueba concluyente en contextos académicos, laborales o legales.

---

## ✨ Características

- 🚀 **Núcleo sin dependencias**: solo librería estándar de Python, funciona en cualquier parte.
- 🧠 **Dos motores combinables**:
  - **Heurístico**: burstiness, conectores discursivos, coloquialismos, puntuación expresiva, repetición de arranques/trigramas…
  - **Neuronal** *(opcional)*: perplejidad con GPT-2 (incluye modelo en español `datificate/gpt2-small-spanish`).
- 🌎 **Español e inglés** con detección automática de idioma.
- 📝 **Análisis por frase** con señales explicadas (transparencia, no caja negra).
- 🌐 **API REST** lista para producción (FastAPI + Docker).
- 🖥️ **Demo web** interactiva (Streamlit).
- ⌨️ **CLI** con informe visual y salida JSON.

## 🧠 Cómo funciona

```
                       ┌────────────────────────┐
   texto ─► preprocesado ─► motor heurístico ───┤
   (frases, tokens,      (7 señales estadísticas) ├─► fusión ─► score 0-100 ─► veredicto
   idioma es/en)         └────────────────────────┘   0.45/0.55
                          motor neuronal (opcional) ──┘
                          (perplejidad GPT-2)
```

**Señales del motor heurístico** (peso entre paréntesis):

| Señal | Qué mide | Peso |
|---|---|---|
| **Burstiness** | Coef. de variación de la longitud de frases: la IA es uniforme, el humano errático | 30% |
| **Conectores formales** | "en primer lugar", "furthermore", "por lo tanto"… por frase | 18% |
| **Coloquialismos** | "o sea", "jaja", "lol"… por 100 palabras (ausencia → IA) | 15% |
| **Puntuación expresiva** | ¡ ¿ … — ( ) por frase (ausencia → IA) | 13% |
| **Arranques repetidos** | Frases que empiezan con la misma palabra | 9% |
| **Longitud de palabra** | Vocabulario formal/largo típico de la IA | 8% |
| **Trigramas repetidos** | Estructuras de 3 palabras recicladas | 7% |

**Motor neuronal**: la perplejidad mide cuánto "sorprende" un texto a GPT-2. Texto generado por IA = muy predecible = perplejidad baja. Además se mide la *uniformidad* de la perplejidad entre frases.

**Interpretación del score**:

| Score | Veredicto |
|---|---|
| 0–19 | Muy probablemente humano |
| 20–39 | Probablemente humano |
| 40–59 | Incierto / texto mixto |
| 60–79 | Probablemente generado por IA |
| 80–100 | Muy probablemente generado por IA |

> ℹ️ El score es un **índice de intensidad**: mide qué tan marcadas están en promedio las señales estilométricas asociadas a IA (uniformidad, conectores formulaicos, arranques repetidos…). **No** es la proporción de texto generado por IA ni una probabilidad calibrada.

## 📦 Instalación

```bash
git clone https://github.com/TU_USUARIO/detector-ia.git
cd detector-ia
pip install -e .              # núcleo (0 dependencias)
pip install -e ".[api]"       # + API REST (FastAPI)
pip install -e ".[demo]"      # + demo web (Streamlit)
pip install -e ".[neural]"    # + motor neuronal (torch + transformers, ~2 GB)
pip install -e ".[all]"       # todo
```

## 🚀 Uso

### Python

```python
from detector_ia import Detector

detector = Detector(mode="auto")  # auto | heuristic | neural
resultado = detector.analyze("Texto a analizar...")

print(resultado.score)      # 0-100
print(resultado.verdict)    # "Probablemente generado por IA"
print(resultado.confidence) # baja | media | alta
for senal in resultado.features:
    print(f"{senal.name}: {senal.score:.2f}")
```

### CLI

```bash
detector-ia "La inteligencia artificial ha transformado numerosos sectores..."
detector-ia --file ensayo.txt
detector-ia --mode heuristic --json "Texto..."   # salida JSON
python -m detector_ia "Texto..."                 # sin instalar entry point
```

### API REST

```bash
uvicorn api.app:app --reload    # http://localhost:8000/docs
```

```bash
curl -X POST http://localhost:8000/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "La inteligencia artificial ha transformado..."}'
```

| Endpoint | Método | Descripción |
|---|---|---|
| `/health` | GET | Estado del servicio y motores disponibles |
| `/v1/analyze` | POST | Analiza un texto (score, veredicto, señales, frases) |
| `/v1/analyze/batch` | POST | Analiza hasta 50 textos |
| `/docs` | GET | Documentación interactiva (Swagger) |

### Demo web

```bash
streamlit run demo/app.py
```

### Docker

```bash
docker build -t detector-ia .
docker run -p 8000:8000 detector-ia
```

## 🔬 Evaluación

```bash
python scripts/evaluate.py                      # sobre data/samples.json
python scripts/download_hc3.py                  # corpus real HC3 (opcional)
```

Las muestras incluidas en `data/samples.json` son **ilustrativas** (exhiben las señales de cada estilo). Para una evaluación realista, descarga el corpus público [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) con el script incluido.

## ⚠️ Limitaciones y uso ético

- Los detectores producen **falsos positivos**: textos humanos formales (académicos, jurídicos) pueden puntuar alto.
- **Textos editados** (IA + revisión humana, o humano + corrector) dan resultados intermedios por diseño.
- El motor heurístico está calibrado para **ensayos y artículos**; es menos fiable en poesía, código, chats o textos muy cortos.
- La perplejidad depende del modelo generador: texto de modelos modernos (GPT-4+) es más difícil de detectar que el de GPT-2/3.
- **No uses este tool para acusar a nadie.** Úsalo para editar, comparar versiones o iniciar conversaciones.

## 🗺️ Hoja de ruta

- [x] Motor heurístico (español + inglés)
- [x] Motor neuronal con perplejidad GPT-2
- [x] API REST + Docker
- [x] Demo Streamlit
- [x] CLI con salida JSON
- [ ] Clasificador fino (RoBERTa/DeBERTa entrenado con HC3)
- [ ] Detección por párrafo con mapa de calor
- [ ] Soporte de más idiomas (pt, fr, de)
- [ ] Widget para navegadores

## 🤝 Contribuir

Las contribuciones son bienvenidas: lee [CONTRIBUTING.md](CONTRIBUTING.md). Ideas especialmente útiles: más idiomas, mejor calibración con HC3, y tests adversariales.

## 📄 Licencia

[MIT](LICENSE) — úsalo, cópialo, mejóralo.

---

## 🚀 Publicar este proyecto en GitHub

```bash
cd detector-ia
git init -b main
git add -A
git commit -m "feat: versión inicial de Detector IA"
git remote add origin https://github.com/TU_USUARIO/detector-ia.git
git push -u origin main
```

> Sustituye `TU_USUARIO` por tu usuario real de GitHub (también en los badges de arriba y en `pyproject.toml`).
