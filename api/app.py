"""API REST de Detector IA (FastAPI).

Ejecutar en desarrollo:
    uvicorn api.app:app --reload

Documentación interactiva:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from detector_ia import __version__
from detector_ia.detector import Detector
from detector_ia.neural_engine import NeuralEngine

app = FastAPI(
    title="Detector IA API",
    description=(
        "API para detectar si un texto fue redactado por inteligencia artificial. "
        "Combina un motor heurístico (burstiness, conectores, coloquialismos) con un "
        "motor neuronal opcional basado en perplejidad GPT-2."
    ),
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia compartida: en modo "auto" usará el motor neuronal solo si hay extras.
detector = Detector(mode="auto")


# ---------------------------------------------------------------------------
# Modelos de petición / respuesta
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000, description="Texto a analizar")
    mode: str = Field("auto", description="Motor: auto | heuristic | neural")
    language: str = Field("auto", description="Idioma: es | en | auto")
    include_sentences: bool = Field(True, description="Incluir análisis por frase")


class FeatureOut(BaseModel):
    name: str
    value: float | None
    score: float
    weight: float
    description: str


class SentenceOut(BaseModel):
    text: str
    score: int
    flags: list[str]


class AnalyzeResponse(BaseModel):
    score: int
    verdict: str
    confidence: str
    engine: str
    language: str
    word_count: int
    sentence_count: int
    features: list[FeatureOut]
    sentences: list[SentenceOut]


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)
    mode: str = "auto"
    language: str = "auto"


class BatchResponse(BaseModel):
    results: list[AnalyzeResponse]


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["meta"])
def health():
    """Estado del servicio y motores disponibles."""
    return {
        "status": "ok",
        "version": __version__,
        "neural_available": NeuralEngine.available(),
    }


@app.post("/v1/analyze", response_model=AnalyzeResponse, tags=["análisis"])
def analyze(request: AnalyzeRequest):
    """Analiza un texto y devuelve la puntuación de IA (0-100) y sus señales."""
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="El texto está vacío.")
    try:
        result = Detector(mode=request.mode).analyze(
            request.text,
            include_sentences=request.include_sentences,
            language=request.language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AnalyzeResponse(**result.to_dict())


@app.post("/v1/analyze/batch", response_model=BatchResponse, tags=["análisis"])
def analyze_batch(request: BatchRequest):
    """Analiza hasta 50 textos en una sola llamada."""
    for i, text in enumerate(request.texts):
        if not text.strip():
            raise HTTPException(
                status_code=422, detail=f"El texto en el índice {i} está vacío."
            )
    detector_local = Detector(mode=request.mode)
    results = []
    for text in request.texts:
        try:
            result = detector_local.analyze(text, include_sentences=False,
                                            language=request.language)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        results.append(AnalyzeResponse(**result.to_dict()))
    return BatchResponse(results=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
