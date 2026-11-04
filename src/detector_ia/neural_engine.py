"""Motor neuronal: perplejidad del texto con modelos de lenguaje (GPT-2).

La perplejidad mide cuánto "le sorprende" al modelo un texto: los textos
generados por IA son muy predecibles (perplejidad baja) y los humanos
tienden a ser más erráticos (perplejidad alta).

Requiere extras opcionales:
    pip install "detector-ia[neural]"   # o: pip install torch transformers
"""

import math

from detector_ia.config import DetectorConfig, Feature
from detector_ia.preprocessing import split_sentences, tokenize


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class NeuralEngine:
    """Calcula perplejidad global y por frase con un modelo causal (GPT-2)."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Disponibilidad (importación perezosa de torch/transformers)
    # ------------------------------------------------------------------
    @staticmethod
    def available() -> bool:
        """True si `torch` y `transformers` están instalados."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = self.model_name or "gpt2"
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name)
        self._model.eval()
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device)

    # ------------------------------------------------------------------
    # Perplejidad
    # ------------------------------------------------------------------
    def perplexity(self, text: str) -> float | None:
        """Perplejidad del texto (None si hay menos de 2 tokens)."""
        self._load()
        import torch

        enc = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        ids = enc["input_ids"]
        seq_len = ids.shape[1]
        if seq_len < 2:
            return None

        max_pos = getattr(self._model.config, "max_position_embeddings", None) or 1024
        window = min(1024, max_pos)

        nll_sum = 0.0
        n_tokens = 0
        # Ventanas sin solaparse: cada token objetivo se cuenta exactamente una vez.
        for begin in range(0, seq_len - 1, window):
            end = min(begin + window, seq_len - 1)
            inputs = ids[:, begin:end]
            targets = ids[:, begin + 1: end + 1]
            with torch.no_grad():
                logits = self._model(inputs).logits
                log_probs = torch.log_softmax(logits, dim=-1)
                nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            nll_sum += float(nll.sum())
            n_tokens += targets.numel()

        if n_tokens == 0:
            return None
        return math.exp(nll_sum / n_tokens)

    # ------------------------------------------------------------------
    # Análisis completo
    # ------------------------------------------------------------------
    def analyze(self, text: str, lang: str = "es") -> tuple[float, list[Feature]]:
        """Devuelve (puntuación 0-1, señales) basada en perplejidad."""
        cfg = DetectorConfig()
        ppl = self.perplexity(text)

        if ppl is None:
            return 0.5, [Feature(
                name="Perplejidad (GPT-2)",
                value=None,
                score=0.5,
                weight=1.0,
                description="Texto demasiado corto para calcular la perplejidad",
            )]

        # Mapa logístico: perplejidad baja → puntuación de IA alta.
        ppl_score = 1 / (1 + math.exp((ppl - cfg.ppl_center) / cfg.ppl_width))

        features = [Feature(
            name="Perplejidad (GPT-2)",
            value=round(ppl, 2),
            score=round(ppl_score, 3),
            weight=0.75,
            description="Perplejidad baja = texto predecible = probable IA",
        )]

        # Uniformidad de la perplejidad por frase (la IA es consistente).
        sentences = [s for s in split_sentences(text) if len(tokenize(s)) >= 3]
        uniform_score = 0.5
        if len(sentences) >= 3:
            ppls = [p for p in (self.perplexity(s) for s in sentences) if p is not None]
            if len(ppls) >= 3 and sum(ppls) > 0:
                mean = sum(ppls) / len(ppls)
                var = sum((p - mean) ** 2 for p in ppls) / len(ppls)
                cv = math.sqrt(var) / mean
                uniform_score = _clamp01((0.45 - cv) / 0.35)
        features.append(Feature(
            name="Uniformidad de la perplejidad por frase",
            value=None,
            score=round(uniform_score, 3),
            weight=0.25,
            description="Variabilidad baja de la perplejidad entre frases → IA",
        ))

        score = 0.75 * ppl_score + 0.25 * uniform_score
        return _clamp01(score), features
