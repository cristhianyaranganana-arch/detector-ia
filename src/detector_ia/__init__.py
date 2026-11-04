"""Detector IA — detección de texto generado por inteligencia artificial.

API pública principal:
    from detector_ia import Detector

    detector = Detector(mode="auto")
    resultado = detector.analyze("Texto a analizar...")
    print(resultado.score, resultado.verdict)
"""

from detector_ia.detector import DetectionResult, Detector, SentenceResult

__version__ = "1.0.1"
__all__ = ["Detector", "DetectionResult", "SentenceResult", "__version__"]
