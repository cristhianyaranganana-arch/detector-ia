"""Evalúa el detector sobre el conjunto de muestras (data/samples.json).

Uso:
    python scripts/evaluate.py
    python scripts/evaluate.py --samples ruta/otras_muestras.json --threshold 50

Nota: las muestras incluidas son ILUSTRATIVAS (están escritas para exhibir las
señales de cada estilo). Para una evaluación realista usa el corpus HC3 con
scripts/download_hc3.py.
"""

import argparse
import json
from pathlib import Path

from detector_ia import Detector


def evaluate(samples_path: Path, mode: str, threshold: int) -> int:
    data = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = data["samples"]

    detector = Detector(mode=mode)
    tp = fp = tn = fn = 0
    rows = []

    for sample in samples:
        result = detector.analyze(sample["text"], include_sentences=False)
        predicted_ai = result.score >= threshold
        is_ai = sample["label"] == "ai"
        if is_ai and predicted_ai:
            tp += 1
        elif is_ai and not predicted_ai:
            fn += 1
        elif not is_ai and predicted_ai:
            fp += 1
        else:
            tn += 1
        rows.append((sample["id"], sample["label"], result.score, result.verdict))

    total = len(rows)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"{'ID':<12} {'real':<7} {'score':>5}  veredicto")
    print("─" * 72)
    for sample_id, label, score, verdict in rows:
        print(f"{sample_id:<12} {label:<7} {score:>5}  {verdict}")
    print("─" * 72)
    print(f"Muestras: {total} · umbral IA: >= {threshold} · motor: {detector.mode}")
    print(f"VP={tp}  FP={fp}  VN={tn}  FN={fn}")
    print(f"Exactitud: {accuracy:.1%} · Precisión: {precision:.1%} · "
          f"Exhaustividad: {recall:.1%} · F1: {f1:.1%}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "samples.json",
    )
    parser.add_argument("--mode", default="auto", choices=["auto", "heuristic", "neural"])
    parser.add_argument("--threshold", type=int, default=50,
                        help="Puntuación a partir de la cual se predice IA")
    args = parser.parse_args()
    return evaluate(args.samples, args.mode, args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
