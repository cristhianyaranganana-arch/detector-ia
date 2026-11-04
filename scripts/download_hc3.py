"""Descarga el corpus público HC3 (Human ChatGPT Comparison Corpus).

Requiere: pip install datasets

Uso:
    python scripts/download_hc3.py [--out data/hc3]

Genera data/hc3/hc3.jsonl con campos {"text": ..., "label": "ai"|"human"}.
HC3 contiene preguntas respondidas por humanos y por ChatGPT: ideal para
calibrar y evaluar el detector con datos reales (no ilustrativos).
"""

import argparse
import json
from pathlib import Path


def main() -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Falta la librería 'datasets': pip install datasets")
        return 2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/hc3"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Hello-SimpleAI/HC3", "all", trust_remote_code=True)

    count = 0
    with (args.out / "hc3.jsonl").open("w", encoding="utf-8") as fh:
        for split in ("train",):
            for row in ds[split]:
                for human_text in row.get("human_answers", []):
                    fh.write(json.dumps(
                        {"text": human_text, "label": "human"}, ensure_ascii=False
                    ) + "\n")
                    count += 1
                for ai_text in row.get("chatgpt_answers", []):
                    fh.write(json.dumps(
                        {"text": ai_text, "label": "ai"}, ensure_ascii=False
                    ) + "\n")
                    count += 1

    print(f"Guardadas {count} muestras en {args.out / 'hc3.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
