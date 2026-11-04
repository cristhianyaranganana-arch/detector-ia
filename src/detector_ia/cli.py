"""Interfaz de línea de comandos de Detector IA.

Uso:
    detector-ia "Texto a analizar..."
    detector-ia --file documento.txt
    detector-ia --mode heuristic --json "Texto..."
"""

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="detector-ia",
        description="Detecta si un texto fue redactado por inteligencia artificial.",
    )
    parser.add_argument("text", nargs="?", help="Texto a analizar (entre comillas)")
    parser.add_argument("--file", "-f", help="Ruta a un archivo .txt a analizar")
    parser.add_argument(
        "--mode",
        choices=["auto", "heuristic", "neural"],
        default="auto",
        help="Motor de análisis (por defecto: auto)",
    )
    parser.add_argument("--language", default="auto", help="Idioma: es | en | auto")
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Imprime el resultado como JSON",
    )
    args = parser.parse_args(argv)

    # Obtener el texto.
    if args.file:
        try:
            with open(args.file, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            print(f"Error leyendo {args.file}: {exc}", file=sys.stderr)
            return 2
    elif args.text:
        text = args.text
    else:
        parser.error("Indica un texto entre comillas o usa --file ARCHIVO")
        return 2

    from detector_ia import Detector

    detector = Detector(mode=args.mode)
    try:
        result = detector.analyze(text, language=args.language)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_report(result, text)

    return 0


def _print_report(result, text: str) -> None:
    """Imprime un informe legible en la terminal."""
    width = 56
    print("╭" + "─" * width + "╮")
    title = "  🤖 DETECTOR IA — Resultado del análisis"
    print("│" + title + " " * (width - len(title) - 1) + "│")
    print("╰" + "─" * width + "╯")
    print()
    preview = text.strip().replace("\n", " ")
    if len(preview) > 70:
        preview = preview[:70] + "…"
    print(f'  Texto: "{preview}"')
    print()
    print(f"  Puntuación IA :  {result.score}/100")
    print(f"  Veredicto     :  {result.verdict}")
    print(f"  Confianza     :  {result.confidence}")
    print(f"  Motor         :  {result.engine}")
    print(f"  Idioma        :  {result.language}  ({result.word_count} palabras,"
          f" {result.sentence_count} frases)")
    print()
    print("  Señales analizadas (0 = humano … 1 = IA):")
    for feature in result.features:
        bar = _bar(feature.score)
        print(f"    {feature.name:<42} {bar} {feature.score:.2f}")
    print()
    print("  ⚠️  Ningún detector es 100% fiable: usa el resultado como indicio,"
          " no como prueba.")


def _bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


if __name__ == "__main__":
    sys.exit(main())
