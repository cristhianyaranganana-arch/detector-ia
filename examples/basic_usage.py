"""Ejemplo básico de uso de la API de Python de Detector IA.

Ejecutar (desde la raíz del repo, con el paquete instalado):
    python examples/basic_usage.py
"""

from detector_ia import Detector

TEXTOS = {
    "estilo IA": (
        "La inteligencia artificial ha transformado numerosos sectores de la sociedad "
        "en la última década. En primer lugar, ha optimizado procesos que antes "
        "requerían horas de trabajo manual. Además, permite analizar grandes volúmenes "
        "de datos en cuestión de segundos. Por otra parte, su implementación plantea "
        "desafíos éticos importantes. Por lo tanto, es fundamental establecer marcos "
        "regulatorios adecuados. En conclusión, la inteligencia artificial constituye "
        "una herramienta poderosa."
    ),
    "estilo humano": (
        "Ayer fui al mercado con hambre y compré de todo, en serio. Mi abuela siempre "
        "dice «compra poco pero compra bien»... como si fuera tan fácil. Total que "
        "gasté cuarenta euros ¡en fruta! ¿Quién me avisa que las cerezas de junio "
        "están carísimas? Nadie. Para la próxima hago lista (y no me desvío ni un "
        "metro del presupuesto, jaja)."
    ),
}


def main() -> None:
    detector = Detector(mode="auto")

    for etiqueta, texto in TEXTOS.items():
        resultado = detector.analyze(texto)
        print(f"[{etiqueta:>13}] {resultado.score:>3}/100 — {resultado.verdict}")
        print(f"{'':>15} motor: {resultado.engine} · confianza: {resultado.confidence}")
        print()


if __name__ == "__main__":
    main()
