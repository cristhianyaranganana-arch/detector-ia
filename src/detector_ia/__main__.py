"""Permite ejecutar el paquete como módulo: python -m detector_ia "texto"."""

import sys

from detector_ia.cli import main

if __name__ == "__main__":
    sys.exit(main())
