# Cómo contribuir a Detector IA

¡Gracias por tu interés! 🎉

## Flujo básico

1. Haz **fork** del repositorio y clona tu copia.
2. Crea una rama descriptiva: `git checkout -b feat/mas-idiomas`.
3. Instala el entorno: `pip install -e ".[dev]"`.
4. Haz tus cambios **con tests**.
5. Verifica: `make lint && make test`.
6. Abre un **Pull Request** describiendo el cambio.

## Reglas del proyecto

- **Tests obligatorios**: todo cambio de lógica va acompañado de tests en `tests/`.
- **Lint**: el CI ejecuta `ruff check .`; ejecútalo localmente antes de subir.
- **Sin dependencias nuevas en el núcleo**: `src/detector_ia` debe funcionar solo con
  la librería estándar. Las dependencias van en extras (`api`, `demo`, `neural`).
- **Español primero**: la documentación y los mensajes de usuario van en español;
  el código (nombres de variables, funciones) en inglés.

## Ideas bienvenidas

- Más idiomas (portugués, francés, alemán…): añade listas en `config.py`.
- Mejor calibración usando el corpus HC3 (`scripts/download_hc3.py`).
- Tests adversariales: textos difíciles (humano formal, IA "humanizada").
- Clasificador fino con RoBERTa/DeBERTa.

## Reportar bugs

Abre un *issue* con: versión de Python, comando ejecutado, texto analizado
(si no es privado) y salida obtenida vs. esperada.
