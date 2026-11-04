.PHONY: install test lint api demo evaluate

install:          ## Instala el paquete + dependencias de desarrollo
	pip install -e ".[dev]"

test:             ## Ejecuta la suite de tests
	pytest -q

lint:             ## Lint con ruff
	ruff check .

api:              ## Levanta la API en http://localhost:8000/docs
	uvicorn api.app:app --reload

demo:             ## Levanta la demo web en http://localhost:8501
	streamlit run demo/app.py

evaluate:         ## Evalúa el detector sobre data/samples.json
	python scripts/evaluate.py
