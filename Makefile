.PHONY: install run mcp test evals lint docker-build docker-run clean help

PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
STREAMLIT := $(VENV)/bin/streamlit

help:
	@echo "Career Intelligence Assistant — available targets:"
	@echo "  make install      Create venv and install dependencies"
	@echo "  make run          Start the Streamlit app"
	@echo "  make mcp          Start the MCP server (for Claude Desktop / Cursor)"
	@echo "  make test         Run pytest test suite"
	@echo "  make evals        Run LLM evaluation suite (requires OPENAI_API_KEY)"
	@echo "  make lint         Run ruff linter"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-run   Run Docker container"
	@echo "  make clean        Remove venv, chroma_db, and __pycache__"

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete. Copy .env.example to .env and add your keys."

run:
	$(STREAMLIT) run app.py

mcp:
	$(VENV)/bin/python -m src.mcp_server

test:
	$(PYTEST) tests/ -v --tb=short

evals:
	$(VENV)/bin/python evals/run_evals.py --verbose

lint:
	$(VENV)/bin/ruff check src/ tests/ evals/ app.py

docker-build:
	docker build -t career-intel:latest .

docker-run:
	docker run --rm -p 8501:8501 \
		--env-file .env \
		-v $(PWD)/chroma_db:/app/chroma_db \
		career-intel:latest

clean:
	rm -rf $(VENV) chroma_db __pycache__ src/__pycache__ \
		tests/__pycache__ evals/__pycache__ \
		src/ingestion/__pycache__ src/rag/__pycache__ src/qa/__pycache__ \
		.pytest_cache evals/eval_results.json
