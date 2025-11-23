PYTHON ?= python3
VENV ?= .venv
ACTIVATE = source $(VENV)/bin/activate
DATA_DIR ?= data/raw
PROCESSED_DIR ?= data/processed

.PHONY: help venv install data ingest agent eval test clean compose-up compose-down

help:
	@echo "Available targets:"
	@echo "  make venv        # create virtualenv"
	@echo "  make install     # install deps"
	@echo "  make data        # download dataset into $(DATA_DIR)"
	@echo "  make ingest      # run ingestion pipeline"
	@echo "  make agent       # launch your agent controller"
	@echo "  make eval        # run retrieval/agent evals"
	@echo "  make test        # run pytest"
	@echo "  make compose-up  # start optional pgvector stack"
	@echo "  make compose-down# stop optional pgvector stack"

venv:
	@if [ ! -d $(VENV) ]; then $(PYTHON) -m venv $(VENV); fi

install: venv
	$(ACTIVATE) && pip install -U pip
	$(ACTIVATE) && pip install -r requirements.txt

data: install
	$(ACTIVATE) && $(PYTHON) scripts/download_dataset.py --output $(DATA_DIR)

ingest: install
	$(ACTIVATE) && agentic-rag ingest --raw-dir $(DATA_DIR) --output-dir $(PROCESSED_DIR)

agent: install
	$(ACTIVATE) && agentic-rag agent

_eval_cmd = $(ACTIVATE) && agentic-rag evaluate

eval: install
	$(_eval_cmd)

test: install
	$(ACTIVATE) && pytest -q

compose-up:
	docker compose up -d vectorstore

compose-down:
	docker compose down

clean:
	rm -rf $(VENV) $(DATA_DIR) $(PROCESSED_DIR)
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
