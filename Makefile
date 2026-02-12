.PHONY: help setup install test run run-limit clean

help:
	@echo "K-Number Extractor - Available Commands"
	@echo ""
	@echo "  make setup          - Initial setup (creates venv and installs dependencies)"
	@echo "  make install        - Install dependencies only"
	@echo "  make test           - Run with --limit 5 (test mode)"
	@echo "  make run            - Run with all K-numbers from Snowflake"
	@echo "  make run-limit N=10 - Run with N K-numbers limit"
	@echo "  make clean          - Remove cache and temporary files"
	@echo ""

setup:
	@bash setup.sh

install:
	@pip install -r requirements.txt

test:
	@python k_number_extractor_batch.py --limit 5

run:
	@python k_number_extractor_batch.py

run-limit:
	@python k_number_extractor_batch.py --limit $(N)

clean:
	@rm -rf __pycache__
	@rm -rf .cache
	@rm -rf *.pyc
	@rm -rf predicate_extraction_results_*.json
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned up temporary files"

venv:
	@python3 -m venv venv
	@source venv/bin/activate && pip install --upgrade pip setuptools wheel
	@echo "✓ Virtual environment created"
