# Makefile for Kaggle Loan Prediction Project

.PHONY: t p f train predict feature clean help

help:
	@echo "Available commands:"
	@echo "  make t       - Train model"
	@echo "  make p       - Run prediction"
	@echo "  make f       - Feature engineering"
	@echo "  make clean   - Clean output files"

t:
	@echo "Starting training..."
	@uv run python -m src.train

p:
	@echo "Running prediction..."
	@uv run python -m src.predict

f:
	@echo "Creating features..."
	@uv run python -m src.feature_engineering

train: t
predict: p
feature: f

clean:
	@echo "Cleaning output files..."
	@rm -rf models/*.pkl
	@rm -rf results/*.csv

all: f t p
	@echo "Pipeline complete!"