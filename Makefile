# Makefile for Kaggle Loan Prediction Project

# 防止同名檔案干擾
.PHONY: t p f train predict feature clean help

# 預設顯示幫助
help:
	@echo "Available commands:"
	@echo "  make t       - Train model"
	@echo "  make p       - Run prediction"
	@echo "  make f       - Feature engineering"
	@echo "  make clean   - Clean output files"

# 簡短命令
t:
	@echo "Starting training..."
	@uv run python -m src.train

p:
	@echo "Running prediction..."
	@uv run python -m src.predict

f:
	@echo "Creating features..."
	@uv run python -m src.feature_engineering

# 完整名稱（可選）
train: t
predict: p
feature: f

# 清理輸出
clean:
	@echo "Cleaning output files..."
	@rm -rf models/*.pkl
	@rm -rf results/*.csv

# 執行完整流程
all: f t p
	@echo "Pipeline complete!"