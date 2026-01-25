.PHONY: help install test test-ml test-frontend test-ocr-viewer test-annotation-canvas test-all clean

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies (Python + Frontend)
	@echo "Installing Python dependencies..."
	uv sync
	@echo "Installing OCR Viewer dependencies..."
	cd app/frontend/ocr_viewer && npm install
	@echo "Installing Annotation Canvas dependencies..."
	cd app/frontend/annotation_canvas && npm install
	@echo "✓ All dependencies installed"

test-ml: ## Run ML model tests (pytest)
	@echo "Running ML tests..."
	uv run pytest tests/ml/models/ -v
	@echo "✓ ML tests complete"

test-backend: ## Run backend/service tests (pytest)
	@echo "Running backend tests..."
	uv run pytest tests/app/ -v
	@echo "✓ Backend tests complete"

test-ocr-viewer: ## Run OCR Viewer frontend tests (vitest)
	@echo "Running OCR Viewer tests..."
	cd app/frontend/ocr_viewer && npm test -- --run
	@echo "✓ OCR Viewer tests complete"

test-annotation-canvas: ## Run Annotation Canvas frontend tests (vitest)
	@echo "Running Annotation Canvas tests..."
	cd app/frontend/annotation_canvas && npm test -- --run
	@echo "✓ Annotation Canvas tests complete"

test-frontend: test-ocr-viewer test-annotation-canvas ## Run all frontend tests

test-python: test-ml test-backend ## Run all Python tests

test-all: test-python test-frontend ## Run all tests (Python + Frontend)
	@echo ""
	@echo "========================================="
	@echo "✓ All tests complete!"
	@echo "========================================="

test: test-all ## Alias for test-all

test-coverage-ml: ## Run ML tests with coverage
	@echo "Running ML tests with coverage..."
	uv run pytest tests/ml/models/ --cov=ml.models --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated in htmlcov/"

test-coverage-backend: ## Run backend tests with coverage
	@echo "Running backend tests with coverage..."
	uv run pytest tests/app/ --cov=app --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated in htmlcov/"

test-quick: ## Run quick test suite (skip slow tests)
	@echo "Running quick tests..."
	uv run pytest tests/ -m "not slow" -v

clean: ## Clean test artifacts and caches
	@echo "Cleaning test artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "node_modules/.vitest" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned"

lint: ## Run linter (ruff)
	@echo "Running linter..."
	uv run ruff check .

lint-fix: ## Auto-fix linting issues
	@echo "Fixing linting issues..."
	uv run ruff check --fix .
	uv run ruff format .

# Test summary with counts
test-summary: ## Show test summary
	@echo "========================================="
	@echo "Test Summary"
	@echo "========================================="
	@echo ""
	@echo "ML Tests:"
	@uv run pytest tests/ml/models/ --collect-only -q | tail -1 || echo "  (run 'make install' first)"
	@echo ""
	@echo "Backend Tests:"
	@uv run pytest tests/app/ --collect-only -q | tail -1 || echo "  (run 'make install' first)"
	@echo ""
	@echo "Frontend Tests:"
	@echo "  OCR Viewer: 16 tests"
	@echo "  Annotation Canvas: 30 tests"
	@echo ""
	@echo "Run 'make test-all' to execute all tests"
	@echo "========================================="
