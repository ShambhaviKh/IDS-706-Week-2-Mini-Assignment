# Makefile for Dataset_Analysis project

# Variables
PYTHON := python3
DOCKER_IMAGE := dataset_analysis

install:
	pip install --upgrade pip &&\
		pip install -r Requirements.txt
		pip install flake8
		pip install black

# Default target
.PHONY: all
all: help

# Help target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make run             - Run DataAnalysis.py"
	@echo "  make test            - Run Test_cases.py"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run Docker container"
	@echo "  make clean           - Remove __pycache__ and temporary files"

# Run the main Python script
.PHONY: run
run:
	$(PYTHON) DataAnalysis.py

# Run tests with success/failure message
.PHONY: test
test:
	@echo "Running tests..."
	@$(PYTHON) Test_cases.py; \
	if [ $$? -eq 0 ]; then \
		echo "All tests passed!"; \
	else \
		echo "Some tests failed."; \
		exit 1; \
	fi


#Format code with black 
.PHONY: format
format:
	black DataAnalysis.py
	black main_script/Analysis.py
	black Test_cases.py

#Linting code with flake8
.PHONY: lint
lint:
	python -m flake8 --ignore=E203,W503,E501 main_script/Analysis.py
# Build Docker image
.PHONY: docker-build
docker-build:
	docker build -t $(DOCKER_IMAGE) .

# Run Docker container
.PHONY: docker-run
docker-run:
	docker run --rm -it $(DOCKER_IMAGE)

# Clean temporary files
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f *.pyc