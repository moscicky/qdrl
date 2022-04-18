.PHONY: clean clean-test clean-pyc clean-build

clean: clean-build clean-pyc clean-test

clean-build:
	rm -rf .mypy_cache/
	rm -rf test-reports/
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -rf {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +
	find . -name '*~' -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-test:
	rm -rf .pytest_cache

mypy: ## run mypy checks
	mypy --junit-xml test-reports/junit-mypy.xml .

clean: clean-build

install-dev: clean ## install dev packages
	pip install -r requirements_dev.txt

lint:
	flake8 qdrl tests
