.PHONY: install data features train alert test test-fast test-e2e baseline clean

install:
	pip install -r requirements.txt

data:
	python3 simulator/generator.py

features:
	python3 src/feature_engineering.py

train:
	python3 src/train_model.py

train-fast:
	python3 src/train_model.py --no-plots

alert:
	python3 -m src.argus_alert --scenario ransomware_like

baseline:
	python3 argus_baseline.py

test:
	pytest -v

test-fast:
	pytest -m "not e2e" -v

test-e2e:
	pytest -m e2e -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
