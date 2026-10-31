# make run    -> sets up .venv on first use, then serves the app on http://127.0.0.1:7860
#                (the weights download from the GitHub release the first time the app starts)
# make test   -> pytest;  make check -> the accuracy check (needs DENTEX in data/, see README)
PORT ?= 7860
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: run install test check docker

run: $(VENV)/.installed
	$(PY) -m uvicorn thirtytwo.server:app --host 127.0.0.1 --port $(PORT) --no-access-log

install: $(VENV)/.installed

$(VENV)/.installed: requirements.txt requirements-dev.txt
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt
	$(PY) -m pip uninstall -y opencv-python
	$(PY) -m pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93
	touch $@

test: $(VENV)/.installed
	$(PY) -m pytest -q

check: $(VENV)/.installed
	$(PY) tools/accuracy_check.py

docker:
	docker build -t thirtytwo .
	docker run --rm -p $(PORT):7860 thirtytwo
