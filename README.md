# GeoAnnotator-Backend

## Purpose
This repository is the backend of the GeoAnnotator (GA). It is mainly used for geoparsing
plain texts, requesting coordinates of detected/given locations and more.

## Installation
### Linux
```bash
conda create -n GeoAnnotatorBackend \
    httpx openai mlflow \
    python=3.10 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y

conda activate GeoAnnotatorBackend

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

pip install fastapi "uvicorn[standard]" geocoder celery[redis]
```
or with YAML-file GeoAnnotatorBackend_Linux.yml:
```bash
conda env create --file GeoAnnotatorBackend_Linux.yml --name GeoAnnotatorBackend
```

### Windows (no LLM-finetuning)
```bash
conda create -n GeoAnnotatorBackend \
    httpx openai mlflow \
    python=3.10 \
    -y

conda activate GeoAnnotatorBackend

pip install fastapi "uvicorn[standard]" geocoder
```
or with YAML-file GeoAnnotatorBackend_Windows.yml:
```bash
conda env create --file GeoAnnotatorBackend_Windows.yml --name GeoAnnotatorBackend
```

### Modules
- python (3.10) as the main programming language
- pytorch (and rest of related modules) for finetuning LLM [Linux]
- httpx for asynchrone HTTP-Requests
- OpenAI for usage of GPT-models
- MLflow for registering and tracking ML-experiments and metrics
- Geocoder for extracting optional coordinates of a place
- FastAPI for Request-APIs and Uvicorn webserver for hosting and logging Request-APIs

## Usage
In bash, run the backend with following command (Win/Lin/Os):
```bash
uvicorn main:app --reload
```
The paths for the endpoint API's are under 'localhost:8000/api/...'. Feel free to explore.

## Project status
Project finished.