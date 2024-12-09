# GeoSense Annotator-Backend
## Purpose
This repository is the backend of the GeoSense Annotator (GSA). It is mainly used for geoparsing
plain texts, requesting coordinates of place-names and initializing the Retrain-Job of LLM-models.

## Installation
In bash, run the following (Win/Lin/Os):
```bash
conda create -n GeoSenseAnnotatorBackend httpx openai mlflow python -y && activate GeoSenseAnnotatorBackend

pip install fastapi "uvicorn[standard]" geocoder
```
or with YAML-file GeoSenseAnnotatorBackend.yml:
```bash
conda env create --file GeoSenseAnnotatorBackend.yml --name GeoSenseAnnotatorBackend
```

### Modules
- python as the main programming language
- httpx for (asynchrone) HTTP-Requests
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
Ongoing.