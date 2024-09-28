# GeoAnnotator-Backend

## Purpose
This repository is the backend of the GeoAnnotator (GA). It is mainly used for geoparsing
plain texts, requesting coordinates of detected/given locations and more.

## Installation
To install and run the backend, following packages/modules are required:
- Python 3.6 or higher
```bash
conda create -n GeoAnnotatorBackend python && conda activate GeoAnnotatorBackend
```
- FastAPI (for Request-API's) and Uvicorn (ASGI, webserver for hosting and logging Request-API's)
```bash
pip install fastapi "uvicorn[standard]"
```
- Geocoder, for "geocoding" exact position of a place
```bash
pip install geocoder
```
- httpx, for asynchrone HTTP-Requests
```bash
conda install httpx
```
- OpenAI, for usage of GPT-models
```bash
conda install openai
```
- MLflow, for registering and tracking ML-experiments and metrics
```bash
conda install openai
```

## Usage
In bash, run the backend with following command (Win/Lin/Os):
```bash
uvicorn main:app --reload
```
The paths for the endpoint API's are under 'localhost:8000/api/...'. Feel free to explore.

## Project status
Project finished.