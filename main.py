import geocoder

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from utils.geoparseFunctions import *
from utils.dictFunctions import *
from utils.baseModels import *

from dotenv import load_dotenv
import json
import os

# Load .env variables
load_dotenv()

PROVIDER_FILE_PATH = os.getenv("PROVIDER_FILE_PATH")

# Initialize LLMs
# initializeLLMs()


# Initiating the router
app = FastAPI()

# App configs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods or specify with ["POST", "GET"]
    allow_headers=["*"],  # Allow all headers or specify with ["X-Custom-Header"]
)

@app.get('/api/geolocations')
async def get_geolocations(placename: str):
    # Static parameters (FIXME: outsource if necessary)
    params = {
        'q': placename,
        'maxRows': 5,
        'username': os.getenv("GEONAMES_USERNAME")
    }
    # Geocoded location
    geoObjects = geocoder.geonames(location=params['q'], maxRows=params['maxRows'], key=params['username'])
    # Extracting further details from detected toponyms
    g_details = [geocoder.geonames(location=row.geonames_id, method='details', key=params['username']) for row in geoObjects]
    # Array, which includes geolocations and their respective coordinates
    gls = structuredGeolocations(g_details)

    if len(gls) == 0:
        raise HTTPException(status_code=404, detail="No location available")

    return gls

@app.post("/api/geoparse")
async def geoparse_text(request: GeoparseRequest):
    try:
        model = request.model
        match model:
            case "gpt":
                extracted_locations = geoparseTextGPT(request.text)
            case "bert":
                extracted_locations = geoparseTextBERT(request.text)
            case _:
                extracted_locations = geoparseTextGPT(request.text)

        return extracted_locations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/provider/save")
async def save_provider(request: ProviderRequest):
    try:
        # Create directory, if it does not exist
        os.makedirs(os.path.dirname(PROVIDER_FILE_PATH), exist_ok=True)

        existing_data = load_existing_provider_data(PROVIDER_FILE_PATH) if os.path.exists(PROVIDER_FILE_PATH) else []

        # Append new provider onto list
        existing_data.append(request.model_dump())

        # Save new provider
        with open(PROVIDER_FILE_PATH, "w") as file:
            json.dump(existing_data, file, indent=2)

        return { "message": "Provider has been saved succesfully!" }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save data: " + str(e))
    
@app.get("/api/provider/all")
async def get_all_providers():
    try:
        data = load_existing_provider_data(PROVIDER_FILE_PATH)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to read providers: " + str(e))