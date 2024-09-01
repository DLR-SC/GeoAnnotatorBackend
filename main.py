import geocoder

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from utils.geoparseFunctions import *
from utils.dictFunctions import *
from utils.baseModels import *

from dotenv import load_dotenv
import os

load_dotenv()

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

# TODO: Organize the requests into different route-files
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
async def geoparse_text(request: TextRequest):
    try:
        match request.model:
            case 'gpt':
                response = geoparseTextGPT(request.text)
            case _:
                response = geoparseTextGPT(request.text)

        # Parse the response to a json-list
        extracted_locations = parseGeoreferences(response.choices[0].message.content)

        # return extracted_locations
        return extracted_locations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))