from fastapi import APIRouter, HTTPException

from utils.baseModels import *
from utils.dictFunctions import *

from geoparse.geoparseFunctions import *

from dotenv import load_dotenv

import geocoder
import os

load_dotenv()

router = APIRouter()

@router.get('/geolocations')
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

@router.post("/geoparse")
async def geoparse_text(request: GeoparseRequest):
    try:
        option = request.provider["option"]
        match option:
            case "openai":
                extracted_locations = geoparseTextGPT(request.text, request.provider)
            case "selfhosted":
                extracted_locations = await geoparseTextSelfHosted(request.text, request.provider)
            case _:
                raise Exception("No provider selected.")

        return extracted_locations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))