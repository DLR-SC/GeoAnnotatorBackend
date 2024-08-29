import geocoder
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils.dictFunctions import *
from utils.baseModels import *
import openai

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

GEONAMES_USERNAME = 'siri_yu'

# TODO: Organize the requests into different route-files
@app.get('/api/geolocations')
async def get_geolocations(placename: str):
    # Static parameters (FIXME: outsource if necessary)
    params = {
        'q': placename,
        'maxRows': 5,
        'username': GEONAMES_USERNAME
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

@app.post("/geoparse", response_model=list[GeoReference])
async def geoparse_text(request: TextRequest):
    try:
        # Call OpenAI API with the unstructured text
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can use GPT-3.5 or the latest model
            prompt=f"Extract geographic references from the following text: {request.text}. For each location, provide the name, latitude, and longitude.",
            max_tokens=150,
            temperature=0.0
        )

        # Parse the response to extract location names and coordinates
        extracted_locations = parse_georeferences(response.choices[0].text)

        return extracted_locations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))