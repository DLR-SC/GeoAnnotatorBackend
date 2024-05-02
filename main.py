import geocoder
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils.dictFunctions import getAddressAndCoordinates

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

# TODO: Store sensible data in an .env file or get the username from request body
GEONAMES_USERNAME = 'siri_yu'

@app.get('/api/coordinates')
async def get_coordinates(placename: str):
    # Static parameters (FIXME: outsource if necessary)
    params = {
        'q': placename,
        'maxRows': 5,
        'username': GEONAMES_USERNAME
    }
    # Geocoded geoname
    g = geocoder.geonames(location=params['q'], maxRows=params['maxRows'], key=params['username'])
    # Constructing a dic with the addresses and their coordinates
    g_dict = getAddressAndCoordinates(g)

    return g_dict