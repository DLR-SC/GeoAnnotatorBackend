import geocoder
from fastapi import FastAPI
from utils.dictFunctions import getAddressAndCoordinates

# Initiating the router
app = FastAPI()

# TODO: Store sensible data in an .env file
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
    
# Experimental
# @app.get('/')
# def test():
#     g = geocoder.geonames(location='New York', key=GEONAMES_USERNAME)
#     return {
#         'address': g.address,
#         'geonames_id': g.geonames_id,
#         'description': g.description,
#         'population': g.population,
    # }

# @app.get("/coordinates/")
# async def get_coordinates(placename: str):
#     params = {
#         'q': placename,
#         'maxRows': 5,
#         'username': GEONAMES_USERNAME
#     }
#     try:
#         response = requests.get(GEONAMES_API_URL, params=params)
#         response.raise_for_status()
#         data = response.json()
#         # Extract coordinates and other desired info from response
#         coordinates = [
#             {"name": place['toponymName'], "lat": place['lat'], "lng": place['lng']}
#             for place in data.get('geonames', [])
#         ]
#         return coordinates
#     except requests.RequestException as e:
#         raise HTTPException(status_code=400, detail=str(e))