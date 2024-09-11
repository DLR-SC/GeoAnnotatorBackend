from utils.baseModels import Provider
import json

'''
    Extracting the addresses and coordinates of the readen geonames 
    and returning an array with dicts with corresponding attributes [location → (lat, lng)...] 
'''
def structuredGeolocations(geodata):
    return [{
        'name': r.address,
        'position': [round(float(r.lat), 2), round(float(r.lng), 2)],
        'continent': r.continent,
        'country': r.country,
        'state': r.state,
    } for r in geodata ]

'''
    Load provider data or return an empty list
'''
def load_existing_provider_data(FILE_PATH) -> list[Provider]:
    with open(FILE_PATH, "r") as file:
        return json.load(file)