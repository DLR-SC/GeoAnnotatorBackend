import json

'''
    Extractin the addresses and coordinates of the readen geonames 
    and returning an array with dicts with corresponding attributes [location → (lat, lng)...] 
'''
def structuredGeolocations(geoDict):
    return [{
        'name': r.address,
        'position': [round(float(r.lat), 2), round(float(r.lng), 2)],
        'continent': r.continent,
        'country': r.country,
        'state': r.state,
    } for r in geoDict ]

'''
    Extract the geoparsed text and structure it in an array
'''
def parseGeoreferences(response_text):
    return json.loads(response_text)