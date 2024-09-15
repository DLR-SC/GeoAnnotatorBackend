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