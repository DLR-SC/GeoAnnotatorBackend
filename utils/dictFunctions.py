from utils.baseModels import GeoReference

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
def parse_georeferences(response_text):
    # Implement a simple parser to extract location data from the response
    # Example: If the LLM response is structured as a JSON-like output or plain text
    locations = []
    # This parsing logic will depend on how the LLM formats its output
    # Below is a dummy implementation that assumes a specific output format
    for line in response_text.split('\n'):
        if line.strip():  # Ignore empty lines
            parts = line.split(',')
            location = GeoReference(
                location_name=parts[0].strip(),
                latitude=float(parts[1].strip()),
                longitude=float(parts[2].strip())
            )
            locations.append(location)
    return locations