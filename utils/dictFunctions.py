'''
    Extractin the addresses and coordinates of the readen geonames 
    and returning a dict with the attributes [location → (lat, lng)...] 
'''
def getAddressAndCoordinates(geoDict):
    g = {}
    for r in geoDict:
        g[r.address] = (float(r.lat), float(r.lng))
    return g