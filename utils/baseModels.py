from pydantic import BaseModel

'Here, you find the response structures'

# TextRequest
class TextRequest(BaseModel):
    text: str
    model: str

# Georeference
class GeoReference(BaseModel):
    location_name: str
    latitude: float
    longitude: float
