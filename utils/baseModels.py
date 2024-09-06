from pydantic import BaseModel

'Here, you find the response structures'

# GeoparseRequest
class ProviderRequest(BaseModel):
    option: str
    instance_name: str
    data: dict

# GeoparseRequest
class GeoparseRequest(BaseModel):
    text: str
    model: str

# Georeference
class GeoReference(BaseModel):
    location_name: str
    latitude: float
    longitude: float
