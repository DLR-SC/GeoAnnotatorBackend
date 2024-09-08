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
    provider: dict

# Georeference
class GeoReference(BaseModel):
    location_name: str
    latitude: float
    longitude: float
