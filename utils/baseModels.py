from pydantic import BaseModel

'Request structures'

# Provider
class Provider(BaseModel):
    option: str
    instance_name: str
    data: dict

# GeoparseRequest
class GeoparseRequest(BaseModel):
    text: str
    provider: dict

# Feedback for active learning
class FeedbackRequest(BaseModel):
    text: str
    predictions: list
    corrections: list
    provider: Provider