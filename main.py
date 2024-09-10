from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from provider import routes as provider
from geoparse import routes as geoparse

# Initiating the router
app = FastAPI()

# App configs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

# Include the API-paths
app.include_router(provider.router, prefix="/api")
app.include_router(geoparse.router, prefix="/api")