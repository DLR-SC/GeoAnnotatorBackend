from fastapi import APIRouter, HTTPException, Query

from utils.dictFunctions import load_existing_provider_data
from utils.baseModels import *

from dotenv import load_dotenv
import json
import os

load_dotenv()
PROVIDER_FILE_PATH = os.getenv("PROVIDER_FILE_PATH")

router = APIRouter()

@router.get("/provider/all")
async def get_all_providers():
    try:
        data = load_existing_provider_data(PROVIDER_FILE_PATH)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to read providers: " + str(e))

@router.post("/provider")
async def save_provider(request: ProviderRequest):
    try:
        # Create directory, if it does not exist
        os.makedirs(os.path.dirname(PROVIDER_FILE_PATH), exist_ok=True)

        existing_data = load_existing_provider_data(PROVIDER_FILE_PATH) if os.path.exists(PROVIDER_FILE_PATH) else []

        # Append new provider onto list
        existing_data.append(request.model_dump())

        # Save new provider
        with open(PROVIDER_FILE_PATH, "w") as file:
            json.dump(existing_data, file, indent=2)

        return { "message": "Provider has been saved succesfully!" }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save provider: " + str(e))

@router.put("/provider")
async def update_provider(request: ProviderRequest):
    providers = load_existing_provider_data(PROVIDER_FILE_PATH) if os.path.exists(PROVIDER_FILE_PATH) else None

    if(providers is None): 
        raise HTTPException(status_code=404, detail="Provider not found!")
    
    print(request.model_dump())

    for i, provider in enumerate(providers):
        if provider["instance_name"] == request.instance_name:
            providers[i] = request.model_dump()
            break
            
    # Save new provider
    with open(PROVIDER_FILE_PATH, "w") as file:
        json.dump(providers, file, indent=2)

    return {"message": "Provider updated successfully!"}

@router.delete("/provider")
async def delete_provider(index: int = Query(..., description="The index of the provider")):
    try:
        providers = load_existing_provider_data(PROVIDER_FILE_PATH)

        del providers[index]

        # Save updated provider list
        with open(PROVIDER_FILE_PATH, "w") as file:
            json.dump(providers, file, indent=2)

        return {"message": "Provider deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete provider: " + str(e))