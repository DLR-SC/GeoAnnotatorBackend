from utils.baseModels import Provider
import json

async def load_existing_provider_data(FILE_PATH) -> list[Provider]:
    '''
        Load provider data or return an empty list
    '''
    with open(FILE_PATH, "r") as file:
        return json.load(file)