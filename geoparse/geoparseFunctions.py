from utils.baseModels import Provider
from openai import OpenAI
import requests
import json

system_content = "You are an assistant that strictly and exclusively extracts geographic references mentioned in the user-input. For each location, provide the exact place-name as it appears in the input, along with its latitude and longitude, as a JSON object (e.g., { 'name': 'place-name', 'position': [latitude, longitude] }). Only return locations mentioned in the text. Under no circumstances should you add or generate locations not present in the text. The list must only contain the exact places mentioned and must be as precise as possible. Please return the result in JSON format without any explanations or labels."
'Command for LLM-system'

async def geoparseTextGPT(text: str, provider: Provider):
    '''
        Geoparsing text with an OpenAI-GPT model
    '''
    response = OpenAI(api_key=provider["data"]["api_key"]).chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_content,
                    },
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "georeferences",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "georeferences": {
                                    "type": "array",  
                                    "items": {       
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string"
                                            },
                                            "position": {
                                                "type": "array",  
                                                "items": {
                                                    "type": "number"
                                                }
                                            }
                                        },
                                        "required": ["name", "position"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["georeferences"],
                            "additionalProperties": False
                        }
                    }
                },
                model=provider["data"]["model"],
                temperature=provider["temperature"]
            ).choices[0].message.content
    output=json.loads(response)

    return output["georeferences"]

async def geoparseTextSelfHosted(text: str, provider: dict):
    '''
        Geoparsing text with a selfhosted LLM
    '''
    response = requests.post(
        url=provider["data"]["hostserver_url"] + '/chat/completions',
        json={
            "model": provider["data"]["model"],
            "messages": [
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": text,
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "georeferences",
                    "strict": "true",
                    "schema": {
                        "type": "array",  
                        "items": {       
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string"
                                },
                                "position": {
                                    "type": "array",  
                                    "items": {
                                        "type": "number",
                                    },
                                    "minItems": 2,
                                    "maxItems": 2
                                }
                            },
                            "required": ["name", "position"]
                        }
                    }
                }
            },
            "temperature": provider["temperature"],
            "max_tokens": -1,
            "stream": False
        }, 
        headers={"Content-Type": "application/json"},
    )
    output=response.json()
    
    return json.loads(output['choices'][0]['message']['content'])