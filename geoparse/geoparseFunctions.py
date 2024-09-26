from utils.baseModels import Provider
from openai import OpenAI
import requests
import json

system_content = "You are an assitant that strictly extracts geographic references from the input. For each location, provide the place-name (exactly as in the text), the latitude and the longitude of the place as a json-object, like { name: place-name, position: [latitude, longitude] }. Create a json-list out of these objects. In the list, there should be no repetitive places with the same place-name. Please only return the value with no explanation or further information and as a normal text without labeling it as json."
'Command for LLM-system'

'''
    Geoparsing text with an OpenAI-GPT model
'''
def geoparseTextGPT(text: str, provider: Provider):
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

'''
    Geoparsing text with a selfhosted LLM
'''
async def geoparseTextSelfHosted(text: str, provider: dict):
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