from openai import OpenAI
import requests
import json

'''
    Geoparsing text with an OpenAI-GPT model
'''
def geoparseTextGPT(text: str, provider: dict):
    response = OpenAI(api_key=provider["data"]["api_key"]).chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Extract geographic references from the input. For each location, provide the name (how it appears in the text), latitude, and longitude as a json-object, like { name: ..., position: [latitude, longitude] } and create a json-list out of these objects. Please only return the value with no explanation or further information and as a normal text, without labeling it as json.",
                    },
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
                model=provider["data"]["model"]
            ).choices[0].message.content

    return json.loads(response)

'''
    Geoparsing text with a selfhosted LLM
'''
async def geoparseTextSelfHosted(text: str, provider: dict):
    response = requests.post(
        url=provider["data"]["hostserver_url"] + '/chat/completions',
        json={
            "model": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "messages": [
                {
                    "role": "system",
                    "content": "Extract geographic references from the input. For each location, provide the name (how it appears in the text), latitude, and longitude as a json-object, like { name: ..., position: [latitude, longitude] } and create a json-list out of these objects. Please only return the value with no explanation or further information and as a normal text, without labeling it as json.",
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
                                        "type": "number"
                                    }
                                }
                            },
                            "required": ["name", "position"]
                        }
                    }
                }
            },
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": False
        }, 
        headers={"Content-Type": "application/json"},
    )
    output=response.json()

    return json.loads(output['choices'][0]['message']['content'])

# '''
#     Geoparsing text with a BERT model
# '''
# def geoparseTextBERT(text: str):
#     # List of extracted locations
#     locations = []

#     # Extract placenames and append the corresponding coordinates
#     for entity in bert_model(text):
#         if entity['entity'] == 'I-LOC':  # Search for Locations
#             location = entity['word']
#             location_obj = geolocator.geocode(location)
#             if location_obj:
#                 lat = round(float(location_obj.latitude), 2)
#                 long = round(float(location_obj.longitude), 2)
#                 locations.append({ "name": location, "position": [lat, long] })
#             time.sleep(1)  # Delay, to respect the API-rate limit

#     return locations