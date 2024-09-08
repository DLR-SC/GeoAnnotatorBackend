from openai import OpenAI
import json

'''
    Geoparsing text with an OpenAI-GPT model
'''
def geoparseTextGPT(text: str, provider: dict):
    response = OpenAI(api_key=provider['data']['api_key']).chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f'''
                            Extract geographic references from the following text: {text}.
                            For each location, provide the name, latitude, and longitude as a json-object, like {{ name: ..., position: [latitude, longitude] }}. 
                            Please only return the json-object with no explanation or further information and as a normal text, without labeling it as json.
                        ''',
                    }
                ],
                model=provider['data']['model']
            ).choices[0].message.content

    return json.loads(response)

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