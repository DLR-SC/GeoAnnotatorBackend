from geopy.geocoders import Nominatim
from transformers import pipeline
from openai import OpenAI
import time
import json
import os

def initializeLLMs():
    # OpenAI
    global openai_model 
    openai_model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # BERT
    global bert_model
    ner_model_name  = "dbmdz/bert-large-cased-finetuned-conll03-english"
    bert_model = pipeline("ner", model=ner_model_name, tokenizer="bert-base-cased")

    # Gazetteer
    global geolocator
    geolocator = Nominatim(user_agent="geoapiExercises")

'''
    Geoparsing text with an OpenAI-GPT model
'''
def geoparseTextGPT(text: str):
    response = openai_model.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f'''
                            Extract geographic references from the following text: {text}.
                            Locations can be seperated with a symbol like a comma. But you need to pay attention to the context, if its referring to the same place.
                            For each location, provide the name, latitude, and longitude as a json-object, like {{ name: ..., position: [latitude, longitude] }}. 
                            Please only return the json-object with no explanation or further information and as a normal text, without labeling it as json.
                        ''',
                    }
                ],
                model="gpt-4o-mini"
            ).choices[0].message.content

    return json.loads(response)

'''
    Geoparsing text with a BERT model
'''
def geoparseTextBERT(text: str):
    # List of extracted locations
    locations = []

    # Extract placenames and append the corresponding coordinates
    for entity in bert_model(text):
        if entity['entity'] == 'I-LOC':  # Search for Locations
            location = entity['word']
            location_obj = geolocator.geocode(location)
            if location_obj:
                locations.append({ "name": location, "position": [location_obj.latitude, location_obj.longitude] })
            time.sleep(1)  # Delay, to respect the API-rate limit

    return locations