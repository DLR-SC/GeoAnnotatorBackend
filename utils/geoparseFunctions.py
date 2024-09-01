from openai import OpenAI
import os

'''
    Geoparsing text with an OpenAI-GPT model
'''
def geoparseTextGPT(text: str):
    return OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            ).chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Extract geographic references from the following text: {text}. For each location, provide the name, latitude, and longitude as a json-object, like {{ name: ..., latitude: ..., longitude: ...}}. Please only return the json-object with no explanation or further information and as a normal text, without labeling it as json.",
                    }
                ],
                model="gpt-4o-mini"
            )