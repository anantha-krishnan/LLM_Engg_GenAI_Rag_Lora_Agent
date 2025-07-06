# --- Tool Definitions (as instance methods) ---

import base64
import json, os
from io import BytesIO
from typing import Literal
import global_vars
from typing import get_origin, get_args, Literal, Callable
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv

from google import genai as g_genai
from google.genai import types
from PIL import Image as PILImage

def create_image_with_text_model_gemini(prompt: str):
    """
    Generates a high-quality image using Google's Gemini model gemini-2.0-flash-preview-image-generation.
    The generated image is returned in a dictionary by saving in a key content.

    Args:
        prompt (str): A detailed text description of the image to create.
                      For best results, be specific about the subject,
                      setting, and artistic style.
    """
    return_dict={
        'type':'image',
        'content':'',
    }    
    client = g_genai.Client()

    contents = prompt
    
    response = client.models.generate_content(
        model=global_vars.gemini_image_model,
        contents=contents,
        config=types.GenerateContentConfig(
          response_modalities=['TEXT', 'IMAGE']
        )
    )

    for part in response.candidates[0].content.parts:
      if part.text is not None:
        print(part.text)
      elif part.inline_data is not None:
        pil_image = PILImage.open(BytesIO((part.inline_data.data)))
        return_dict['content']=pil_image
        #image.save('gemini-native-image.png')
        #image.show()
    del client
    return return_dict
    
def create_image_with_vertex_ai(self, prompt: str) -> dict:
    """
    Generates a high-quality image using Google's Imagen model on Vertex AI.

    Args:
        prompt (str): A detailed text description of the image to create.
                      For best results, be specific about the subject,
                      setting, and artistic style.
    """
    print(f"--- Generating image with Vertex AI for prompt: '{prompt}' ---")
    
    response = self.image_gen_model.generate_images(
        prompt=prompt,
        number_of_images=1,
        # aspect_ratio="1:1", # "1:1", "9:16", "16:9"
    )
    
    if response.images:
        generated_image = response.images[0]
        pil_image = PILImage.open(BytesIO(generated_image._image_bytes))
        return {'type': 'image', 'content': pil_image}
    else:
        return {'type': 'json', 'content': '{"status": "failed", "message": "API returned no images."}'}
        
def get_current_weather(location: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> dict:
    """Gets the current weather in a given location.
    It can be a city, state format.

    Args:
        location (str): The city and state, e.g., San Francisco, CA.
        a example if of form San Francisco, CA
        unit (str): The unit of temperature, either 'celsius' or 'fahrenheit'.
    """
    content = {}
    if "san francisco" in location.lower():
        content = {"location": location, "temperature": "15", "unit": unit}
    elif "tokyo" in location.lower():
        content = {"location": location, "temperature": "22", "unit": unit}
    else:
        content = {"location": location, "temperature": "unknown"}
    
    return {'type': 'json', 'content': json.dumps(content)}

def get_stock_price(symbol: str) -> dict:
    """Retrieves the current stock price for a given ticker symbol.
    
    Args:
        symbol (str): The stock ticker symbol, e.g., AAPL for Apple, GOOG for Google.
    """
    if symbol.upper() == "AAPL":
        content = {"symbol": "AAPL", "price": "175.28"}
    else:
        content = {"symbol": symbol, "price": "not found"}

    return {'type': 'json', 'content': json.dumps(content)}

def create_image(msg: str) -> dict:
    """Generates an image from a detailed text prompt. Use this tool when a user explicitly asks to draw or create a picture,
    OR when a visual aid would significantly help in explaining a complex concept or answering a question.

    Args:
        msg (str): the string that can be passed as a input to openai image generation (openai.image.generate) call's keyword argument prompt. 
        It has to obtained from the user's message or intention requesting for a iamge
    """
    load_dotenv('C:/Projects/llm_engg/llm_engineering/.env',override=True)
    openai_api_key=os.getenv('OPEN_API_KEY')
    
    client = OpenAI()
    #print(f"Generating image with prompt: '{msg}'")
    image_response = client.images.generate(
        model="dall-e-3",
        prompt=msg,
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )
    
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    del client
    return {'type': 'image', 'content': image}