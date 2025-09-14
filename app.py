# from flask import Flask, render_template, request, jsonify
# import os
# import requests
# import base64
# from dotenv import load_dotenv
# import logging
# import json
# import time
# from requests.adapters import HTTPAdapter
# from requests.packages.urllib3.util.retry import Retry

# app = Flask(__name__)
# load_dotenv()

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)

# # Use the best model by default
# DEFAULT_MODEL = 'stabilityai/stable-diffusion-xl-base-1.0'

# # High-quality prompt enhancement
# PROMPT_ENHANCEMENT = 'professional photography, 8k uhd, highly detailed, photorealistic, masterpiece, trending on artstation, sharp focus, dramatic lighting, rule of thirds'

# # Get tokens

# read_token = os.getenv('read_token')
# write_token = os.getenv('write_token')

# read_headers = {
#     "Authorization": f"Bearer {read_token}",
#     "Content-Type": "application/json"
# }

# write_headers = {
#     "Authorization": f"Bearer {write_token}",
#     "Content-Type": "application/json"
# }

# # Configure retry strategy
# retry_strategy = Retry(
#     total=3,  # number of retries
#     backoff_factor=1,  # wait 1, 2, 4 seconds between retries
#     status_forcelist=[429, 500, 502, 503, 504]  # status codes to retry on
# )

# # Create session with retry strategy
# session = requests.Session()
# adapter = HTTPAdapter(max_retries=retry_strategy)
# session.mount("http://", adapter)
# session.mount("https://", adapter)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/generate', methods=['POST'])
# def generate_image():
#     try:
#         data = request.json
#         prompt = data.get('prompt')

#         if not prompt:
#             return jsonify({'error': 'No prompt provided'}), 400

#         # Enhance prompt with high-quality parameters
#         enhanced_prompt = f"{prompt}, {PROMPT_ENHANCEMENT}"

#         logging.info(f"Generating image for prompt: {enhanced_prompt}")
        
#         api_url = f"https://api-inference.huggingface.co/models/{DEFAULT_MODEL}"
        
#         # First try with write token
#         response = try_generate_image(enhanced_prompt, api_url, write_headers)
#         if response.status_code == 200:
#             return process_successful_response(response)
            
#         # If write token fails, try with read token
#         logging.info("Write token failed, trying with read token...")
#         response = try_generate_image(enhanced_prompt, api_url, read_headers)
#         if response.status_code == 200:
#             return process_successful_response(response)
            
#         # If both failed, return error
#         error_message = f"API Error: {response.text}"
#         logging.error(error_message)
#         return jsonify({'error': error_message}), 500

#     except Exception as e:
#         error_message = f"Error: {str(e)}"
#         logging.error(error_message)
#         return jsonify({'error': error_message}), 500

# def try_generate_image(prompt, api_url, headers):
#     payload = {
#         "inputs": prompt,
#         "options": {
#             "wait_for_model": True,
#             "use_cache": False,
#             "num_inference_steps": 50,  # More steps for better quality
#             "guidance_scale": 7.5,  # Better prompt following
#             "negative_prompt": "blurry, bad quality, distorted, deformed, ugly, bad anatomy, watermark, signature, extra limbs",
#         }
#     }
    
#     max_attempts = 3
#     attempt = 0
    
#     while attempt < max_attempts:
#         try:
#             logging.info(f"Attempt {attempt + 1} of {max_attempts}")
#             response = session.post(
#                 api_url,
#                 headers=headers,
#                 json=payload,
#                 timeout=90  # Increased timeout to 90 seconds
#             )
            
#             logging.info(f"Response status: {response.status_code}")
#             logging.info(f"Response headers: {dict(response.headers)}")
            
#             # Check for specific error cases
#             if response.status_code == 401:
#                 raise Exception("Authentication failed. Please check your Hugging Face API tokens.")
#             elif response.status_code == 403:
#                 raise Exception("Access denied. Your API token might not have the required permissions.")
#             elif response.status_code == 503:
#                 if attempt < max_attempts - 1:
#                     wait_time = (attempt + 1) * 5  # Progressive waiting: 5s, 10s, 15s
#                     logging.info(f"Model is loading. Waiting {wait_time} seconds before retry...")
#                     time.sleep(wait_time)
#                     attempt += 1
#                     continue
#                 else:
#                     raise Exception("Model is still loading after multiple attempts. Please try again later.")
#             elif response.status_code != 200:
#                 error_msg = "Unknown error occurred"
#                 try:
#                     error_data = response.json()
#                     error_msg = error_data.get('error', error_msg)
#                 except:
#                     pass
#                 raise Exception(f"API Error ({response.status_code}): {error_msg}")
            
#             # Check response content
#             content_type = response.headers.get('content-type', '')
#             if 'image' not in content_type and 'application/json' in content_type:
#                 json_response = response.json()
#                 if 'error' in json_response:
#                     raise Exception(f"API Error: {json_response['error']}")
                
#             return response
            
#         except requests.exceptions.Timeout:
#             if attempt < max_attempts - 1:
#                 wait_time = (attempt + 1) * 5
#                 logging.info(f"Request timed out. Waiting {wait_time} seconds before retry...")
#                 time.sleep(wait_time)
#                 attempt += 1
#             else:
#                 raise Exception("Request timed out after multiple attempts. The server might be experiencing high load.")
#         except requests.exceptions.ConnectionError:
#             raise Exception("Connection failed. Please check your internet connection.")
#         except Exception as e:
#             if "Authentication failed" in str(e) or "API Error" in str(e):
#                 raise  # Re-raise authentication and API errors immediately
#             if attempt < max_attempts - 1:
#                 wait_time = (attempt + 1) * 5
#                 logging.info(f"Error occurred: {str(e)}. Waiting {wait_time} seconds before retry...")
#                 time.sleep(wait_time)
#                 attempt += 1
#             else:
#                 logging.error(f"Error in try_generate_image after {max_attempts} attempts: {str(e)}")
#                 raise Exception(f"Failed to generate image after multiple attempts: {str(e)}")

# def process_successful_response(response):
#     image_bytes = response.content
#     image_base64 = base64.b64encode(image_bytes).decode()
    
#     return jsonify({
#         'success': True,
#         'image': f'data:image/jpeg;base64,{image_base64}'
#     })

# if __name__ == '__main__':
#     # Only use debug mode in development
#     is_development = os.getenv('FLASK_ENV') == 'development'
#     app.run(debug=is_development)







from flask import Flask, render_template, request, jsonify
import os
import requests
import base64
from dotenv import load_dotenv
import logging
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

app = Flask(__name__)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- API Configuration ---
# Hugging Face
HF_DEFAULT_MODEL = 'stabilityai/stable-diffusion-xl-base-1.0'
hf_read_token = os.getenv('read_token')
hf_write_token = os.getenv('write_token')

hf_read_headers = {
    "Authorization": f"Bearer {hf_read_token}",
    "Content-Type": "application/json"
}
hf_write_headers = {
    "Authorization": f"Bearer {hf_write_token}",
    "Content-Type": "application/json"
}

# Gemini
gemini_api_key = os.getenv('GEMINI_API_KEY')
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={gemini_api_key}"
gemini_headers = {"Content-Type": "application/json"}

# --- Request Session with Retry Strategy ---
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

# --- Helper Functions ---

def generate_image_from_hf(prompt):
    """Generates an image using the Hugging Face Inference API."""
    api_url = f"https://api-inference.huggingface.co/models/{HF_DEFAULT_MODEL}"
    enhanced_prompt = f"A professional, vibrant, high-resolution photograph of a freshly made '{prompt}', presented beautifully on a clean plate, with natural lighting and a slightly blurred background."
    
    payload = {
        "inputs": enhanced_prompt,
        "options": {
            "wait_for_model": True,
            "use_cache": False,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "negative_prompt": "blurry, bad quality, distorted, deformed, ugly, bad anatomy, watermark, signature, extra limbs, text, letters",
        }
    }

    # First attempt with write token
    logging.info("Attempting image generation with write token.")
    response = session.post(api_url, headers=hf_write_headers, json=payload, timeout=90)
    
    if response.status_code == 200 and 'image' in response.headers.get('content-type', ''):
        logging.info("Image successfully generated with write token.")
        image_bytes = response.content
        image_base64 = base64.b64encode(image_bytes).decode()
        return f'data:image/jpeg;base64,{image_base64}'

    # Fallback to read token if write token fails
    logging.warning("Write token failed or returned non-image response. Retrying with read token.")
    response = session.post(api_url, headers=hf_read_headers, json=payload, timeout=90)

    if response.status_code == 200 and 'image' in response.headers.get('content-type', ''):
        logging.info("Image successfully generated with read token.")
        image_bytes = response.content
        image_base64 = base64.b64encode(image_bytes).decode()
        return f'data:image/jpeg;base64,{image_base64}'

    # Handle final failure
    error_message = f"Failed to generate image from Hugging Face. Status: {response.status_code}, Response: {response.text}"
    logging.error(error_message)
    raise Exception(error_message)


def generate_text_from_gemini(payload):
    """Generates text using the Google Gemini API."""
    response = session.post(GEMINI_API_URL, headers=gemini_headers, json=payload, timeout=60)
    if response.status_code != 200:
        error_message = f"Gemini API Error. Status: {response.status_code}, Response: {response.json()}"
        logging.error(error_message)
        raise Exception(error_message)
    
    try:
        data = response.json()
        text = data['candidates'][0]['content']['parts'][0]['text']
        return text
    except (KeyError, IndexError) as e:
        error_message = f"Could not parse Gemini response: {data}. Error: {e}"
        logging.error(error_message)
        raise Exception(error_message)

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    """Handles the main recipe generation request."""
    try:
        data = request.json
        dish_name = data.get('dishName')
        language = data.get('language', 'English')

        if not dish_name:
            return jsonify({'error': 'No dish name provided'}), 400

        # 1. Generate Recipe Text with Gemini
        system_prompt_recipe = "You are a world-class chef. Provide a clear, concise, and easy-to-follow recipe. The recipe should have two main sections, marked ONLY with '### Ingredients' and '### Instructions'. These specific English markers MUST NOT be translated, even when the rest of the recipe is in another language. Under '### Ingredients', list each item on a new line starting with a dash. Under '### Instructions', list each step on a new line. Do not add numbers to the steps. Do not use any markdown formatting like bolding."
        user_query_recipe = f"Generate a recipe for: {dish_name}. Provide the entire response, including all text, in {language}."
        
        recipe_payload = {
            "contents": [{"parts": [{"text": user_query_recipe}]}],
            "systemInstruction": {"parts": [{"text": system_prompt_recipe}]},
        }
        
        recipe_text = generate_text_from_gemini(recipe_payload)
        
        # 2. Generate Image with Hugging Face
        # We run this in parallel with text generation for speed, but for simplicity here we do it sequentially.
        # In a production app, consider async requests.
        recipe_image_base64 = generate_image_from_hf(dish_name)

        return jsonify({
            'recipeText': recipe_text,
            'recipeImage': recipe_image_base64
        })

    except Exception as e:
        logging.error(f"Error in /generate_recipe: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/solve_doubt', methods=['POST'])
def solve_doubt():
    """Handles a user's question about the current recipe."""
    try:
        data = request.json
        context = data.get('context')
        doubt = data.get('doubt')
        language = data.get('language', 'English')

        if not all([context, doubt]):
            return jsonify({'error': 'Missing context or doubt'}), 400

        system_prompt_doubt = "You are a friendly and knowledgeable cooking assistant. You will be given a recipe and a user's question about it. Answer the user's question concisely and helpfully, based *only* on the provided recipe context. If the answer isn't in the context, say you don't have that information but offer a general cooking tip if possible."
        user_query_doubt = f"Recipe Context:\n- Dish: {context.get('name')}\n- Full Recipe: {context.get('recipe')}\n\nUser's Question: \"{doubt}\". Please provide your entire answer in {language}."
        
        doubt_payload = {
            "contents": [{"parts": [{"text": user_query_doubt}]}],
            "systemInstruction": {"parts": [{"text": system_prompt_doubt}]},
        }
        
        answer_text = generate_text_from_gemini(doubt_payload)

        return jsonify({'answer': answer_text})

    except Exception as e:
        logging.error(f"Error in /solve_doubt: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    is_development = os.getenv('FLASK_ENV') == 'development'
    app.run(debug=is_development, host='0.0.0.0', port=5000)
