from flask import Flask, render_template, request, jsonify
import os
import requests
import base64
from dotenv import load_dotenv
import logging
import json
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

app = Flask(__name__)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Use the best model by default
DEFAULT_MODEL = 'stabilityai/stable-diffusion-xl-base-1.0'

# High-quality prompt enhancement
PROMPT_ENHANCEMENT = 'professional photography, 8k uhd, highly detailed, photorealistic, masterpiece, trending on artstation, sharp focus, dramatic lighting, rule of thirds'

# Get tokens
read_token = os.getenv('HUGGINGFACE_READ_TOKEN')
write_token = os.getenv('HUGGINGFACE_WRITE_TOKEN')

read_headers = {
    "Authorization": f"Bearer {read_token}",
    "Content-Type": "application/json"
}

write_headers = {
    "Authorization": f"Bearer {write_token}",
    "Content-Type": "application/json"
}

# Configure retry strategy
retry_strategy = Retry(
    total=3,  # number of retries
    backoff_factor=1,  # wait 1, 2, 4 seconds between retries
    status_forcelist=[429, 500, 502, 503, 504]  # status codes to retry on
)

# Create session with retry strategy
session = requests.Session()
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        # Enhance prompt with high-quality parameters
        enhanced_prompt = f"{prompt}, {PROMPT_ENHANCEMENT}"

        logging.info(f"Generating image for prompt: {enhanced_prompt}")
        
        api_url = f"https://api-inference.huggingface.co/models/{DEFAULT_MODEL}"
        
        # First try with write token
        response = try_generate_image(enhanced_prompt, api_url, write_headers)
        if response.status_code == 200:
            return process_successful_response(response)
            
        # If write token fails, try with read token
        logging.info("Write token failed, trying with read token...")
        response = try_generate_image(enhanced_prompt, api_url, read_headers)
        if response.status_code == 200:
            return process_successful_response(response)
            
        # If both failed, return error
        error_message = f"API Error: {response.text}"
        logging.error(error_message)
        return jsonify({'error': error_message}), 500

    except Exception as e:
        error_message = f"Error: {str(e)}"
        logging.error(error_message)
        return jsonify({'error': error_message}), 500

def try_generate_image(prompt, api_url, headers):
    payload = {
        "inputs": prompt,
        "options": {
            "wait_for_model": True,
            "use_cache": False,
            "num_inference_steps": 50,  # More steps for better quality
            "guidance_scale": 7.5,  # Better prompt following
            "negative_prompt": "blurry, bad quality, distorted, deformed, ugly, bad anatomy, watermark, signature, extra limbs",
        }
    }
    
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        try:
            logging.info(f"Attempt {attempt + 1} of {max_attempts}")
            response = session.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=90  # Increased timeout to 90 seconds
            )
            
            logging.info(f"Response status: {response.status_code}")
            logging.info(f"Response headers: {dict(response.headers)}")
            
            # Check for specific error cases
            if response.status_code == 401:
                raise Exception("Authentication failed. Please check your Hugging Face API tokens.")
            elif response.status_code == 403:
                raise Exception("Access denied. Your API token might not have the required permissions.")
            elif response.status_code == 503:
                if attempt < max_attempts - 1:
                    wait_time = (attempt + 1) * 5  # Progressive waiting: 5s, 10s, 15s
                    logging.info(f"Model is loading. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                else:
                    raise Exception("Model is still loading after multiple attempts. Please try again later.")
            elif response.status_code != 200:
                error_msg = "Unknown error occurred"
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', error_msg)
                except:
                    pass
                raise Exception(f"API Error ({response.status_code}): {error_msg}")
            
            # Check response content
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type and 'application/json' in content_type:
                json_response = response.json()
                if 'error' in json_response:
                    raise Exception(f"API Error: {json_response['error']}")
                
            return response
            
        except requests.exceptions.Timeout:
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 5
                logging.info(f"Request timed out. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                attempt += 1
            else:
                raise Exception("Request timed out after multiple attempts. The server might be experiencing high load.")
        except requests.exceptions.ConnectionError:
            raise Exception("Connection failed. Please check your internet connection.")
        except Exception as e:
            if "Authentication failed" in str(e) or "API Error" in str(e):
                raise  # Re-raise authentication and API errors immediately
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 5
                logging.info(f"Error occurred: {str(e)}. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                attempt += 1
            else:
                logging.error(f"Error in try_generate_image after {max_attempts} attempts: {str(e)}")
                raise Exception(f"Failed to generate image after multiple attempts: {str(e)}")

def process_successful_response(response):
    image_bytes = response.content
    image_base64 = base64.b64encode(image_bytes).decode()
    
    return jsonify({
        'success': True,
        'image': f'data:image/jpeg;base64,{image_base64}'
    })

if __name__ == '__main__':
    # Only use debug mode in development
    is_development = os.getenv('FLASK_ENV') == 'development'
    app.run(debug=is_development)
