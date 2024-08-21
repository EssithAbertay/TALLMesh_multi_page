from openai import OpenAI, AzureOpenAI
import anthropic
import streamlit as st
from api_key_management import load_api_keys, load_azure_settings
import time
import random
import logging
import json
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def exponential_backoff(attempt, max_attempts=5, base_delay=5, max_delay=120):
    if attempt >= max_attempts:
        raise Exception("Max retry attempts reached")
    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.5 * (2 ** attempt)), max_delay)
    logger.info(f"Backing off for {delay:.2f} seconds (attempt {attempt + 1}/{max_attempts})")
    time.sleep(delay)

def extract_json(text):
    try:
        # First, try to parse the entire text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If that fails, try to find a JSON-like structure
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None

def llm_call(model, full_prompt, model_temperature, model_top_p):
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            if model.startswith("gpt"):
                client = OpenAI(api_key=load_api_keys().get('OpenAI'))
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    response_format={ "type": "json_object" },
                    temperature=model_temperature,
                    top_p=model_top_p
                )
                return response.choices[0].message.content
            
            elif model.startswith("claude"):
                client = anthropic.Anthropic(api_key=load_api_keys().get('Anthropic'))
                response = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=8192,
                    temperature=model_temperature,
                    top_p=model_top_p,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                content = response.content[0].text
                json_content = extract_json(content)
                if json_content is None:
                    logger.warning(f"Failed to extract valid JSON from Claude's response. Raw content: {content}")
                    raise ValueError("Failed to extract valid JSON from Claude's response")
                return json.dumps(json_content)

            elif model.startswith("azure_"):
                azure_settings = load_azure_settings()
                if not azure_settings:
                    st.error("Azure settings are not configured. Please set them up in the Azure Settings page.")
                    return None

                deployment_name = model.split("azure_")[1]
                deployment = next((d for d in azure_settings['deployments'] if d['deployment_name'] == deployment_name), None)
                
                if not deployment:
                    st.error(f"Selected Azure deployment '{deployment_name}' not found in settings.")
                    return None

                client = AzureOpenAI(
                    api_key=azure_settings['api_key'],
                    api_version="2024-02-01",
                    azure_endpoint=azure_settings['endpoint']
                )
                response = client.chat.completions.create(
                    model=deployment['deployment_name'],
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=model_temperature,
                    top_p=model_top_p
                )
                content = response.choices[0].message.content
                json_content = extract_json(content)
                if json_content is None:
                    logger.warning(f"Failed to extract valid JSON from Azure's response. Raw content: {content}")
                    raise ValueError("Failed to extract valid JSON from Azure's response")
                return json.dumps(json_content)

            else:
                st.error(f"Unsupported model type: {model}")
                return None

        except Exception as e:
            if hasattr(e, 'status_code') and e.status_code == 429:
                logger.info(f"Rate limit error (429) hit... backing off gracefully (attempt {attempt + 1}/{max_attempts})")
                exponential_backoff(attempt)
            elif isinstance(e, ValueError) and "Failed to extract valid JSON" in str(e):
                logger.warning(f"JSON extraction failed (attempt {attempt + 1}/{max_attempts}): {str(e)}")
                if attempt < max_attempts - 1:
                    logger.info("Retrying with the same prompt...")
                    time.sleep(2)  # Short delay before retrying
                else:
                    logger.error("Max attempts reached for JSON extraction")
                    return None
            else:
                logger.error(f"Unexpected error occurred: {str(e)}")
                st.error(f"An unexpected error occurred: {str(e)}")
                return None

    st.error("Max retry attempts reached. Unable to complete the API call.")
    return None