from openai import OpenAI, AzureOpenAI
import anthropic
import streamlit as st
from api_key_management import load_api_keys, load_azure_settings
import time
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def exponential_backoff(attempt, max_attempts=5, base_delay=5, max_delay=120):
    if attempt >= max_attempts:
        raise Exception("Max retry attempts reached")
    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.5 * (2 ** attempt)), max_delay)
    logger.info(f"Backing off for {delay:.2f} seconds (attempt {attempt + 1}/{max_attempts})")
    time.sleep(delay)

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
                    model="claude-3-sonnet-20240620",
                    max_tokens=1500,
                    temperature=model_temperature,
                    top_p=model_top_p,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                return response.content[0].text

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
                processed_output = client.chat.completions.create(
                    model=deployment['deployment_name'],
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=model_temperature,
                    top_p=model_top_p
                ).choices[0].message.content
                return processed_output

            else:
                st.error(f"Unsupported model type: {model}")
                return None

        except Exception as e:
            if hasattr(e, 'status_code') and e.status_code == 429:
                logger.info(f"Rate limit error (429) hit... backing off gracefully (attempt {attempt + 1}/{max_attempts})")
                exponential_backoff(attempt)
            else:
                logger.error(f"Unexpected error occurred: {str(e)}")
                st.error(f"An unexpected error occurred: {str(e)}")
                return None

    st.error("Max retry attempts reached. Unable to complete the API call.")
    return None