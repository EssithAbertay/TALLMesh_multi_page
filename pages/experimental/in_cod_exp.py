# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:51:05 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""
import streamlit as st
import openai
import pandas as pd
import ast
import zipfile
import io

from openai import AzureOpenAI

def get_completion(prompt, model):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def process_zip_file(zip_file, prompt, selected_model, api_key, azure_endpoint=None, azure_deployment=None):
    # Unzip the uploaded zip file
    zip_data = zipfile.ZipFile(zip_file)
    
    # Initialize an empty DataFrame to store the processed data
    final_df = pd.DataFrame(columns=['code', 'description', 'quote'])
    
    # Iterate through each file in the zip folder
    for file_name in zip_data.namelist():
        if file_name.endswith('.txt'):
            st.write(f"Processing file: {file_name}")
            with zip_data.open(file_name) as file:
                # Read the text file
                file_content = file.read().decode("utf-8")
                
                # Process the content
                prompt_content = prompt + f"{{}}{{}}File Content:{{}}{{}}{file_content}"
                
                if selected_model == "azure-gpt35":
                    client_azure = AzureOpenAI(
                        api_key=api_key,
                        api_version="2023-12-01-preview",
                        azure_endpoint=azure_endpoint
                    )
                    processed_output = client_azure.chat.completions.create(
                        model=azure_deployment,
                        messages=[{"role": "user", "content": prompt_content}],
                        temperature=0,
                    ).choices[0].message.content
                else:
                    client.api_key = api_key
                    processed_output = get_completion(prompt_content, model=selected_model)
                
                # Parse the processed output as JSON
                try:
                    json_output = ast.literal_eval(processed_output)
                    df = pd.json_normalize(json_output['final_codes'])
                    df.columns = ['code', 'description', 'quote']
                    final_df = pd.concat([final_df, df])
                except (ValueError, SyntaxError) as e:
                    st.warning(f"Unable to parse the output as JSON for file {file_name}. Error: {e}")
    
    return final_df


def main():
    global client  # Declare client as a global variable

    # Sidebar for additional options
    st.sidebar.title("Select Model and Provide Your Key")

    # Model selection menu
    model_options = ["gpt-3.5-turbo-16k", "azure-gpt35", "llama-2", "mistral"]
    selected_model = st.sidebar.selectbox("Select Model", model_options)

    # Input form for OpenAI API key (password type)
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

    # For Azure model, ask for endpoint and deployment
    azure_endpoint = None
    azure_deployment = None
    if selected_model == "azure-gpt35":
        azure_endpoint = st.sidebar.text_input("Enter Azure Endpoint",  type="password")
        azure_deployment = st.sidebar.text_input("Enter Azure Deployment",  type="password")

    # Confirm button for updating the API key and model
    if st.sidebar.button("Confirm Key and Model"):
        st.sidebar.success("API Key and Model Confirmed!")
       
    # File uploader for selecting a zip folder containing TXT files
    st.header(":orange[Initial Coding]")
    zip_file = st.file_uploader("Upload a zip folder containing TXT files", type=["zip"])

    # Central text input form for the prompt
    st.header("Enter prompt:")
    prompt_input = st.text_area("Type your prompt here", value="Can you assist in the generation of a very broad range of initial codes "
                                                            "(generate as many initial codes as needed - at least 15 codes - to capture all the significant explicit or latent meaning, "
                                                            "or events in the text, focus on the respondent and not the interviewer), "
                                                            "aiming to encompass a wide spectrum of themes and ideas "
                                                            "present in the text below, to assist me with my thematic analysis. "
                                                            "Provide a name for each code in no more than 4 words, 25 words "
                                                            "dense description of the code and a quote from the respondent for each topic no longer than 4 words. "
                                                            "Format the response as a json file keeping codes, descriptions and quotes together in the json, and keep them together in 'final_codes'.")

    # Display the processed output
    if st.button("Process") and zip_file:
        if prompt_input:
            final_df = process_zip_file(zip_file, prompt_input, selected_model, api_key, azure_endpoint, azure_deployment)
            st.subheader("Processed Output:")
            st.write("You can download your codes as CSV now =========>")
            st.write(final_df)

if __name__ == "__main__":
    main()
