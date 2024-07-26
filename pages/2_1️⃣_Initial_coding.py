# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:51:05 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""
import streamlit as st
import openai
#import json #I could not parse the json with json loader had to use ast, this needs fixed
import pandas as pd
import ast
from openai import AzureOpenAI
from api_key_management import manage_api_keys


def get_completion(prompt, model):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def main():
    global client  # Declare client as a global variable
    #st.title("Page 1")

    # Sidebar for additional options
    #st.sidebar.title("Select Model and Provide Your Key")

    # Model selection menu
    #model_options = ["gpt-3.5-turbo-16k", "azure-gpt35", "llama-2", "mistral"]
    model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
    selected_model = st.sidebar.selectbox("Select Model", model_options)


    # Input form for OpenAI API key (password type)
    #api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

    # Call the API key management function instead of asking to reinput keys
    manage_api_keys()

    # For Azure model, ask for endpoint and deployment
    azure_endpoint = None
    azure_deployment = None
    if selected_model == "azure-gpt35":
        azure_endpoint = st.sidebar.text_input("Enter Azure Endpoint",  type="password")
        azure_deployment = st.sidebar.text_input("Enter Azure Deployment",  type="password")

    # Confirm button for updating the API key and model
    #if st.sidebar.button("Confirm Key and Model"):
    #    st.sidebar.success("API Key and Model Confirmed!")
       

    # File uploader for selecting multiple .txt files
    st.header(":orange[Initial Coding]")
    uploaded_files = st.file_uploader("Upload .txt files of your interviews", type=["txt"], accept_multiple_files=True)
    
    # Dropdown for selecting which file to process
    selected_file = st.selectbox("Select an interview to process:", [file.name for file in uploaded_files])
    
    # Display the selected file
    st.write(f"Selected File: {selected_file}")
    
    
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
    if st.button("Process"):
        if uploaded_files and prompt_input:
            # Find the selected file based on the user's choice
            selected_file_object = next(file for file in uploaded_files if file.name == selected_file)
            st.write("Selected File:")
            st.write(selected_file_object.name)
    
            # Read the content of the selected file
            file_content = selected_file_object.read()
    
            # Combine the file content and the entered prompt for processing
            prompt = f"{prompt_input}{{}}{{}}File Content:{{}}{{}}{file_content}"
    
            # Call the get_completion function to get the processed output
            with st.spinner("Processing..."):
                if selected_model == "azure-gpt35":
                    client_azure = AzureOpenAI(
                        api_key=api_key,
                        api_version="2023-12-01-preview",
                        azure_endpoint=azure_endpoint
                    )
                    processed_output = client_azure.chat.completions.create(
                        model=azure_deployment,
                        messages = [{"role": "user", "content": prompt}],
                        temperature=0,
                    ).choices[0].message.content
                    #st.write(type(processed_output))
                else:
                    client = openai
                    client.api_key = api_key
                    #client = openai
                    processed_output = get_completion(prompt, model=selected_model)
    
            # Display the input and output
            
            #st.write("Output:") #for debugging and see the json before is pased to df
            #st.write(processed_output)
    
            try:
                json_output = ast.literal_eval(processed_output)
                st.subheader("Processed JSON Output:")
                st.write("You can download your codes as csv now =========>")
               
                # Save JSON as CSV
                          
                df = pd.json_normalize(json_output['final_codes'])
                df.columns = ['code', 'description', 'quote']
                st.write(df)
                
                        
            except (ValueError, SyntaxError) as e:
                st.warning(f"Unable to parse the output as JSON. Error: {e}")
                st.text("Processed Output:")
                st.text(processed_output)  # Print the processed output for debugging

  

if __name__ == "__main__":
    main()
