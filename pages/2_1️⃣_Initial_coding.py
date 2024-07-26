# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:51:05 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""
import streamlit as st
import os
import pandas as pd
import json
from openai import OpenAI, AzureOpenAI
import anthropic
from api_key_management import manage_api_keys, load_api_keys

PROJECTS_DIR = 'projects'

def get_projects():
    return [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]

def get_project_files(project_name):
    data_folder = os.path.join(PROJECTS_DIR, project_name, 'data')
    return [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

def save_uploaded_files(uploaded_files, project_name):
    data_folder = os.path.join(PROJECTS_DIR, project_name, 'data')
    saved_files = []
    for file in uploaded_files:
        file_path = os.path.join(data_folder, file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            saved_files.append(file.name)
    return saved_files

def save_initial_codes(project_name, file_name, df):
    initial_codes_folder = os.path.join(PROJECTS_DIR, project_name, 'initial_codes')
    os.makedirs(initial_codes_folder, exist_ok=True)
    
    # strip extension from filename ... could be problematic in cases of here.is.some.file.extension
    file_name_stripped = str(os.path.splitext(file_name)[0])

    output_file_path = os.path.join(initial_codes_folder, f"{file_name_stripped}_initial_codes.csv")
    df.to_csv(output_file_path, index=False, encoding='utf-8')
    return output_file_path

def process_file(file_path, model, prompt):
    # Specify encoding to handle potential UnicodeDecodeError
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    full_prompt = f"{prompt}\n\nFile Content:\n{content}"
    
    if model.startswith("gpt"):
        client = OpenAI(api_key=load_api_keys().get('OpenAI'))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content
    
    elif model.startswith("claude"):
        client = anthropic.Anthropic(api_key=load_api_keys().get('Anthropic'))
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            temperature=0.1,
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response.content[0].text
    # Add handling for Azure if needed

def main():
    st.header(":orange[Initial Coding]")
    
    # Project selection
    projects = get_projects()
    selected_project = st.selectbox("Select a project:", ["Select a project..."] + projects)
    
    if selected_project != "Select a project...":
        # File upload
        uploaded_files = st.file_uploader("Upload additional files or select below", type=["txt"], accept_multiple_files=True)
        if uploaded_files:
            saved_files = save_uploaded_files(uploaded_files, selected_project)
            if saved_files:
                st.success(f"Files uploaded successfully: {', '.join(saved_files)}")
            else:
                st.info("No new files were uploaded. They may already exist in the project.")

        # File selection
        project_files = get_project_files(selected_project)
        
        with st.expander("Select files to process", expanded=True):
            col1, col2 = st.columns([0.9, 0.1])
            select_all = col2.checkbox("Select All", value=True)
            
            file_checkboxes = {}
            for i, file in enumerate(project_files):
                col1, col2 = st.columns([0.9, 0.1])
                col1.write(file)
                file_checkboxes[file] = col2.checkbox(".", key=f"checkbox_{file}", value=select_all, label_visibility="hidden")
        
        selected_files = [file for file, checked in file_checkboxes.items() if checked]
        
        # Model selection
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "claude-sonnet-3.5"]
        selected_model = st.selectbox("Select Model", model_options)
        
        # Prompt input
        prompt_input = st.text_area("Type your prompt here", value="Can you assist in the generation of a very broad range of initial codes "
                                                            "(generate as many initial codes as needed - at least 15 codes - to capture all the significant explicit or latent meaning, "
                                                            "or events in the text, focus on the respondent and not the interviewer), "
                                                            "aiming to encompass a wide spectrum of themes and ideas "
                                                            "present in the text below, to assist me with my thematic analysis. "
                                                            "Provide a name for each code in no more than 4 words, 25 words "
                                                            "dense description of the code and a quote from the respondent for each topic no longer than 4 words. "
                                                            "Format the response as a json file keeping codes, descriptions and quotes together in the json, and keep them together in 'final_codes'.")
        
        if st.button("Process"):
            with st.spinner("Processing... beep boop beep"):
                if selected_files and prompt_input:
                    for file in selected_files:
                        file_path = os.path.join(PROJECTS_DIR, selected_project, 'data', file)
                        processed_output = process_file(file_path, selected_model, prompt_input)
                        
                        try:
                            # Use json.loads instead of ast.literal_eval
                            json_output = json.loads(processed_output)
                            df = pd.json_normalize(json_output['final_codes'])
                            df.columns = ['code', 'description', 'quote']
                            
                            st.subheader(f"Processed Output for {file}:")
                            st.write(df)
                            
                            # Save initial codes
                            saved_file_path = save_initial_codes(selected_project, file, df)
                            st.success(f"Initial codes saved to {saved_file_path}")
                            
                            '''
                            # Add download button for each file's results
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label=f"Download CSV for {file}",
                                data=csv,
                                file_name=f"{file}_codes.csv",
                                mime="text/csv"
                            )
                            '''

                        except (ValueError, json.JSONDecodeError) as e:
                            st.warning(f"Error processing {file}: {e}")
                            st.text(processed_output)
                else:
                    st.warning("Please select files and enter a prompt.")
    else:
        st.write("Please select a project to continue.")

    # Call API key management function
    manage_api_keys()

if __name__ == "__main__":
    main()
