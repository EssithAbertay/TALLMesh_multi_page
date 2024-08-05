# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:16:46 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""
import streamlit as st
from openai import OpenAI, AzureOpenAI
import pandas as pd
import json # replacing ast
from api_key_management import manage_api_keys, load_api_keys
from project_utils import get_projects, get_project_files, get_processed_files
import os
from prompts import finding_themes_prompts
import anthropic
from azure_model_mapping import azure_model_maps

PROJECTS_DIR = 'projects'

def extract_json(text):
    import re
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return match.group(0)
    return None

# reduction of codes leaves duplicate codes and descriptions (intended but becomes problematic when identifying similar codes)
def preprocess_codes(df):
    # Create a dictionary to store unique codes
    unique_codes = {}

    # Iterate through the DataFrame
    for _, row in df.iterrows():
        code = row['code']
        description = row['description']
        quote = row['quote']
        source = row['source']

        # If the code is not in the dictionary, add it
        if code not in unique_codes:
            unique_codes[code] = {
                'description': description,
                'quotes': [],
                'sources': []
            }

        # Add the quote and source to the code's entry
        unique_codes[code]['quotes'].append(quote)
        unique_codes[code]['sources'].append(source)

    # Create a new DataFrame from the unique_codes dictionary
    preprocessed_df = pd.DataFrame([
        {
            'code': code,
            'description': data['description'],
            'quotes': ', '.join(data['quotes']),
            'sources': ', '.join(data['sources'])
        }
        for code, data in unique_codes.items()
    ])

    #st.write(preprocessed_df)
    return preprocessed_df

# Function to process codes
def process_codes(selected_files, model, prompt, model_temperature, model_top_p):
    all_codes = []
    for file in selected_files:
        df = pd.read_csv(file)
        all_codes.append(df)
    
    # Combine all DataFrames
    combined_df = pd.concat(all_codes, ignore_index=True)
    
    # Preprocess the combined df
    preprocessed_df = preprocess_codes(combined_df)
    
    # Create the codes list for the prompt
    codes_list = [f"[{i}]: {row['code']}: {row['description']}" for i, (_, row) in enumerate(preprocessed_df.iterrows())]
    
    full_prompt = f"{prompt}\n\nCodes:\n{', '.join(codes_list)}"
    
    if model.startswith("gpt"):
        client = OpenAI(api_key=load_api_keys().get('OpenAI'))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            response_format={ "type": "json_object" },
            temperature=model_temperature,
            top_p=model_top_p
        )
        processed_output = response.choices[0].message.content
    elif model.startswith("claude"):
        client = anthropic.Anthropic(api_key=load_api_keys().get('Anthropic'))
        response = client.messages.create(
            model="claude-3-sonnet-20240620",
            max_tokens=1500,
            temperature=model_temperature,
            top_p=model_top_p,
            messages=[{"role": "user", "content": full_prompt}]
        )
        processed_output = response.content[0].text

    elif model.startswith("azure"): # will need a dict of names : models as azure models share names with gpt models
        azure_key = st.session_state.api_keys['Azure']['key']
        azure_endpoint = st.session_state.api_keys['Azure']['endpoint']
        client = AzureOpenAI(
            api_key = azure_key,
            api_version="2024-02-01", # 2023-12-01-preview
            azure_endpoint = azure_endpoint
        )
        processed_output = client.chat.completions.create(
                model=azure_model_maps[model],
                messages = [{"role": "user", "content": prompt}],
                temperature=0,
                top_p=model_top_p
            ).choices[0].message.content
    
    json_string = extract_json(processed_output)

    if json_string:
        return json.loads(json_string), preprocessed_df
    else:
        st.warning("No valid JSON found in the response")
        return None
    
def save_themes(project_name, df):
    themes_folder = os.path.join(PROJECTS_DIR, project_name, 'themes')
    os.makedirs(themes_folder, exist_ok=True)
    
    output_file_path = os.path.join(themes_folder, f"themes_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(output_file_path, index=False, encoding='utf-8')
    return output_file_path

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

def main():
    # session_state persists through page changes so need to reset the text input message 
    if 'current_prompt' in st.session_state:
        del st.session_state.current_prompt

    st.header(":orange[Finding Themes]")
    st.subheader(":orange[Project & Data Selection]")

    projects = get_projects()
    
    # Initialize session state for selected project if it doesn't exist
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = "Select a project..."

    # Calculate the index for the selectbox
    project_options = ["Select a project..."] + projects
    if st.session_state.selected_project in project_options:
        index = project_options.index(st.session_state.selected_project)
    else:
        index = 0

    # Use selectbox with the session state as the default value
    selected_project = st.selectbox(
        "Select a project:", 
        project_options,
        index=index,
        key="project_selector"
    )

    # Update session state when a new project is selected
    if selected_project != st.session_state.selected_project:
        st.session_state.selected_project = selected_project
        st.rerun()

    if selected_project != "Select a project...":
        project_files = get_project_files(selected_project, 'reduced_codes')
        
        with st.expander("Select files to process", expanded=True):
            col1, col2 = st.columns([0.9, 0.1])
            select_all = col2.checkbox("Select All", value=True)
            
            file_checkboxes = {}
            for i, file in enumerate(project_files):
                col1, col2 = st.columns([0.9, 0.1])
                col1.write(file)
                file_checkboxes[file] = col2.checkbox(".", key=f"checkbox_{file}", value=select_all, label_visibility="hidden")

        selected_files = [os.path.join(PROJECTS_DIR, selected_project, 'reduced_codes', file) for file, checked in file_checkboxes.items() if checked]

        st.divider()
        st.subheader(":orange[LLM Settings]")

        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "claude-sonnet-3.5", "azure_model_1"]
        selected_model = st.selectbox("Select Model", model_options)

        max_temperature_value = 2.0 if selected_model.startswith('gpt') else 1.0

        selected_preset = st.selectbox("Select a preset prompt:", list(finding_themes_prompts.keys()))

        if 'current_prompt' not in st.session_state or selected_preset != st.session_state.get('last_selected_preset'):
            st.session_state.current_prompt = finding_themes_prompts[selected_preset]
            st.session_state.last_selected_preset = selected_preset

        prompt_input = st.text_area("Edit prompt if needed:", value=st.session_state.current_prompt, height=200)
        
        settings_col1, settings_col2 = st.columns([0.5, 0.5])
        with settings_col1:
            model_temperature = st.slider(label="Model Temperature", min_value=float(0), max_value=float(max_temperature_value), step=0.01, value=1.0)
        with settings_col2:
            model_top_p = st.slider(label="Model Top P", min_value=float(0), max_value=float(1), step=0.01, value=1.0)

        

        if st.button("Process"):
            st.divider()
            st.subheader(":orange[Output]")
            with st.spinner("Finding themes... please wait..."):
                themes_output, processed_df = process_codes(selected_files, selected_model, prompt_input, model_temperature, model_top_p)
                
                if themes_output is not None:
                    themes_df = pd.json_normalize(themes_output['themes'])
                    
                    st.write(":orange[Generated Themes:]")
                    for _, theme in themes_df.iterrows():
                        with st.expander(f"{theme['name']}"):
                            st.write(f"Description: {theme['description']}")
                            st.write("Codes:")
                            for code_index in theme['codes']:
                                code_row = processed_df.iloc[code_index]
                                st.write(f"- [{code_index}] {code_row['code']}: {code_row['description']}")
                    st.write(":orange[Codes & Descriptions (for reference):]")
                    with st.expander("Codes & Descriptions:"):
                        st.write(processed_df)
                
                    
                    saved_file_path = save_themes(selected_project, themes_df)
                    st.success(f"Themes saved to {saved_file_path}")
                    
                    csv = themes_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download themes",
                        data=csv,
                        file_name="themes.csv",
                        mime="text/csv"
                    )

        # View previously processed files
        processed_files = get_processed_files(selected_project, 'themes')
        with st.expander("Saved Themes", expanded=False):
            for processed_file in processed_files:
                col1, col2 = st.columns([0.9, 0.1])
                col1.write(processed_file)
                if col2.button("Delete", key=f"delete_{processed_file}"):
                    os.remove(os.path.join(PROJECTS_DIR, selected_project, 'themes', processed_file))
                    st.success(f"Deleted {processed_file}")
                    st.rerun()
                
                df = pd.read_csv(os.path.join(PROJECTS_DIR, selected_project, 'themes', processed_file))
                st.write(df)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download {processed_file}",
                    data=csv,
                    file_name=processed_file,
                    mime="text/csv"
                )

    else:
        st.write("Please select a project to continue. If you haven't set up a project yet, head over to the '🏠 Folder Set Up' page to get started.")

    manage_api_keys()

if __name__ == "__main__":
    main()
