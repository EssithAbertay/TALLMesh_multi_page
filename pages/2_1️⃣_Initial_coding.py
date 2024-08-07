# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:51:05 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""
import streamlit as st
import os
import pandas as pd
import json
import re
from openai import OpenAI, AzureOpenAI
import anthropic
from api_key_management import manage_api_keys, load_api_keys
from prompts import initial_coding_prompts
from project_utils import get_projects, get_project_files, get_processed_files
from azure_model_mapping import azure_model_maps

PROJECTS_DIR = 'projects'

# Function to find the JSON in AI responses (OpenAI can be set to json format, but anthropic with higher temp sometimes prefaces json)
def extract_json(text):
    # Find the first occurrence of a JSON-like structure in response
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return match.group(0)
    return None

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
    
    file_name_stripped = str(os.path.splitext(file_name)[0])
    output_file_path = os.path.join(initial_codes_folder, f"{file_name_stripped}_initial_codes.csv")
    df.to_csv(output_file_path, index=False, encoding='utf-8')
    return output_file_path

def process_file(file_path, model, prompt, model_temperature, model_top_p):
    # Specify encoding to handle potential UnicodeDecodeError
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    full_prompt = f"{prompt}\n\nFile Content:\n{content}"
    
    if model.startswith("gpt"):
        client = OpenAI(api_key=load_api_keys().get('OpenAI'))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            response_format={ "type": "json_object" },
            temperature = model_temperature,
            top_p = model_top_p
        )
        return response.choices[0].message.content
    
    elif model.startswith("claude"):
        client = anthropic.Anthropic(api_key=load_api_keys().get('Anthropic'))
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1500,
            temperature=model_temperature,
            top_p = model_top_p,
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response.content[0].text
    

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
        return processed_output




    _="""
    elif model.startswith("azure"):
        client_azure = AzureOpenAI(api_key=api_key, api_version="2023-12-01-preview", azure_endpoint=azure_endpoint)
        processed_output = client_azure.chat.completions.create(
                model=azure_deployment,
                messages = [{"role": "user", "content": prompt}],
                temperature=0,
            ).choices[0].message.content"""



    # Add handling for Azure if needed

def main():
    # session_state persists through page changes so need to reset the text input message 
    if 'current_prompt' in st.session_state:
        del st.session_state.current_prompt 

    st.header(":orange[Initial Coding]")

    with st.expander("Instructions"):
        st.write("""
        The Initial Coding page is where you begin the analysis of your data. This step involves generating initial codes for each of your uploaded files using AI assistance. Here's how to use this page:
        """)

        st.subheader(":orange[1. Project Selection]")
        st.write("""
        - Use the dropdown menu to select the project you want to work on.
        - If you haven't set up a project yet, you'll be prompted to go to the 'Folder Set Up' page first.
        """)

        st.subheader(":orange[2. File Selection]")
        st.write("""
        - Once a project is selected, you'll see a list of files available for processing.
        - You can select individual files or use the "Select All" checkbox to choose all files at once.
        - :orange[Files that have already been processed will be marked with a warning icon.]
        """)

        st.subheader(":orange[3. LLM Settings]")
        st.write("""
        - Choose the AI model you want to use for the analysis from the dropdown menu. 
        - Select a preset prompt or edit the provided prompt to customize your analysis.
        - Adjust the model temperature and top_p values using the sliders. These parameters control the creativity and randomness of the AI's output.
        """)

        st.info("Make sure you have provided an API key for the provider of the model you have selected (e.g., Anthropic, Azure, OpenAI)")

        st.subheader(":orange[4. Processing Files]")
        st.write("""
        - After configuring your settings, click the "Process" button to start the initial coding.
        - :orange[The system will process each selected file and generate initial codes.]
        - A progress bar will show you the status of the processing.
        """)

        st.subheader(":orange[5. Viewing Results]")
        st.write("""
        - Once processing is complete, you'll see the results for each file in expandable sections.
        - Each section will show a table with the generated codes, their descriptions, and relevant quotes.
        - You can download the results for each file individually using the provided download buttons.
        """)

        st.subheader(":orange[6. Saved Initial Codes]")
        st.write("""
        - At the bottom of the page, you'll find an expandable section showing previously processed files.
        - You can view, delete, or download these saved initial codes.
        """)

        st.subheader(":orange[Tips]")
        st.write("""
        - :orange[Experiment with different prompts and model settings to get the best results for your data.]
        - If you're not satisfied with the initial codes, you can delete them and reprocess the file with different settings.
        - Remember that this is just the first step in the analysis process. The codes generated here will be refined in later stages.
        """)

        st.info("Initial coding sets the foundation for your thematic analysis. Take your time to review the generated codes and ensure they capture the essence of your data before moving on to the next stage.")


    st.subheader(":orange[Project & Data Selection]")

    # Project selection
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
        # File upload
        uploaded_files = st.file_uploader("Upload additional files or select below", type=["txt"], accept_multiple_files=True)
        if uploaded_files:
            saved_files = save_uploaded_files(uploaded_files, selected_project)
            if saved_files:
                st.success(f"Files uploaded successfully: {', '.join(saved_files)}")
            else:
                st.info("No new files were uploaded. They may already exist in the project.")

        # File selection
        project_files = get_project_files(selected_project, 'data')
        processed_files = get_processed_files(selected_project, 'initial_codes')
        processed_file_names = [os.path.splitext(f)[0].replace('_initial_codes', '') for f in processed_files]
        
        with st.expander("Select files to process", expanded=True):
            col1, col2 = st.columns([0.9, 0.2])
            select_all = col2.checkbox("Select All", value=True)
            
            file_checkboxes = {}
            for i, file in enumerate(project_files):
                col1, col2 = st.columns([0.9, 0.2])
                col1.write(file)
                is_processed = file.replace('.txt', '') in processed_file_names
                if is_processed:
                    col2.warning("Already processed", icon="⚠️")
                else:
                    file_checkboxes[file] = col2.checkbox(".", key=f"checkbox_{file}", value=select_all, label_visibility="hidden")
        
        selected_files = [file for file, checked in file_checkboxes.items() if checked]
        
        st.divider()
        st.subheader(":orange[LLM Settings]")
        
        # Model selection
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "claude-sonnet-3.5", "azure_model_1"]
        selected_model = st.selectbox("Select Model", model_options)

        # OpenAI & Anthropic Models have different max temperature settings (2 & 1, respectively)
        max_temperature_value = 2.0 if selected_model.startswith('gpt') else 1.0
        
        # Prompt selection and input
        selected_preset = st.selectbox("Select a preset prompt:", list(initial_coding_prompts.keys()))

        if 'current_prompt' not in st.session_state or selected_preset != st.session_state.get('last_selected_preset'):
            st.session_state.current_prompt = initial_coding_prompts[selected_preset]
            st.session_state.last_selected_preset = selected_preset

        prompt_input = st.text_area("Edit prompt if needed:", value=st.session_state.current_prompt, height=200)
        settings_col1, settings_col2 = st.columns([0.5, 0.5])
        with settings_col1:
            model_temperature = st.slider(label="Model Temperature", min_value=float(0), max_value=float(max_temperature_value),step=0.01,value=0.1)

        with settings_col2:
            model_top_p = st.slider(label="Model Top P", min_value=float(0), max_value=float(1),step=0.01,value=1.0)

        if st.button("Process"):
            st.divider()
            st.subheader(":orange[Output]")
            with st.spinner("Generating initial codes ... please wait ..."):
                prog_bar = st.progress(0)
                if selected_files and prompt_input:
                    for i, file in enumerate(selected_files):
                        file_path = os.path.join(PROJECTS_DIR, selected_project, 'data', file)
                        try:
                            processed_output = process_file(file_path, selected_model, prompt_input, model_temperature, model_top_p)
                            json_string = extract_json(processed_output)
                            if json_string:
                                json_output = json.loads(json_string)
                                df = pd.json_normalize(json_output['final_codes'])
                                df.columns = ['code', 'description', 'quote']
                                
                                with st.expander(f"Processed Output for {file}:", expanded=True):
                                    st.write(df)
                                
                                    # Save initial codes
                                    saved_file_path = save_initial_codes(selected_project, file, df)
                                    st.success(f"Initial codes saved to {saved_file_path}")
                                
                                    # Add download button for each file's results
                                    csv = df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label=f"Download initial codes for {str(os.path.splitext(file)[0])}",
                                        data=csv,
                                        file_name=f"{str(os.path.splitext(file)[0])}_initial_codes.csv",
                                        mime="text/csv"
                                    )
                            else:
                                st.warning(f"No valid JSON found in the response for {file}")
                                st.text(processed_output)
                            
                        except (ValueError, json.JSONDecodeError) as e:
                            st.warning(f"Error processing {file}: {e}")
                            st.text(processed_output)
                        progress = (i + 1) / len(selected_files)
                        prog_bar.progress(progress)
                        
                else:
                    st.warning("Please select files and enter a prompt.")

        # View previously processed files
        with st.expander("Saved Initial Codes", expanded=False):
            for processed_file in processed_files:
                col1, col2 = st.columns([0.9, 0.1])
                col1.write(processed_file)
                if col2.button("Delete", key=f"delete_{processed_file}"):
                    os.remove(os.path.join(PROJECTS_DIR, selected_project, 'initial_codes', processed_file))
                    st.success(f"Deleted {processed_file}")
                    st.rerun()
                
                df = pd.read_csv(os.path.join(PROJECTS_DIR, selected_project, 'initial_codes', processed_file))
                st.write(df)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download {processed_file}",
                    data=csv,
                    file_name=processed_file,
                    mime="text/csv"
                )
    else:
        st.write(f"Please select a project to continue. If you haven't set up a project yet, head over to the '🏠 Folder Set Up' page to get started")

    # Call API key management function
    manage_api_keys()

if __name__ == "__main__":
    main()