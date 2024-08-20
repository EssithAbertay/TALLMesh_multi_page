# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:51:05 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk

This script implements the Initial Coding page of the TALLMesh application.
It allows users to process text files using AI models for initial thematic coding.
"""

import streamlit as st
import os
import pandas as pd
import json
import re
from openai import OpenAI, AzureOpenAI
import anthropic
from api_key_management import manage_api_keys, load_api_keys, load_azure_settings, get_azure_models, AZURE_SETTINGS_FILE
from prompts import initial_coding_prompts
from project_utils import get_projects, get_project_files, get_processed_files, PROJECTS_DIR
from llm_utils import llm_call
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_custom_prompts():
    """
    Load user-defined custom prompts from a JSON file.
    
    Returns:
        dict: A dictionary of custom prompts, or an empty dict if the file is not found.
    """
    try:
        with open('custom_prompts.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def extract_json(text):
    """
    Extract the first occurrence of a JSON-like structure from the given text.
    
    Args:
        text (str): The text to search for JSON.
    
    Returns:
        str: The extracted JSON string, or None if no JSON is found.
    """
    match = re.search(r'\{[\s\S]*\}', text)
    return match.group(0) if match else None

def save_uploaded_files(uploaded_files, project_name):
    """
    Save uploaded files to the project's data folder.
    
    Args:
        uploaded_files (list): List of uploaded file objects.
        project_name (str): Name of the current project.
    
    Returns:
        list: Names of the files that were successfully saved.
    """
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
    """
    Save the initial codes DataFrame to a CSV file in the project's initial_codes folder.
    
    Args:
        project_name (str): Name of the current project.
        file_name (str): Name of the original file being processed.
        df (pandas.DataFrame): DataFrame containing the initial codes.
    
    Returns:
        str: Path of the saved CSV file.
    """
    initial_codes_folder = os.path.join(PROJECTS_DIR, project_name, 'initial_codes')
    os.makedirs(initial_codes_folder, exist_ok=True)
    
    file_name_stripped = str(os.path.splitext(file_name)[0])
    output_file_path = os.path.join(initial_codes_folder, f"{file_name_stripped}_initial_codes.csv")
    df.to_csv(output_file_path, index=False, encoding='utf-8')
    return output_file_path

def split_text(text, max_chunk_size=50000):
    """
    Split the input text into chunks of approximately max_chunk_size characters.
    
    Args:
        text (str): The input text to be split.
        max_chunk_size (int): Maximum size of each chunk in characters.
    
    Returns:
        list: A list of text chunks.
    """
    logger.info("Starting to split text into chunks.")
    
    chunks = []
    current_chunk = ""
    
    logger.info(f"Using max_chunk_size: {max_chunk_size} characters.")
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    logger.info(f"Total number of sentences to process: {len(sentences)}")

    for i, sentence in enumerate(sentences):
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            logger.info(f"Chunk {len(chunks)} created with size {len(current_chunk.strip())} characters.")
            current_chunk = sentence + " "
            logger.info(f"Starting a new chunk with sentence {i+1}.")
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        logger.info(f"Final chunk created with size {len(current_chunk.strip())} characters.")
    
    logger.info(f"Total number of chunks created: {len(chunks)}")
    return chunks

def process_file(file_path, model, prompt, model_temperature, model_top_p):
    """
    Process a single file using the specified AI model and prompt.
    
    Args:
        file_path (str): Path to the file to be processed.
        model (str): Name of the AI model to use.
        prompt (str): The prompt to use for processing.
        model_temperature (float): Temperature setting for the AI model.
        model_top_p (float): Top P setting for the AI model.
    
    Returns:
        str: JSON string containing the processed output (initial codes).
    """
    # Read file content with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split content into manageable chunks
    chunks = split_text(content)
    all_codes = []
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        chunk_prompt = f"{prompt}\n\nFile Content (Part {i+1}/{len(chunks)}):\n{chunk}"
        chunk_response = llm_call(model, chunk_prompt, model_temperature, model_top_p)
        
        # Extract and parse JSON from the response
        json_string = extract_json(chunk_response)
        if json_string:
            json_output = json.loads(json_string)
            all_codes.extend(json_output.get('final_codes', []))
        else:
            st.warning(f"No valid JSON found in the response for chunk {i+1}")
    
    # Combine all codes from different chunks
    combined_output = {'final_codes': all_codes}
    return json.dumps(combined_output)

def main():
    """
    Main function to run the Initial Coding page of the TALLMesh application.
    """
    # Reset the current prompt in session state
    if 'current_prompt' in st.session_state:
        del st.session_state.current_prompt 

    st.header(":orange[Initial Coding]")

    # Display instructions
    with st.expander("Instructions"):
        display_instructions()

    st.subheader(":orange[Project & Data Selection]")

    # Project selection
    selected_project = select_project()

    if selected_project != "Select a project...":
        # File upload and processing
        handle_file_upload(selected_project)
        process_selected_files(selected_project)

        # LLM Settings
        st.divider()
        st.subheader(":orange[LLM Settings]")
        model, prompt, temperature, top_p = configure_llm_settings()

        # Process button
        if st.button("Process"):
            process_files(selected_project, model, prompt, temperature, top_p)

        # View previously processed files
        display_saved_initial_codes(selected_project)
    else:
        st.write("Please select a project to continue. If you haven't set up a project yet, head over to the '🏠 Folder Set Up' page to get started")

    # Manage API keys
    manage_api_keys()

def display_instructions():
    """
    Display instructions for using the Initial Coding page.
    """
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

def select_project():
    """
    Allow the user to select a project from the available projects.
    
    Returns:
        str: The name of the selected project.
    """
    projects = get_projects()
    
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = "Select a project..."

    project_options = ["Select a project..."] + projects
    index = project_options.index(st.session_state.selected_project) if st.session_state.selected_project in project_options else 0

    selected_project = st.selectbox(
        "Select a project:", 
        project_options,
        index=index,
        key="project_selector"
    )

    if selected_project != st.session_state.selected_project:
        st.session_state.selected_project = selected_project
        st.rerun()

    return selected_project

def handle_file_upload(project_name):
    """
    Handle file upload for the selected project.
    
    Args:
        project_name (str): Name of the current project.
    """
    uploaded_files = st.file_uploader("Upload additional files or select below", type=["txt"], accept_multiple_files=True)
    if uploaded_files:
        saved_files = save_uploaded_files(uploaded_files, project_name)
        if saved_files:
            st.success(f"Files uploaded successfully: {', '.join(saved_files)}")
        else:
            st.info("No new files were uploaded. They may already exist in the project.")

def process_selected_files(project_name):
    """
    Allow user to select files for processing and display their status.
    
    Args:
        project_name (str): Name of the current project.
    """
    project_files = get_project_files(project_name, 'data')
    processed_files = get_processed_files(project_name, 'initial_codes')
    processed_file_names = [os.path.splitext(f)[0].replace('_initial_codes', '') for f in processed_files]
    
    with st.expander("Select files to process", expanded=True):
        col1, col2 = st.columns([0.9, 0.2])
        select_all = col2.checkbox("Select All", value=True)
        
        file_checkboxes = {}
        for file in project_files:
            col1, col2 = st.columns([0.9, 0.2])
            col1.write(file)
            is_processed = file.replace('.txt', '') in processed_file_names
            if is_processed:
                col2.warning("Already processed", icon="⚠️")
            else:
                file_checkboxes[file] = col2.checkbox(".", key=f"checkbox_{file}", value=select_all, label_visibility="hidden")
    
    return [file for file, checked in file_checkboxes.items() if checked]

def configure_llm_settings():
    """
    Configure the settings for the Language Model.
    
    Returns:
        tuple: (selected_model, prompt_input, model_temperature, model_top_p)
    """
    # Model selection
    default_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "claude-sonnet-3.5"]
    azure_models = get_azure_models()
    model_options = default_models + azure_models
    selected_model = st.selectbox("Select Model", model_options)
    
    # Set max temperature based on model type
    max_temperature_value = 2.0 if selected_model.startswith('gpt') else 1.0
    
    # Load and combine prompts
    custom_prompts = load_custom_prompts().get('Initial Coding', {})
    all_prompts = {**initial_coding_prompts, **custom_prompts}

    # Prompt selection and editing
    selected_prompt = st.selectbox("Select a prompt:", list(all_prompts.keys()))
    selected_prompt_data = all_prompts[selected_prompt]
    prompt_input = st.text_area("Edit prompt if needed:", value=selected_prompt_data["prompt"], height=200)
    
    # Temperature and Top P settings
    settings_col1, settings_col2 = st.columns([0.5, 0.5])
    with settings_col1:
        model_temperature = st.slider("Model Temperature", min_value=0.0, max_value=max_temperature_value, step=0.01, value=selected_prompt_data["temperature"])
    with settings_col2:
        model_top_p = st.slider("Model Top P", min_value=0.0, max_value=1.0, step=0.01, value=selected_prompt_data["top_p"])

    return selected_model, prompt_input, model_temperature, model_top_p

def process_files(project_name, model, prompt, temperature, top_p):
    """
    Process the selected files using the configured LLM settings.
    
    Args:
        project_name (str): Name of the current project.
        model (str): Selected AI model.
        prompt (str): Configured prompt for processing.
        temperature (float): Model temperature setting.
        top_p (float): Model top_p setting.
    """
    st.divider()
    st.subheader(":orange[Output]")
    with st.spinner("Generating initial codes ... please wait ..."):
        prog_bar = st.progress(0)
        selected_files = process_selected_files(project_name)
        if selected_files and prompt:
            for i, file in enumerate(selected_files):
                file_path = os.path.join(PROJECTS_DIR, project_name, 'data', file)
                try:
                    processed_output = process_file(file_path, model, prompt, temperature, top_p)
                    json_output = json.loads(processed_output)
                    df = pd.json_normalize(json_output['final_codes'])
                    df.columns = ['code', 'description', 'quote']
                    
                    with st.expander(f"Processed Output for {file}:", expanded=True):
                        st.write(df)
                        
                        # Save and provide download for initial codes
                        saved_file_path = save_initial_codes(project_name, file, df)
                        st.success(f"Initial codes saved to {saved_file_path}")
                        
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Download initial codes for {str(os.path.splitext(file)[0])}",
                            data=csv,
                            file_name=f"{str(os.path.splitext(file)[0])}_initial_codes.csv",
                            mime="text/csv"
                        )
                    
                except (ValueError, json.JSONDecodeError) as e:
                    st.warning(f"Error processing {file}: {e}")
                    st.text(processed_output)
                
                progress = (i + 1) / len(selected_files)
                prog_bar.progress(progress)
        else:
            st.warning("Please select files and enter a prompt.")

def display_saved_initial_codes(project_name):
    """
    Display and manage previously processed files (saved initial codes).
    
    Args:
        project_name (str): Name of the current project.
    """
    with st.expander("Saved Initial Codes", expanded=False):
        processed_files = get_processed_files(project_name, 'initial_codes')
        for processed_file in processed_files:
            col1, col2 = st.columns([0.9, 0.1])
            col1.write(processed_file)
            if col2.button("Delete", key=f"delete_{processed_file}"):
                os.remove(os.path.join(PROJECTS_DIR, project_name, 'initial_codes', processed_file))
                st.success(f"Deleted {processed_file}")
                st.rerun()
            
            df = pd.read_csv(os.path.join(PROJECTS_DIR, project_name, 'initial_codes', processed_file))
            st.write(df)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download {processed_file}",
                data=csv,
                file_name=processed_file,
                mime="text/csv"
            )

if __name__ == "__main__":
    main()