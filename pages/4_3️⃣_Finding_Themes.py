# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:16:46 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk

This script is part of the TALLMesh project and focuses on finding themes from reduced codes.
It provides a Streamlit interface for users to process their coded data and generate themes using AI models.
"""

# Import necessary libraries
import streamlit as st
import pandas as pd
import json
import os
from api_key_management import manage_api_keys, load_api_keys, load_azure_settings, get_azure_models, AZURE_SETTINGS_FILE
from project_utils import get_projects, get_project_files, get_processed_files
from prompts import finding_themes_prompts
from llm_utils import llm_call
import logging
import tooltips
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PROJECTS_DIR = 'projects'

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
    Extract a JSON object from a string using regex.
    
    Args:
        text (str): The input string containing a JSON object.
    
    Returns:
        str: The extracted JSON string, or None if no JSON object is found.
    """
    import re
    match = re.search(r'\{[\s\S]*\}', text)
    return match.group(0) if match else None

def preprocess_codes(df):
    """
    Preprocess the codes dataframe to remove duplicates and combine quotes and sources.
    
    Args:
        df (pandas.DataFrame): The input dataframe containing codes, descriptions, quotes, and sources.
    
    Returns:
        pandas.DataFrame: A preprocessed dataframe with unique codes and combined quotes and sources.
    """
    logger.info("Starting preprocessing of codes")
    unique_codes = {}

    # Iterate through the DataFrame to collect unique codes and their associated data
    for _, row in df.iterrows():
        code = row['code']
        if code not in unique_codes:
            unique_codes[code] = {
                'description': row['description'],
                'quotes': [],
                'sources': []
            }
        unique_codes[code]['quotes'].append(row['quote'])
        unique_codes[code]['sources'].append(row['source'])

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

    logger.info(f"Preprocessing complete. Number of unique codes: {len(preprocessed_df)}")
    return preprocessed_df

def process_codes(selected_files, model, prompt, model_temperature, model_top_p):
    """
    Process the selected code files to find themes using an AI model.
    
    Args:
        selected_files (list): List of file paths to process.
        model (str): The AI model to use for processing.
        prompt (str): The prompt to guide the AI in finding themes.
        model_temperature (float): The temperature setting for the AI model.
        model_top_p (float): The top_p setting for the AI model.
    
    Returns:
        tuple: A tuple containing the processed output (dict) and the preprocessed dataframe.
    """
    logger.info(f"Starting to process codes from {len(selected_files)} files")
    logger.info(f"Model: {model}, Temperature: {model_temperature}, Top P: {model_top_p}")

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Combine all selected files into a single DataFrame
    all_codes = []
    for i, file in enumerate(selected_files):
        progress = (i + 1) / (len(selected_files) + 3)  # +3 for preprocessing, AI processing, and JSON parsing
        progress_bar.progress(progress)
        status_text.info(f"Reading file: {file}")
        logger.info(f"Reading file: {file}")
        df = pd.read_csv(file)
        logger.info(f"File {file} read. Shape: {df.shape}")
        all_codes.append(df)
    combined_df = pd.concat(all_codes, ignore_index=True)
    logger.info(f"Combined DataFrame shape: {combined_df.shape}")
    
    # Preprocess the combined DataFrame
    status_text.info("Preprocessing codes...")
    progress_bar.progress((len(selected_files) + 1) / (len(selected_files) + 3))
    preprocessed_df = preprocess_codes(combined_df)
    
    # Create a list of codes for the prompt
    codes_list = [f"[{i}]: {row['code']}: {row['description']}" for i, (_, row) in enumerate(preprocessed_df.iterrows())]
    
    # Construct the full prompt
    full_prompt = f"{prompt}\n\nCodes:\n{', '.join(codes_list)}"
    logger.info(f"Full prompt constructed. Length: {len(full_prompt)}")
    
    # Process the codes using the AI model
    status_text.info(f"Processing codes with {model} model...")
    progress_bar.progress((len(selected_files) + 2) / (len(selected_files) + 3))
    logger.info("Calling AI model to process codes")
    processed_output = llm_call(model, full_prompt, model_temperature, model_top_p)
    
    # Extract and parse the JSON response
    status_text.info("Parsing AI response...")
    progress_bar.progress(1.0)
    json_string = extract_json(processed_output)
    if json_string:
        logger.info("Successfully extracted JSON from AI response")
        parsed_output = json.loads(json_string)
        logger.info(f"Number of themes found: {len(parsed_output.get('themes', []))}")
        status_text.info("Theme finding process completed successfully.")
        return parsed_output, preprocessed_df
    else:
        logger.warning("No valid JSON found in the response")
        status_text.info("Failed to extract themes from AI response.")
        return None, preprocessed_df

def save_themes(project_name, df):
    """
    Save the generated themes to a CSV file in the project's themes folder.
    
    Args:
        project_name (str): The name of the current project.
        df (pandas.DataFrame): The dataframe containing the generated themes.
    
    Returns:
        str: The path to the saved themes file.
    """
    themes_folder = os.path.join(PROJECTS_DIR, project_name, 'themes')
    os.makedirs(themes_folder, exist_ok=True)
    
    output_file_path = os.path.join(themes_folder, f"themes_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(output_file_path, index=False, encoding='utf-8')
    return output_file_path

@st.cache_data
def convert_df(df):
    """
    Convert a dataframe to a CSV string for downloading.
    
    Args:
        df (pandas.DataFrame): The dataframe to convert.
    
    Returns:
        bytes: The CSV-encoded dataframe as bytes.
    """
    return df.to_csv().encode('utf-8')


# Functions to generate theme code book(s) - previously a separate page


def format_quotes(quotes_json):
    """
    Parses the JSON string of quotes, extracts the text,
    and joins each quote with a newline for better readability.
    
    Args:
    quotes_json (str): A JSON string containing quote information.
    
    Returns:
    str: A formatted string of quotes, each on a new line.
    """
    try:
        quotes = json.loads(quotes_json)
        formatted_quotes = "\n".join(quote['text'] for quote in quotes)
        return formatted_quotes
    except (json.JSONDecodeError, KeyError, TypeError):
        # Return the original string if there's an error in parsing or formatting
        return quotes_json

def load_data(project_name):
    """
    Loads the most recent themes and reduced codes files for a given project.
    
    Args:
    project_name (str): The name of the project to load data for.
    
    Returns:
    tuple: A tuple containing two pandas DataFrames (themes_df, codes_df) or (None, None) if files are not found.
    """
    # Define paths for themes and codes folders
    themes_folder = os.path.join(PROJECTS_DIR, project_name, 'themes')
    codes_folder = os.path.join(PROJECTS_DIR, project_name, 'reduced_codes')
    
    # Get the most recent themes file
    themes_files = get_processed_files(project_name, 'themes')
    if not themes_files:
        return None, None
    latest_themes_file = max(themes_files, key=lambda f: os.path.getmtime(os.path.join(themes_folder, f)))
    themes_df = pd.read_csv(os.path.join(themes_folder, latest_themes_file))
    
    # Get the most recent reduced codes file
    codes_files = get_processed_files(project_name, 'reduced_codes')
    if not codes_files:
        return None, None
    latest_codes_file = max(codes_files, key=lambda f: os.path.getmtime(os.path.join(codes_folder, f)))
    codes_df = pd.read_csv(os.path.join(codes_folder, latest_codes_file))
    
    return themes_df, codes_df

def process_data(themes_df, codes_df):
    """
    Processes the themes and codes data to create a final theme-codes book.
    
    Args:
    themes_df (pandas.DataFrame): DataFrame containing theme data.
    codes_df (pandas.DataFrame): DataFrame containing code data.
    
    Returns:
    pandas.DataFrame: A processed DataFrame combining themes, codes, and associated information.
    """
    # Initialize empty DataFrame for final theme-codes book with correct column names
    final_df = pd.DataFrame(columns=['Theme', 'Theme Description', 'Code', 'Code Description', 'Merge Explanation', 'Quotes', 'Source'])

    # Iterate through each theme and its associated codes
    for _, theme_row in themes_df.iterrows():
        theme = theme_row['name']
        theme_description = theme_row['description']
        code_indices = [int(idx) for idx in theme_row['codes'].strip('[]').split(',')]
        
        # For each code associated with the theme, create a new row in the final DataFrame
        for idx in code_indices:
            if idx < len(codes_df):
                code_row = codes_df.iloc[idx]
                new_row = pd.Series({
                    'Theme': theme,
                    'Theme Description': theme_description,
                    'Code': code_row['code'],
                    'Code Description': code_row['description'],
                    'Merge Explanation': code_row['merge_explanation'],
                    'Quotes': code_row.get('quote', code_row.get('source', '')),  # Try 'quote', then 'source' if 'quote' doesn't exist
                    'Source': code_row.get('source', code_row.get('quote_2', ''))  # Try 'source', then 'quote_2' if 'source' doesn't exist
                })
                final_df = pd.concat([final_df, new_row.to_frame().T], ignore_index=True)

    return final_df

def main():
    """
    The main function that sets up the Streamlit interface and handles user interactions.
    """
    # Reset the current prompt in the session state
    if 'current_prompt' in st.session_state:
        del st.session_state.current_prompt

    st.header(":orange[Finding Themes]")

    # Display instructions in an expandable section
    with st.expander("Instructions"):
        st.write("""
        The Finding Themes page is where you identify overarching themes from your reduced codes. This step helps you synthesize your data into meaningful patterns. Here's how to use this page:
        """)

        st.subheader(":orange[1. Project and File Selection]")
        st.write("""
        - Select your project from the dropdown menu.
        - Once a project is selected, you'll see a list of reduced code files available for processing.
        - Choose the files you want to analyze. You can select individual files or use the "Select All" checkbox.
        """)

        st.subheader(":orange[2. LLM Settings]")
        st.write("""
        - Choose the AI model you want to use for theme identification.
        - Select a preset prompt or edit the provided prompt to guide the theme finding process.
        - Adjust the model temperature and top_p values using the sliders. These parameters influence the AI's creativity and output variability.
        """)

        st.subheader(":orange[3. Processing and Results]")
        st.write("""
        - Click the "Process" button to start finding themes.
        - The system will analyze the selected reduced code files and generate themes.
        - Once complete, you'll see:
        - An expandable section for each generated theme, showing the theme name, description, and associated codes.
        - A reference section showing all codes and their descriptions used in the analysis.
        - You can download the generated themes as a CSV file.
        """)

        st.subheader(":orange[4. Saved Themes]")
        st.write("""
        - At the bottom of the page, you'll find an expandable section showing previously generated theme files.
        - You can view, delete, or download these saved theme files.
        """)

        st.subheader("Key Features")
        st.write("""
        - :orange[Automated theme generation:] The AI identifies patterns across your reduced codes to suggest overarching themes.
        - :orange[Theme descriptions:] Each theme comes with a detailed description to explain its meaning and relevance.
        - :orange[Code mapping:] The system shows which codes are associated with each theme, maintaining the connection between your data and the higher-level themes.
        - :orange[Flexibility:] You can adjust the prompt and model settings to influence how themes are generated and organized.
        """)

        st.subheader(":orange[Tips]")
        st.write("""
        - Review the generated themes carefully. While the AI is helpful, your expertise and understanding of the context are crucial for validating and refining these themes.
        - :orange[Experiment with different prompts and settings] if you're not satisfied with the initial results. Different approaches can yield different insights.
        - Consider the number of themes generated. Too few might oversimplify your data, while too many might make it difficult to draw meaningful conclusions.
        - Use the reference section to understand how individual codes contribute to the larger themes.
        - Remember that theme generation is an iterative process. You may need to run this step multiple times, adjusting your approach based on the results.
        """)

        st.info("Finding themes is a crucial step in synthesizing your analysis. It helps you move from detailed codes to broader, more conceptual understanding of your data. Take your time to reflect on the themes and how they relate to your research questions.")

    st.subheader(":orange[Project & Data Selection]")

    # Get available projects and handle project selection
    projects = get_projects()
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = "Select a project..."

    project_options = ["Select a project..."] + projects
    index = project_options.index(st.session_state.selected_project) if st.session_state.selected_project in project_options else 0

    selected_project = st.selectbox(
        "Select a project:", 
        project_options,
        index=index,
        key="project_selector",
        help= tooltips.project_tooltip
    )

    # Update session state when a new project is selected
    if selected_project != st.session_state.selected_project:
        st.session_state.selected_project = selected_project
        st.rerun()

    if selected_project != "Select a project...":
        # Get and display project files for selection
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

        # Model selection
        default_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "claude-sonnet-3.5"]
        azure_models = get_azure_models()
        model_options = default_models + azure_models
        selected_model = st.selectbox("Select Model", model_options, help=tooltips.model_tooltip)

        max_temperature_value = 2.0 if selected_model.startswith('gpt') else 1.0

        # Load custom prompts
        custom_prompts = load_custom_prompts().get('Finding Themes', {})

        # Combine preset and custom prompts
        all_prompts = {**finding_themes_prompts, **custom_prompts}

        # Prompt selection
        selected_prompt = st.selectbox("Select a prompt:", list(all_prompts.keys()), help=tooltips.presets_tooltip)

        # Load selected prompt values
        selected_prompt_data = all_prompts[selected_prompt]
        prompt_input = selected_prompt_data["prompt"]
        model_temperature = selected_prompt_data["temperature"]
        model_top_p = selected_prompt_data["top_p"]

        prompt_input = st.text_area("Edit prompt if needed:", value=prompt_input, height=200, help=tooltips.prompt_tooltip)
        
        settings_col1, settings_col2 = st.columns([0.5, 0.5])
        with settings_col1:
            model_temperature = st.slider(label="Model Temperature", min_value=float(0), max_value=float(max_temperature_value), step=0.01, value=model_temperature, help = tooltips.model_temp_tooltip)
        with settings_col2:
            model_top_p = st.slider(label="Model Top P", min_value=float(0), max_value=float(1), step=0.01, value=model_top_p, help=tooltips.top_p_tooltip)

        if st.button("Process"):
            st.divider()
            st.subheader(":orange[Output]")
            with st.spinner("Finding themes... please wait..."):
                # Process the selected files and display results
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
                
                    # Save and offer download of themes
                    saved_file_path = save_themes(selected_project, themes_df)
                    st.success(f"Themes saved to {saved_file_path}")
                    
                    csv = themes_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download themes",
                        data=csv,
                        file_name="themes.csv",
                        mime="text/csv"
                    )

            # Load data for the selected project
            themes_df, codes_df = load_data(selected_project)
            
            if themes_df is None or codes_df is None:
                st.error("Error: Required files not found in the project directory.")
            else:
                st.success(f"Theme-codes generates successfully for project: {selected_project}")
                
                # Process data to create the final theme-codes book
                final_df = process_data(themes_df, codes_df)
                
                # Display various views of the data
                st.write("Condensed Themes")
                st.write(themes_df)
                
                st.write("Expanded Themes w/ Codes, Quotes & Sources")
                final_display_df = final_df.copy()
                final_display_df['Quotes'] = final_display_df['Quotes'].apply(format_quotes)
                st.write(final_df)
                
                with st.expander("Merged Codes (for reference)"):
                    st.write("Merged Codes")
                    st.write(codes_df)
                
                # Save the final DataFrame
                output_folder = os.path.join(PROJECTS_DIR, selected_project, 'theme_books')
                os.makedirs(output_folder, exist_ok=True)

                # Save condensed theme book
                output_file_condensed = os.path.join(output_folder, f"{selected_project}_condensed_theme_book_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
                themes_df.to_csv(output_file_condensed, index=False)

                # Save expanded theme book
                output_file_expanded = os.path.join(output_folder, f"{selected_project}_expanded_theme_book_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
                final_df.to_csv(output_file_expanded, index=False)

                st.success(f"Theme books (condensed and expanded) saved to: \n-{output_file_condensed} \n{output_file_expanded}")
                
                # Provide download button for the final theme-codes book
                csv = final_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Final Theme-Codes Book",
                    data=csv,
                    file_name="final_theme_codes_book.csv",
                    mime="text/csv"
                )

        # Display previously processed files
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

    # Manage API keys
    manage_api_keys()

if __name__ == "__main__":
    main()
