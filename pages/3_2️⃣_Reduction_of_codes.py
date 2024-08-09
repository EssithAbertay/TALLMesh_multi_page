# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:30:28 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""

import os
import streamlit as st
import pandas as pd
import json # got rid of ast
from openai import OpenAI, AzureOpenAI
import anthropic
from api_key_management import manage_api_keys, load_api_keys
from project_utils import get_projects, get_project_files, get_processed_files, PROJECTS_DIR
from prompts import reduce_duplicate_codes_prompts
#from azure_model_mapping import azure_model_maps
from api_key_management import manage_api_keys, load_api_keys, load_azure_settings, get_azure_models, AZURE_SETTINGS_FILE
from llm_utils import llm_call



#PROJECTS_DIR = 'projects' # should probably set this in a config or something instead of every single page

def extract_json(text):
    import re
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return match.group(0)
    return None

def format_quotes(quotes_json):
    """
    Parses the JSON string of quotes, extracts the text,
    and joins each quote with a newline for better readability.
    """
    try:
        quotes = json.loads(quotes_json)
        formatted_quotes = "\n".join(quote['text'] for quote in quotes)
        return formatted_quotes
    except (json.JSONDecodeError, KeyError, TypeError):
        return quotes_json  # Return the original if there's an error

# Merge rows with duplicate codes, retaining separarte quotes and sources...
def amalgamate_duplicate_codes(df):
    # Group by 'code' and aggregate other columns
    amalgamated_df = df.groupby('code').agg({
        'description': 'first',
        'merge_explanation': 'first',
        'original_code': lambda x: list(set(x)),
        'quote': lambda x: json.dumps([{'text': q, 'source': s} for q, s in zip(x, df.loc[x.index, 'source'])]),
        'source': lambda x: list(set(x))  # Keep this for backward compatibility
    }).reset_index()

    # Function to flatten lists in cells
    def flatten_list(cell):
        if isinstance(cell, list):
            return ', '.join(map(str, cell))
        return cell

    # Apply flattening to relevant columns
    for col in ['original_code', 'quote', 'source']:
        amalgamated_df[col] = amalgamated_df[col].apply(flatten_list)

    return amalgamated_df


def match_reduced_to_original_codes(reduced_df, initial_codes_directory):
    # Check if reduced_df is a string (file path) or DataFrame
    if isinstance(reduced_df, str):
        reduced_df = pd.read_csv(reduced_df)
    elif not isinstance(reduced_df, pd.DataFrame):
        raise ValueError("reduced_df must be either a file path or a pandas DataFrame")
    
    # Dictionary to store initial codes dataframes
    initial_codes_dfs = {}
    
    # Read all initial codes files
    for filename in os.listdir(initial_codes_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(initial_codes_directory, filename)
            initial_codes_dfs[filename] = pd.read_csv(file_path)
    
    # Function to find the original code
    def find_original_code(row):
        source_file = row['source']
        quote = row['quote']
        
        if source_file in initial_codes_dfs:
            source_df = initial_codes_dfs[source_file]
            matching_row = source_df[source_df['quote'] == quote]
            
            if not matching_row.empty:
                return matching_row['code'].values[0]
        
        return 'Original code not found'
    
    # Apply the function to each row in the reduced codes dataframe
    reduced_df['original_code'] = reduced_df.apply(find_original_code, axis=1)
    
    return reduced_df

def save_reduced_codes(project_name, df, folder):
    reduced_codes_folder = os.path.join(PROJECTS_DIR, project_name, folder)
    os.makedirs(reduced_codes_folder, exist_ok=True)
    
    output_file_path = os.path.join(reduced_codes_folder, f"reduced_codes_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(output_file_path, index=False, encoding='utf-8')
    return output_file_path


# Sequentially analyse each of the initial_code files, recursively reducing duplicate codes. 
# Added tracking for unique and total codes for saturation metric calculation later on
def compare_and_reduce_codes(df1, df2, model, prompt, model_temperature, model_top_p):
    combined_codes = pd.concat([df1, df2], ignore_index=True)
    
    # Ensure 'source' column exists and is populated correctly
    if 'source' not in combined_codes.columns:
        combined_codes['source'] = combined_codes.apply(lambda row: row.name.split('_')[0] + '.csv', axis=1)

    codes_list = [{"code": code, "description": description, "quote": quote, "source": source} 
                  for code, description, quote, source in zip(combined_codes['code'], combined_codes['description'], combined_codes['quote'], combined_codes['source'])]
    
    full_prompt = f"{prompt}\n\nCodes:\n{json.dumps(codes_list)}"
    
    processed_output = llm_call(model, full_prompt, model_temperature, model_top_p)

    # Format returned output - should be own function
    json_string = extract_json(processed_output)
    if json_string:
        json_output = json.loads(json_string)
        reduced_codes = json_output['reduced_codes']
        
        # Create a mapping of original codes to reduced codes
        code_mapping = {}
        for reduced_code in reduced_codes:
            for original_code in reduced_code.get('original_codes', [reduced_code['code']]):
                code_mapping[original_code] = reduced_code['code']
        
        # Create the reduced DataFrame
        reduced_rows = []
        for reduced_code in reduced_codes:
            original_codes = reduced_code.get('original_codes', [reduced_code['code']])
            for original_code in original_codes:
                original_rows = combined_codes[combined_codes['code'] == original_code]
                for _, row in original_rows.iterrows():
                    reduced_rows.append({
                        'code': reduced_code['code'],
                        'description': reduced_code['description'],
                        'merge_explanation': reduced_code.get('merge_explanation', ''),
                        'original_code': original_code,
                        'quote': row['quote'],
                        'source': row['source']
                    })
        
        reduced_df = pd.DataFrame(reduced_rows)
        
        # Count total and unique codes correctly
        total_codes = len(combined_codes)
        unique_codes = len(reduced_df['code'].unique())
        
        return reduced_df, total_codes, unique_codes
    else:
        st.warning("No valid JSON found in the response")
        return None, None, None

# Process files and include a progress bar so users feel in the loop
def process_files(selected_project, selected_files, model, prompt, model_temperature, model_top_p):
    reduced_df = None
    total_codes_list = []
    unique_codes_list = []
    cumulative_total = 0
    progress_bar = st.progress(0)
    for i, file in enumerate(selected_files):
        df = pd.read_csv(file)
        # Add source column if it doesn't exist
        if 'source' not in df.columns:
            df['source'] = os.path.basename(file)
        
        file_total_codes = len(df)
        cumulative_total += file_total_codes
        
        if reduced_df is None:
            reduced_df = df
            total_codes_list.append(cumulative_total)
            unique_codes_list.append(len(df['code'].unique()))
        else:
            reduced_df, _, _ = compare_and_reduce_codes(reduced_df, df, model, prompt, model_temperature, model_top_p)
            total_codes_list.append(cumulative_total)
            unique_codes_list.append(len(reduced_df['code'].unique()))
        
        progress = (i + 1) / len(selected_files)
        progress_bar.progress(progress)
    
    # Save intermediate results
    results_df = pd.DataFrame({
        'total_codes': total_codes_list,
        'unique_codes': unique_codes_list
    })
    results_df.to_csv(os.path.join(PROJECTS_DIR, selected_project, 'code_reduction_results.csv'), index=False)
    
    return reduced_df, results_df



@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

def main():
    # session_state persists through page changes so need to reset the text input message 
    if 'current_prompt' in st.session_state:
        del st.session_state.current_prompt 

    st.header(":orange[Reduction of Codes]")

    with st.expander("Instructions"):
        st.write("""
        The Reduction of Codes page is where you refine and consolidate the initial codes generated in the previous step. This process helps to identify patterns and reduce redundancy in your coding. Here's how to use this page:
        """)

        st.subheader(":orange[1. Project and File Selection]")
        st.write("""
        - Select your project from the dropdown menu.
        - Once a project is selected, you'll see a list of files containing initial codes.
        - Choose the files you want to process. You can select individual files or use the "Select All" checkbox.
        """)

        st.subheader(":orange[2. LLM Settings]")
        st.write("""
        - Choose the AI model you want to use for code reduction.
        - Select a preset prompt or edit the provided prompt to guide the reduction process.
        - Adjust the model temperature and top_p values using the sliders. These parameters influence the AI's output.
        """)

        st.subheader(":orange[3. Processing and Results]")
        st.write("""
        - Click the "Process" button to start the code reduction.
        - The system will analyze the selected files sequentially, comparing and merging similar codes.
        - A progress bar will show the status of the processing.
        - Once complete, you'll see:
        - A table of reduced codes with their descriptions, merged explanations, and associated quotes.
        - A "Code Reduction Results" table showing the number of total and unique codes for each processed file.
        - You can download both the reduced codes and the code reduction results as CSV files.
        """)

        st.subheader(":orange[4. Saved Reduced Codes]")
        st.write("""
        - At the bottom of the page, you'll find an expandable section showing previously processed reduced code files.
        - You can view, delete, or download these saved reduced codes.
        """)

        st.subheader(":orange[Key Features]")
        st.write("""
        - :orange[Automatic merging:] The AI identifies similar codes and combines them, providing explanations for the merges.
        - :orange[Quote preservation:] All quotes associated with the original codes are retained and linked to the reduced codes.
        - :orange[Tracking changes:] The system keeps track of how initial codes map to reduced codes, maintaining traceability.
        - :orange[Saturation analysis:] The code reduction results can be used to assess thematic saturation in your analysis (see 'Saturation Metric).
        """)

        st.subheader(":orange[Tips]")
        st.write("""
        - Review the merged codes carefully to ensure the AI's decisions align with your understanding of the data.
        - If you're not satisfied with the reduction, you can adjust the prompt or model settings and reprocess the files.
        - :orange[Pay attention to the "Code Reduction Results" table.] A plateauing number of unique codes may indicate approaching saturation in your analysis.
        - Consider running the reduction process multiple times with different settings to compare results and ensure thorough analysis.
        """)

        st.info("Code reduction is a critical step in refining your analysis. It helps to consolidate your findings and prepare for the identification of overarching themes in the next stage.")



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
        project_files = get_project_files(selected_project, 'initial_codes')
        
        with st.expander("Select files to process", expanded=True):
            col1, col2 = st.columns([0.9, 0.1])
            select_all = col2.checkbox("Select All", value=True)
            
            file_checkboxes = {}
            for i, file in enumerate(project_files):
                col1, col2 = st.columns([0.9, 0.1])
                col1.write(file)
                file_checkboxes[file] = col2.checkbox(".", key=f"checkbox_{file}", value=select_all, label_visibility="hidden")
        
        selected_files = [os.path.join(PROJECTS_DIR, selected_project, 'initial_codes', file) for file, checked in file_checkboxes.items() if checked]

        st.divider()
        st.subheader(":orange[LLM Settings]")

        # Model selection
        default_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "claude-sonnet-3.5"]
        azure_models = get_azure_models()
        model_options = default_models + azure_models
        selected_model = st.selectbox("Select Model", model_options)

        max_temperature_value = 2.0 if selected_model.startswith('gpt') else 1.0
        
        selected_preset = st.selectbox("Select a preset prompt:", list(reduce_duplicate_codes_prompts.keys()))

        if 'current_prompt' not in st.session_state or selected_preset != st.session_state.get('last_selected_preset'):
            st.session_state.current_prompt = reduce_duplicate_codes_prompts[selected_preset]
            st.session_state.last_selected_preset = selected_preset

        prompt_input = st.text_area("Edit prompt if needed:", value=st.session_state.current_prompt, height=200)
        
        settings_col1, settings_col2 = st.columns([0.5, 0.5])
        with settings_col1:
            model_temperature = st.slider(label="Model Temperature", min_value=float(0), max_value=float(max_temperature_value), step=0.01, value=0.1)
        with settings_col2:
            model_top_p = st.slider(label="Model Top P", min_value=float(0), max_value=float(1), step=0.01, value=1.0)

        if st.button("Process"):
            st.divider()
            st.subheader(":orange[Output]")
            with st.spinner("Reducing codes... depending on the number of initial code files, this could take some time ..."):
                reduced_df, results_df = process_files(selected_project, selected_files, selected_model, prompt_input, model_temperature, model_top_p)

                # Messy but works to map initial codes to new reduced codes. revisit this
                initial_codes_directory = os.path.join(PROJECTS_DIR, selected_project, 'initial_codes')
                updated_df = match_reduced_to_original_codes(reduced_df, initial_codes_directory) # Needed for visualisations later on as it matches reduced code - initial code - quote(s)
                #print(updated_df)
                amalgamated_df = amalgamate_duplicate_codes(updated_df)
                amalgamated_df_for_display = amalgamated_df.copy()
                amalgamated_df_for_display['quote'] = amalgamated_df_for_display['quote'].apply(format_quotes)

                if reduced_df is not None:
                    # Display results
                    st.write(amalgamated_df_for_display)
                    
                    # Display intermediate results
                    st.write("Code Reduction Results:")
                    st.write(results_df)
                    
                    save_reduced_codes(selected_project, updated_df, 'expanded_reduced_codes') # we need this view for visualisations later

                    saved_file_path = save_reduced_codes(selected_project, amalgamated_df, 'reduced_codes')
                    st.success(f"Reduced codes saved to {saved_file_path}")
                    
                    csv = amalgamated_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download reduced codes",
                        data=csv,
                        file_name="reduced_codes.csv",
                        mime="text/csv"
                    )
                    
                    # Save intermediate results
                    results_csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download code reduction results",
                        data=results_csv,
                        file_name="code_reduction_results.csv",
                        mime="text/csv"
                    )


        # View previously processed files
        processed_files = get_processed_files(selected_project, 'reduced_codes')
        with st.expander("Saved Reduced Codes", expanded=False):
            for processed_file in processed_files:
                col1, col2 = st.columns([0.9, 0.1])
                col1.write(processed_file)
                if col2.button("Delete", key=f"delete_{processed_file}"):
                    os.remove(os.path.join(PROJECTS_DIR, selected_project, 'reduced_codes', processed_file))
                    st.success(f"Deleted {processed_file}")
                    st.rerun()
                
                df = pd.read_csv(os.path.join(PROJECTS_DIR, selected_project, 'reduced_codes', processed_file))
                df['quote'] = df['quote'].apply(format_quotes)
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