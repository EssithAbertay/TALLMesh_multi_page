# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:30:28 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""

import os
import streamlit as st
import pandas as pd
import json # got rid of ast
from openai import OpenAI
import anthropic
from api_key_management import manage_api_keys, load_api_keys
from project_utils import get_projects, get_project_files, get_processed_files
from prompts import reduce_duplicate_codes_prompts

PROJECTS_DIR = 'projects' # should probably set this in a config or something instead of every single page

def extract_json(text):
    import re
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return match.group(0)
    return None


def save_reduced_codes(project_name, df):
    reduced_codes_folder = os.path.join(PROJECTS_DIR, project_name, 'reduced_codes')
    os.makedirs(reduced_codes_folder, exist_ok=True)
    
    output_file_path = os.path.join(reduced_codes_folder, f"reduced_codes_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(output_file_path, index=False, encoding='utf-8')
    return output_file_path


# Sequentially analyse each of the initial_code files, recursively reducing duplicate codes. 
# Added tracking for unique and total codes for saturation metric calculation later on
def compare_and_reduce_codes(df1, df2, model, prompt, model_temperature, model_top_p):
    combined_codes = pd.concat([df1, df2], ignore_index=True)
    
    # Ensure 'source' column exists
    if 'source' not in combined_codes.columns:
        combined_codes['source'] = 'Unknown'

    codes_list = [{"code": code, "description": description, "quote": quote, "source": source} 
                  for code, description, quote, source in zip(combined_codes['code'], combined_codes['description'], combined_codes['quote'], combined_codes['source'])]
    
    full_prompt = f"{prompt}\n\nCodes:\n{json.dumps(codes_list)}"
    
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
    
    json_string = extract_json(processed_output)
    if json_string:
        json_output = json.loads(json_string)
        reduced_df = pd.json_normalize(json_output['reduced_codes'])
        # Explode the quotes column to create separate rows for each quote
        reduced_df = reduced_df.explode('quotes')
        reduced_df['quote'] = reduced_df['quotes'].apply(lambda x: x['text'])
        reduced_df['source'] = reduced_df['quotes'].apply(lambda x: x['source'])
        reduced_df = reduced_df.drop(columns=['quotes'])
        
        # Count total and unique codes correctly
        total_codes = len(combined_codes)
        unique_codes = len(reduced_df['code'].unique())  # Count unique codes, not rows
        
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
            unique_codes_list.append(len(df['code'].unique()))  # Count unique codes in first file
        else:
            reduced_df, _, _ = compare_and_reduce_codes(reduced_df, df, model, prompt, model_temperature, model_top_p)
            total_codes_list.append(cumulative_total)
            unique_codes_list.append(len(reduced_df['code'].unique()))  # Count unique codes after reduction
        
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
    st.subheader(":orange[Project & Data Selection]")
    
    projects = get_projects()
    selected_project = st.selectbox("Select a project:", ["Select a project..."] + projects, label_visibility="hidden")

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
        

        #selected_files = [file for file, checked in file_checkboxes.items() if checked]
        selected_files = [os.path.join(PROJECTS_DIR, selected_project, 'initial_codes', file) for file, checked in file_checkboxes.items() if checked]

        st.divider()
        st.subheader(":orange[LLM Settings]")

        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "claude-sonnet-3.5"]
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
            model_top_p = st.slider(label="Model Top P", min_value=float(0), max_value=float(1), step=0.01, value=0.1)

        if st.button("Process"):
            st.divider()
            st.subheader(":orange[Output]")
            with st.spinner("Reducing codes... depending on the number of initial code files, this could take some time ..."):
                reduced_df, results_df = process_files(selected_project, selected_files, selected_model, prompt_input, model_temperature, model_top_p)
                
                if reduced_df is not None:
                    # Display results
                    st.write(reduced_df)
                    
                    # Display intermediate results
                    st.write("Code Reduction Results:")
                    st.write(results_df)
                    
                    saved_file_path = save_reduced_codes(selected_project, reduced_df)
                    st.success(f"Reduced codes saved to {saved_file_path}")
                    
                    csv = reduced_df.to_csv(index=False).encode('utf-8')
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