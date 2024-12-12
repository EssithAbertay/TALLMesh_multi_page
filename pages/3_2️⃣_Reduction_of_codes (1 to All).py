# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:30:28 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk

This script is part of the TALLMesh multi-page application for qualitative data analysis.
It focuses on the reduction of initial codes generated in the previous step of the analysis process.
The main purpose is to refine and consolidate codes, identifying patterns and reducing redundancy.


"""

import os
import streamlit as st
import pandas as pd
import json
import re
from api_key_management import manage_api_keys, load_api_keys, load_azure_settings, get_azure_models, AZURE_SETTINGS_FILE
from project_utils import get_projects, get_project_files, get_processed_files, PROJECTS_DIR
from prompts import reduce_duplicate_codes_1_v_all
from llm_utils import llm_call, default_models
import logging
import tooltips
import time
from ui_utils import centered_column_with_number
import uuid
from time import sleep
import networkx as nx

logo = "pages/static/tmeshlogo.png"
st.logo(logo)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

process_gif = "pages/animations/process_rounded.gif"
compare_gif = "pages/animations/compare_rounded.gif"
merge_gif = "pages/animations/merge_rounded.gif"

process_text = 'The LLM compares each set of initial codes...'
compare_text = '...to identify duplicates based on the prompt...'
merge_text = "...which are merged into a set of unique codes."


def collect_all_initial_codes(selected_files):
    """
    Collect and normalize all initial codes from all files into a single DataFrame.
    """
    logger = logging.getLogger(__name__)
    all_codes = []
    required_columns = {'code', 'description', 'quote'}
    for file_path in selected_files:
        df = pd.read_csv(file_path)
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"File {file_path} missing required columns: {missing_cols}")
        if 'source' not in df.columns:
            df['source'] = os.path.basename(file_path)
        df['code_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        df['original_code'] = df['code']
        all_codes.append(df)
        logger.info(f"Processed {file_path} with {len(df)} codes")
    if not all_codes:
        raise ValueError("No valid code files were processed")
    master_codes_df = pd.concat(all_codes, ignore_index=True)
    logger.info(f"Master codes DataFrame with {len(master_codes_df)} total codes")
    return master_codes_df

def generate_comparison_prompt(target_code, comparison_codes, prompt, include_quotes=False):
    base_prompt = prompt
    target_quote = f',\n        "quote": "{target_code["quote"]}"' if include_quotes else ''
    comparison_text = []
    for code in comparison_codes:
        if include_quotes:
            comparison_text.append(
                f'{{"code": "{code["code"]}", "description": "{code["description"]}", "quote": "{code["quote"]}"}}'
            )
        else:
            comparison_text.append(
                f'{{"code": "{code["code"]}", "description": "{code["description"]}"}}'
            )
    comparison_list = ',\n    '.join(comparison_text)
    final_prompt = base_prompt % (
        target_code["code"],
        target_code["description"],
        target_quote,
        f'[\n    {comparison_list}\n]'
    )
    return final_prompt

def process_similarity_comparisons(master_codes_df, model, prompt, model_temperature, model_top_p, include_quotes=False, resume_data=None):
    """
    Perform 1-vs-all similarity checks on all codes.
    """
    logger = logging.getLogger(__name__)

    def initialize_similarity_results(resume_data):
        if resume_data and 'similarity_results' in resume_data:
            return resume_data['similarity_results']
        return {code_id: {'similar_codes': [], 'comparison_status': False} 
                for code_id in master_codes_df['code_id']}

    def process_comparison_response(response_text, target_code_id):
        try:
            response_json = json.loads(response_text)
            comparisons = response_json.get('comparisons', {})
            if not isinstance(comparisons, dict):
                logger.error(f"Invalid response format for {target_code_id}: {response_text}")
                return None
            return comparisons
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {target_code_id}: {str(e)}")
            return None
    
    similarity_results = initialize_similarity_results(resume_data)
    total_codes = len(master_codes_df)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, target_code in master_codes_df.iterrows():
        target_code_id = target_code['code_id']
        if similarity_results[target_code_id]['comparison_status']:
            continue
        target_code_dict = target_code.to_dict()
        comparison_codes = master_codes_df[master_codes_df['code_id'] != target_code_id]

        comparison_prompt = generate_comparison_prompt(
            target_code_dict,
            [code.to_dict() for _, code in comparison_codes.iterrows()],
            prompt,
            include_quotes
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm_call(model, comparison_prompt, model_temperature, model_top_p)
                comparison_results = process_comparison_response(response, target_code_id)
                if comparison_results is not None:
                    for i, is_similar in enumerate(comparison_results.values(), start=1):
                        if is_similar:
                            # Positionally map code_id_x to actual codes:
                            # The order in comparison_results corresponds to the comparison list's order.
                            # We must find the corresponding code from the comparison.
                            # We'll rely on the order from the prompt generation:
                            # Extract keys in order, match them to the comparison_codes iterrows.
                            # Actually we do not need the actual code_id from the prompt. Just track similarity internally.
                            # We'll just re-run grouping later. For now, trust that code_id_x mapping is positional.
                            # The final approach: after we get True/False, we must identify which code was true.
                            # We rely on the position in comparison_results. Let's get a list of comparison code_ids:
                            comp_df = comparison_codes.reset_index(drop=True)
                            # code_id_x keys are ordered as code_id_1, code_id_2,... Let's get keys sorted:
                            comp_keys = sorted(comparison_results.keys(), key=lambda k: int(k.split('_')[-1]))
                            # Map each code_id_x to corresponding row in comp_df
                            matched_code_id = comp_df.iloc[i-1]['code_id']
                            similarity_results[target_code_id]['similar_codes'].append(matched_code_id)
                            similarity_results[matched_code_id]['similar_codes'].append(target_code_id)
                    break
            except Exception as e:
                logger.error(f"Error in API call: {str(e)}")
                if attempt < max_retries - 1:
                    sleep(2 ** attempt)
                else:
                    raise

        similarity_results[target_code_id]['comparison_status'] = True
        progress = (idx + 1) / total_codes
        progress_bar.progress(progress)
        status_text.text(f"Processing code {idx + 1}/{total_codes}")

        if idx % 5 == 0:
            st.session_state['intermediate_results'] = {
                'similarity_results': similarity_results,
                'last_processed_idx': idx
            }

    progress_bar.empty()
    status_text.empty()
    return similarity_results

def reduce_based_on_similarities(similarity_results, master_codes_df, model, model_temperature, model_top_p, include_quotes=False):
    logger = logging.getLogger(__name__)

    def log_group_details(group_idx, group, group_codes):
        logger.info(f"\nProcessing group {group_idx}")
        logger.info(f"Group size: {len(group)}")
        logger.info("Group code IDs: " + ", ".join(group))
        logger.info("Original codes in group:")
        for _, code in group_codes.iterrows():
            logger.info(f"  Code ID: {code['code_id']}")
            logger.info(f"    Code: {code['code']}")
            logger.info(f"    Original code: {code.get('original_code', 'Not found')}")

    def generate_merge_prompt(codes_to_merge):
        merge_prompt = """Create a merged code from the following similar codes. Provide:
            1. A concise name for the merged code (maximum 6 words)
            2. A detailed description (up to 25 words) explaining the merged code's meaning
            3. A brief explanation (max 50 words) of why these codes were merged

            The codes to merge are:
            {codes}

            Format the response as a JSON object with this structure:
            {{
                "merged_code": {{
                    "code": "name of merged code",
                    "description": "merged description",
                    "merge_explanation": "explanation of merge"
                }}
            }}

            Important! Your response must be a valid JSON object with no additional text."""
        
        codes_text = []
        for _, code in codes_to_merge.iterrows():
            code_text = (f'Code ID: {code["code_id"]}\n'
                         f'Code: "{code["code"]}"\n'
                         f'Description: "{code["description"]}"\n'
                         f'Original Code: "{code.get("original_code", "Not found")}"')
            if include_quotes and 'quote' in code:
                code_text += f'\nQuote: "{code["quote"]}"'
            codes_text.append(code_text)
        
        return merge_prompt.format(codes='\n\n'.join(codes_text))

    def find_code_groups():
        processed_codes = set()
        groups = []
        
        for code_id in similarity_results:
            if code_id in processed_codes:
                continue
                
            # Start a new group with this code
            current_group = {code_id}
            similar_codes = set(similarity_results[code_id]['similar_codes'])
            
            # Add any codes that were marked as similar
            current_group.update(similar_codes)
            
            # Only add this group if it's not already a subset of an existing group
            if not any(current_group.issubset(existing_group) for existing_group in groups):
                groups.append(current_group)
                processed_codes.update(current_group)
            
            # If this code has no similar codes, treat it as its own group
            if not similar_codes:
                groups.append({code_id})
                processed_codes.add(code_id)
        
        logger.info(f"\nFound {len(groups)} code groups")
        return groups

    reduced_codes = []
    code_groups = find_code_groups()
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, group in enumerate(code_groups):
        status_text.text(f"Processing group {idx + 1}/{len(code_groups)}")
        progress_bar.progress((idx + 1) / len(code_groups))
        group_codes = master_codes_df[master_codes_df['code_id'].isin(group)]
        log_group_details(idx, group, group_codes)

        if len(group_codes) > 1:
            merge_prompt = generate_merge_prompt(group_codes)
            logger.info("\nGenerated merge prompt:")
            logger.info(merge_prompt)
            try:
                response = llm_call(model, merge_prompt, model_temperature, model_top_p)
                logger.info("\nLLM Response:")
                logger.info(response)
                merged_details = json.loads(response)['merged_code']
                all_original_codes = []
                for _, code in group_codes.iterrows():
                    if isinstance(code.get('original_code'), str):
                        try:
                            original_codes = json.loads(code['original_code'])
                            if isinstance(original_codes, list):
                                all_original_codes.extend(original_codes)
                            else:
                                all_original_codes.append(original_codes)
                        except json.JSONDecodeError:
                            all_original_codes.append(code['original_code'])
                    else:
                        all_original_codes.append(code['code'])

                reduced_code = {
                    'code': merged_details['code'],
                    'description': merged_details['description'],
                    'merge_explanation': merged_details['merge_explanation'],
                    'original_code': json.dumps(all_original_codes),
                    'quote': json.dumps([
                        {'text': row['quote'], 'source': row['source']}
                        for _, row in group_codes.iterrows()
                    ]),
                    'source': ', '.join(group_codes['source'].unique())
                }
                logger.info("\nCreated reduced code:")
                logger.info(json.dumps(reduced_code, indent=2))
            except Exception as e:
                logger.error(f"\nError merging codes in group {idx}: {str(e)}")
                # Gather all original codes from the group
                all_original_codes = []
                quotes_list = []
                unique_sources = set()
                for _, code_row in group_codes.iterrows():
                    # Attempt to load original_code as JSON; if not JSON, treat as a single code
                    orig_val = code_row.get('original_code', code_row['code'])
                    if isinstance(orig_val, str):
                        try:
                            parsed = json.loads(orig_val)
                            if isinstance(parsed, list):
                                all_original_codes.extend(parsed)
                            else:
                                all_original_codes.append(parsed)
                        except json.JSONDecodeError:
                            all_original_codes.append(orig_val)
                    else:
                        all_original_codes.append(orig_val)
                    
                    quotes_list.append({'text': code_row['quote'], 'source': code_row['source']})
                    unique_sources.add(code_row['source'])

                # Use the first code in the group as the representative code name and description
                fallback_code = group_codes.iloc[0]
                reduced_code = {
                    'code': fallback_code['code'],
                    'description': fallback_code['description'],
                    'merge_explanation': 'Merge failed - using original code',
                    'original_code': json.dumps(all_original_codes),
                    'quote': json.dumps(quotes_list),
                    'source': ', '.join(unique_sources)
                }

                logger.info("\nUsing fallback code:")
                logger.info(json.dumps(reduced_code, indent=2))
        else:
            single_code = group_codes.iloc[0]
            reduced_code = {
                'code': single_code['code'],
                'description': single_code['description'],
                'merge_explanation': '',
                'original_code': json.dumps([single_code['code']]),
                'quote': json.dumps([{'text': single_code['quote'], 'source': single_code['source']}]),
                'source': single_code['source']
            }
            logger.info("\nSingle code (no merge needed):")
            logger.info(json.dumps(reduced_code, indent=2))

        reduced_codes.append(reduced_code)

    progress_bar.empty()
    status_text.empty()

    reduced_df = pd.DataFrame(reduced_codes)
    required_columns = ['code', 'description', 'merge_explanation', 'original_code', 'quote', 'source']
    for col in required_columns:
        if col not in reduced_df.columns:
            reduced_df[col] = ''
    reduced_df = reduced_df[required_columns]

    logger.info("\nFinal reduced DataFrame:")
    logger.info(f"Shape: {reduced_df.shape}")
    logger.info("Sample of reduced codes:")
    logger.info(reduced_df.head().to_string())
    return reduced_df


class AutoSaveResume:
    def __init__(self, project_name):
        self.project_name = project_name
        self.save_path = os.path.join(PROJECTS_DIR, project_name, 'code_reduction_progress.json')

    def save_progress(self, processed_files, reduced_df, total_codes_list, unique_codes_list, cumulative_total, mode,
                  master_codes_df=None, similarity_results=None, selected_files=None):
        progress = {
            'processed_files': processed_files,
            'reduced_df': reduced_df.to_json(),
            'total_codes_list': total_codes_list,
            'unique_codes_list': unique_codes_list,
            'cumulative_total': cumulative_total,
            'mode': mode
        }
        if master_codes_df is not None:
            progress['master_codes_df'] = master_codes_df.to_json()
        if similarity_results is not None:
            progress['similarity_results'] = similarity_results
        if selected_files is not None:
            progress['selected_files'] = selected_files

        with open(self.save_path, 'w') as f:
            json.dump(progress, f)


    def load_progress(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                progress = json.load(f)
            progress['reduced_df'] = pd.read_json(progress['reduced_df'])
            if 'master_codes_df' in progress:
                progress['master_codes_df'] = pd.read_json(progress['master_codes_df'])
            return progress
        return None


    def clear_progress(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

def amalgamate_duplicate_codes(df):
    required_columns = ['code', 'description', 'merge_explanation', 'original_code', 'quote', 'source']
    for col in required_columns:
        if col not in df.columns:
            if col == 'merge_explanation':
                df[col] = ''
            elif col == 'original_code':
                df[col] = df['code']
            else:
                logger.error(f"Required column {col} missing in DataFrame")
                return None
    try:
        amalgamated_df = df.groupby('code').agg({
            'description': 'first',
            'merge_explanation': 'first',
            'original_code': lambda x: sum((json.loads(item) if isinstance(item, str) and item.strip().startswith('[') else [item] for item in x), []),
            'quote': lambda x: [{'text': q, 'source': s} for q, s in zip(x, df.loc[x.index, 'source'])],
            'source': lambda x: list(x)
        }).reset_index()

        amalgamated_df['original_code'] = amalgamated_df['original_code'].apply(json.dumps)
        amalgamated_df['quote'] = amalgamated_df['quote'].apply(json.dumps)
        amalgamated_df['source'] = amalgamated_df['source'].apply(lambda x: ', '.join(set(x)))

        return amalgamated_df
    except Exception as e:
        logger.error(f"Error in amalgamate_duplicate_codes: {str(e)}")
        return None


def save_reduced_codes(project_name, df, folder):
    reduced_codes_folder = os.path.join(PROJECTS_DIR, project_name, folder)
    os.makedirs(reduced_codes_folder, exist_ok=True)
    output_file_path = os.path.join(reduced_codes_folder, f"reduced_codes_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(output_file_path, index=False, encoding='utf-8')
    return output_file_path

def process_files_with_autosave(selected_project, selected_files, model, prompt, 
                                model_temperature, model_top_p, include_quotes, 
                                resume_data=None, mode='Automatic'):
    """
    Process the codes in either automatic or incremental mode, with the ability to resume.
    In automatic mode, all files are processed at once.
    In incremental mode, process one file at a time, then stop and allow resumption.
    """
    auto_save = AutoSaveResume(selected_project)

    if resume_data:
        processed_files = resume_data['processed_files']
        reduced_df = resume_data['reduced_df']
        total_codes_list = resume_data['total_codes_list']
        unique_codes_list = resume_data['unique_codes_list']
        cumulative_total = resume_data['cumulative_total']
        similarity_results = resume_data.get('similarity_results', None)
        master_codes_df = resume_data.get('master_codes_df', None)                                                                              

        # If selected_files were saved in progress, override current selected_files to maintain consistency
        if 'selected_files' in resume_data and resume_data['selected_files']:
            selected_files = resume_data['selected_files']

    else:
        processed_files = []
        reduced_df = None
        total_codes_list = []
        unique_codes_list = []
        cumulative_total = 0
        similarity_results = None
        master_codes_df = None

    status_message = st.empty()

    if mode == 'Automatic':
        # Automatic mode: Process all selected files at once
        # Collect all codes
        status_message.info("Collecting codes from all files...")
        master_codes_df = collect_all_initial_codes(selected_files)
        cumulative_total = len(master_codes_df)
        total_codes_list.append(cumulative_total)

        # Run similarity on all at once
        status_message.info("Processing code comparisons...")
        similarity_results = process_similarity_comparisons(
            master_codes_df=master_codes_df,
            model=model,
            prompt=prompt,
            model_temperature=model_temperature,
            model_top_p=model_top_p,
            include_quotes=include_quotes,
            resume_data=resume_data
        )

        status_message.info("Reducing codes based on similarities...")
        reduced_df = reduce_based_on_similarities(
            similarity_results=similarity_results,
            master_codes_df=master_codes_df,
            model=model,
            model_temperature=model_temperature,
            model_top_p=model_top_p,
            include_quotes=include_quotes
        )

        unique_codes = len(reduced_df['code'].unique())
        unique_codes_list.append(unique_codes)
        processed_files = selected_files

        results_df = pd.DataFrame({
            'total_codes': total_codes_list,
            'unique_codes': unique_codes_list
        })
        results_path = os.path.join(PROJECTS_DIR, selected_project, 'code_reduction_results.csv')
        results_df.to_csv(results_path, index=False)
        auto_save.clear_progress()
        return reduced_df, results_df, processed_files

    else:
        # Incremental mode:
        # If we have no resume_data and haven't processed any files yet:
        if not resume_data and not processed_files:
            if not selected_files:
                st.error("No files selected.")
                return None, None, []

        status_message.info(f"Processing first file incrementally: {os.path.basename(selected_files[0])}")
        first_file_df = pd.read_csv(selected_files[0])
        if 'source' not in first_file_df.columns:
            first_file_df['source'] = os.path.basename(selected_files[0])
        if 'original_code' not in first_file_df.columns:
            first_file_df['original_code'] = first_file_df['code']
        if 'code_id' not in first_file_df.columns:
            first_file_df['code_id'] = [str(uuid.uuid4()) for _ in range(len(first_file_df))]
        if 'quote' not in first_file_df.columns:
            first_file_df['quote'] = ''

        # Treat this first set of codes as we do in subsequent steps:
        # Perform similarity comparisons and reduction even if there's only one file.
        master_codes_df = first_file_df.copy()
        cumulative_total = len(master_codes_df)
        total_codes_list.append(cumulative_total)

        status_message.info("Processing code comparisons for the first file...")
        similarity_results = process_similarity_comparisons(
            master_codes_df=master_codes_df,
            model=model,
            prompt=prompt,
            model_temperature=model_temperature,
            model_top_p=model_top_p,
            include_quotes=include_quotes
        )

        status_message.info("Reducing codes based on similarities for the first file...")
        reduced_df = reduce_based_on_similarities(
            similarity_results=similarity_results,
            master_codes_df=master_codes_df,
            model=model,
            model_temperature=model_temperature,
            model_top_p=model_top_p,
            include_quotes=include_quotes
        )

        unique_codes = len(reduced_df['code'].unique())
        unique_codes_list.append(unique_codes)
        processed_files = [selected_files[0]]

        # Save progress and return
        results_df = pd.DataFrame({
            'total_codes': total_codes_list,
            'unique_codes': unique_codes_list
        })
        auto_save.save_progress(
            processed_files=processed_files,
            reduced_df=reduced_df,
            total_codes_list=total_codes_list,
            unique_codes_list=unique_codes_list,
            cumulative_total=cumulative_total,
            mode=mode,
            selected_files=selected_files
        )


        results_path = os.path.join(PROJECTS_DIR, selected_project, 'code_reduction_results.csv')
        results_df.to_csv(results_path, index=False)

        # Stop after first file in incremental mode
        return reduced_df, results_df, processed_files



def load_custom_prompts():
    try:
        with open('custom_prompts.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def format_quotes(quotes_json):
    try:
        quotes = json.loads(quotes_json)
        formatted_quotes = "\n".join(f"{quote['text']} (Source: {quote['source']})" for quote in quotes)
        return formatted_quotes
    except (json.JSONDecodeError, KeyError, TypeError):
        return quotes_json

def format_original_codes(original_codes):
    try:
        codes = json.loads(original_codes)
        return ', '.join(codes) if isinstance(codes, list) else original_codes
    except json.JSONDecodeError:
        return original_codes

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

def main():
    if 'current_prompt' in st.session_state:
        del st.session_state.current_prompt 

    st.header(":orange[Reduction of Codes]")

    with st.expander("Instructions"):
        st.write("""
        The Reduction of Codes page is where you refine and consolidate the initial codes generated in the previous step. 
        This process helps to identify patterns and reduce redundancy in your coding.
        """)
        col1, col2, col3 = st.columns(3)
        centered_column_with_number(col1, 1, process_text, process_gif)
        centered_column_with_number(col2, 2, compare_text, compare_gif)
        centered_column_with_number(col3, 3, merge_text, merge_gif)

        st.markdown(
            """
            <p style="font-size: 8px; color: gray; text-align: center;">
            <a href="https://www.flaticon.com/animated-icons" title="document animated icons" style="color: gray; text-decoration: none;">
            Animated icons created by Freepik - Flaticon
            </a>
            </p>
            """,
            unsafe_allow_html=True
        )

        st.subheader(":orange[1. Project and File Selection]")
        st.write("""
        - Select your project.
        - Choose the files you want to process.
        """)

        st.subheader(":orange[2. LLM Settings]")
        st.write("""
        - Choose the model.
        - Select or edit the prompt.
        - Adjust temperature and top_p.
        """)

        st.subheader(":orange[3. Processing and Results]")
        st.write("""
        - Choose 'automatic' or 'incremental' processing.
        - Click 'Process' to start.
        - Once complete, view and download results.
        """)

        st.subheader(":orange[4. Saved Reduced Codes]")
        st.write("""
        - View previously processed reduced code files.
        - Download or delete them as needed.
        """)

        st.info("Code reduction helps refine your analysis and prepare for thematic identification.")

    st.subheader(":orange[Project & Data Selection]")
    projects = get_projects()
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = "Select a project..."
    project_options = ["Select a project..."] + projects
    if st.session_state.selected_project in project_options:
        index = project_options.index(st.session_state.selected_project)
    else:
        index = 0

    selected_project = st.selectbox(
        "Select a project:", 
        project_options,
        index=index,
        key="project_selector",
        help=tooltips.project_tooltip
    )

    if selected_project != st.session_state.selected_project:
        st.session_state.selected_project = selected_project
        st.rerun()

    if selected_project != "Select a project...":
        project_files = get_project_files(selected_project, 'initial_codes')
        with st.expander("Select files to process", expanded=True):
            col1, col2 = st.columns([0.9, 0.2])
            select_all = col2.checkbox("Select All", value=True)
            file_checkboxes = {}
            for i, file in enumerate(project_files):
                col1, col2 = st.columns([0.9, 0.2])
                col1.write(file)
                file_checkboxes[file] = col2.checkbox(".", key=f"checkbox_{file}", value=select_all, label_visibility="hidden")

        selected_files = [os.path.join(PROJECTS_DIR, selected_project, 'initial_codes', file) 
                         for file, checked in file_checkboxes.items() if checked]

        st.divider()
        st.subheader(":orange[LLM Settings]")
        azure_models = get_azure_models()
        model_options = default_models + azure_models
        selected_model = st.selectbox("Select Model", model_options, help=tooltips.model_tooltip)
        max_temperature_value = 2.0 if selected_model.startswith('gpt') else 1.0
        custom_prompts = load_custom_prompts().get('Reduction of Codes', {})
        all_prompts = {**reduce_duplicate_codes_1_v_all, **custom_prompts}
        selected_prompt = st.selectbox("Select a prompt:", list(all_prompts.keys()), help=tooltips.presets_tooltip)
        selected_prompt_data = all_prompts[selected_prompt]
        prompt_input = selected_prompt_data["prompt"]
        model_temperature = selected_prompt_data["temperature"]
        model_top_p = selected_prompt_data["top_p"]

        prompt_input = st.text_area(
            "Edit prompt if needed:",
            value=prompt_input,
            height=200,
            help="In the 1-vs-all approach, this prompt guides merging."
        )

        settings_col1, settings_col2 = st.columns([0.5, 0.5])
        with settings_col1:
            model_temperature = st.slider(
                label="Model Temperature",
                min_value=float(0),
                max_value=float(max_temperature_value),
                step=0.01,
                value=model_temperature,
                help=tooltips.model_temp_tooltip
            )
        with settings_col2:
            model_top_p = st.slider(
                label="Model Top P",
                min_value=float(0),
                max_value=float(1),
                step=0.01,
                value=model_top_p,
                help=tooltips.top_p_tooltip
            )

        include_quotes = st.checkbox(
            label="Include Quotes",
            value=False,
            help='Include quotes in comparisons for more context.'
        )

        st.subheader(":orange[Processing Mode]")
        processing_mode = st.radio(
            "Choose processing mode:",
            ("Automatic", "Incremental"),
            help="Automatic: all files at once. Incremental: one file at a time."
        )

        auto_save = AutoSaveResume(selected_project)
        progress = auto_save.load_progress()
        resume_data = None
        if progress:
            processed_files = progress['processed_files']
            saved_mode = progress.get('mode', 'Automatic')
            if saved_mode == processing_mode:
                st.warning("Previous unfinished progress found.")
                st.info(f"Processed files: {[os.path.basename(f) for f in processed_files]}")
                st.info(f"Remaining files: {[os.path.basename(f) for f in selected_files if f not in processed_files]}")
                resume = st.checkbox("Resume from last checkpoint", value=True, key="resume_checkbox")
                if resume:
                    resume_data = progress
                else:
                    auto_save.clear_progress()
            else:
                st.warning(f"Previous unfinished progress found in a different mode: '{saved_mode}'.")
                st.info(f"Processed files: {[os.path.basename(f) for f in processed_files]}")
                st.info(f"Remaining files: {[os.path.basename(f) for f in selected_files if f not in processed_files]}")
                choice = st.radio(
                    "What would you like to do?",
                    (
                        f"Resume previous progress in '{saved_mode}' mode",
                        f"Discard previous progress and start fresh in '{processing_mode}' mode"
                    ),
                    key="mode_switch_choice"
                )
                if choice == f"Resume previous progress in '{saved_mode}' mode":
                    processing_mode = saved_mode
                    resume_data = progress
                    st.info(f"Switching back to '{saved_mode}' mode to resume.")
                else:
                    auto_save.clear_progress()
                    st.info(f"Starting fresh in '{processing_mode}' mode.")

        if st.button("Process"):
            st.divider()
            st.subheader(":orange[Output]")
            with st.spinner("Reducing codes... this may take time."):
                reduced_df, results_df, processed_files = process_files_with_autosave(
                    selected_project=selected_project,
                    selected_files=selected_files,
                    model=selected_model,
                    prompt=prompt_input,
                    model_temperature=model_temperature,
                    model_top_p=model_top_p,
                    include_quotes=include_quotes,
                    resume_data=resume_data,
                    mode=processing_mode
                ) 

                if reduced_df is not None:
                    status_message = st.empty()
                    status_message.info("Finalizing reduced codes...")
                    amalgamated_df = amalgamate_duplicate_codes(reduced_df)
                    amalgamated_df_for_display = amalgamated_df.copy()
                    amalgamated_df_for_display['quote'] = amalgamated_df_for_display['quote'].apply(format_quotes)
                    amalgamated_df_for_display['original_code'] = amalgamated_df_for_display['original_code'].apply(format_original_codes)

                    st.write("Reduced Codes:")
                    st.write(amalgamated_df_for_display)

                    st.write("Code Reduction Results:")
                    st.write(results_df)

                    status_message.info("Saving reduced codes...")
                    saved_file_path = save_reduced_codes(selected_project, amalgamated_df, 'reduced_codes')
                    st.success(f"Reduced codes saved to {saved_file_path}")

                    csv = amalgamated_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download reduced codes",
                        data=csv,
                        file_name="reduced_codes.csv",
                        mime="text/csv"
                    )

                    results_csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download code reduction results",
                        data=results_csv,
                        file_name="code_reduction_results.csv",
                        mime="text/csv"
                    )

                    status_message.success("Code reduction process completed successfully!")

                    if processing_mode == 'Incremental' and len(selected_files) > len(processed_files):
                        remaining_files = len(selected_files) - len(processed_files)
                        st.warning(f"Processing paused after current file in incremental mode. {remaining_files} file(s) remaining. Click 'Process' again to continue.")
                else:
                    st.error("Failed to reduce codes. Check logs for more info.")

        processed_files_list = get_processed_files(selected_project, 'reduced_codes')
        with st.expander("Saved Reduced Codes", expanded=False):
            for processed_file in processed_files_list:
                col1, col2 = st.columns([0.9, 0.1])
                col1.write(processed_file)
                if col2.button("Delete", key=f"delete_{processed_file}"):
                    os.remove(os.path.join(PROJECTS_DIR, selected_project, 'reduced_codes', processed_file))
                    st.success(f"Deleted {processed_file}")
                    st.rerun()
                df = pd.read_csv(os.path.join(PROJECTS_DIR, selected_project, 'reduced_codes', processed_file))
                df['quote'] = df['quote'].apply(format_quotes)
                df['original_code'] = df['original_code'].apply(format_original_codes)
                st.write(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download {processed_file}",
                    data=csv,
                    file_name=processed_file,
                    mime="text/csv"
                )

    else:
        st.write("Please select a project. If none, go to '🏠 Folder Set Up' to create one.")

    manage_api_keys()

if __name__ == "__main__":
    main()
