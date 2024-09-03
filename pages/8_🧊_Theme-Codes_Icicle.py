# -*- coding: utf-8 -*-
"""
Theme-Codes Icicle Visualization

This script generates an interactive icicle plot to visualize the hierarchical structure of themes,
reduced codes, initial codes, quotes, and sources in a qualitative data analysis project.

Created on Tue Mar 26 08:48:38 2024
@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from project_utils import get_projects, PROJECTS_DIR, get_processed_files
from api_key_management import manage_api_keys, load_api_keys

# Constants
THEME_BOOKS_FOLDER = 'theme_books'
EXPANDED_REDUCED_CODES_FOLDER = 'expanded_reduced_codes'

def load_data(project_name):
    """
    Load the latest theme and code data for a given project.

    Args:
        project_name (str): Name of the project to load data for.

    Returns:
        tuple: A tuple containing two pandas DataFrames (themes_df, codes_df) or (None, None) if data is not available.
    """
    themes_folder = os.path.join(PROJECTS_DIR, project_name, THEME_BOOKS_FOLDER)
    codes_folder = os.path.join(PROJECTS_DIR, project_name, EXPANDED_REDUCED_CODES_FOLDER)
    
    # Load the latest expanded themes file
    themes_files = get_processed_files(project_name, THEME_BOOKS_FOLDER)
    expanded_themes_files = [f for f in themes_files if 'expanded' in f]
    if not expanded_themes_files:
        return None, None
    latest_themes_file = max(expanded_themes_files, key=lambda f: os.path.getmtime(os.path.join(themes_folder, f)))
    themes_df = pd.read_csv(os.path.join(themes_folder, latest_themes_file))
    
    # Load the latest codes file
    codes_files = get_processed_files(project_name, EXPANDED_REDUCED_CODES_FOLDER)
    if not codes_files:
        return None, None
    latest_codes_file = max(codes_files, key=lambda f: os.path.getmtime(os.path.join(codes_folder, f)))
    codes_df = pd.read_csv(os.path.join(codes_folder, latest_codes_file))

    return themes_df, codes_df 

def prepare_icicle_data(themes_df, codes_df):
    """
    Prepare the data for the icicle plot by combining theme and code information.

    Args:
        themes_df (pd.DataFrame): DataFrame containing theme data.
        codes_df (pd.DataFrame): DataFrame containing code data.

    Returns:
        pd.DataFrame: A DataFrame structured for use in the icicle plot.
    """
    icicle_data = []
    
    for _, theme_row in themes_df.iterrows():
        theme = theme_row['Theme']
        theme_description = theme_row['Theme Description']
        reduced_code = theme_row['Code']
        reduced_code_description = theme_row['Code Description']
        
        # Filter codes_df for the current reduced code
        relevant_codes = codes_df[codes_df['code'] == reduced_code]
        
        for _, code_row in relevant_codes.iterrows():
            initial_code = code_row['original_code']
            quote = code_row['quote']
            source = code_row['source']
            
            icicle_data.append({
                'Theme': theme,
                'Theme Description': theme_description,
                'Reduced Code': reduced_code,
                'Reduced Code Description': reduced_code_description,
                'Initial Code': initial_code,
                'Quote': quote,
                'Source': source,
                'Value': 1  # Each entry has a value of 1 for equal weighting in the icicle plot
            })
    
    return pd.DataFrame(icicle_data)

def create_icicle_plot(df_filtered):
    """
    Create an icicle plot using the filtered data.

    Args:
        df_filtered (pd.DataFrame): Filtered DataFrame containing data for the selected theme.

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure object containing the icicle plot.
    """
    fig = px.icicle(
        df_filtered, 
        path=['Theme', 'Reduced Code', 'Initial Code', 'Quote', 'Source'], 
        values='Value',
        branchvalues='total',
        maxdepth=5,
        hover_data=['Theme Description', 'Reduced Code Description']
    )
    fig.update_traces(
        root_color="lightgrey", 
        textinfo="label+value",
        textfont=dict(size=14),
        textposition='middle center',
    )
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25), 
        height=800,
    )
    return fig

def main():
    """
    Main function to run the Streamlit app for the Theme-Codes Icicle visualization.
    """
    st.header(":orange[Theme-Codes Icicle]")
    st.subheader(":orange[Structure: Theme > Reduced Codes > Initial Code(s) > Quote(s) > Source]")

    # Project selection
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

    # Rerun the app if a new project is selected
    if selected_project != st.session_state.selected_project:
        st.session_state.selected_project = selected_project
        st.rerun()

    if selected_project != "Select a project...":
        # Load and process data
        themes_df, codes_df = load_data(selected_project)

        if themes_df is None or codes_df is None:
            st.error("No data available for the selected project.")
            return

        icicle_data = prepare_icicle_data(themes_df, codes_df)

        # Theme selection
        unique_themes = icicle_data['Theme'].unique()
        selected_theme = st.selectbox('Select a Theme to Visualise', unique_themes)
    
        # Filter data for the selected theme
        df_filtered = icicle_data[icicle_data['Theme'] == selected_theme]
    
        # Create and display the icicle plot
        fig = create_icicle_plot(df_filtered)
        st.write('You can click on the components to see them in more detail. Hover for descriptions.')
        st.plotly_chart(fig, use_container_width=True)

    # API key management
    manage_api_keys()
    
if __name__ == "__main__":
    main()