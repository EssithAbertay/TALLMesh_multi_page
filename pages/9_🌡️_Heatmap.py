"""
Heatmap of Theme and Code Frequencies

This module creates an interactive heatmap showing the frequency of codes within themes
from a qualitative data analysis project. It uses Streamlit for the web interface and Plotly
for the interactive chart.

Dependencies:
- streamlit
- pandas
- plotly
- numpy
- project_utils (custom module)
- api_key_management (custom module)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set logo
logo = "pages/static/tmeshlogo.png"
st.logo(logo)

# Custom modules
from project_utils import get_projects, PROJECTS_DIR, get_processed_files
from api_key_management import manage_api_keys, load_api_keys

def load_data(project_name):
    """
    Load the latest theme and code data for a given project.

    Args:
        project_name (str): Name of the project to load data for.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the frequency data.
                      Returns None if no data is available.
    """
    themes_folder = os.path.join(PROJECTS_DIR, project_name, 'theme_books')
    codes_folder = os.path.join(PROJECTS_DIR, project_name, 'expanded_reduced_codes')
    
    # Load themes data
    themes_files = get_processed_files(project_name, 'theme_books')
    expanded_themes_files = [f for f in themes_files if 'expanded' in f]
    if not expanded_themes_files:
        return None
    latest_themes_file = max(expanded_themes_files, key=lambda f: os.path.getmtime(os.path.join(themes_folder, f)))
    themes_df = pd.read_csv(os.path.join(themes_folder, latest_themes_file))
    
    # Load codes data
    codes_files = get_processed_files(project_name, 'expanded_reduced_codes')
    if not codes_files:
        return None
    latest_codes_file = max(codes_files, key=lambda f: os.path.getmtime(os.path.join(codes_folder, f)))
    codes_df = pd.read_csv(os.path.join(codes_folder, latest_codes_file))
    
    # Merge dataframes to associate codes with themes
    merged_df = themes_df.merge(codes_df, left_on='Code', right_on='code', how='left')
    
    return merged_df

def create_heatmap(data):
    """
    Create a heatmap of code frequencies within themes.

    Args:
        data (pd.DataFrame): The merged DataFrame containing all data.

    Returns:
        plotly.graph_objects.Figure: The heatmap figure.
    """
    # Count the frequency of initial codes within themes
    freq_table = data.groupby(['Theme', 'original_code']).size().reset_index(name='Frequency')
    
    # Pivot the table to create a matrix suitable for a heatmap
    heatmap_data = freq_table.pivot(index='Theme', columns='original_code', values='Frequency')
    heatmap_data = heatmap_data.fillna(0)

    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Initial Codes", y="Themes", color="Frequency"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        xaxis={'side': 'top'},
        margin=dict(t=50, l=100, r=50, b=100),
        height=800
    )

    return fig

def main():
    """
    Main function to run the Streamlit app.

    This function sets up the user interface, handles project selection,
    loads data, creates the Heatmap, and displays it in the Streamlit app.
    """
    st.header(":orange[Heatmap of Theme vs. Initial Code Frequencies]")

    # Get list of projects and set up project selection
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
        st.experimental_rerun()

    if selected_project != "Select a project...":
        # Load data for the selected project
        data = load_data(selected_project)

        if data is None or data.empty:
            st.error("No data available for the selected project.")
            return

        # Create and display the Heatmap
        fig = create_heatmap(data)
        st.plotly_chart(fig, use_container_width=True)

        st.write("Hover over cells to see the frequency of codes within themes.")

        # Option to display data as a table
        with st.expander("Show Data Table"):
            st.dataframe(data)

    # Manage API keys
    manage_api_keys()

if __name__ == "__main__":
    main()
