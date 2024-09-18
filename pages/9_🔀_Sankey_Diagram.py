"""
Sankey Diagram (Initial Codes -> Reduced Codes -> Themes) Generator

This module creates an interactive Sankey diagram visualization of the flow from initial codes
to reduced codes to themes in a qualitative data analysis project. It uses Streamlit for the web interface
and Plotly for the interactive diagram.

Dependencies:
- streamlit
- pandas
- plotly
- project_utils (custom module)
- api_key_management (custom module)
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components

# Custom modules
from project_utils import get_projects, PROJECTS_DIR, get_processed_files
from api_key_management import manage_api_keys, load_api_keys

def load_data(project_name):
    """
    Load the latest theme and code data for a given project.

    Args:
        project_name (str): Name of the project to load data for.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the merged data.
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
    merged_df = codes_df.merge(themes_df, left_on='code', right_on='Code', how='left')
    
    return merged_df

def hex_to_rgba(hex_color, opacity):
    """Convert hex color to rgba string with the given opacity."""
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    rgb = tuple(int(hex_color[i:i+h_len//3], 16) for i in range(0, h_len, h_len//3))
    return f'rgba{rgb + (opacity,)}'

def create_sankey_diagram(data, level_selection):
    # Define the labels based on the selected levels
    label_list = []
    color_list = []
    node_indices = {}

    # Build the nodes and links based on selected levels
    levels = level_selection  # For example: ['Initial Code', 'Reduced Code', 'Theme']
    source_indices = []
    target_indices = []
    values = []
    link_colors = []

    # Prepare the data for Sankey diagram
    if levels == ['Initial Code', 'Reduced Code', 'Theme']:
        grouped_data = data.groupby(['original_code', 'code', 'Theme']).size().reset_index(name='count')

        # Nodes for Initial Codes
        initial_codes = grouped_data['original_code'].unique()
        for code in initial_codes:
            node_indices[code] = len(label_list)
            label_list.append(code)
            color_list.append('#FFC300')  # Color for initial codes (yellow)

        # Nodes for Reduced Codes
        reduced_codes = grouped_data['code'].unique()
        for code in reduced_codes:
            if code not in node_indices:
                node_indices[code] = len(label_list)
                label_list.append(code)
                color_list.append('#FF5733')  # Color for reduced codes (orange-red)

        # Nodes for Themes
        themes = grouped_data['Theme'].unique()
        for theme in themes:
            if theme not in node_indices:
                node_indices[theme] = len(label_list)
                label_list.append(theme)
                color_list.append('#C70039')  # Color for themes (dark red)

        # Links from Initial Codes to Reduced Codes
        for _, row in grouped_data.iterrows():
            source_indices.append(node_indices[row['original_code']])
            target_indices.append(node_indices[row['code']])
            values.append(row['count'])
            # Use the color of the source node with opacity
            source_color = color_list[node_indices[row['original_code']]]
            link_colors.append(hex_to_rgba(source_color, 0.4))

        # Links from Reduced Codes to Themes
        for _, row in grouped_data.drop_duplicates(subset=['code', 'Theme']).iterrows():
            source_indices.append(node_indices[row['code']])
            target_indices.append(node_indices[row['Theme']])
            value = grouped_data[(grouped_data['code'] == row['code']) & (grouped_data['Theme'] == row['Theme'])]['count'].sum()
            values.append(value)
            # Use the color of the source node with opacity
            source_color = color_list[node_indices[row['code']]]
            link_colors.append(hex_to_rgba(source_color, 0.4))
    else:
        st.error("Currently, only the 'Initial Code -> Reduced Code -> Theme' flow is implemented.")
        return None

    # Create the Sankey diagram
    link = dict(
        source=source_indices,
        target=target_indices,
        value=values,
        color=link_colors
    )
    node = dict(
        label=label_list,
        pad=15,
        thickness=20,
        color=color_list,
        line=dict(color='black', width=0.5)
    )

    fig = go.Figure(data=[go.Sankey(link=link, node=node)])

    fig.update_layout(
        title_text="Sankey Diagram of Codes Flow",
        font_size=12,
        height=800
    )

    return fig

def main():
    """
    Main function to run the Streamlit app.

    This function sets up the user interface, handles project selection,
    loads data, creates the Sankey diagram, and displays it in the Streamlit app.
    """
    st.header(":orange[Sankey Diagram of Codes Flow]")

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
        st.rerun()

    if selected_project != "Select a project...":
        # Load data for the selected project
        data = load_data(selected_project)

        if data is None or data.empty:
            st.error("No data available for the selected project.")
            return

        selected_levels = ['Initial Code', 'Reduced Code', 'Theme'] # Here for now until I can work out how to skip a hierarchical level

        # Advanced settings in an expander
        with st.expander("Advanced Settings"):
            # Level selection
            #available_levels = ['Initial Code', 'Reduced Code', 'Theme']
            #default_levels = ['Initial Code', 'Reduced Code', 'Theme']
            #selected_levels = st.multiselect(
            #    "Select levels to display in the Sankey diagram",
            #    available_levels,
            #    default=default_levels,
            #    key="level_selector"
            #)

            #if selected_levels != ['Initial Code', 'Reduced Code', 'Theme']:
            #    st.warning("Currently, only the 'Initial Code -> Reduced Code -> Theme' flow is implemented.")
            #    return

            # Filtering options
            filter_theme = st.multiselect("Filter by Theme", options=data['Theme'].unique())
            filter_reduced_code = st.multiselect("Filter by Reduced Code", options=data['code'].unique())
            filter_initial_code = st.multiselect("Filter by Initial Code", options=data['original_code'].unique())

        # Apply filters
        df_filtered = data.copy()
        if filter_theme:
            df_filtered = df_filtered[df_filtered['Theme'].isin(filter_theme)]
        if filter_reduced_code:
            df_filtered = df_filtered[df_filtered['code'].isin(filter_reduced_code)]
        if filter_initial_code:
            df_filtered = df_filtered[df_filtered['original_code'].isin(filter_initial_code)]

        # Create and display the Sankey diagram
        fig = create_sankey_diagram(df_filtered, selected_levels)

        if fig:
            #st.plotly_chart(fig, use_container_width=True)
            #st.write("Hover over nodes and flows to see details.")

            # Optionally, save the figure as an HTML file and render it
            html_content = fig.to_html(full_html=False)
            components.html(html_content, height=900)

            # Option to display data as a table
            with st.expander("Show Data Table"):
                st.dataframe(df_filtered)
    
    # Manage API keys
    manage_api_keys()

if __name__ == "__main__":
    main()
