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



def create_sankey_diagram(data, level_selection, style_settings=None):
    """
    Create a Sankey diagram with customizable styling options and improved layout.
    
    Args:
        data: DataFrame containing the hierarchical data
        level_selection: List of levels to display
        style_settings: Dictionary containing user-defined style parameters
    """
    # Set default style settings
    default_settings = {
        'node_pad': 15,
        'node_thickness': 20,
        'node_line_color': 'white',
        'node_line_width': 1.0,
        'initial_code_color': '#FFC300',
        'reduced_code_color': '#FF5733',
        'theme_color': '#C70039',
        'link_opacity': 0.4,
        'font_size': 16,
        'margin_top': 5,
        'margin_bottom': 10,
        'margin_left': 5,
        'margin_right': 10,
        'nodes_per_height_unit': 20
    }
    
    # Update with user settings if provided
    if style_settings:
        default_settings.update(style_settings)
    
    # Define the labels based on the selected levels
    label_list = []
    color_list = []
    node_indices = {}

    # Build the nodes and links
    source_indices = []
    target_indices = []
    values = []
    link_colors = []

    if level_selection == ['Initial Code', 'Reduced Code', 'Theme']:
        grouped_data = data.groupby(['original_code', 'code', 'Theme']).size().reset_index(name='count')

        # Calculate the number of nodes at each level
        n_initial = len(grouped_data['original_code'].unique())
        n_reduced = len(grouped_data['code'].unique())
        n_themes = len(grouped_data['Theme'].unique())
        
        # Calculate optimal height based on the maximum number of nodes at any level
        max_nodes = max(n_initial, n_reduced, n_themes)
        base_height = max_nodes * default_settings['nodes_per_height_unit']
        
        # Set minimum and maximum heights
        min_height = 400
        max_height = 2000
        dynamic_height = min(max_height, max(min_height, base_height))

        # Nodes for Initial Codes
        initial_codes = grouped_data['original_code'].unique()
        for code in initial_codes:
            node_indices[code] = len(label_list)
            label_list.append(code)
            color_list.append(default_settings['initial_code_color'])

        # Nodes for Reduced Codes
        reduced_codes = grouped_data['code'].unique()
        for code in reduced_codes:
            if code not in node_indices:
                node_indices[code] = len(label_list)
                label_list.append(code)
                color_list.append(default_settings['reduced_code_color'])

        # Nodes for Themes
        themes = grouped_data['Theme'].unique()
        for theme in themes:
            if theme not in node_indices:
                node_indices[theme] = len(label_list)
                label_list.append(theme)
                color_list.append(default_settings['theme_color'])

        # Links from Initial Codes to Reduced Codes
        for _, row in grouped_data.iterrows():
            source_indices.append(node_indices[row['original_code']])
            target_indices.append(node_indices[row['code']])
            values.append(row['count'])
            source_color = color_list[node_indices[row['original_code']]]
            link_colors.append(hex_to_rgba(source_color, default_settings['link_opacity']))

        # Links from Reduced Codes to Themes
        for _, row in grouped_data.drop_duplicates(subset=['code', 'Theme']).iterrows():
            source_indices.append(node_indices[row['code']])
            target_indices.append(node_indices[row['Theme']])
            value = grouped_data[(grouped_data['code'] == row['code']) & 
                               (grouped_data['Theme'] == row['Theme'])]['count'].sum()
            values.append(value)
            source_color = color_list[node_indices[row['code']]]
            link_colors.append(hex_to_rgba(source_color, default_settings['link_opacity']))

        # Create the Sankey diagram with improved positioning
        link = dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors
        )
        
        node = dict(
            label=label_list,
            pad=default_settings['node_pad'],
            thickness=default_settings['node_thickness'],
            color=color_list,
            line=dict(
                color=default_settings['node_line_color'],
                width=default_settings['node_line_width']
            )
        )

        # Create the Sankey diagram with improved layout
        fig = go.Figure(data=[go.Sankey(
            link=link,
            node=node,
            arrangement="snap"
        )])

        # Update layout with improved positioning and spacing
        fig.update_layout(
            font_color='black',
            font_size=default_settings['font_size'],
            height=dynamic_height,
            margin=dict(
                t=default_settings['margin_top'],
                l=default_settings['margin_left'],
                r=default_settings['margin_right'],
                b=default_settings['margin_bottom'],
                pad=0  # Remove padding
            ),
            paper_bgcolor='#0A0A0A',
            plot_bgcolor='#0A0A0A',
            autosize=True,
            # Remove any default padding
            xaxis=dict(
                automargin=True,
                constrain='domain'
            ),
            yaxis=dict(
                automargin=True,
                constrain='domain'
            )
        )

        return fig
    else:
        st.error("Currently, only the 'Initial Code -> Reduced Code -> Theme' flow is implemented.")
        return None

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
        data = load_data(selected_project)
        
        if data is None or data.empty:
            st.error("No data available for the selected project.")
            return

        selected_levels = ['Initial Code', 'Reduced Code', 'Theme']

        # Advanced settings in an expander
        with st.expander("Advanced Settings"):
            # Create a single tab for settings
            tab, = st.tabs(["Settings"])
            with tab:
                # Filters Section
                st.markdown("#### Filters")
                filter_theme = st.multiselect(
                    "Filter by Theme", 
                    options=data['Theme'].unique(), 
                    help="Select themes to display in the Sankey diagram"
                )
                filter_reduced_code = st.multiselect(
                    "Filter by Reduced Code", 
                    options=data['code'].unique(),
                    help="Select reduced codes to display in the Sankey diagram"
                )
                filter_initial_code = st.multiselect(
                    "Filter by Initial Code", 
                    options=data['original_code'].unique(),
                    help="Select initial codes to display in the Sankey diagram"
                )
                
                # Colours Section
                st.markdown("#### Colours")
                col1, col2, col3 = st.columns(3)
                with col1:
                    initial_code_color = st.color_picker(
                        "Initial Codes Color", 
                        '#F7F069', 
                        help="Choose the color for initial code nodes"
                    )
                with col2:
                    reduced_code_color = st.color_picker(
                        "Reduced Codes Color", 
                        '#DE7863', 
                        help="Choose the color for reduced code nodes"
                    )
                with col3:
                    theme_color = st.color_picker(
                        "Themes Color", 
                        '#6BE6D3',
                        help="Choose the color for theme nodes"
                    )
                link_opacity = st.slider(
                    "Link Opacity", 
                    0.0, 1.0, 1.0, 0.1, 
                    help="Adjust the opacity of the links between nodes (0 is fully transparent, 1 is fully opaque)"
                )
                
                # Layout Section
                st.markdown("#### Layout")
                col1, col2 = st.columns(2)
                with col1:
                    font_size = st.number_input(
                        "Font Size", 
                        8, 24, 16, 
                        help="Set the font size of the labels in the diagram"
                    )
                    nodes_per_height = st.number_input(
                        "Height per Node", 
                        5, 50, 20,
                        help="Adjust the vertical spacing between nodes; higher values increase spacing"
                    )
                with col2:
                    margin_top = st.number_input(
                        "Top Margin", 
                        0, 100, 5, 
                        help="Set the top margin of the diagram in pixels"
                    )
                    margin_bottom = st.number_input(
                        "Bottom Margin", 
                        0, 100, 10, 
                        help="Set the bottom margin of the diagram in pixels"
                    )
                    margin_left = st.number_input(
                        "Left Margin", 
                        0, 100, 5, 
                        help="Set the left margin of the diagram in pixels"
                    )
                    margin_right = st.number_input(
                        "Right Margin", 
                        0, 100, 10, 
                        help="Set the right margin of the diagram in pixels"
                    )
                
                # Node Style Section
                st.markdown("#### Node Style")
                col1, col2 = st.columns(2)
                with col1:
                    node_pad = st.number_input(
                        "Node Padding", 
                        5, 50, 15, 
                        help="Set the amount of padding between nodes"
                    )
                    node_thickness = st.number_input(
                        "Node Thickness", 
                        5, 50, 20, 
                        help="Set the thickness of the nodes in pixels"
                    )
                with col2:
                    node_line_color = st.color_picker(
                        "Node Line Color", 
                        '#ffffff', 
                        help="Choose the color of the node borders"
                    )
                    node_line_width = st.number_input(
                        "Node Line Width", 
                        0.0, 5.0, 1.0, 0.1, 
                        help="Set the width of the node borders"
                    )

        # Apply filters
        df_filtered = data.copy()
        if filter_theme:
            df_filtered = df_filtered[df_filtered['Theme'].isin(filter_theme)]
        if filter_reduced_code:
            df_filtered = df_filtered[df_filtered['code'].isin(filter_reduced_code)]
        if filter_initial_code:
            df_filtered = df_filtered[df_filtered['original_code'].isin(filter_initial_code)]

        # Collect style settings
        style_settings = {
            'initial_code_color': initial_code_color,
            'reduced_code_color': reduced_code_color,
            'theme_color': theme_color,
            'link_opacity': link_opacity,
            'font_size': font_size,
            'margin_top': margin_top,
            'margin_bottom': margin_bottom,
            'margin_left': margin_left,
            'margin_right': margin_right,
            'node_pad': node_pad,
            'node_thickness': node_thickness,
            'node_line_color': node_line_color,
            'node_line_width': node_line_width,
            'nodes_per_height_unit': nodes_per_height
        }

        # Create and display the Sankey diagram
        fig = create_sankey_diagram(df_filtered, selected_levels, style_settings)

        if fig:

            st.plotly_chart(fig, use_container_width=True)

            # Option to display data as a table
            with st.expander("Show Data Table"):
                st.dataframe(df_filtered)
    
    # Manage API keys
    manage_api_keys()

if __name__ == "__main__":
    main()
