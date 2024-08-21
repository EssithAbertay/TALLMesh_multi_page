import streamlit as st
import pandas as pd
from itertools import combinations
from graphviz import Digraph
import os

# Import custom utility functions and constants
from project_utils import get_projects, PROJECTS_DIR, get_processed_files
from api_key_management import manage_api_keys, load_api_keys

def load_data(project_name):
    """
    Load the latest expanded themes data for a given project.

    Args:
    project_name (str): The name of the project to load data for.

    Returns:
    pandas.DataFrame or None: A DataFrame containing the themes data if available, None otherwise.
    """
    # Construct the path to the themes folder for the specified project
    themes_folder = os.path.join(PROJECTS_DIR, project_name, 'theme_books')
    
    # Get all processed theme book files for the project
    themes_files = get_processed_files(project_name, 'theme_books')
    
    # Filter for expanded theme files
    expanded_themes_files = [f for f in themes_files if 'expanded' in f]
    
    # If no expanded theme files are found, return None
    if not expanded_themes_files:
        return None
    
    # Get the most recently modified expanded themes file
    latest_themes_file = max(expanded_themes_files, key=lambda f: os.path.getmtime(os.path.join(themes_folder, f)))
    
    # Load and return the themes data as a pandas DataFrame
    themes_df = pd.read_csv(os.path.join(themes_folder, latest_themes_file))
    return themes_df

def common_codes(codes1, codes2):
    """
    Calculate the number of common codes between two sets of codes.

    Args:
    codes1 (set): First set of codes.
    codes2 (set): Second set of codes.

    Returns:
    int: The number of common codes between the two sets.
    """
    return len(set(codes1) & set(codes2))

def create_graph(data, min_common_codes):
    """
    Create a DOT language representation of the thematic overlap graph.

    Args:
    data (pandas.DataFrame): The themes data.
    min_common_codes (int): The minimum number of common codes required for a connection.

    Returns:
    str or None: A string containing the DOT language representation of the graph, or None if no edges are found.
    """
    # Get unique themes from the data
    themes = data['Theme'].unique()
    edges = []
    
    # Generate all possible pairs of themes
    for pair in combinations(themes, 2):
        # Get the codes for each theme in the pair
        theme1_codes = set(data[data['Theme'] == pair[0]]['Code'])
        theme2_codes = set(data[data['Theme'] == pair[1]]['Code'])
        theme1, theme2 = pair
        
        # Calculate the number of common codes
        common = common_codes(theme1_codes, theme2_codes)
        
        # If the number of common codes meets the threshold, add an edge
        if common >= min_common_codes:
            edges.append((theme1, theme2, common))
    
    # If no edges are found, return None
    if not edges:
        return None

    # Create the DOT language representation of the graph
    output_dot = "digraph {\n"
    for edge in edges:
        output_dot += f'\t"{edge[0]}" -> "{edge[1]}" [label="Common Codes: {edge[2]}" penwidth={max(1, edge[2])} dir=none]\n'
    output_dot += "}"
    
    return output_dot

def main():
    """
    Main function to run the Streamlit app for the Thematic Overlap Map.
    """
    st.header(":orange[Thematic Map based on Shared Codes]")
    st.subheader("Visualizing relationships between themes based on common codes")

    # Get available projects
    projects = get_projects()
    
    # Initialize session state for selected project if not already set
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = "Select a project..."

    # Create list of project options with default option
    project_options = ["Select a project..."] + projects
    
    # Get index of currently selected project
    index = project_options.index(st.session_state.selected_project) if st.session_state.selected_project in project_options else 0

    # Create selectbox for project selection
    selected_project = st.selectbox(
        "Select a project:", 
        project_options,
        index=index,
        key="project_selector"
    )

    # If a new project is selected, update session state and rerun
    if selected_project != st.session_state.selected_project:
        st.session_state.selected_project = selected_project
        st.rerun()

    # If a valid project is selected, proceed with data loading and visualization
    if selected_project != "Select a project...":
        themes_df = load_data(selected_project)

        # Check if theme data is available
        if themes_df is None:
            st.error("No theme data available for the selected project.")
            return

        # Create slider for minimum common codes
        min_common_codes = st.slider("Minimum Common Codes", min_value=0, max_value=10, value=0)

        # Create the graph
        graph_dot = create_graph(themes_df, min_common_codes)
        
        # Display the graph or a warning if no connections are found
        if graph_dot is None:
            st.warning("No connections found with the current minimum common codes. Try lowering the threshold.")
        else:
            st.graphviz_chart(graph_dot)
            
            st.write("This graph shows connections between themes based on their shared codes. "
                     "The thickness of the lines represents the number of common codes. "
                     "Adjust the 'Minimum Common Codes' slider to filter connections.")

    # Display API key management section
    manage_api_keys()

if __name__ == "__main__":
    main()