import streamlit as st
import pandas as pd
from itertools import combinations
from graphviz import Digraph
from project_utils import get_projects, PROJECTS_DIR, get_processed_files
import os
from api_key_management import manage_api_keys, load_api_keys

def load_data(project_name):
    themes_folder = os.path.join(PROJECTS_DIR, project_name, 'theme_books')
    
    themes_files = get_processed_files(project_name, 'theme_books')
    expanded_themes_files = [f for f in themes_files if 'expanded' in f]
    if not expanded_themes_files:
        return None
    latest_themes_file = max(expanded_themes_files, key=lambda f: os.path.getmtime(os.path.join(themes_folder, f)))
    themes_df = pd.read_csv(os.path.join(themes_folder, latest_themes_file))
    
    return themes_df

def common_codes(codes1, codes2):
    return len(set(codes1) & set(codes2))

def create_graph(data, min_common_codes):
    themes = data['Theme'].unique()
    edges = []
    
    for pair in combinations(themes, 2):
        theme1_codes = set(data[data['Theme'] == pair[0]]['Code'])
        theme2_codes = set(data[data['Theme'] == pair[1]]['Code'])
        theme1 = pair[0]
        theme2 = pair[1]
        common = common_codes(theme1_codes, theme2_codes)
        if common >= min_common_codes:
            edges.append((theme1, theme2, common))
    
    output_dot = "digraph {\n"
    for edge in edges:
        output_dot += f'\t"{edge[0]}" -> "{edge[1]}" [label="Common Codes: {edge[2]}" penwidth={edge[2]} dir=none]\n'
    output_dot += "}"
    
    return output_dot

def main():
    st.header(":orange[Thematic Map based on Shared Codes]")
    st.subheader("Visualizing relationships between themes based on common codes")

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

    if selected_project != "Select a project...":
        themes_df = load_data(selected_project)

        if themes_df is None:
            st.error("No theme data available for the selected project.")
            return

        min_common_codes = st.slider("Minimum Common Codes", min_value=1, max_value=10, value=1)

        graph_dot = create_graph(themes_df, min_common_codes)
        
        st.graphviz_chart(graph_dot)
        
        st.write("This graph shows connections between themes based on their shared codes. "
                 "The thickness of the lines represents the number of common codes. "
                 "Adjust the 'Minimum Common Codes' slider to filter connections.")

    manage_api_keys()

if __name__ == "__main__":
    main()