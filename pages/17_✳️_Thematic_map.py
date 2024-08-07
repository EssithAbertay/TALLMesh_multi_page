import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from project_utils import get_projects, PROJECTS_DIR, get_processed_files
import os
import tempfile

def load_data(project_name):
    themes_folder = os.path.join(PROJECTS_DIR, project_name, 'theme_books')
    codes_folder = os.path.join(PROJECTS_DIR, project_name, 'expanded_reduced_codes')
    
    themes_files = get_processed_files(project_name, 'theme_books')
    expanded_themes_files = [f for f in themes_files if 'expanded' in f]
    if not expanded_themes_files:
        return None, None
    latest_themes_file = max(expanded_themes_files, key=lambda f: os.path.getmtime(os.path.join(themes_folder, f)))
    themes_df = pd.read_csv(os.path.join(themes_folder, latest_themes_file))
    
    codes_files = get_processed_files(project_name, 'expanded_reduced_codes')
    if not codes_files:
        return None, None
    latest_codes_file = max(codes_files, key=lambda f: os.path.getmtime(os.path.join(codes_folder, f)))
    codes_df = pd.read_csv(os.path.join(codes_folder, latest_codes_file))

    return themes_df, codes_df

def create_thematic_map(themes_df, codes_df, min_common_codes):
    G = nx.Graph()
    
    # Add theme nodes
    for _, theme_row in themes_df.iterrows():
        theme = theme_row['Theme']
        theme_id = f"theme_{theme}"
        theme_description = theme_row['Theme Description']
        G.add_node(theme_id, color='#99CCFF', size=30, title=f"{theme}\n\n{theme_description}", label=theme)
        
        # Add reduced code nodes
        reduced_code = theme_row['Code']
        reduced_code_id = f"reduced_{reduced_code}"
        reduced_code_description = theme_row['Code Description']
        G.add_node(reduced_code_id, color='#99FF99', size=20, title=f"{reduced_code}\n\n{reduced_code_description}", label=reduced_code)
        G.add_edge(theme_id, reduced_code_id)
    
    # Create edges between themes based on common codes
    for i, theme1 in enumerate(themes_df['Theme']):
        for j, theme2 in enumerate(themes_df['Theme']):
            if i < j:
                theme1_codes = set(themes_df[themes_df['Theme'] == theme1]['Code'])
                theme2_codes = set(themes_df[themes_df['Theme'] == theme2]['Code'])
                common = len(theme1_codes & theme2_codes)
                if common >= min_common_codes:
                    G.add_edge(f"theme_{theme1}", f"theme_{theme2}", weight=common, title=f"Common Codes: {common}")
    
    # Create a Pyvis network from our NetworkX graph
    net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white')
    net.from_nx(G)
    net.toggle_physics(True)
    net.set_options('''
    var options = {
      "nodes": {
        "font": {
          "size": 12
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    ''')
    
    # Save and read the HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        net.save_graph(tmpfile.name)
        with open(tmpfile.name, 'r', encoding='utf-8') as f:
            html_string = f.read()
    
    return html_string

def main():
    st.header(":orange[Thematic Map based on shared codes]")

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
        themes_df, codes_df = load_data(selected_project)

        if themes_df is None or codes_df is None:
            st.error("No data available for the selected project.")
            return

        min_common_codes = st.slider("Minimum Common Codes", min_value=1, max_value=10, value=1)

        thematic_map_html = create_thematic_map(themes_df, codes_df, min_common_codes)

        st.components.v1.html(thematic_map_html, height=600)

        st.write("Drag nodes to rearrange. Zoom with mouse wheel. Hover over nodes and edges for more information.")

if __name__ == "__main__":
    main()