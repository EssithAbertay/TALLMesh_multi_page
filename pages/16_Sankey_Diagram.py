import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

def create_sankey(themes_df, codes_df):
    # Prepare the data for the Sankey Diagram
    theme_labels = list(themes_df['Theme'].unique())
    code_labels = list(themes_df['Code'].unique())
    initial_code_labels = list(codes_df['original_code'].unique())
    
    labels = theme_labels + code_labels + initial_code_labels
    
    # Create a mapping of labels to indices
    label_to_index = {label: i for i, label in enumerate(labels)}

    # Prepare source and target data
    sources = []
    targets = []
    values = []

    # Connect themes to reduced codes
    for _, row in themes_df.iterrows():
        theme_index = label_to_index[row['Theme']]
        code_index = label_to_index[row['Code']]
        sources.append(theme_index)
        targets.append(code_index)
        values.append(1)  # Use actual data if available

    # Connect reduced codes to initial codes
    for _, row in codes_df.iterrows():
        code_index = label_to_index[row['code']]
        initial_code_index = label_to_index[row['original_code']]
        sources.append(code_index)
        targets.append(initial_code_index)
        values.append(1)  # Use actual data if available

    # Create a Sankey Diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=["#99CCFF"] * len(theme_labels) + 
                  ["#99FF99"] * len(code_labels) + 
                  ["#FFCC99"] * len(initial_code_labels)
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    ))

    fig.update_layout(
        title_text="Sankey Diagram of Themes and Codes", 
        font_size=10,
        height=800  # Increase the height for better visibility
    )

    return fig

def main():
    st.header("Sankey Diagram Visualization")

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

        fig = create_sankey(themes_df, codes_df)
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
