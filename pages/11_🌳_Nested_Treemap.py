import streamlit as st
import pandas as pd
import plotly.express as px
from project_utils import get_projects, PROJECTS_DIR, get_processed_files
import os
from api_key_management import manage_api_keys, load_api_keys

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

def prepare_treemap_data(themes_df, codes_df):
    treemap_data = []
    
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
            #source = code_row['source']
            
            treemap_data.append({
                'Theme': theme,
                'Theme Description': theme_description,
                'Reduced Code': reduced_code,
                'Reduced Code Description': reduced_code_description,
                'Initial Code': initial_code,
                'Quote': quote,
                'Value': 1
            })
    
    return pd.DataFrame(treemap_data)

def main():
    st.header(":orange[Nested Treemap Visualization]")

    st.subheader(":orange[Structure: Theme > Reduced Codes > Initial Code > Quote]")

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

        treemap_data = prepare_treemap_data(themes_df, codes_df)

        fig = px.treemap(
            treemap_data,
            path=['Theme', 'Reduced Code', 'Initial Code', 'Quote'],
            values='Value',
            hover_data=['Theme Description', 'Reduced Code Description'],
            color='Theme',
            color_continuous_scale='RdBu',
        )

        fig.update_traces(
            textinfo="label",
            textfont=dict(size=14),
            hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Description: %{customdata[0]}<extra></extra>'
        )

        fig.update_layout(
            margin=dict(t=50, l=25, r=25, b=25),
            height=800,
            title="Nested Treemap of Themes, Codes, and Quotes"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write("Click on the rectangles to zoom in. Double-click to zoom out. Hover for descriptions.")
        st.write("To return to the original view, click in the top left of the vsisualisation or refresh the page.")

    manage_api_keys()

if __name__ == "__main__":
    main()