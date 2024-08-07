import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from project_utils import get_projects, PROJECTS_DIR, get_processed_files
import os
import json
import numpy as np


def load_data(project_name):
    themes_folder = os.path.join(PROJECTS_DIR, project_name, 'theme_books')
    codes_folder = os.path.join(PROJECTS_DIR, project_name, 'expanded_reduced_codes')
    
    themes_files = get_processed_files(project_name, 'theme_books')
    expanded_themes_files = [f for f in themes_files if 'expanded' in f]
    if not expanded_themes_files:
        st.warning(f"No expanded theme files found in {themes_folder}")
        return None, None
    latest_themes_file = max(expanded_themes_files, key=lambda f: os.path.getmtime(os.path.join(themes_folder, f)))
    themes_df = pd.read_csv(os.path.join(themes_folder, latest_themes_file))
    
    codes_files = get_processed_files(project_name, 'expanded_reduced_codes')
    if not codes_files:
        st.warning(f"No code files found in {codes_folder}")
        return None, None
    latest_codes_file = max(codes_files, key=lambda f: os.path.getmtime(os.path.join(codes_folder, f)))
    codes_df = pd.read_csv(os.path.join(codes_folder, latest_codes_file))

    st.write("Themes DataFrame:")
    st.write(themes_df.head())
    st.write("Codes DataFrame:")
    st.write(codes_df.head())

    return themes_df, codes_df

def prepare_sunburst_data(themes_df, codes_df):
    sunburst_data = {
        'ids': [],
        'labels': [],
        'parents': [],
        'text': [],
        'values': []
    }
    
    # Add themes
    for _, theme_row in themes_df.iterrows():
        theme = theme_row['Theme']
        sunburst_data['ids'].append(f"theme_{theme}")
        sunburst_data['labels'].append(theme)
        sunburst_data['parents'].append("")
        sunburst_data['text'].append(theme_row['Theme Description'])
        sunburst_data['values'].append(1)
        
        # Add reduced codes for each theme
        reduced_codes = codes_df[codes_df['code'] == theme_row['Code']]
        for _, code_row in reduced_codes.iterrows():
            reduced_code = code_row['code']
            sunburst_data['ids'].append(f"reduced_{reduced_code}")
            sunburst_data['labels'].append(reduced_code)
            sunburst_data['parents'].append(f"theme_{theme}")
            sunburst_data['text'].append(code_row['description'])
            sunburst_data['values'].append(1)
            
            # Add initial codes for each reduced code
            initial_codes = codes_df[codes_df['code'] == reduced_code]
            for _, initial_code_row in initial_codes.iterrows():
                initial_code = initial_code_row['original_code']
                sunburst_data['ids'].append(f"initial_{initial_code}")
                sunburst_data['labels'].append(initial_code)
                sunburst_data['parents'].append(f"reduced_{reduced_code}")
                sunburst_data['text'].append(initial_code_row['quote'])
                sunburst_data['values'].append(1)
    
    result_df = pd.DataFrame(sunburst_data)
    
    # Convert 'values' column to numeric
    result_df['values'] = pd.to_numeric(result_df['values'], errors='coerce')
    
    # Check for negative values
    if (result_df['values'] < 0).any():
        st.warning("Negative values found in the data. Setting them to 0.")
        result_df['values'] = result_df['values'].clip(lower=0)
    
    # Ensure leaf node values are not larger than parent nodes
    def adjust_values(group):
        sorted_group = group.sort_values('values', ascending=False)
        for i in range(1, len(sorted_group)):
            if sorted_group.iloc[i]['values'] > sorted_group.iloc[i-1]['values']:
                sorted_group.iloc[i, sorted_group.columns.get_loc('values')] = sorted_group.iloc[i-1]['values']
        return sorted_group

    result_df = result_df.groupby('parents', group_keys=False).apply(adjust_values)
    
    st.write("Sunburst Data:")
    st.write(result_df.head())
    st.write(f"Total rows in sunburst data: {len(result_df)}")
    
    # Additional checks
    st.write("Data types:")
    st.write(result_df.dtypes)
    st.write("Summary statistics:")
    st.write(result_df.describe())
    
    return result_df

def main():
    st.header(":orange[Theme-Codes Sunburst]")

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

        sunburst_data = prepare_sunburst_data(themes_df, codes_df)

        if len(sunburst_data) == 0:
            st.warning("No data to display in the sunburst chart.")
            return

        try:
            fig = go.Figure(go.Sunburst(
                ids=sunburst_data['ids'],
                labels=sunburst_data['labels'],
                parents=sunburst_data['parents'],
                text=sunburst_data['text'],
                values=sunburst_data['values'],
                branchvalues="total",
                hovertemplate='<b>%{label}</b><br>%{text}<extra></extra>',
                maxdepth=3
            ))

            fig.update_layout(
                margin=dict(t=0, l=0, r=0, b=0),
                height=800,
            )

            st.write('Click on segments to zoom in. Double click to zoom out. Hover for descriptions.')
            st.plotly_chart(fig, use_container_width=True)
            
            # If the chart doesn't appear, display the raw data
            st.write("Raw chart data:")
            st.write(sunburst_data)
            
        except Exception as e:
            st.error(f"An error occurred while creating the chart: {str(e)}")
            st.write("Sunburst data sample:")
            st.write(sunburst_data.head())

if __name__ == "__main__":
    main()