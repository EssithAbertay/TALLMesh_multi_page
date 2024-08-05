# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:48:38 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from project_utils import get_projects

def main():
    st.header(":orange[Theme-Codes Icicle]")

    # Project selection
    projects = get_projects()
    
    # Initialize session state for selected project if it doesn't exist
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = "Select a project..."

    # Calculate the index for the selectbox
    project_options = ["Select a project..."] + projects
    if st.session_state.selected_project in project_options:
        index = project_options.index(st.session_state.selected_project)
    else:
        index = 0

    # Use selectbox with the session state as the default value
    selected_project = st.selectbox(
        "Select a project:", 
        project_options,
        index=index,
        key="project_selector"
    )

    # Update session state when a new project is selected
    if selected_project != st.session_state.selected_project:
        st.session_state.selected_project = selected_project
        st.rerun()

    uploaded_file = st.file_uploader("Upload a complete Theme Book", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("File Uploaded Successfully!")

        # Get unique themes
        unique_themes = df['Theme'].unique()
    
        # Let user select a theme
        selected_theme = st.selectbox('Select a Theme to Visualise', unique_themes)
    
        # Filter DataFrame based on selected theme
        df_filtered = df[df['Theme'] == selected_theme]
    
        # Create icicle plot
        fig = px.icicle(df_filtered, 
                        path=['Theme', 'Code', 'Quote', 'Source'], 
                        values='Code',  # Using 'Code' as a placeholder for values
                        branchvalues='total')
        fig.update_traces(root_color="black")
        fig.update_layout(margin=dict(t=5, l=2, r=2, b=2), uniformtext=dict(minsize=20))
    
        # Tell to click
        st.write('You can click on the components to see them in more details')
        
        # Display the icicle plot
        col1, col2 = st.columns([5, 250])
        with col2:
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
