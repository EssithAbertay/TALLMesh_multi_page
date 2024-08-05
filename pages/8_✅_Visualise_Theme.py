# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:48:38 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""
import streamlit as st
import pandas as pd
import plotly.express as px

def main():
    st.header(":orange[Theme-Codes Icicle]")

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
