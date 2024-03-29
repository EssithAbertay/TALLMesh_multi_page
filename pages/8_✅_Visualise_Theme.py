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

    #df = pd.read_csv('C:/Users/s516331/Desktop/LLMs_research/BA_project/text_example_final/after_each/rb.csv')
    
    uploaded_file = st.file_uploader("Upload a complete Theme Book", type=["csv"])
    
    if uploaded_file:
       
        df = pd.read_csv(uploaded_file)#rename columns just in case
        
        #add index column
       # df = df.reset_index(drop=True)

        st.write("File Uploaded Successfully!")

        # Get unique themes
        unique_themes = df['Theme'].unique()
    
        # Let user select a theme
        selected_theme = st.selectbox('Select a Theme to Visualise', unique_themes)
    
        # Filter DataFrame based on selected theme
        df_filtered = df[df['Theme'] == selected_theme]
    
        # Get all columns starting with 'quote_'
        quote_columns = [col for col in df_filtered.columns if col.startswith('quote_')]
    
        # Reshape DataFrame to have quote columns stacked vertically
        df_stacked = df_filtered.melt(id_vars=['Theme', 'code'], value_vars=quote_columns, var_name='quote_number', value_name='quote')
    
        # Group by Theme, Code, and Quote Number, then count occurrences
        df_grouped = df_stacked.groupby(['Theme', 'code', 'quote_number', 'quote']).size().reset_index(name='count')
    
        # Create icicle plot
        fig = px.icicle(df_grouped, path=['Theme', 'code', 'quote_number', 'quote'], values='count', branchvalues='total')
        fig.update_traces(root_color="black")
        #fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    
        fig.update_layout(margin=dict(t=5, l=2, r=2, b=2), uniformtext=dict(minsize=20))
    
    
        #tell to click
        st.write ('You can click on the components to see them in more details')
        # Display the icicle plot
        col1, col2 = st.columns([5, 250])
        with col2:
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
