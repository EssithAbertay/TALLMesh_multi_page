# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:16:46 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import os
from zipfile import ZipFile
from io import BytesIO
import plotly.graph_objects as go
from api_key_management import manage_api_keys, load_api_keys
from project_utils import get_projects, get_project_files, get_processed_files

PROJECTS_DIR = 'projects'

def count_rows_in_folder(folder_path):
    file_counts = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            file_counts.append(len(df))
    return file_counts

def cumulative_sum(file_counts):
    cumulative_counts = [sum(file_counts[:i+1]) for i in range(len(file_counts))]
    return cumulative_counts

def main():
    st.header(":orange[Measure saturation]")
    
    st.write("See our paper on saturation and LLMs (https://arxiv.org/pdf/2401.03239) for more information.")
    
    st.subheader("This version uses the files from initial coding (to derive total codes) and the files from the reduction of codes (unique codes) stored in your project folder.")

    projects = get_projects()
    selected_project = st.selectbox("Select a project:", ["Select a project..."] + projects)

    if selected_project != "Select a project...":
        results_file = os.path.join(PROJECTS_DIR, selected_project, 'code_reduction_results.csv')

        if os.path.exists(results_file):
            with st.spinner("Processing..."):
                results_df = pd.read_csv(results_file)
                
                unique_counts = results_df['unique_codes'].tolist()
                total_counts = results_df['total_codes'].tolist()
                cumulative_total_counts = cumulative_sum(total_counts)

                st.success("Files processed successfully!")

                # Calculate ITS Metric (Saturation)
                its_metric = round(unique_counts[-1] / cumulative_total_counts[-1], 3)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(":orange[ITS Metric (Saturation):]")
                with col2:
                    st.subheader(f":green[{its_metric}]")

                # Create plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(1, len(unique_counts) + 1)), y=unique_counts, mode='lines+markers', name='Unique Codes', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=list(range(1, len(total_counts) + 1)), y=total_counts, mode='lines+markers', name='Total Initial Codes'))
                fig.update_layout(title='Unique Codes vs Total Codes Cumulative Sum', xaxis_title='File Index', yaxis_title='Count')
                st.plotly_chart(fig)

                # Create two columns
                col1, col2 = st.columns(2)

                # Display data in the first column
                with col2:
                    st.write("Unique Codes Counts:")
                    st.write(unique_counts)

                # Display data in the second column
                with col1:
                    st.write("Total Codes Cumulative Sum:")
                    st.write(total_counts)
        else:
            st.error("Error: Code reduction results file not found. Please run the code reduction process first.")
    else:
        st.write("Please select a project to continue.")

    manage_api_keys()

if __name__ == "__main__":
    main()