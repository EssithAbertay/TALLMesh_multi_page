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

def extract_numeric_part(filename):
    return int(filename.split('_')[0])

def count_rows_in_folder(folder_path):
    file_counts = []
    for filename in sorted(os.listdir(folder_path), key=extract_numeric_part):
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
    
    st.subheader("For this the current version requires two zip folders, one with the files from initial coding (Total) and one with the files from the reduction of codes (Unique)")

    st.write("Upload the 'unique codes' folder:")
    unique_codes_folder = st.file_uploader("Upload ZIP file", type="zip")

    st.write("Upload the 'total codes' folder:")
    total_codes_folder = st.file_uploader("Upload ZIP file2", type="zip")

    if unique_codes_folder and total_codes_folder:
        with st.spinner("Processing..."):
            unique_counts = []
            total_counts = []

            # Extract and process files from 'unique codes' folder
            with ZipFile(BytesIO(unique_codes_folder.read()), 'r') as zip_ref:
                for filename in sorted(zip_ref.namelist(), key=extract_numeric_part):
                    if filename.endswith('.csv'):
                        with zip_ref.open(filename) as file:
                            df = pd.read_csv(file)
                            unique_counts.append(len(df))

            # Extract and process files from 'total codes' folder
            with ZipFile(BytesIO(total_codes_folder.read()), 'r') as zip_ref:
                for filename in sorted(zip_ref.namelist(), key=extract_numeric_part):
                    if filename.endswith('.csv'):
                        with zip_ref.open(filename) as file:
                            df = pd.read_csv(file)
                            total_counts.append(len(df))

            if len(unique_counts) != len(total_counts):
                st.error("Error: Number of files in the two folders does not match.")
            else:
                st.success("Files processed successfully!")

                st.write("Unique Codes Counts:")
                st.write(unique_counts)

                st.write("Total Codes Cumulative Sum:")
                st.write(cumulative_sum(total_counts))
                
                

                # Calculate ITS Metric (Saturation)
                its_metric = round(unique_counts[-1] / cumulative_sum(total_counts)[-1], 3)
                
                st.write("ITS Metric (Saturation):", its_metric)

                # Create plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(1, len(unique_counts) + 1)), y=unique_counts, mode='lines+markers', name='Unique Codes', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=list(range(1, len(total_counts) + 1)), y=cumulative_sum(total_counts), mode='lines+markers', name='Total Codes Cumulative Sum'))
                fig.update_layout(title='Unique Codes vs Total Codes Cumulative Sum', xaxis_title='File Index', yaxis_title='Count')
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()
