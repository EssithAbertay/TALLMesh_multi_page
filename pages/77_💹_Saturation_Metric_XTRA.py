# -*- coding: utf-8 -*-
"""
Saturation Metric Analysis Module

This module provides functionality for analyzing and visualizing the saturation
of qualitative coding in research projects. It calculates and displays various
metrics related to code saturation, including the Information Tracking Score (ITS),
rolling ITS, and rate of change.

Created on Fri Feb  2 15:16:46 2024
@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""

import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from api_key_management import manage_api_keys
from project_utils import get_projects, get_project_files, get_processed_files

# Constants
PROJECTS_DIR = 'projects'

def calculate_rolling_its(unique_counts, total_counts):
    """
    Calculate the rolling Information Tracking Score (ITS) for each file.

    Args:
    unique_counts (list): Cumulative count of unique codes for each file.
    total_counts (list): Cumulative count of total codes for each file.

    Returns:
    list: Rolling ITS values for each file.
    """
    return [round(u / t, 3) for u, t in zip(unique_counts, total_counts)]

def calculate_rate_of_change(its_values):
    """
    Calculate the rate of change in ITS values between consecutive files.

    Args:
    its_values (list): List of ITS values for each file.

    Returns:
    list: Rate of change in ITS values, with 0 as the first element.
    """
    return [0] + [its_values[i] - its_values[i-1] for i in range(1, len(its_values))]

def bootstrap_its(unique_counts, total_counts, n_iterations=1000):
    """
    Perform bootstrap analysis to calculate confidence interval for ITS.

    Args:
    unique_counts (list): Cumulative count of unique codes for each file.
    total_counts (list): Cumulative count of total codes for each file.
    n_iterations (int): Number of bootstrap iterations to perform.

    Returns:
    tuple: Lower and upper bounds of the 95% confidence interval.
    """
    its_values = []
    for _ in range(n_iterations):
        indices = np.random.choice(len(unique_counts), len(unique_counts), replace=True)
        sampled_unique = [unique_counts[i] for i in indices]
        sampled_total = [total_counts[i] for i in indices]
        its_values.append(sampled_unique[-1] / sampled_total[-1])
    return np.percentile(its_values, [2.5, 97.5])

def find_elbow_point(x, y):
    """
    Find the elbow point in a curve using the maximum distance method.

    Args:
    x (list): X-coordinates of the curve points.
    y (list): Y-coordinates of the curve points.

    Returns:
    int: Index of the elbow point.
    """
    nPoints = len(x)
    allCoord = np.vstack((x, y)).T
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.tile(lineVecNorm, (nPoints, 1)), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint

def create_saturation_plot(unique_counts, total_counts, rolling_its, rate_of_change):
    """
    Create a plotly figure for visualizing saturation metrics.

    Args:
    unique_counts (list): Cumulative count of unique codes for each file.
    total_counts (list): Cumulative count of total codes for each file.
    rolling_its (list): Rolling ITS values for each file.
    rate_of_change (list): Rate of change in ITS values for each file.

    Returns:
    plotly.graph_objs._figure.Figure: The created plotly figure.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=list(range(1, len(unique_counts) + 1)), y=unique_counts, mode='lines+markers', name='Unique Codes'))
    fig.add_trace(go.Scatter(x=list(range(1, len(total_counts) + 1)), y=total_counts, mode='lines+markers', name='Total Codes'))
    fig.add_trace(go.Scatter(x=list(range(1, len(rolling_its) + 1)), y=rolling_its, mode='lines+markers', name='Rolling ITS', yaxis='y2'))
    fig.add_trace(go.Scatter(x=list(range(1, len(rate_of_change) + 1)), y=rate_of_change, mode='lines+markers', name='Rate of Change', yaxis='y2'))

    fig.update_layout(
        title='Codes, ITS, and Rate of Change',
        xaxis_title='File Index',
        yaxis_title='Code Count',
        yaxis2_title='ITS / Rate of Change'
    )

    return fig

def main():
    """
    Main function to run the Streamlit app for saturation metric analysis.
    """
    st.header(":orange[Measure saturation]")
    
    st.write("See our paper on saturation and LLMs (https://arxiv.org/pdf/2401.03239) for more information.")
    
    st.subheader("This version uses the files from initial coding (to derive total codes) and the files from the reduction of codes (unique codes) stored in your project folder.")

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
        results_file = os.path.join(PROJECTS_DIR, selected_project, 'code_reduction_results.csv')

        if os.path.exists(results_file):
            with st.spinner("Processing..."):
                results_df = pd.read_csv(results_file)
                
                unique_counts = results_df['unique_codes'].tolist()
                total_counts = results_df['total_codes'].tolist()

                st.success("Files processed successfully!")

                # Calculate ITS Metric (Saturation)
                its_metric = round(unique_counts[-1] / total_counts[-1], 3)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(":orange[ITS Metric (Saturation):]")
                with col2:
                    st.subheader(f":green[{its_metric}]")

                # Calculate rolling ITS and rate of change
                rolling_its = calculate_rolling_its(unique_counts, total_counts)
                rate_of_change = calculate_rate_of_change(rolling_its)

                # Saturation threshold
                saturation_threshold = st.slider("Set saturation threshold", 0.0, 1.0, 0.05, 0.01)
                saturation_point = next((i for i, its in enumerate(rolling_its) if its >= 1 - saturation_threshold), len(rolling_its))

                # Create and display plot
                fig = create_saturation_plot(unique_counts, total_counts, rolling_its, rate_of_change)
                st.plotly_chart(fig)

                # Calculate and display confidence interval
                confidence_interval = bootstrap_its(unique_counts, total_counts)
                st.write(f"95% Confidence Interval for ITS: {confidence_interval}")

                # Find and display elbow point
                elbow_point = find_elbow_point(range(len(rolling_its)), rolling_its)
                st.write(f"Suggested saturation point (elbow method): File {elbow_point + 1}")

                # Display summary statistics
                st.subheader("Summary Statistics")
                st.write(f"Total number of files processed: {len(total_counts)}")
                st.write(f"Final number of unique codes: {unique_counts[-1]}")
                st.write(f"Final number of total codes: {total_counts[-1]}")
                st.write(f"Maximum rate of change: {max(rate_of_change):.3f}")
                st.write(f"Minimum rate of change: {min(rate_of_change):.3f}")

                # Display data in two columns
                col1, col2 = st.columns(2)
                with col2:
                    st.write("Unique Codes Counts:")
                    st.write(unique_counts)
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