# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:16:46 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk

This script implements a Streamlit page for measuring and visualizing the saturation
of qualitative coding in a research project. It calculates and displays the ITS
(Incremental Theme Saturation) Metric, which is a measure of coding saturation.

The script uses data from initial coding (total codes) and the reduction of codes
(unique codes) stored in the project folder to generate these metrics and visualizations.
"""

import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from api_key_management import manage_api_keys
from project_utils import get_projects

# Constants
PROJECTS_DIR = 'projects'

def count_rows_in_folder(folder_path):
    """
    Count the number of rows in each CSV file within a given folder.
    
    Args:
    folder_path (str): Path to the folder containing CSV files.
    
    Returns:
    list: A list of row counts for each CSV file in the folder.
    """
    file_counts = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            file_counts.append(len(df))
    return file_counts

def cumulative_sum(file_counts):
    """
    Calculate the cumulative sum of a list of counts.
    
    Args:
    file_counts (list): A list of counts.
    
    Returns:
    list: A list of cumulative sums.
    """
    return [sum(file_counts[:i+1]) for i in range(len(file_counts))]

def main():
    """
    Main function to run the Streamlit app for measuring saturation.
    """
    st.header(":orange[Measure saturation]")

    # Display guide in an expandable section
    with st.expander("Guide to ITS"):
        
        st.subheader(":orange[Introduction to Thematic Analysis and Saturation]")

        st.write("""
        Thematic Analysis (TA) is a method used in qualitative research to identify and analyze patterns or themes within data. When conducting TA, researchers often seek to ensure that their analysis is comprehensive and captures the breadth of themes present in the data. One way to measure the completeness of this analysis is by assessing **saturation**. Saturation is a concept indicating that further analysis would yield little to no new information or themes, suggesting that the researcher has fully explored the data.
        """)

        st.subheader(":orange[What is the Saturation Metric?]")

        st.info("""
        In the context of this paper, saturation is quantitatively assessed using a metric called **Initial Thematic Saturation (ITS)**. ITS focuses on the emergence of new codes during the analysis. The metric is particularly important when using Large Language Models (LLMs) for TA, as it helps determine if the model's analysis is thorough.
        """)

        st.subheader(":orange[Understanding ITS: A Simplified Explanation]")

        st.write(":green[1. Themes and Codes]")
        st.write("""
        - In TA, **codes** are labels given to specific pieces of data that represent a concept or idea. Multiple codes can combine into a **theme**, which is a broader pattern identified in the data.
        - When analyzing data, the goal is to identify these themes by coding the data. Initially, many new codes are created, but as the analysis continues, the occurrence of new codes should decrease if the data is being thoroughly analyzed—this decrease indicates saturation.
        """)

        st.write(":green[2. How ITS is Measured]")
        st.write("""
        - **Cumulative Total Codes**: As you analyze more data (e.g., more interviews), you continue to add to a list of all codes identified—this is your cumulative total.
        - **Cumulative Unique Codes**: Simultaneously, you track how many of these codes are unique (i.e., not duplicates). As analysis continues, fewer new unique codes should appear, indicating that the analysis is approaching saturation.
        - **Saturation Indicator**: The saturation metric is calculated as the ratio between the rate at which new unique codes are identified and the total number of codes. As you progress, this ratio decreases, indicating that you are identifying fewer new ideas—the data is becoming saturated.
        """)

        st.write(":green[3. Interpreting ITS]")
        st.write("""
        - A higher ITS ratio indicates less saturation (many new codes are still emerging), while a lower ITS ratio indicates more saturation (few new codes are emerging). 
        """)

        st.subheader(":orange[Why ITS is Important]")

        st.warning("""
        For naive users, understanding ITS is crucial because it provides a measure of how well an analysis captures the richness of the data. If the ITS metric shows that saturation has been reached, it suggests that the themes identified are likely to be robust and comprehensive. Conversely, if saturation has not been reached, it may indicate that more data or analysis is needed.
        """)

        st.subheader(":orange[Conclusion]")

        st.success("""
        In summary, ITS is a valuable tool for ensuring the validity of thematic analysis, particularly when using advanced tools like LLMs. By tracking the emergence of new codes, researchers can quantitatively assess whether their analysis is complete and whether they have thoroughly explored the themes within their data.
        """)

        st.info("See our paper on saturation and LLMs (https://arxiv.org/pdf/2401.03239) for more information.")

    
    st.write("See our paper on saturation and LLMs (https://arxiv.org/pdf/2401.03239) for more information.")
    
    st.subheader("This version uses the files from initial coding (to derive total codes) and the files from the reduction of codes (unique codes) stored in your project folder.")

    # Project selection
    projects = get_projects()
    
    # Initialize session state for selected project if it doesn't exist
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = "Select a project..."

    # Calculate the index for the selectbox
    project_options = ["Select a project..."] + projects
    index = project_options.index(st.session_state.selected_project) if st.session_state.selected_project in project_options else 0

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

    if selected_project != "Select a project...":
        results_file = os.path.join(PROJECTS_DIR, selected_project, 'code_reduction_results.csv')

        if os.path.exists(results_file):
            with st.spinner("Processing..."):
                # Read the results file
                results_df = pd.read_csv(results_file)
                
                # Extract unique and total code counts
                unique_counts = results_df['unique_codes'].tolist()
                total_counts = results_df['total_codes'].tolist()

                st.success("Files processed successfully!")

                # Calculate ITS Metric (Saturation)
                its_metric = round(unique_counts[-1] / total_counts[-1], 3)
                
                # Display ITS Metric
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(":orange[ITS Metric (Saturation):]")
                with col2:
                    st.subheader(f":green[{its_metric}]")

                # Create plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(1, len(unique_counts) + 1)), y=unique_counts, mode='lines+markers', name='Unique Codes', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=list(range(1, len(total_counts) + 1)), y=total_counts, mode='lines+markers', name='Total Initial Codes'))
                fig.update_layout(
                    title='Unique Codes vs Total Codes Cumulative Sum',
                    xaxis_title='File Index',
                    yaxis_title='Count',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig)

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

    # Manage API keys
    manage_api_keys()

if __name__ == "__main__":
    main()