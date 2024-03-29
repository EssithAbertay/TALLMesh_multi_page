# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:31:06 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""
import streamlit as st
import pandas as pd
from itertools import combinations
from graphviz import Digraph

# Function to calculate the number of common codes between two topics
def common_codes(topic1, topic2):
    return len(set(topic1) & set(topic2))

# Function to convert the data into a graph representation
def create_graph(data, min_common_codes):
    # Extracting unique topic names
    topics = data['theme'].unique()

    # Initialize empty lists for edges
    edges = []

    # Create edges
    for pair in combinations(topics, 2):
        topic1_codes = data[data['theme'] == pair[0]]['codes'].iloc[0].split(',')
        topic2_codes = data[data['theme'] == pair[1]]['codes'].iloc[0].split(',')
        topic1 = pair[0]
        topic2 = pair[1]
        common = common_codes(topic1_codes, topic2_codes)
        if common >= min_common_codes:
            edges.append((topic1, topic2, common))

    # Construct the string in the desired format
    output_dot = "digraph {\n"
    for edge in edges:
        output_dot += f"\t\"{edge[0]}\" -> \"{edge[1]}\" [label=\"Common Codes {edge[2]}\" penwidth={edge[2]}]\n [dir=none] "

    output_dot += "}"
    
    return output_dot

# Main function
def main():
    st.header(":orange[Thematic Map based on shared codes]")
    st.subheader("Input should be the CSV from phase 3 - Finding Themes")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read CSV
        data = pd.read_csv(uploaded_file)

        # Set minimum common codes threshold
        min_common_codes = st.slider("Minimum Common Codes", min_value=1, max_value=10, value=1)

        # Render graph using Graphviz
        graph_dot = create_graph(data, min_common_codes)
             
        st.graphviz_chart(graph_dot)
        

# Run the app
if __name__ == "__main__":
    main()
