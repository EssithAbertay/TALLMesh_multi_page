# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:29:16 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""
import streamlit as st
import pandas as pd

def load_data(themes_file, codes_file):
    # Load the first DataFrame
    themes_df = pd.read_csv(themes_file)

    # Load the second DataFrame from CSV
    codes_df = pd.read_csv(codes_file)

    #nr of columns in codes_df
    num_columns = codes_df.shape[1]

    # Generate column labels, this is needed for multiple quotes in particular
    column_labels = ['code_index', 'code', 'description'] + [f'quote_{i}' for i in range(1, num_columns - 2)]

    codes_df.columns = column_labels

    return themes_df, codes_df, column_labels


def process_data(themes_df, codes_df, column_labels):
    #nr of themes length to iterate over set of codes
    l=len(themes_df.codes)

    # Initialize empty lists to store theme, description, and codes
    theme_list = []
    description_list = []
    codes_list = []

    # Declare an empty DataFrame
    final_df = pd.DataFrame(columns=column_labels)

    #iterate over each list of codes per theme
    for i in range(l):
        #read each batch of codes
        indexes = themes_df.codes[i]
        #read theme name & description
        theme = themes_df.theme[i]
        description = themes_df.description[i]

        #split each code index
        numbers_list = [int(num) for num in indexes.split(',')]

        for re in numbers_list:
            row_data = codes_df.iloc[re]

            elements_to_add = pd.Series([theme, description], index=['Theme', 'Description'])

            # Concatenate the two series
            new_series = pd.concat([elements_to_add, row_data])

            # Appending the list as a new row for final theme-codes book
            final_df = pd.concat([final_df, new_series.to_frame().T], ignore_index=True)

    return final_df


def main():
    st.header(":orange[Theme-Codes book rebuild]")

    # File uploaders
    st.write("Upload Themes CSV")
    themes_file = st.file_uploader("File1", type=["csv"])
    st.write("Upload Codes CSV")
    codes_file = st.file_uploader("File2", type=["csv"])

    if themes_file is not None and codes_file is not None:
        # Load data
        themes_df, codes_df, column_labels = load_data(themes_file, codes_file)
        
        # Process button
        if st.button("Process"):
           # Process data
           final_df = process_data(themes_df, codes_df, column_labels)
           
           # Get list of all column names except 'Theme' and 'Description'
           other_columns = [col for col in final_df.columns if col not in ['Theme', 'Description']]

           # Reorder columns
           final_df = final_df[['Theme', 'Description'] + other_columns]

           # Display the final DataFrame
           st.write(final_df)
        

        

if __name__ == "__main__":
    main()

