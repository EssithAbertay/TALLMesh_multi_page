# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:29:16 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""
import streamlit as st
import pandas as pd
import os
from api_key_management import manage_api_keys, load_api_keys
from project_utils import get_projects, get_project_files, get_processed_files

PROJECTS_DIR = 'projects'

def load_data(project_name):
    themes_folder = os.path.join(PROJECTS_DIR, project_name, 'themes')
    codes_folder = os.path.join(PROJECTS_DIR, project_name, 'reduced_codes')
    
    # Get the most recent themes file
    themes_files = get_processed_files(project_name, 'themes')
    if not themes_files:
        return None, None, None
    latest_themes_file = max(themes_files, key=lambda f: os.path.getmtime(os.path.join(themes_folder, f)))
    themes_df = pd.read_csv(os.path.join(themes_folder, latest_themes_file))
    
    # Get the most recent reduced codes file
    codes_files = get_processed_files(project_name, 'reduced_codes')
    if not codes_files:
        return None, None, None
    latest_codes_file = max(codes_files, key=lambda f: os.path.getmtime(os.path.join(codes_folder, f)))
    codes_df = pd.read_csv(os.path.join(codes_folder, latest_codes_file))
    
    # Generate column labels for codes_df
    num_columns = codes_df.shape[1]
    column_labels = ['code', 'description', 'merge_explanation'] + [f'quote_{i}' for i in range(1, num_columns - 2)]
    codes_df.columns = column_labels

    return themes_df, codes_df, column_labels

def process_data(themes_df, codes_df, column_labels):
    # Initialize empty DataFrame for final theme-codes book
    final_df = pd.DataFrame(columns=['Theme', 'Description'] + column_labels)

    for _, theme_row in themes_df.iterrows():
        theme = theme_row['name']
        description = theme_row['description']
        code_indices = [int(idx) for idx in theme_row['codes'].strip('[]').split(',')]
        
        for idx in code_indices:
            if idx < len(codes_df):
                row_data = codes_df.iloc[idx]
                new_row = pd.Series([theme, description] + list(row_data), index=final_df.columns)
                final_df = pd.concat([final_df, new_row.to_frame().T], ignore_index=True)

    return final_df

def main():
    st.header(":orange[Theme-Codes book rebuild]")

    projects = get_projects()
    selected_project = st.selectbox("Select a project:", ["Select a project..."] + projects)

    if selected_project != "Select a project...":
        themes_df, codes_df, column_labels = load_data(selected_project)
        
        if themes_df is None or codes_df is None:
            st.error("Error: Required files not found in the project directory.")
        else:
            st.success(f"Files loaded successfully for project: {selected_project}")
            
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
                
                # Save the final DataFrame
                output_folder = os.path.join(PROJECTS_DIR, selected_project, 'theme_book')
                os.makedirs(output_folder, exist_ok=True)
                output_file = os.path.join(output_folder, f"final_theme_book_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
                final_df.to_csv(output_file, index=False)
                st.success(f"Final Theme-Codes book saved to: {output_file}")
                
                # Download button
                csv = final_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Final Theme-Codes Book",
                    data=csv,
                    file_name="final_theme_codes_book.csv",
                    mime="text/csv"
                )
    else:
        st.write("Please select a project to continue.")

    manage_api_keys()

if __name__ == "__main__":
    main()

