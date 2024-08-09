# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:29:16 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""
import streamlit as st
import pandas as pd
import os
from api_key_management import manage_api_keys, load_api_keys
from project_utils import get_projects, get_project_files, get_processed_files, PROJECTS_DIR
from openai import OpenAI, AzureOpenAI
import json


def format_quotes(quotes_json):
    """
    Parses the JSON string of quotes, extracts the text,
    and joins each quote with a newline for better readability.
    """
    try:
        quotes = json.loads(quotes_json)
        formatted_quotes = "\n".join(quote['text'] for quote in quotes)
        return formatted_quotes
    except (json.JSONDecodeError, KeyError, TypeError):
        return quotes_json  # Return the original if there's an error

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
    #num_columns = codes_df.shape[1]
    #column_labels = ['code', 'description', 'merge_explanation'] + [f'quote_{i}' for i in range(1, num_columns - 2)]
    #codes_df.columns = column_labels

    return themes_df, codes_df #, column_labels

def process_data(themes_df, codes_df):
    # Print column names for debugging
    #print("Themes DataFrame columns:", themes_df.columns)
    #print("Codes DataFrame columns:", codes_df.columns)

    # Initialize empty DataFrame for final theme-codes book with correct column names
    final_df = pd.DataFrame(columns=['Theme', 'Theme Description', 'Code', 'Code Description', 'Merge Explanation', 'Quotes', 'Source'])

    for _, theme_row in themes_df.iterrows():
        theme = theme_row['name']
        theme_description = theme_row['description']
        code_indices = [int(idx) for idx in theme_row['codes'].strip('[]').split(',')]
        
        for idx in code_indices:
            if idx < len(codes_df):
                code_row = codes_df.iloc[idx]
                new_row = pd.Series({
                    'Theme': theme,
                    'Theme Description': theme_description,
                    'Code': code_row['code'],
                    'Code Description': code_row['description'],
                    'Merge Explanation': code_row['merge_explanation'],
                    'Quotes': code_row.get('quote', code_row.get('source', '')),  # Try 'quote', then 'source' if 'quote' doesn't exist
                    'Source': code_row.get('source', code_row.get('quote_2', ''))  # Try 'source', then 'quote_2' if 'source' doesn't exist
                })
                final_df = pd.concat([final_df, new_row.to_frame().T], ignore_index=True)

    return final_df

def main():
    st.header(":orange[Theme-Codes book rebuild]")

    with st.expander("Instructions"):
        st.write("""
        The Finalised Theme Book page is where you compile and organize all your themes, codes, and associated data into a comprehensive structure. This step provides a clear overview of your entire analysis. Here's how to use this page:
        """)

        st.subheader(":orange[1. Project Selection]")
        st.write("""
        - Select your project from the dropdown menu.
        - The system will automatically load the most recent themes and reduced codes files for your project.
        """)

        st.subheader(":orange[2. Data Processing]")
        st.write("""
        - Once you've selected a project, the system will automatically process the data to create your theme book.
        - This process combines your themes with their associated codes, descriptions, and quotes.
        - :orange[No additional input is required] - the theme book is generated based on your previous work in the earlier stages.
        """)

        st.subheader(":orange[3. Viewing Results]")
        st.write("""
        - The results are presented in three main sections:
        1. :orange[Condensed Themes:] A concise view of your themes without the associated codes and quotes.
        2. :orange[Expanded Themes:] A detailed view including themes, codes, quotes, and sources.
        3. :orange[Merged Codes:] A reference section showing all your reduced codes.
        - Each section is displayed in a table format for easy reading and comparison.
        """)

        st.subheader(":orange[4. Saving and Downloading]")
        st.write("""
        - The system automatically saves two versions of your theme book:
        1. A condensed version with just the themes.
        2. An expanded version with all details including codes and quotes.
        - You can download the final theme book as a CSV file using the provided download button.
        """)

        st.subheader("Key Features")
        st.write("""
        - :orange[Automatic compilation:] The system pulls together all your work from previous stages into a coherent structure.
        - :orange[Multiple views:] You can see your analysis at different levels of detail, from high-level themes to specific quotes.
        - :orange[Traceability:] The expanded view allows you to trace each theme back to its constituent codes and original data sources.
        - :orange[Easy export:] You can easily save and share your final analysis as a CSV file.
        """)

        st.subheader("Tips")
        st.write("""
        - Take time to review the expanded themes carefully. This is your opportunity to see how everything fits together.
        - Use the condensed view for a quick overview, and the expanded view when you need to dive into the details.
        - :orange[Consider how your themes relate to each other.] Are there any overarching patterns or relationships between themes?
        - If you notice any inconsistencies or areas that need refinement, you can go back to earlier stages of the analysis and make adjustments.
        - The final theme book is an excellent resource for writing up your findings or preparing presentations about your analysis.
        """)

        st.info("The Finalised Theme Book represents the culmination of your thematic analysis. It provides a structured overview of your themes and their grounding in the data, which is crucial for ensuring the validity and reliability of your qualitative research. The following pages make use of these finalised themes for metrics and visualisations")


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

    if selected_project != "Select a project...":
        themes_df, codes_df  = load_data(selected_project)
        
        if themes_df is None or codes_df is None:
            st.error("Error: Required files not found in the project directory.")
        else:
            st.success(f"Files loaded successfully for project: {selected_project}")
            
            #if st.button("Process"): # redudant, just generate the theme book & other info
            # Process data
            final_df = process_data(themes_df, codes_df)
            
            #Print themes, ondensed code view, everything together
            st.write("Condensed Themes")
            st.write(themes_df)
            st.write("Expanded Themes w/ Codes, Quotes & Sources")
            final_display_df = final_df.copy()
            final_display_df['Quotes'] = final_display_df['Quotes'].apply(format_quotes)
            st.write(final_df)
            with st.expander("Merged Codes (for reference)"):
                st.write("Merged Codes")
                st.write(codes_df)
            
            # Save the final DataFrame
            output_folder = os.path.join(PROJECTS_DIR, selected_project, 'theme_books')
            os.makedirs(output_folder, exist_ok=True)

            output_file_condensed = os.path.join(output_folder, f"{selected_project}_condensed_theme_book_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
            themes_df.to_csv(output_file_condensed, index=False)

            output_file_expanded = os.path.join(output_folder, f"{selected_project}_expanded_theme_book_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
            final_df.to_csv(output_file_expanded, index=False)

            st.success(f"Theme books (condensed and expanded) saved to: \n-{output_file_condensed} \n{output_file_expanded}")
            
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

