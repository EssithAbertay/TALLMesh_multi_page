import os
import streamlit as st
import pandas as pd
import plotly.express as px
from project_utils import get_projects, PROJECTS_DIR, get_processed_files
from api_key_management import manage_api_keys, load_api_keys

# Constants
THEME_BOOKS_FOLDER = 'theme_books'
EXPANDED_REDUCED_CODES_FOLDER = 'expanded_reduced_codes'

def load_data(project_name):
    """Load the latest theme and code data for a given project."""
    themes_folder = os.path.join(PROJECTS_DIR, project_name, THEME_BOOKS_FOLDER)
    codes_folder = os.path.join(PROJECTS_DIR, project_name, EXPANDED_REDUCED_CODES_FOLDER)
    
    themes_files = get_processed_files(project_name, THEME_BOOKS_FOLDER)
    expanded_themes_files = [f for f in themes_files if 'expanded' in f]
    if not expanded_themes_files:
        return None, None
    latest_themes_file = max(expanded_themes_files, key=lambda f: os.path.getmtime(os.path.join(themes_folder, f)))
    themes_df = pd.read_csv(os.path.join(themes_folder, latest_themes_file))
    
    codes_files = get_processed_files(project_name, EXPANDED_REDUCED_CODES_FOLDER)
    if not codes_files:
        return None, None
    latest_codes_file = max(codes_files, key=lambda f: os.path.getmtime(os.path.join(codes_folder, f)))
    codes_df = pd.read_csv(os.path.join(codes_folder, latest_codes_file))

    return themes_df, codes_df 

def prepare_icicle_data(themes_df, codes_df):
    """Prepare the data for the icicle plot by combining theme and code information."""
    icicle_data = []
    
    for _, theme_row in themes_df.iterrows():
        theme = theme_row['Theme']
        theme_description = theme_row['Theme Description']
        reduced_code = theme_row['Code']
        reduced_code_description = theme_row['Code Description']
        
        relevant_codes = codes_df[codes_df['code'] == reduced_code]
        
        for _, code_row in relevant_codes.iterrows():
            initial_code = code_row['original_code']
            quote = code_row['quote']
            source = code_row['source']
            
            icicle_data.append({
                'Theme': theme,
                'Theme Description': theme_description,
                'Reduced Code': reduced_code,
                'Reduced Code Description': reduced_code_description,
                'Initial Code': initial_code,
                'Quote': quote,
                #'Source': source, # remvoing the source allows for more quote text... 
                'Value': 1
            })
    
    df = pd.DataFrame(icicle_data)
    
    # Adjust values to make quotes wider and make theme level narrow
    df.loc[df['Quote'].notna(), 'Value'] = 3
    df.loc[df['Theme'].notna(), 'Value'] = 0.2
    
    return df

def create_icicle_plot(df_filtered, color_scheme, text_size):
    """Create an improved icicle plot with better accessibility."""
    fig = px.icicle(
        df_filtered, 
        path=['Theme', 'Reduced Code', 'Initial Code', 'Quote'],#, 'Source'], 
        values='Value',
        color_discrete_sequence=color_scheme,
        branchvalues='total',
        maxdepth=4,
        hover_data=['Theme Description', 'Reduced Code Description'],
    )

    fig.update_traces(
        textfont=dict(size=text_size, color="white"),
        textposition='middle center',
        hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Description: %{customdata[0]}<extra></extra>'
    )

    for i, trace in enumerate(fig.data):
        if i == 0:  # The first trace corresponds to the outermost level (Theme)
            trace.textposition = 'middle left'
            #trace.insidetextorientation = 'radial' # display text vertically

    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25), 
        height=800,
        font=dict(size=text_size),
    )

    return fig

def main():
    """Main function to run the Streamlit app for the Theme-Codes Icicle visualization."""
    st.header(":orange[Theme-Codes Icicle]")
    st.subheader(":orange[Structure: Theme > Reduced Codes > Initial Code(s) > Quote(s) > Source]")

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
        themes_df, codes_df = load_data(selected_project)

        if themes_df is None or codes_df is None:
            st.error("No data available for the selected project.")
            return

        icicle_data = prepare_icicle_data(themes_df, codes_df)

        unique_themes = icicle_data['Theme'].unique()
        selected_theme = st.selectbox('Select a Theme to Visualise', unique_themes)
    
        df_filtered = icicle_data[icicle_data['Theme'] == selected_theme]
    
        # Accessibility options
        #use_colorblind_scheme = st.checkbox("Use colour blind-friendly scheme")

        # Define a colorblind-friendly color scheme ... check these for WCAG2.1
        colorblind_safe_colors = [
            "#1170AA", "#FC7D0B", "#A3ACB9", "#57606C", "#5FA2CE", 
            "#C85200", "#7B848F", "#A3CCE9", "#FFBC79", "#C8D0D9"
        ]
        
        # Use the defined colours or default plotyl qualitative palette
        #color_scheme = colorblind_safe_colors if use_colorblind_scheme else px.colors.qualitative.Plotly

        color_scheme = px.colors.qualitative.Plotly
        
        text_size = st.slider("Adjust text size", min_value=14, max_value=26, value=20, step=1)
        
        fig = create_icicle_plot(df_filtered, color_scheme, text_size)
        st.write('You can click on the components to see them in more detail. Hover for descriptions.')
        st.plotly_chart(fig, use_container_width=True)

        # Option to display data as a table
        show_table = st.checkbox("Show data as table",help='Table view of the currently displayed theme')
        if show_table:
            st.dataframe(df_filtered)

    # API key management
    manage_api_keys()
    
if __name__ == "__main__":
    main()