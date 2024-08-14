import streamlit as st
import os
import chardet
from project_utils import get_projects

PROJECTS_DIR = 'projects'

def detect_encoding(file_content):
    result = chardet.detect(file_content)
    return result['encoding']

def convert_to_txt(file, project_name):
    file_content = file.read()
    encoding = detect_encoding(file_content)
    
    # Use 'utf-8' as default if encoding detection fails
    if encoding is None:
        encoding = 'utf-8'
    
    try:
        # Attempt to decode the content using the detected encoding
        decoded_content = file_content.decode(encoding)
    except UnicodeDecodeError:
        # If decoding fails, try with 'utf-8' as a fallback
        try:
            decoded_content = file_content.decode('utf-8')
        except UnicodeDecodeError:
            # If 'utf-8' also fails, use 'latin-1' which can decode any byte string
            decoded_content = file_content.decode('latin-1')
    
    # Create the file path
    file_name = os.path.splitext(file.name)[0] + '.txt'
    file_path = os.path.join(PROJECTS_DIR, project_name, 'data', file_name)
    
    # Write the content to a new .txt file with UTF-8 encoding
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(decoded_content)
    
    return file_name

def get_project_files(project_name):
    data_folder = os.path.join(PROJECTS_DIR, project_name, 'data')
    return [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

def main():
    st.header(":orange[File Upload and Conversion]")
    
    with st.expander("Instructions"):
        st.write("""
        This page allows you to upload files and convert them to .txt format with appropriate encoding. Here's how to use it:
        
        1. Select your project from the dropdown menu.
        2. Upload your files using the file uploader.
        3. The app will automatically convert the files to .txt format with UTF-8 encoding.
        4. You can view the list of converted files below the uploader.
        
        Note: This tool supports various file formats and automatically detects the original encoding to ensure accurate conversion.
        """)
    
    # Project selection
    projects = get_projects()
    selected_project = st.selectbox("Select a project:", ["Select a project..."] + projects)
    
    if selected_project != "Select a project...":
        st.write(f"Selected project: {selected_project}")
        
        # File upload
        uploaded_files = st.file_uploader("Upload files to convert to .txt", accept_multiple_files=True)
        
        if uploaded_files:
            st.write("Converting files...")
            converted_files = []
            for file in uploaded_files:
                converted_file = convert_to_txt(file, selected_project)
                converted_files.append(converted_file)
            
            st.success(f"Successfully converted {len(converted_files)} file(s) to .txt format.")
        
        # Display existing files
        st.subheader("Existing files in the project:")
        existing_files = get_project_files(selected_project)
        if existing_files:
            for file in existing_files:
                st.write(file)
        else:
            st.write("No files in this project yet.")
    else:
        st.write("Please select a project to continue.")

if __name__ == "__main__":
    main()