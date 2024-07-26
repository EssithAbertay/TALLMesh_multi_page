import os
import streamlit as st
import json
from api_key_management import manage_api_keys

PROJECTS_DIR = 'projects'

# messages (St.info, success, toast, etc) wont persist through st.rerun() so need session_state vars to store success / fail messages
if 'message' not in st.session_state:
    st.session_state.message = None
    st.session_state.message_type = None

def handle_file_upload(uploaded_files, project_name):
    if uploaded_files:
        saved_files = save_uploaded_files(uploaded_files, project_name)
        if saved_files:
            st.session_state.message = f"Files uploaded successfully: {', '.join(saved_files)}"
            st.session_state.message_type = "success"
        else:
            st.session_state.message = "No new files were uploaded. They may already exist in the project."
            st.session_state.message_type = "info"
    else:
        st.session_state.message = "Please select files to upload."
        st.session_state.message_type = "warning"


def create_project(project_name):
    project_path = os.path.join(PROJECTS_DIR, project_name)
    os.makedirs(project_path, exist_ok=True)
    
    subfolders = ['data', 'initial_codes', 'reduced_codes', 'themes']
    for folder in subfolders:
        os.makedirs(os.path.join(project_path, folder), exist_ok=True)

def get_projects():
    if not os.path.exists(PROJECTS_DIR):
        os.makedirs(PROJECTS_DIR)
    return [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]

def save_uploaded_files(uploaded_files, project_name):
    data_folder = os.path.join(PROJECTS_DIR, project_name, 'data')
    saved_files = []
    for file in uploaded_files:
        file_path = os.path.join(data_folder, file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            saved_files.append(file.name)
    return saved_files

def get_project_files(project_name):
    data_folder = os.path.join(PROJECTS_DIR, project_name, 'data')
    return [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

def remove_files(project_name, filenames):
    for filename in filenames:
        file_path = os.path.join(PROJECTS_DIR, project_name, 'data', filename)
        if os.path.exists(file_path):
            os.remove(file_path)

def main():
    st.header(":orange[Project Set Up & File Management]")
    st.write("Welcome! This page allows you to:")
    st.write("1. Create new projects or select existing ones.")
    st.write("2. Upload files to your selected project.")
    st.write("3. Manage your project files.")
    st.write(":green[Select an existing project or create a new one to get started.]")
    
    if 'projects' not in st.session_state:
        st.session_state.projects = get_projects()
    
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = None

    # Project creation
    new_project = st.text_input("Enter new project name:")
    if st.button("Create Project"):
        if new_project and new_project not in st.session_state.projects:
            create_project(new_project)
            st.success(f"Project '{new_project}' created successfully!")
            st.session_state.projects.append(new_project)
            st.session_state.selected_project = new_project
            st.rerun()
        else:
            st.error("Invalid project name or project already exists.")

    # Project selection
    project_options = ["Select a project..."] + st.session_state.projects
    index = 0 if st.session_state.selected_project is None else project_options.index(st.session_state.selected_project)
    selected_project = st.selectbox("Select a project:", project_options, key='project_selector', index=index)
    
    if selected_project != "Select a project...":
        st.session_state.selected_project = selected_project
    else:
        st.session_state.selected_project = None

    if st.session_state.selected_project:
        col1, col2 = st.columns([0.95,0.05])
        col1.subheader(f"Project: {st.session_state.selected_project}")
        delete_button = col2.empty()
        
        # Display existing files with checkboxes
        existing_files = get_project_files(st.session_state.selected_project)
        if existing_files:
            files_to_delete = []
            
            # List files with checkboxes
            for file in existing_files:
                file_col, checkbox_col = st.columns([0.95, 0.05])
                file_col.write(file)
                if checkbox_col.checkbox(".", key=f"checkbox_{file}", label_visibility="hidden"):
                    files_to_delete.append(file)
            
            # Show delete button if files are selected
            if files_to_delete:
                if delete_button.button("🗑️"):
                    remove_files(st.session_state.selected_project, files_to_delete)
                    st.success(f"Deleted {len(files_to_delete)} file(s)")
                    st.rerun()
        else:
            st.write("No files in this project yet. Upload files below to get started")
        
        # File upload using form
        with st.form("upload_form", clear_on_submit=True, border=False):
            uploaded_files = st.file_uploader("Upload interviews .txt files", accept_multiple_files=True, label_visibility="hidden")
            submitted = st.form_submit_button("Upload Files")

            if submitted:
                handle_file_upload(uploaded_files, st.session_state.selected_project)
                st.rerun()

        # Display message if it exists in session state
        if st.session_state.message:
            if st.session_state.message_type == "success":
                st.toast(st.session_state.message, icon="😍")
            elif st.session_state.message_type == "info":
                st.toast(st.session_state.message, icon="⚠️")
            elif st.session_state.message_type == "warning":
                st.warning(st.session_state.message)
            
            # Clear the message after displaying
            st.session_state.message = None
            st.session_state.message_type = None

    else:
        st.write("Please select or create a project to continue.")

    # Call API key saving function
    manage_api_keys()

if __name__ == "__main__":
    main()