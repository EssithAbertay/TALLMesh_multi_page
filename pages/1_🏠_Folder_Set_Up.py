import os
import streamlit as st
import json
from api_key_management import manage_api_keys
import shutil
from project_utils import get_projects

PROJECTS_DIR = 'projects'

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
    
    subfolders = ['data', 'initial_codes', 'reduced_codes', 'themes', 'refined_themes', 'theme_books', 'expanded_reduced_codes']
    for folder in subfolders:
        os.makedirs(os.path.join(project_path, folder), exist_ok=True)

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

def remove_project(project_name):
    project_path = os.path.join(PROJECTS_DIR, project_name)
    if os.path.exists(project_path):
        try:
            shutil.rmtree(project_path)
            st.session_state.message = f"Project '{project_name}' has been successfully removed."
            st.session_state.message_type = "success"
            # Update the projects list in session state
            st.session_state.projects = get_projects()
            st.session_state.selected_project = None
            st.session_state.delete_project = None  # Reset the delete_project flag
        except Exception as e:
            st.session_state.message = f"Error removing project '{project_name}': {str(e)}"
            st.session_state.message_type = "error"
    else:
        st.session_state.message = f"Project '{project_name}' does not exist."
        st.session_state.message_type = "warning"

# messages (St.info, success, toast, etc) wont persist through st.rerun() so need session_state vars to store success / fail messages
# Initialize session state variables
if 'message' not in st.session_state:
    st.session_state.message = None
    st.session_state.message_type = None

if 'projects' not in st.session_state:
    st.session_state.projects = get_projects()

if 'selected_project' not in st.session_state:
    st.session_state.selected_project = None

if 'delete_project' not in st.session_state:
    st.session_state.delete_project = None



def main():
    st.header(":orange[Project Set Up & File Management]")
    
    with st.expander("Instructions"):
        st.write("""
        The "Folder Set Up" page is where you'll begin your thematic analysis journey. This page allows you to create new projects, manage existing ones, and upload the files you want to analyze. Here's how to use it:
        """)

        st.subheader(":orange[1. Creating a New Project]")
        st.write("""
        - Enter a unique name for your project in the "Enter new project name:" text box.
        - Click the "Create Project" button to set up your project.
        - The system will create a new folder structure for your project, including subfolders for data, initial codes, reduced codes, themes, and more.
        """)

        st.subheader(":orange[2. Selecting an Existing Project]")
        st.write("""
        - Use the dropdown menu labeled "Select a project:" to choose from your existing projects.
        - Once selected, you'll see the project name displayed and have options to manage its files.
        """)

        st.subheader(":orange[3. Uploading Files]")
        st.write("""
        - With a project selected, you'll see a file uploader labeled "Upload interviews .txt files".
        - You can drag and drop multiple .txt files or click to browse and select them.
        - Click the "Upload Files" button to add these files to your project's data folder.
        - Successful uploads will be confirmed with a message.
        """)

        st.subheader(":orange[4. Managing Existing Files]")
        st.write("""
        - Below the file uploader, you'll see a list of all files currently in your project.
        - Each file has a checkbox next to it. Select the checkbox for any files you want to delete.
        - Click the trash can icon (🗑️) that appears to remove selected files from your project.
        """)

        st.subheader(":orange[5. Deleting a Project]")
        st.write("""
        - If you need to remove an entire project, select it from the dropdown and click the "Delete Project" button.
        - This action will remove all files and folders associated with the project, so use it carefully.
        """)

        st.subheader(":orange[6. API Key Management]")
        st.write("""
        - In the sidebar, you'll find options to manage your API keys for different AI providers.
        - You can add new keys, view existing ones (last few digits only for security), and delete keys as needed.
        """)


        st.subheader(":orange[Tips]")
        st.write("""
        - Always double-check your project selection before uploading or deleting files.
        - It's recommended to use descriptive names for your projects to easily identify them later.
        - If you're analyzing interviews, consider naming your files consistently (e.g., "1_Interview_topic.txt", "2_Interview_topic.txt").
        - Remember that deleting files or projects is permanent and cannot be undone.
        """)

        st.info(":bulb: By properly setting up your project and managing your files here, you'll have a solid foundation for the subsequent steps in your thematic analysis process.")

    
    st.write(":green[Select an existing project or create a new one to get started.]")
    
    # Update projects list at the start of each run
    st.session_state.projects = get_projects()

    # Initialize selected_project in session state if it doesn't exist
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = None

    # Check if a project needs to be deleted
    if 'delete_project' in st.session_state and st.session_state.delete_project:
        remove_project(st.session_state.delete_project)
        st.session_state.delete_project = None
        st.session_state.selected_project = None
        st.rerun()

    # Project creation
    new_project = st.text_input("Enter new project name:")
    if st.button("Create Project"):
        if new_project and new_project not in st.session_state.projects:
            create_project(new_project)
            st.session_state.message = f"Project '{new_project}' created successfully!"
            st.session_state.message_type = "success"
            st.session_state.projects = get_projects()
            st.session_state.selected_project = new_project
            st.rerun()
        else:
            st.session_state.message = "Invalid project name or project already exists."
            st.session_state.message_type = "error"

    # Project selection
    project_options = ["Select a project..."] + st.session_state.projects
    index = 0 if st.session_state.selected_project is None else project_options.index(st.session_state.selected_project)
    selected_project = st.selectbox("Select a project:", project_options, index=index, key="project_selector")
    
    if selected_project != "Select a project...":
        st.session_state.selected_project = selected_project
    else:
        st.session_state.selected_project = None

    if st.session_state.selected_project:
        col1, col2, col3 = st.columns([0.2,0.2,0.05])
        col1.subheader(f"Project: {st.session_state.selected_project}")
        if col2.button("Delete Project"):
            st.session_state.delete_project = st.session_state.selected_project
            st.rerun()
        delete_button = col3.empty()
        
        # Display existing files with checkboxes
        try:
            existing_files = get_project_files(st.session_state.selected_project)
        except:
            existing_files = []

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

    else:
        st.write("Please select or create a project to continue.")

    # Display message if it exists in session state
    if 'message' in st.session_state and st.session_state.message:
        if st.session_state.message_type == "success":
            st.toast(st.session_state.message, icon="😍")
        elif st.session_state.message_type == "info":
            st.toast(st.session_state.message, icon="⚠️")
        elif st.session_state.message_type == "warning":
            st.warning(st.session_state.message)
        elif st.session_state.message_type == "error":
            st.error(st.session_state.message)
        
        # Clear the message after displaying
        st.session_state.message = None
        st.session_state.message_type = None

    # Call API key saving function
    manage_api_keys()

if __name__ == "__main__":
    main()