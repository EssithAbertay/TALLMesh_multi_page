'''
Just some util functions for dealing with project files and directories
'''
import os 
PROJECTS_DIR = "projects"

def get_projects():
    if not os.path.exists(PROJECTS_DIR):
        os.makedirs(PROJECTS_DIR)
    return [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]

def get_project_files(project_name, folder):
    data_folder = os.path.join(PROJECTS_DIR, project_name, folder)
    return [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

def get_processed_files(project_name, folder):
    initial_codes_folder = os.path.join(PROJECTS_DIR, project_name, folder)
    return [f for f in os.listdir(initial_codes_folder)]