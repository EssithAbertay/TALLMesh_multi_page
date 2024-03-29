# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 08:14:47 2024

@author: Laptop
"""
import os
import streamlit as st

def create_folders(folder_path, folder_name):
    # Create the main folder
    main_folder = os.path.join(folder_path, folder_name)
    os.makedirs(main_folder, exist_ok=True)
    
    # Create subfolders
    subfolders = ['data', 'initial_codes', 'reduced_codes', 'themes']
    for folder in subfolders:
        os.makedirs(os.path.join(main_folder, folder), exist_ok=True)
    
    # Save folder path and name to a text file
    with open(os.path.join(main_folder, 'folder_info.txt'), 'w') as f:
        f.write(f"Folder Path: {folder_path}\n")
        f.write(f"Folder Name: {folder_name}\n")

def save_uploaded_files(uploaded_files, folder_path, folder_name):
    data_folder = os.path.join(folder_path, folder_name, 'data')
    for file in uploaded_files:
        with open(os.path.join(data_folder, file.name), "wb") as f:
            f.write(file.getbuffer())

def main():
    st.header(":orange[Project Folder Creation & File Upload App]")
    st.write("This allows you to create a project folder structure with specified subfolders and upload .txt files such as interview.")
    st.write("The folder structure is fixed for all projects. :green[If you have created one previously, just proceed with the other steps of the analysis.]")
    
    # User input for folder path and name
    folder_path = st.text_input("Enter the project folder path:", "/path/to/parent/folder")
    folder_name = st.text_input("Enter the folder name:")
    
    uploaded_files = st.file_uploader("Upload interviews .txt files", accept_multiple_files=True)
    
    if st.button("Create Folder Structure & Save Files"):
        if folder_path.strip() == "" or folder_name.strip() == "":
            st.error("Please enter both folder path and name.")
        else:
            try:
                create_folders(folder_path, folder_name)
                if uploaded_files:
                    save_uploaded_files(uploaded_files, folder_path, folder_name)
                st.success("Project folder structure created successfully and files saved! - You can proceed with your analysis")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
