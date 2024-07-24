# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 08:14:47 2024

@author: Laptop
"""
import os
import streamlit as st
import json

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

# Replace setting API key on every page with single setting in set up page

# File to store API keys
API_KEYS_FILE = 'api_keys.json' # currently git ignored

# List of LLM providers
providers = ['OpenAI','Anthropic']

# Function to load api keys from json file; for deployment, keys should be in db
def load_api_keys():
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_api_keys(api_keys):
    with open(API_KEYS_FILE, 'w') as f:
        json.dump(api_keys, f)

def manage_api_keys():
    st.sidebar.header("API Key Management")

    # Load existing API keys
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = load_api_keys()

    # Add new API key
    new_provider = st.sidebar.selectbox("Provider", providers)
    new_key = st.sidebar.text_input("API Key", type="password")
    if st.sidebar.button("Add API Key"):
        if new_provider and new_key:
            st.session_state.api_keys[new_provider] = new_key
            save_api_keys(st.session_state.api_keys)
            st.sidebar.success(f"API Key for {new_provider} added successfully!")
        else:
            st.sidebar.error("Please enter both provider and key.")

    # Display and manage existing API keys
    st.sidebar.subheader("Saved API Keys")
    for provider, key in st.session_state.api_keys.items():
        col1, col2 = st.sidebar.columns([3, 1])
        masked_key = '*' * (len(key) - 3) + key[-3:]  # show the last n digits of the API key
        col1.text(f"{provider}: {masked_key[-7:]}") # show the last n digits of the masked key (saves users scrolling right to check last digits)
        if col2.button("❌", key=f"delete_{provider}"):
            del st.session_state.api_keys[provider]
            save_api_keys(st.session_state.api_keys)
            st.sidebar.success(f"API Key for {provider} deleted.")
            st.rerun()

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

    # Call API key saving function
    manage_api_keys()

if __name__ == "__main__":
    main()
