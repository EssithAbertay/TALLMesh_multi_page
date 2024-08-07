import streamlit as st
from api_key_management import manage_api_keys, load_api_keys
import json
import os


AZURE_SETTINGS_FILE = 'azure_settings.json'

def load_azure_settings():
    if os.path.exists(AZURE_SETTINGS_FILE):
        with open(AZURE_SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_azure_settings(settings):
    with open(AZURE_SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)

def get_azure_models():
    azure_settings = load_azure_settings()
    return [f"azure_{deployment}" for deployment in azure_settings.get('deployments', [])]

def azure_settings():
    st.header("Azure API Settings")

    with st.expander("Instructions"):
        st.header("Setting Up Azure OpenAI")

        st.write("""
        To use Azure OpenAI with our application, you'll need to configure your Azure settings. Follow these steps:

        1. Enter your :orange[Azure API Key] and :orange[Azure Endpoint]. These can be found in your Azure portal.

        2. Click 'Save Azure Settings' to store your credentials securely.

        3. To add an Azure deployment:
        - Enter the :orange[Deployment Name] in the provided field.
        - Click 'Add Deployment' to save it.

        4. You can add multiple deployments if needed.

        5. To remove a deployment, click the 'Delete' button next to its name.

        Once set up, your Azure deployments will appear as options in the model selection dropdowns throughout the application, prefixed with :orange['azure_'].

        :warning: Remember to keep your API key confidential and never share it publicly.
        """)

        st.info("""
        **Note:** The Deployment Name is the identifier you chose when deploying a model in Azure. It's not the same as the model name (e.g., GPT-4). Make sure to use the correct Deployment Name for each of your Azure OpenAI deployments.
        """)

        st.success("""
        With your Azure settings configured, you're ready to use Azure OpenAI models in your analysis!
        """)


    azure_settings = load_azure_settings()

    api_key = st.text_input("Azure API Key", value=azure_settings.get('api_key', ''), type="password")
    endpoint = st.text_input("Azure Endpoint", value=azure_settings.get('endpoint', ''))

    if st.button("Save Azure Settings", key="save_azure_settings"):
        if api_key and endpoint:
            azure_settings['api_key'] = api_key
            azure_settings['endpoint'] = endpoint
            save_azure_settings(azure_settings)
            st.success("Azure settings saved successfully!")
        else:
            st.error("Please fill in both API Key and Endpoint.")

    st.divider()

    st.subheader("Manage Azure Deployments")

    if 'deployments' not in azure_settings:
        azure_settings['deployments'] = []

    deployment_name = st.text_input("Deployment Name")

    if st.button("Add Deployment", key="add_deployment"):
        if deployment_name:
            if deployment_name not in azure_settings['deployments']:
                azure_settings['deployments'].append(deployment_name)
                save_azure_settings(azure_settings)
                st.success(f"Deployment '{deployment_name}' added successfully!")
            else:
                st.error(f"Deployment '{deployment_name}' already exists.")
        else:
            st.error("Please enter a Deployment Name.")

    if azure_settings['deployments']:
        st.subheader("Current Azure Deployments")
        for i, deployment in enumerate(azure_settings['deployments']):
            col1, col2 = st.columns([3, 1])
            col1.text(f"Deployment: {deployment}")
            if col2.button("Delete", key=f"delete_deployment_{i}"):
                azure_settings['deployments'].remove(deployment)
                save_azure_settings(azure_settings)
                st.success(f"Deployment '{deployment}' deleted.")
                st.rerun()

    st.divider()
    st.subheader("Current Azure Settings")
    if azure_settings.get('api_key'):
        st.text(f"API Key: {'*' * len(azure_settings['api_key'])}")
    if azure_settings.get('endpoint'):
        st.text(f"Endpoint: {azure_settings['endpoint']}")

if __name__ == "__main__":
    azure_settings()


    

    
