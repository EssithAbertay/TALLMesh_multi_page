import streamlit as st
from api_key_management import manage_api_keys, load_api_keys

def main():

    st.header(":orange[Azure Deployment Settings]")
    #st.subheader(":orange[Project & Data Selection]")

    manage_api_keys()

if __name__ == "__main__":
    main()