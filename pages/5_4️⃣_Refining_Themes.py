# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:16:46 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""
import streamlit as st
import openai
import pandas as pd
import ast
from api_key_management import manage_api_keys

client = openai

client = openai
def get_completion(prompt, model, temperature):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

def main():
    #st.sidebar.title("Select Model and Provide Your Key")

    model_options = ["gpt-3.5-turbo-16k", "text-davinci-002", "llama-2", "mistral"]
    selected_model = st.sidebar.selectbox("Select Model", model_options)

    #api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

    #if st.sidebar.button("Confirm Key and Model"):
    #    st.sidebar.success("API Key and Model Confirmed!")
    #    client.api_key = api_key


    manage_api_keys()    

    st.header(":orange[Refining Themes]")
    st.subheader("Changing the temperature will result in different creative for the model")

    uploaded_file = st.file_uploader("Upload a CSV file with columns: Code, Description, Quote", type=["csv"])

    # Temperature slider
    temperature = st.slider("Select Temperature", min_value=0.1, max_value=1.0, step=0.1, value=0.5)

    if uploaded_file:

        df = pd.read_csv(uploaded_file, names=['Code', 'Description', 'Quote'])  # rename columns just in case

        # add index column
        df = df.reset_index(drop=True)

        st.write("File Uploaded Successfully!")

        # Pre-populate prompt
        codes_list = [f"[{index}]: {row['Code']}: {row['Description']}" for index, row in df.iterrows()]

        st.write("If you wish you can check the codes that will be used")
        st.write(codes_list)

        prepopulated_prompt = f"Read first the list of initial codes of my Thematic Analysis: {{', '.join(codes_list)}}.\nInitial codes are in the following format: [index]: code_name. code_description.\n\nDetermine all the possible themes by sorting, comparing, and grouping initial codes.\nProvide a suitable number of themes along with a name, a dense description (70 words), and a list of codes (index) for each theme.\nEnsure the themes capture the richness and diversity of the initial codes.\n\nFormat the text below as a json dictionary, grouped in the main keys: 1) 'themes', which should include the theme, theme description, and list of codes (including their index)"
        st.header("Enter prompt:")
        prompt_input = st.text_area("Edit the prompt if needed:", value=prepopulated_prompt)

        if st.button("Process"):
            # Combine the file content and the entered prompt for processing
            prompt = f"{prompt_input}{{}}{{}}File Content:{{}}{{}}{uploaded_file.getvalue().decode('utf-8')}"

            with st.spinner("Processing..."):
                processed_output = get_completion(prompt, model=selected_model, temperature=temperature)

            try:
                # st.write(processed_output)  # debugging
                json_output = ast.literal_eval(processed_output)
                st.subheader("Processed JSON Output:")
                st.write("You can download your themes as CSV now =========>")

                # Save JSON as CSV
                df_themes = pd.json_normalize(json_output['themes'])
                st.write(df_themes)

            except (ValueError, SyntaxError) as e:
                st.warning(f"Unable to parse the output as JSON. Error: {e}")
                st.text("Processed Output:")
                st.text(processed_output)  # Print the processed output for debugging

if __name__ == "__main__":
    main()
