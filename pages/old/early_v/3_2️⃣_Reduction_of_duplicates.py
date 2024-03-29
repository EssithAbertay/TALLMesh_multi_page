import streamlit as st
import pandas as pd
import ast
import openai
from openai import AzureOpenAI

client = openai
def get_completion(prompt, model):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

def main():
    st.sidebar.title("Select Model and Provide Your Key")

    model_options = ["gpt-3.5-turbo-16k", "azure-gpt35", "llama-2", "mistral"]
    selected_model = st.sidebar.selectbox("Select Model", model_options)

    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

    azure_endpoint = None
    azure_deployment = None
    if selected_model == "azure-gpt35":
        azure_endpoint = st.sidebar.text_input("Enter Azure Endpoint")
        azure_deployment = st.sidebar.text_input("Enter Azure Deployment")

    if st.sidebar.button("Confirm Key and Model"):
        st.sidebar.success("API Key and Model Confirmed!")
        

    st.header(":orange[Reduction of Duplicates]")
    st.subheader("Upload CSV Files:")
    form = st.form(key='my_form')
    
    # File uploader for the first CSV file
    uploaded_file_1 = form.file_uploader("Upload the first Total Codebook CSV file", type=["csv"], key="csv_uploader_1")

    # File uploader for the second CSV file
    uploaded_file_2 = form.file_uploader("Upload the codes of the interview we are checking for duplicates", type=["csv"], key="csv_uploader_2")

    submit_button = form.form_submit_button('Process')

    if uploaded_file_1 and uploaded_file_2 and submit_button:
        st.write("Files Uploaded:")
        st.write(f"1. {uploaded_file_1.name}")
        st.write(f"2. {uploaded_file_2.name}")

        # Read CSV files
        col_names=['index', 'code', 'description', 'quote']
        df1 = pd.read_csv(uploaded_file_1, names=col_names)
        df2 = pd.read_csv(uploaded_file_2, names=col_names)
        
        #join code-description
        code_description_list = [f"{code}: {description}" for code, description in zip(df1['code'], df1['description'])]
        
        st.write('List of codes we are checking:', code_description_list)

        st.header("Comparison:")
        
        # Compare each row of the second CSV with all rows of the first
        for index, row in df2.iterrows():
            #value = row['your_column_name']  # Replace 'your_column_name' with the actual column name

            # Join code: description for each row in df2
            value = row['code']+': '+ row['description']
            st.write('value '+value)


            prompt = f"""
            Then, determine if value: ```{value}``` conveys a similar idea or meaning
            to any element in the list combined_previous: {", ".join(code_description_list)}. 
            Your response should be either asserting 'true' (Similar idea or meaning) or a string 'false' (no similarity),
            and the index of the similar code in combined_previous (write 9999 when 'false').
            
            Format the response as a JSON file using the main key value_in_combined_previous, and subkeys 'comparison' and 'index'
            """

            with st.spinner(f"Processing for value: {value}..."):
                if selected_model == "azure-gpt35":
                    client_azure = AzureOpenAI(
                        api_key=api_key,
                        api_version="2023-12-01-preview",
                        azure_endpoint=azure_endpoint
                    )
                    processed_output = client_azure.chat.completions.create(
                        model=azure_deployment,
                        messages = [{"role": "user", "content": prompt}],
                        temperature=0,
                    ).choices[0].message.content
                else:
                    #client = openai
                    client.api_key = api_key
                    processed_output = get_completion(prompt, model=selected_model)

            try:
                json_output = ast.literal_eval(processed_output)
                st.subheader(f"Processed JSON Output for value: ")
                st.write(json_output)

                # You can further process the JSON output as needed
                if json_output['value_in_combined_previous']== 'true': 
                    st.write ('true')

            except (ValueError, SyntaxError) as e:
                st.warning(f"Unable to parse the output as JSON. Error: {e}")
                st.text("Processed Output:")
                st.text(processed_output)  # Print the processed output for debugging
            
            

    else:
        st.warning("Please upload exactly two CSV files.")

if __name__ == "__main__":
    main()
