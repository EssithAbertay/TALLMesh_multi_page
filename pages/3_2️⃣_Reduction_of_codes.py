# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:30:28 2024

@author: Stefano De Paoli - s.depaoli@abertay.ac.uk
"""

import streamlit as st
import pandas as pd
import ast #I am using this rather than json loads, but this needs to be fixed
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
        st.write(f"1 Master Cumulative Codebook. {uploaded_file_1.name}")
        st.write(f"2 Single Interview {uploaded_file_2.name}")
        
              
        #this is the cumulative
        df_cumul = pd.read_csv(uploaded_file_1)
        
        # Reset the index to turn the index into a regular column named 'index'
        # Rename the first column to 'index'
        df_cumul = df_cumul.rename(columns={df_cumul.columns[0]: 'index'})
        
        #check the cumulative codebook
        st.write(df_cumul)
        
       
        df_to_check = pd.read_csv(uploaded_file_2)
        
        
        #check the cumulative codebook
        #st.write(df_cumul)
        
        #join code-description
        code_description_list = [f"{code}: {description}" for code, description in zip(df_cumul['code'], df_cumul['description'])]
        
       # st.write('List of codes we are checking:', code_description_list)

        st.header("Comparison:")
        
        # Create a list to store values for which the comparison is false
        false_values = []
        quotes = []
        codes= []
       

        # Compare each row of the second CSV with all rows of the first
        for index, row in df_to_check.iterrows():
            
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
                if json_output['value_in_combined_previous']['comparison']== 'false': 
                   
                    # If comparison is false, add the value to the list
                    false_values.append(value)
                else:
                     # If comparison is true, retrieve the quote from df2 and add it to the quotes list
                    quote = df_to_check.loc[df_to_check['code'] == row['code'], 'quote'].iloc[0]
                    st.write(quote)
                    quotes.append(quote)
                    
                    #retrieve codes which will need to be checked in df1
                    index = json_output['value_in_combined_previous']['index']
                    codes.append(df_cumul['code'][index])
                   

            except (ValueError, SyntaxError) as e:
                st.warning(f"Unable to parse the output as JSON. Error: {e}")
                st.text("Processed Output:")
                st.text(processed_output)  # Print the processed output for debugging

        # Append false values to df1, these are the Unique Codes
        if false_values:
            
            st.subheader("False Values:")
           
            false_records = []
            for value in false_values:
                code, description = value.split(':')
                # Retrieve the corresponding quote from df2
                quote = df_to_check.loc[df_to_check['code'] == code, 'quote'].iloc[0]
                false_records.append({'code': code, 'description': description, 'quote': quote})
            false_df = pd.DataFrame(false_records)
            df_cumul = pd.concat([df_cumul, false_df], ignore_index=True)
            df_cumul = df_cumul.drop(columns=['index'], axis=1)  # Drop the 'index' column
            df_cumul.index = range(len(df_cumul))  # Set the index to the actual index
            #st.write(df1)

        if codes:
            # Function to add quotes to the DataFrame
           # def add_quotes(df, codes, quotes):
           #    quotes_map = {}
            #    for code, quote in zip(codes, quotes):
            #        if code not in quotes_map:
            #            quotes_map[code] = []
           #         quotes_map[code].append(quote+' #')
           #     
           #     for i, (code, quote_list) in enumerate(quotes_map.items(), start=1):
           #         new_quote_col = f'quotes_{i}'
           #         df[new_quote_col] = df['code'].apply(lambda x: quote_list if x == code else None)
           #     
           #     return df
           # def add_quotes(df, codes, quotes):
           #     quotes_map = {}
           #     for code, quote in zip(codes, quotes):
           #         if code not in quotes_map:
           #             quotes_map[code] = []
           #         quotes_map[code].append(quote + ' #')
           #     
           #     for i, (code, quote_list) in enumerate(quotes_map.items(), start=1):
           #         new_quote_col = f'quotes_{i}'
           #         # Iterate over each row and append the quote to the next blank cell in the column
           #         for index, row in df.iterrows():
           #             if row['code'] == code:
           #                 for j, quote in enumerate(quote_list):
           #                     if pd.isnull(row[new_quote_col]):
            #                        df.at[index, new_quote_col] = quote
           #                         break
           #                 #else:
           #                     # If no blank cell found in the column, add a new row with the quote
           #                  #   df.at[index, new_quote_col] = quote_list[0]
           #                   #  df = df.append(row)
           #     return df

           # df_cumul = add_quotes(df_cumul, codes, quotes)  
            
            
            #code to shift quotes
            
                                              
            #show output
            st.write(df_cumul)

        else: st.write('There are no duplicates')
        
    else:
        st.warning("Please upload exactly two CSV files.")


if __name__ == "__main__":
    main()
