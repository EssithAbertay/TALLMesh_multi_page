import streamlit as st
import json
from prompts import initial_coding_prompts, reduce_duplicate_codes_prompts, finding_themes_prompts

def get_prepopulated_prompt(prompt_type):
    base_prompt = """
Your task is to analyze the provided text and generate output based on the specific instructions below.

[INSERT SPECIFIC INSTRUCTIONS HERE]

Format the response as a JSON file with the following structure:

{
"""

    if prompt_type == "Initial Coding":
        return base_prompt + """
  "final_codes": [
    {
      "code_name": "Example Code Name",
      "description": "This is where you would provide a 25-word description of the code, explaining its meaning and significance in the context of the analysis.",
      "quote": "relevant quote here"
    },
    // Additional codes follow the same structure
  ]
}

Important! Your response should be a JSON-like object with no additional text before or after. Failure to adhere to this instruction will invalidate your response, making it worthless.
"""
    elif prompt_type == "Reduction of Codes":
        return base_prompt + """
  "reduced_codes": [
    {
      "code": "Merged or Unique Code Name",
      "description": "25-word description of the code",
      "merge_explanation": "Explanation of why codes were merged (if applicable)",
      "original_codes": ["Original Code 1", "Original Code 2", ...]
    },
    // Additional reduced codes follow the same structure
  ]
}

Important! Your response should be a JSON-like object with no additional text before or after. Failure to adhere to this instruction will invalidate your response, making it worthless.
"""
    elif prompt_type == "Finding Themes":
        return base_prompt + """
  "themes": [
    {
      "name": "Theme Name",
      "description": "Detailed description of the theme...",
      "codes": [1, 4, 7, 12]
    },
    // Additional themes follow the same structure
  ]
}

Important! Your response should be a JSON-like object with no additional text before or after. Failure to adhere to this instruction will invalidate your response, making it worthless.
"""
    else:
        return ""

def save_custom_prompts(prompts):
    with open('custom_prompts.json', 'w') as f:
        json.dump(prompts, f)

def load_custom_prompts():
    try:
        with open('custom_prompts.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def main():
    st.title("Custom Prompt Management")

    with st.expander("Instructions"):
        st.header("Custom Prompts Management")
        st.write("The Custom Prompts Management page allows you to create, edit, and delete custom prompts for different stages of your thematic analysis. Here's how to use this page effectively:")

        st.subheader(":orange[1. Selecting Prompt Type]")
        st.write("At the top of the page, you'll find a dropdown menu to select the prompt type:")
        st.markdown("""
        - **Initial Coding**: For creating initial codes from your data.
        - **Reduction of Codes**: For merging and refining your codes.
        - **Finding Themes**: For identifying overarching themes from your codes.
        """)
        st.write("Select the prompt type you want to work with.")

        st.subheader(":orange[2. Viewing Existing Custom Prompts]")
        st.write("After selecting a prompt type, you'll see a list of existing custom prompts (if any) for that category.")
        st.markdown("""
        - Each prompt is displayed in an expandable section.
        - Click on a prompt name to view its details.
        """)

        st.subheader(":orange[3. Editing Existing Prompts]")
        st.write("To edit an existing custom prompt:")
        st.markdown("""                                                 
        1. Expand the prompt you want to edit.
        2. Modify the prompt text in the text area.
        3. Adjust the Temperature and Top P values using the number input fields.
        4. Click the "Update" button to save your changes.
        """)
        st.warning("Remember to maintain the JSON structure in your prompt to ensure proper functionality.")

        st.subheader(":orange[4. Deleting Custom Prompts]")
        st.write("To delete a custom prompt:")
        st.markdown("""
        1. Expand the prompt you want to delete.
        2. Click the "Delete" button at the bottom of the expanded section.
        3. Confirm the deletion when prompted.
        """)
        st.warning("Deleted prompts cannot be recovered, so be sure before deleting.")

        st.subheader(":orange[5. Creating New Custom Prompts]")
        st.write("To create a new custom prompt:")
        st.markdown("""
        1. Scroll to the "Add New Custom Prompt" section at the bottom of the page.
        2. Enter a name for your new prompt in the "Name for new prompt" field.
        3. Edit the pre-populated prompt in the "New prompt" text area. 
        - The pre-populated text includes the required JSON structure for the selected prompt type.
        - Modify the [INSERT SPECIFIC INSTRUCTIONS HERE] section with your custom instructions.
        4. Set the desired Temperature and Top P values.
        5. Click the "Add Custom Prompt" button to create your new prompt.
        """)
        st.info("Tip: The pre-populated prompt structure helps ensure your custom prompt will work correctly with the system. Try to maintain this structure while customizing your instructions.")

        st.subheader(":orange[6. Using Custom Prompts]")
        st.write("After creating custom prompts, you can use them in their respective analysis stages:")
        st.markdown("""
        - Custom Initial Coding prompts will appear in the prompt selection on the Initial Coding page.
        - Custom Reduction of Codes prompts will be available on the Reduction of Codes page.
        - Custom Finding Themes prompts can be selected on the Finding Themes page.
        """)
        st.info("Your custom prompts will appear alongside the preset prompts in these pages.")

        st.write("Remember, well-crafted prompts can significantly improve the quality of your analysis. Take time to refine your prompts based on your specific research needs and the nature of your data.")

    custom_prompts = load_custom_prompts()

    prompt_types = {
        "Initial Coding": initial_coding_prompts,
        "Reduction of Codes": reduce_duplicate_codes_prompts,
        "Finding Themes": finding_themes_prompts
    }

    selected_type = st.selectbox("Select prompt type:", list(prompt_types.keys()))

    st.subheader(f"Custom Prompts for {selected_type}")

    # Display existing custom prompts
    for name, data in custom_prompts.get(selected_type, {}).items():
        with st.expander(name):
            st.text_area("Prompt", value=data["prompt"], key=f"prompt_{name}", height=300)
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Temperature", value=data["temperature"], key=f"temp_{name}", min_value=0.0, max_value=2.0, step=0.01)
            with col2:
                st.number_input("Top P", value=data["top_p"], key=f"top_p_{name}", min_value=0.0, max_value=1.0, step=0.01)
            if st.button("Update", key=f"update_{name}"):
                custom_prompts[selected_type][name] = {
                    "prompt": st.session_state[f"prompt_{name}"],
                    "temperature": st.session_state[f"temp_{name}"],
                    "top_p": st.session_state[f"top_p_{name}"]
                }
                save_custom_prompts(custom_prompts)
                st.success(f"Updated custom prompt: {name}")
            if st.button("Delete", key=f"delete_{name}"):
                del custom_prompts[selected_type][name]
                save_custom_prompts(custom_prompts)
                st.success(f"Deleted custom prompt: {name}")
                st.rerun()

    # Add new custom prompt
    st.subheader("Add New Custom Prompt")
    new_name = st.text_input("Name for new prompt:")
    new_prompt = st.text_area("New prompt:", value=get_prepopulated_prompt(selected_type), height=300)
    col1, col2 = st.columns(2)
    with col1:
        new_temp = st.slider("Temperature", value=0.1, min_value=0.0, max_value=2.0, step=0.01)
    with col2:
        new_top_p = st.slider("Top P", value=1.0, min_value=0.0, max_value=1.0, step=0.01)
    
    if st.button("Add Custom Prompt"):
        if new_name and new_prompt:
            if selected_type not in custom_prompts:
                custom_prompts[selected_type] = {}
            custom_prompts[selected_type][new_name] = {
                "prompt": new_prompt,
                "temperature": new_temp,
                "top_p": new_top_p
            }
            save_custom_prompts(custom_prompts)
            st.success(f"Added new custom prompt: {new_name}")
            st.rerun()
        else:
            st.error("Please provide both a name and a prompt.")

if __name__ == "__main__":
    main()