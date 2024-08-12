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