'''
File to store preset prompts for AI models

'''

initial_coding_prompts = {  
"Preset 1: >15 Codes, Inductive, Long Quotes" : {"prompt": """Generate a comprehensive set of initial codes (at least 15) for thematic analysis based on the provided text. Focus on capturing all significant explicit and latent meanings or events, emphasizing the respondent's perspective rather than the interviewer's.

For each code, provide:

1. A concise name (maximum 5 words)
2. A detailed description (25 words) explaining the code's meaning and relevance
3. A quote (minimum necessary to capture context and example) from the respondent that exemplifies the code

Important! Format the response as a JSON file with the following structure:

{
  "final_codes": [
    {
      "code_name": "Example Code Name",
      "description": "This is where you would provide a 25-word description of the code, explaining its meaning and significance in the context of the analysis.",
      "quote": "relevant quote here"
    },
    // Additional codes follow the same structure
  ]
}

Ensure that the codes cover a wide range of themes and ideas present in the text, including both obvious and subtle concepts. The goal is to provide a comprehensive starting point for further thematic analysis.

Important! Your response should be a JSON-like object with no additional text before or after. Failure to adhere to this instruction will invalidate your response, making it worthless.
""", "temperature":0.00, "top_p":0.1},
"Preset 2: Deductive, Long Quotes" : {"prompt": """Generate a comprehensive set of initial codes for thematic analysis based on the provided text. Focus on capturing all significant explicit and latent meanings or events relating to [INSERT SEARCHABLE CODES / THEMES HERE]

For each code, provide:

1. A concise name (maximum 5 words)
2. A detailed description (25 words) explaining the code's meaning and relevance
3. A quote (minimum necessary to capture context and example) from the respondent that exemplifies the code

Important! Format the response as a JSON file with the following structure:

{
  "final_codes": [
    {
      "code_name": "Example Code Name",
      "description": "This is where you would provide a 25-word description of the code, explaining its meaning and significance in the context of the analysis.",
      "quote": "relevant quote here"
    },
    // Additional codes follow the same structure
  ]
}

Ensure that the codes cover a wide range of themes and ideas present in the text, including both obvious and subtle concepts. The goal is to provide a comprehensive starting point for further thematic analysis.

Important! Your response should be a JSON-like object with no additional text before or after. Failure to adhere to this instruction will invalidate your response, making it worthless.
""", "temperature":0.00, "top_p":0.1},
"Preset 3: >15 Codes, Short Quotes" : {"prompt": """Generate a comprehensive set of initial codes (at least 15) for thematic analysis based on the provided text. Focus on capturing all significant explicit and latent meanings or events, emphasizing the respondent's perspective rather than the interviewer's.

For each code, provide:

1. A concise name (maximum 4 words)
2. A detailed description (25 words) explaining the code's meaning and relevance
3. A brief quote (maximum 4 words) from the respondent that exemplifies the code

Format the response as a JSON file with the following structure:

{
  "final_codes": [
    {
      "code_name": "Example Code Name",
      "description": "This is where you would provide a 25-word description of the code, explaining its meaning and significance in the context of the analysis.",
      "quote": "relevant quote here"
    },
    // Additional codes follow the same structure
  ]
}

Ensure that the codes cover a wide range of themes and ideas present in the text, including both obvious and subtle concepts. The goal is to provide a comprehensive starting point for further thematic analysis.

Important! Your response should be a JSON-like object with no additional text before or after. Failure to adhere to this instruction will invalidate your response, making it worthless.
""","temperature": 0.00,
        "top_p": 0.1
    },
"Preset 4" :  {"prompt" : """Your task is to assist in the generation of a very broad range of initial codes (generate as many initial codes as needed - at least 15 codes - to capture all the significant explicit or latent meaning, or events in the text, focusing on the respondent and not the interviewer), aiming to encompass a wide spectrum of themes and ideas present in the text below, to assist me with my thematic analysis? Provide a name for each code in no more than 4 words, a 25-word dense description of the code, and a quote from the respondent for each topic no longer than 4 words. Format the response as a JSON file, keeping codes, descriptions, and quotes together in the JSON, and group them under 'final_codes'.

Here is an example of the expected JSON format:

```json
{
  "final_codes": [
    {
      "code_name": "Example Code Name",
      "description": "This is where you would provide a 25-word description of the code, explaining its meaning and significance in the context of the analysis.",
      "quote": "relevant quote here"
    },
    // Additional codes follow the same structure
  ]
}
""", "temperature": 0.00, "top_p": 0.1},
"Preset 5" : {"prompt": """Can you assist in the generation of a very broad range of initial codes (generate as many initial codes as needed - at least 15 codes - to capture all the significant explicit or latent meaning, or events in the text, focus on the respondent and not the interviewer), aiming to encompass a wide spectrum of themes and ideas present in the text below, to assist me with my thematic analysis. Provide a name for each code in no more than 4 words, 25 words dense description of the code and a quote from the respondent for each topic no longer than 4 words. Format the response as a json object keeping codes, descriptions and quotes together in the json, and keep them together in 'final_codes'. Format the response as a JSON file with the following structure:

{
  "final_codes": [
    {
      "code_name": "Example Code Name",
      "description": "This is where you would provide a 25-word description of the code, explaining its meaning and significance in the context of the analysis.",
      "quote": "relevant quote here"
    },
    // Additional codes follow the same structure
  ]
}
""", "temperature": 0.00, "top_p": 0.1},
}

reduce_duplicate_codes_prompts = {
    "Preset 1 - Reduce Duplicates": {"prompt":"""Analyze the following list of codes and their descriptions. Identify and merge any duplicate codes within this chunk. For each set of merged codes, provide:

1. Important! A concise name for the merged code (maximum 4 words)
2. Important! A detailed description (25 words) explaining the merged code's meaning and relevance
3. Important! A brief explanation (max 50 words) of why these codes were merged
4. Important! A list of ALL original code names, including those that were not merged. Every input code must be accounted for in the output. 

Important! If a code already has a merge explanation, incorporate it into the new explanation.
Important! Ensure that the 'original_codes' list for each reduced code includes ALL relevant original codes, even if it's just one code that wasn't merged, or codes with the exact same name.
Important! For unique codes that are not merged, keep them as they are.

The codes to analyze are provided in the following format:
{codes}

Format the response as a JSON file with the following structure:

{
  "reduced_codes": [
    {
      "code": "Merged or Unique Code Name",
      "description": "25-word description of the code",
      "merge_explanation": "Explanation of why codes were merged (if applicable)",
      "original_codes": ["Original Code 1", "Original Code 2", ...]
    }
  ]
}
                                     


Important! Your response should be a JSON-like object with no additional text before or after. Failure to adhere to this instruction will invalidate your response, making it worthless.""",
        "temperature": 0.00,
        "top_p": 0.1
    },
 "Preset 2 - Reduce Duplicates & Highly Similar": {"prompt":"""Analyze the following list of codes and their descriptions. Identify and merge any duplicate or highly similar codes within this chunk. For each set of merged codes, provide:

1. Important! A concise name for the merged code (maximum 4 words)
2. Important! A detailed description (25 words) explaining the merged code's meaning and relevance
3. Important! A brief explanation (max 50 words) of why these codes were merged
4. Important! A list of ALL original code names, including those that were not merged. Every input code must be accounted for in the output.

Important! If a code already has a merge explanation, incorporate it into the new explanation.

Important! For unique codes that are not merged, keep them as they are.

The codes to analyze are provided in the following format:
{codes}

Format the response as a JSON file with the following structure:

{
  "reduced_codes": [
    {
      "code": "Merged or Unique Code Name",
      "description": "25-word description of the code",
      "merge_explanation": "Explanation of why codes were merged (if applicable)",
      "original_codes": ["Original Code 1", "Original Code 2", ...]
    }
  ]
}

Important! Your response should be a JSON-like object with no additional text before or after. Failure to adhere to this instruction will invalidate your response, making it worthless.""",
        "temperature": 0.00,
        "top_p": 0.1
    },
}

finding_themes_prompts = {"Preset 1: Basic Theme Generation (Overlap Allowed)": {"prompt":"""Analyze the provided list of codes and generate themes that capture the main ideas and patterns in the data. For each theme:

1. Provide a concise theme name (maximum 5 words)
2. Write a detailed theme description (50-75 words) explaining the theme's meaning and relevance
3. List the codes (by index number) that are associated with this theme

Important! It is okay for themes to have shared codes, this will facilitate an evaluation of higher order concepts.

Format the response as a JSON file with the following structure:

{
  "themes": [
    {
      "name": "Theme Name",
      "description": "Detailed description of the theme...",
      "codes": [1, 4, 7, 12]
    }
  ]
}

Important! Your response should be a JSON-like object with no additional text before or after.""",
        "temperature": 0.00,
        "top_p": 0.1
    },
    "Preset 2: Basic Theme Generation (No Overlap)": {"prompt":"""Analyze the provided list of codes and generate themes that capture the main ideas and patterns in the data. For each theme:

1. Provide a concise theme name (maximum 5 words)
2. Write a detailed theme description (50-75 words) explaining the theme's meaning and relevance
3. List the codes (by index number) that are associated with this theme

Aim to create unique themes that collectively represent the entire dataset. Ensure that each theme is distinct and coherent.

Format the response as a JSON file with the following structure:

{
  "themes": [
    {
      "name": "Theme Name",
      "description": "Detailed description of the theme...",
      "codes": [1, 4, 7, 12]
    }
  ]
}

Important! Your response should be a JSON-like object with no additional text before or after.""",
        "temperature": 0.00,
        "top_p": 0.1
    },
}