'''
File to store preset prompts for AI models

'''

initial_coding_prompts = {
    "Preset 1" : """Generate a comprehensive set of initial codes (at least 15) for thematic analysis based on the provided text. Focus on capturing all significant explicit and latent meanings or events, emphasizing the respondent's perspective rather than the interviewer's.

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
""", 
"Preset 2" : """Your task is to assist in the generation of a very broad range of initial codes (generate as many initial codes as needed - at least 15 codes - to capture all the significant explicit or latent meaning, or events in the text, focusing on the respondent and not the interviewer), aiming to encompass a wide spectrum of themes and ideas present in the text below, to assist me with my thematic analysis? Provide a name for each code in no more than 4 words, a 25-word dense description of the code, and a quote from the respondent for each topic no longer than 4 words. Format the response as a JSON file, keeping codes, descriptions, and quotes together in the JSON, and group them under 'final_codes'.

Here is an example of the expected JSON format:

```json
{
  "final_codes": [
    {
      "code_name": "Work Stress",
      "description": "The respondent describes high levels of stress and pressure at their workplace, affecting their overall well-being and productivity.",
      "quote": "too much pressure"
    },
    {
      "code_name": "Family Support",
      "description": "The respondent talks about the emotional and practical support they receive from family members during challenging times.",
      "quote": "always there"
    },
    {
      "code_name": "Hobby Enjoyment",
      "description": "The respondent mentions their enjoyment and satisfaction derived from engaging in personal hobbies and activities during their free time.",
      "quote": "love painting"
    }
  ]
}
""",
"Preset 3" : """Can you assist in the generation of a very broad range of initial codes (generate as many initial codes as needed - at least 15 codes - to capture all the significant explicit or latent meaning, or events in the text, focus on the respondent and not the interviewer), aiming to encompass a wide spectrum of themes and ideas present in the text below, to assist me with my thematic analysis. Provide a name for each code in no more than 4 words, 25 words dense description of the code and a quote from the respondent for each topic no longer than 4 words. Format the response as a json object keeping codes, descriptions and quotes together in the json, and keep them together in 'final_codes'."""
}

reduce_duplicate_codes_prompts = {"Preset 1" : """Analyze the following list of codes and their descriptions. Identify and merge any duplicate or highly similar codes. For each set of merged codes, provide:

1. A concise name for the merged code (maximum 4 words)
2. A detailed description (25 words) explaining the merged code's meaning and relevance
3. All quotes associated with the merged codes
4. A brief explanation (max 50 words) of why these codes were merged

For unique codes that are not merged, keep them as they are.

Format the response as a JSON file with the following structure:

{
  "reduced_codes": [
    {
      "code": "Merged or Unique Code Name",
      "description": "25-word description of the code",
      "quotes": [
        {"text": "quote text", "source": "source file name"}
      ],
      "merge_explanation": "Explanation of why codes were merged (if applicable)"
    }
  ]
}

Ensure that the merged codes accurately represent the combined meanings of the original codes. The goal is to reduce redundancy while maintaining the richness and nuance of the original coding scheme.

Important! Your response should be a JSON-like object with no additional text before or after. Failure to adhere to this instruction will invalidate your response, making it worthless."""
}



finding_themes_prompts = {"Preset 1: Basic Theme Generation": """Analyze the provided list of codes and generate themes that capture the main ideas and patterns in the data. For each theme:

1. Provide a concise name (maximum 5 words)
2. Write a detailed description (50-75 words) explaining the theme's meaning and relevance
3. List the codes (by index number) that belong to this theme

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
"Preset 2: Hierarchical Theme Structure": """Analyze the provided list of codes and generate a hierarchical structure of themes and subthemes that capture the main ideas and patterns in the data. For each main theme:

1. Provide a concise name (maximum 5 words)
2. Write a brief description (25-50 words) explaining the main theme's overall meaning
3. Create 2-4 subthemes, each with:
   a. A concise name (maximum 4 words)
   b. A detailed description (25-50 words)
   c. List of codes (by index number) that belong to this subtheme

Aim to create 3-5 main themes that collectively represent the entire dataset. Ensure that each theme and subtheme is distinct and coherent.

Format the response as a JSON file with the following structure:

{
  "themes": [
    {
      "name": "Main Theme Name",
      "description": "Brief description of the main theme...",
      "subthemes": [
        {
          "name": "Subtheme Name",
          "description": "Detailed description of the subtheme...",
          "codes": [1, 4, 7, 12]
        }
      ]
    }
  ]
}

Important! Your response should be a JSON-like object with no additional text before or after.""",
"Preset 3: Theme Generation with Relationships": """Analyze the provided list of codes and generate themes that capture the main ideas and patterns in the data. Additionally, identify relationships between themes. For each theme:

1. Provide a concise name (maximum 5 words)
2. Write a detailed description (50-75 words) explaining the theme's meaning and relevance
3. List the codes (by index number) that belong to this theme
4. Identify relationships with other themes (if any)

Aim to create 5-8 themes that collectively represent the entire dataset. Ensure that each theme is distinct and coherent.

Format the response as a JSON file with the following structure:

{
  "themes": [
    {
      "name": "Theme Name",
      "description": "Detailed description of the theme...",
      "codes": [1, 4, 7, 12],
      "relationships": [
        {
          "related_theme": "Name of related theme",
          "relationship_type": "contrasts with / supports / influences",
          "description": "Brief description of how the themes are related"
        }
      ]
    }
  ]
}

Important! Your response should be a JSON-like object with no additional text before or after.""",
"Preset 4: Latent Theme Identification": """Analyze the provided list of codes and generate themes that capture both explicit and latent patterns in the data. Focus on identifying underlying ideas, assumptions, and conceptualizations that may not be immediately apparent. For each theme:

1. Provide a concise name (maximum 5 words)
2. Write a detailed description (75-100 words) explaining the theme's meaning, relevance, and any latent concepts it represents
3. List the codes (by index number) that belong to this theme
4. Provide a brief explanation of how this theme relates to broader theoretical or conceptual frameworks (if applicable)

Aim to create 4-6 themes that collectively represent the depth and complexity of the dataset. Ensure that each theme is distinct, coherent, and goes beyond surface-level interpretations.

Format the response as a JSON file with the following structure:

{
  "themes": [
    {
      "name": "Theme Name",
      "description": "Detailed description of the theme, including latent concepts...",
      "codes": [1, 4, 7, 12],
      "theoretical_connection": "Brief explanation of theoretical or conceptual connections"
    }
  ]
}

Important! Your response should be a JSON-like object with no additional text before or after."""
}