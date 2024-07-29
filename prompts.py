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
3. A brief quote (maximum 4 words) that exemplifies the merged code

Format the response as a JSON file with the following structure:

{
  "reduced_codes": [
    {
      "code": "Merged Code Name",
      "description": "This is where you would provide a 25-word description of the merged code, explaining its meaning and significance in the context of the analysis.",
      "quote": "relevant quote here"
    },
    // Additional merged codes follow the same structure
  ]
}

Ensure that the merged codes accurately represent the combined meanings of the original codes. The goal is to reduce redundancy while maintaining the richness and nuance of the original coding scheme.

Important! Your response should be a JSON-like object with no additional text before or after. Failure to adhere to this instruction will invalidate your response, making it worthless."""}