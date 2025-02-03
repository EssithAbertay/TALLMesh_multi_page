import json
import re
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_text(text: str) -> str:
    """
    Sanitizes text by removing or replacing problematic characters
    while preserving meaningful content.
    
    Args:
        text (str): The input text to sanitize
        
    Returns:
        str: Sanitized text safe for JSON encoding
    """
    if not isinstance(text, str):
        return str(text)
    
    # Replace various types of quotes with standard quotes
    text = re.sub(r'[''‹›‚]', "'", text)  # Normalize single quotes
    text = re.sub(r'[""„〝〞〟‟]', '"', text)  # Normalize double quotes
    
    # Replace problematic whitespace characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)  # Remove zero-width spaces
    text = re.sub(r'[\r\n\t]+', ' ', text)  # Replace newlines and tabs with spaces
    
    # Remove control characters while preserving standard whitespace
    text = ''.join(char for char in text if char >= ' ' or char in ['\n', '\r', '\t'])
    
    return text.strip()

def prepare_code_dict(code: Dict[str, Any], include_quotes: bool = False) -> Dict[str, Any]:
    """
    Prepares a code dictionary for JSON encoding by sanitizing its values
    and structuring it appropriately.
    
    Args:
        code (Dict[str, Any]): The code dictionary to prepare
        include_quotes (bool): Whether to include quotes in the output
        
    Returns:
        Dict[str, Any]: A prepared dictionary safe for JSON encoding
    """
    prepared_dict = {
        "code": sanitize_text(code["code"]),
        "description": sanitize_text(code["description"]),
        "code_id": str(code["code_id"])  # Ensure code_id is string
    }
    
    if include_quotes and "quote" in code:
        prepared_dict["quote"] = sanitize_text(code["quote"])
    
    return prepared_dict

def generate_comparison_prompt(
    target_code: Dict[str, Any],
    comparison_codes: List[Dict[str, Any]],
    prompt: str,
    include_quotes: bool = False
) -> str:
    """
    Generates a prompt for code comparison with robust JSON handling.
    
    Args:
        target_code (Dict[str, Any]): The code to compare against
        comparison_codes (List[Dict[str, Any]]): List of codes to compare with
        prompt (str): Base prompt template
        include_quotes (bool): Whether to include quotes in comparison
        
    Returns:
        str: A formatted prompt with properly escaped JSON
    """
    try:
        # Prepare the target code
        target_dict = prepare_code_dict(target_code, include_quotes)
        
        # Prepare comparison codes
        comparison_list = []
        for code in comparison_codes:
            try:
                prepared_code = prepare_code_dict(code, include_quotes)
                comparison_list.append(json.dumps(prepared_code))
            except Exception as e:
                logger.error(f"Error preparing comparison code: {str(e)}")
                continue
        
        # Format the comparison list
        comparisons_json = ',\n    '.join(comparison_list)
        
        # Create the final prompt with properly escaped JSON
        final_prompt = prompt % (
            json.dumps(target_dict["code"])[1:-1],  # Remove outer quotes
            json.dumps(target_dict["description"])[1:-1],
            f',\n        "quote": {json.dumps(target_dict["quote"])}' if include_quotes else '',
            f'[\n    {comparisons_json}\n]'
        )
        
        return final_prompt
        
    except Exception as e:
        logger.error(f"Error generating comparison prompt: {str(e)}")
        raise

def validate_json_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Validates and cleans the JSON response from the LLM.
    
    Args:
        response (str): The raw response from the LLM
        
    Returns:
        Optional[Dict[str, Any]]: Parsed JSON if valid, None if invalid
    """
    try:
        # First try direct JSON parsing
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON if embedded in other text
        json_match = re.search(r'\{(?:[^{}]|(?R))*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return None
        return None