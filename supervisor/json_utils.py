"""Utility functions for JSON parsing and extraction."""

import json
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def extract_json_from_response(response: str) -> str:
    """
    Extract JSON from a response that may be wrapped in markdown code blocks.
    
    Args:
        response: The raw response string that may contain JSON in markdown code blocks
        
    Returns:
        The extracted JSON string
    """
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # If no code blocks found, return the response as-is
    return response.strip()


def parse_json_response(response: str, context: str = "") -> Optional[Dict[str, Any]]:
    """
    Parse JSON from a response, handling markdown code blocks.
    
    Args:
        response: The raw response string
        context: Context for error logging
        
    Returns:
        Parsed JSON dictionary or None if parsing fails
    """
    try:
        json_str = extract_json_from_response(response)
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON in {context}: {e}")
        logger.error(f"Raw response: {response}")
        logger.error(f"Extracted JSON string: {json_str}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON in {context}: {e}")
        return None


def validate_json_fields(data: Dict[str, Any], required_fields: list, context: str = "") -> bool:
    """
    Validate that a JSON object contains all required fields.
    
    Args:
        data: The JSON data to validate
        required_fields: List of required field names
        context: Context for error logging
        
    Returns:
        True if all fields are present, False otherwise
    """
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        logger.error(f"Missing required fields in {context}: {missing_fields}")
        return False
    return True
