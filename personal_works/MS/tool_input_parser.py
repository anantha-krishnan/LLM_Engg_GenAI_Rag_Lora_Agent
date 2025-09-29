from langchain.tools import StructuredTool 
from typing import List, Any, Tuple, Dict
import json
import re


class RobustTool(StructuredTool):
    """
    A custom Tool class that handles common LLM failure modes:
    1. Generating JSON-like strings with comments, which are invalid JSON.
    2. Nesting the entire argument dictionary as a JSON string under the first argument's key.
    """
    def _parse_input(
        self,
        tool_input: str | Dict,
        tool_call_id: str | None = None,
    ) -> str | Dict[str, Any]:
        """Override the input parsing to clean and correct common LLM errors."""
        
        # --- NEW LOGIC: Handle raw string input with comments ---
        # If the input is a string, it's likely the raw LLM output.
        if isinstance(tool_input, str):
            # Use regex to remove all Python-style comments (# to end of line)
            # The `re.MULTILINE` flag is crucial for strings with newlines.
            cleaned_input_str = re.sub(r'#.*$', '', tool_input, flags=re.MULTILINE)
            
            try:
                # Attempt to parse the cleaned string.
                tool_input = json.loads(cleaned_input_str)
                print("   [RobustTool Correction] Cleaned comments from JSON string and parsed successfully.")
            except json.JSONDecodeError:
                # If it still fails, it wasn't a comment issue.
                # We'll let the default parser handle it, it might be a different format.
                print("   [RobustTool Info] Input is a string, but not valid JSON even after cleaning. Passing to default parser.")
                pass # Fall through to the super() call with the original string.

        # Let the default LangChain parser have a go.
        # It will receive either our newly parsed dict or the original input.
        parsed_input = super()._parse_input(tool_input, tool_call_id)

        # --- OLD LOGIC: Handle the nested dictionary case (still good to have) ---
        # This handles the case where the input looks like:
        # {'arg1': '{"arg1": "val1", "arg2": "val2"}'}
        if isinstance(parsed_input, dict) and len(parsed_input) == 1:
            key, value = list(parsed_input.items())[0]
            if isinstance(value, str):
                try:
                    # Try to parse the nested string value.
                    corrected_input = json.loads(value)
                    if isinstance(corrected_input, dict):
                        print(f"   [RobustTool Correction] Detected and fixed nested JSON input.")
                        return corrected_input
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, it was probably just a regular string value.
                    pass

        # Return whatever we have at the end of the process.
        return parsed_input