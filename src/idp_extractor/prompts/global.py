from typing import Type, Any, Dict, List, Union
from pydantic import BaseModel
from pydantic.fields import FieldInfo
import json


def generate_document_extraction_prompt(model_class: Type[BaseModel], document_type: str = "document") -> str:
    """
    Generate a generic document extraction prompt based on a Pydantic model.

    Args:
        model_class: The Pydantic model class to generate the prompt for
        document_type: Description of the document type (e.g., "French passport")

    Returns:
        A formatted prompt string explaining field expectations and JSON format
    """

    def _get_field_description(field_name: str, field_info: FieldInfo) -> str:
        """Extract field description and format requirements."""
        description = field_info.description or f"The {field_name.replace('_', ' ')}"

        # Handle special field types
        if hasattr(field_info.annotation, '__origin__') and field_info.annotation.__origin__ is list:
            description += " (as a list/array)"
        elif hasattr(field_info.annotation, '__name__'):
            if field_info.annotation.__name__ == 'date':
                description += " (in YYYY-MM-DD format)"
            elif field_info.annotation.__name__ == 'Gender':
                description += " (use 'M' for male, 'F' for female)"

        return description

    def _build_json_schema(model: Type[BaseModel], indent: int = 0) -> Dict[str, Any]:
        """Build a JSON schema representation of the model."""
        schema = {}

        for field_name, field_info in model.model_fields.items():
            field_type = field_info.annotation

            # Handle nested BaseModel
            if hasattr(field_type, 'model_fields'):
                schema[field_name] = _build_json_schema(field_type, indent + 1)
            # Handle enums
            elif hasattr(field_type, '__members__'):
                # For enums, show the possible values
                enum_values = list(field_type.__members__.keys())
                schema[field_name] = f"{'|'.join(enum_values)}"
            # Handle Optional types
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is type(Union):
                # Get the non-None type
                non_none_type = [arg for arg in field_type.__args__ if arg is not type(None)][0]
                if hasattr(non_none_type, 'model_fields'):
                    schema[field_name] = _build_json_schema(non_none_type, indent + 1)
                else:
                    schema[field_name] = f"{non_none_type.__name__ if hasattr(non_none_type, '__name__') else str(non_none_type)} (optional)"
            # Handle lists
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                schema[field_name] = []
            # Handle basic types
            else:
                type_name = getattr(field_type, '__name__', str(field_type))
                if type_name == 'date':
                    schema[field_name] = "YYYY-MM-DD"
                else:
                    schema[field_name] = type_name

        return schema

    # Build the field descriptions
    field_descriptions = []
    for field_name, field_info in model_class.model_fields.items():
        description = _get_field_description(field_name, field_info)
        field_descriptions.append(f"- {field_name}: {description}")

    # Build the expected JSON schema
    json_schema = _build_json_schema(model_class)

    # Generate the prompt
    prompt = f"""
Extract the following information from the {document_type} image:

{chr(10).join(field_descriptions)}

Return the information in JSON format with the following structure:
{json.dumps(json_schema, indent=2)}

Important notes:
- Ensure all dates are in YYYY-MM-DD format
- For optional fields, use null if the information is not available
- Be as accurate as possible in extracting the information
- If information is unclear or partially visible, make your best estimate
"""

    return prompt.strip()


# Example usage with Passeport class
if __name__ == "__main__":
    from ..document.passport import Passeport

    prompt = generate_document_extraction_prompt(Passeport, "French passport")
    print(prompt)