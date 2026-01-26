from ..base import PromptTemplate


class PassportPromptV1(PromptTemplate):
    def __init__(self):
        system_prompt = "You are an expert at extracting information from French passport images. Provide accurate, structured data."
        user_prompt_template = """
Extract the following information from the French passport image:
- Type (P)
- Country code (FRA)
- Passport number
- Surname
- Given names
- Nationality
- Date of birth (YYYY-MM-DD)
- Sex (M/F)
- Place of birth
- Date of issue (YYYY-MM-DD)
- Date of expiry (YYYY-MM-DD)
- Authority
- MRZ line 1 (if visible)
- MRZ line 2 (if visible)

Return the information in JSON format with MRZ as a nested object:
{
  "type": "...",
  "country_code": "...",
  ...
  "mrz": {
    "line_1": "...",
    "line_2": "..."
  }
}
"""
        super().__init__(system_prompt=system_prompt, user_prompt_template=user_prompt_template)