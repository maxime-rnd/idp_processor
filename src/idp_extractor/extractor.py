from .document import Passeport, ExtractionResult
from .connector.llmaas import LLMaaSConnector
from .prompts import DEFAULT_PROMPT
from typing import Optional


def extract_passport_info(file_path: str, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None) -> ExtractionResult:
    """Extract passport information using MLLM."""
    connector = LLMaaSConnector(api_key=api_key, base_url=base_url, model=model)
    
    return connector.extract_info_from_image(file_path, DEFAULT_PROMPT, Passeport)