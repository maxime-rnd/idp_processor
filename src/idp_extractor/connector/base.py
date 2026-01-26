from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Tuple
import base64
import json
from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium
from ..prompts import PromptTemplate
from ..document import ExtractionResult
from ..config import Settings


class LLMConnector(ABC):
    """Abstract base class for LLM connectors."""

    def __init__(self):
        self.settings = Settings()

    def _encode_file(self, file_path: str) -> str:
        """Encode file to base64. Supports images (jpg, png, etc.) and PDFs."""
        path = Path(file_path)
        file_ext = path.suffix.lower()
        
        if file_ext == '.pdf':
            # Convert PDF first page to image
            pdf = pdfium.PdfDocument(file_path)
            page = pdf[0]  # First page
            pil_image = page.render(scale=self.settings.pdf_scale).to_pil()
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
        else:
            # Assume it's an image file
            pil_image = Image.open(file_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
        
        # Save to bytes and encode
        from io import BytesIO
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _parse_json_from_text(self, text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Parse JSON from LLM output text using pythonic string processing.
        
        This method handles cases where the LLM output contains extra text, markdown formatting,
        or other content around the JSON. It uses brace counting for accurate JSON boundary detection
        instead of regex, making it more robust and pythonic.
        
        Examples of inputs it can handle:
        - Pure JSON: '{"name": "John", "age": 30}'
        - JSON with markdown: '```json\n{"name": "John"}\n```'
        - JSON with extra text: 'Here is the result: {"name": "John"} Hope this helps!'
        
        Args:
            text: The raw text output from the LLM
            
        Returns:
            Tuple of (parsed_json, error_message)
        """
        if not text or not text.strip():
            return None, "Empty or None text provided"
        
        # Clean the text first
        cleaned_text = text.strip()
        
        # Remove markdown code blocks if present
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # First try to parse the cleaned text as JSON
        try:
            return json.loads(cleaned_text), None
        except json.JSONDecodeError:
            return None, f"Could not extract valid JSON from text: {cleaned_text[:200]}..."

    @abstractmethod
    def extract_info_from_file(self, file_path: str, prompt_template: PromptTemplate, model_class: Type[Any], **kwargs) -> ExtractionResult:
        """Extract information from a file using the LLM and return comprehensive result."""
        pass

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generate text response from a prompt."""
        pass