import json
from openai import OpenAI
from typing import Any, Optional, Type
from datetime import datetime
from .base import LLMConnector
from ..prompts import PromptTemplate
from ..document import ExtractionResult
from ..metrics.models import PerformanceMetrics


class LLMaaSConnector(LLMConnector):
    """Connector for OpenAI-compatible LLMaaS (Large Language Model as a Service)."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or self.settings.api_key
        if not self.api_key:
            raise ValueError("API key must be provided either as parameter or in API_KEY environment variable/.env file")
        
        self.base_url = base_url or self.settings.base_url
        self.model = model or self.settings.model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def extract_info_from_file(self, file_path: str, prompt_template: PromptTemplate, model_class: Type[Any], **kwargs) -> ExtractionResult:
        """Extract information from file using vision model and return comprehensive result."""
        base64_image = self._encode_file(file_path)
        user_prompt = prompt_template.format_user_prompt(**kwargs)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt_template.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=self.settings.default_max_tokens,
            logprobs=True  # Enable logprobs for confidence calculation
        )

        content = response.choices[0].message.content
        timestamp = datetime.now()
        
        # Try to parse JSON from the LLM output
        parsed_json = None
        passport = None
        success = False
        error_message = None
        
        parsed_json, parse_error = self._parse_json_from_text(content)
        if parsed_json is not None:
            try:
                passport = model_class(**parsed_json)
                success = True
            except (TypeError, ValueError) as e:
                error_message = f"JSON parsed successfully but validation failed: {e}"
        else:
            error_message = parse_error
        
        # Extract logprobs for confidence
        logprobs = []
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            logprobs = [token.logprob for token in response.choices[0].logprobs.content if token.logprob is not None]
        
        # Calculate tokens used (simplified)
        tokens_used = response.usage.completion_tokens if response.usage else 0
        
        # Create metrics
        metrics = PerformanceMetrics.from_measurements(
            processing_time_seconds=0.0,  # Would need to track this
            energy_consumed_kwh=0.0,  # Would need codecarbon integration
            tokens_used=tokens_used,
            model=self.model,
            logprobs=logprobs,
            settings=self.settings
        )

        return ExtractionResult(
            passport=passport,
            raw_response=content,
            model_used=self.model,
            timestamp=timestamp,
            parsed_json=parsed_json,
            metrics=metrics,
            success=success,
            error_message=error_message
        )

    def generate_text(self, prompt: str) -> str:
        """Generate text response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.settings.default_max_tokens
        )
        return response.choices[0].message.content