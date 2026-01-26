import json
from typing import Any, Optional, Type
from datetime import datetime
from vllm import LLM, SamplingParams
from vllm.multimodal import MultiModalData
from .base import LLMConnector
from ..prompts import PromptTemplate
from ..document import ExtractionResult
from ..metrics.models import PerformanceMetrics


class LocalVLMConnector(LLMConnector):
    """Connector for local Vision Language Model using vLLM."""

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", max_model_len: int = 2048):
        super().__init__()
        self.model_name = model_name
        self.llm = LLM(model=model_name, max_model_len=max_model_len)
        self.sampling_params = SamplingParams(temperature=0.1, max_tokens=self.settings.default_max_tokens)

    def extract_info_from_file(self, file_path: str, prompt_template: PromptTemplate, model_class: Type[Any], **kwargs) -> ExtractionResult:
        """Extract information from file using local VLM and return comprehensive result."""
        user_prompt = prompt_template.format_user_prompt(**kwargs)
        multimodal_data = MultiModalData(image=file_path)

        messages = [
            {"role": "system", "content": prompt_template.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"file://{file_path}"}}
                ]
            }
        ]

        # vLLM handling for multimodal
        # Assuming the model supports multimodal inputs
        outputs = self.llm.chat(messages, sampling_params=self.sampling_params, multimodal_data=multimodal_data)

        content = outputs[0].outputs[0].text
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
        
        # For local models, we don't have token usage or logprobs easily
        # Create basic metrics
        metrics = PerformanceMetrics.from_measurements(
            processing_time_seconds=0.0,
            energy_consumed_kwh=0.0,
            tokens_used=0,  # Not available
            model=self.model_name,
            logprobs=[],  # Not available
            settings=self.settings
        )

        return ExtractionResult(
            passport=passport,
            raw_response=content,
            model_used=self.model_name,
            timestamp=timestamp,
            parsed_json=parsed_json,
            metrics=metrics,
            success=success,
            error_message=error_message
        )

    def generate_text(self, prompt: str) -> str:
        """Generate text response."""
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text