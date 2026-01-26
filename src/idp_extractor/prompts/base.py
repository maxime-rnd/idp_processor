from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel


class PromptTemplate(BaseModel, ABC):
    """Base class for prompt templates."""

    system_prompt: str
    user_prompt_template: str
    _user_prompt: Optional[str] = None

    @property
    def user_prompt(self) -> str:
        """Get the formatted user prompt with validation."""
        if self._user_prompt is None:
            raise ValueError("User prompt has not been formatted yet. Call format_user_prompt() first.")
        return self._user_prompt

    def format_user_prompt(self, **kwargs) -> str:
        """Format the user prompt with dynamic values and store it."""
        self._user_prompt = self.user_prompt_template.format(**kwargs)
        return self._user_prompt