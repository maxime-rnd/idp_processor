from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Configuration settings for the IDP extractor loaded from environment variables and .env file."""
    
    # API Keys
    api_key: Optional[str] = Field(None, description="API key for MLLM services")
    base_url: Optional[str] = Field(None, description="Base URL for API endpoint (optional)")
    
    # Cost settings (in EUR per token)
    cost_per_token: float = Field(0.0000255, description="Default cost per token in euros")
    
    # Energy settings (in kWh per token)
    energy_per_token_kwh: float = Field(0.0003, description="Energy consumption per token in kWh")
    
    # Model settings
    model: str = Field("gpt-4o", description="Default model to use")
    default_max_tokens: int = Field(500, description="Default maximum tokens for API calls")
    
    # PDF processing settings
    pdf_scale: int = Field(2, description="Scale for PDF rendering")
    
    # Confidence calculation settings
    min_confidence_threshold: float = Field(0.7, description="Minimum confidence threshold")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False