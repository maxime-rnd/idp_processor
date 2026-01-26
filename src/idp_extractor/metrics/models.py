from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from ..config import Settings


class PerformanceMetrics(BaseModel):
    """Pydantic model for performance metrics with automatic computation from raw data."""
    
    # Raw input data (provided by user)
    processing_time_seconds: float = Field(..., description="Raw processing time in seconds")
    tokens_used: int = Field(..., description="Number of tokens used")
    model: str = Field(..., description="Model name used")
    logprobs: Optional[List[float]] = Field(None, description="Raw log probabilities from model output")
    
    # Computed KPIs (calculated automatically)
    energy_consumed_kwh: float = Field(..., description="Energy consumption in kWh (computed from tokens)")
    cost_eur: float = Field(..., description="Cost in EUR (computed from tokens)")
    confidence: Optional[float] = Field(None, description="Confidence score (computed from logprobs)")
    tokens_per_second: float = Field(..., description="Token processing rate")
    energy_per_token: float = Field(..., description="Energy efficiency (kWh per token)")
    cost_per_token_eur: float = Field(..., description="Cost per token in EUR")
    
    # Auto-computed timestamp
    timestamp: datetime = Field(default_factory=datetime.now, description="When metrics were recorded")
    
    @classmethod
    def from_measurements(
        cls,
        processing_time_seconds: float,
        tokens_used: int,
        model: str,
        logprobs: Optional[List[float]] = None,
        settings: Optional[Settings] = None
    ) -> "PerformanceMetrics":
        """Create PerformanceMetrics from raw measurements with automatic KPI computation."""
        if settings is None:
            settings = Settings()
        
        # Compute energy consumption based on tokens and energy ratio from settings
        energy_per_token_ratio = settings.energy_per_token_kwh
        energy_consumed_kwh = tokens_used * energy_per_token_ratio
        
        # Compute cost
        cost_eur = tokens_used * settings.cost_per_token
        
        # Compute confidence from logprobs
        confidence = None
        if logprobs:
            probs = [2 ** logprob for logprob in logprobs if logprob is not None]
            if probs:
                confidence = sum(probs) / len(probs)
        
        # Compute token processing rate
        tokens_per_second = tokens_used / processing_time_seconds if processing_time_seconds > 0 else 0.0
        
        # Compute energy efficiency
        energy_per_token = energy_consumed_kwh / tokens_used if tokens_used > 0 else 0.0
        
        # Get cost per token
        cost_per_token_eur = settings.cost_per_token
        
        return cls(
            processing_time_seconds=processing_time_seconds,
            tokens_used=tokens_used,
            model=model,
            logprobs=logprobs,
            energy_consumed_kwh=energy_consumed_kwh,
            cost_eur=cost_eur,
            confidence=confidence,
            tokens_per_second=tokens_per_second,
            energy_per_token=energy_per_token,
            cost_per_token_eur=cost_per_token_eur
        )