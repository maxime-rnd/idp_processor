from pydantic import BaseModel
from typing import Optional, Any, Dict
from datetime import datetime
from .passport import Passeport
from ..metrics.models import PerformanceMetrics


class ExtractionResult(BaseModel):
    """Result of document information extraction."""
    passport: Optional[Passeport] = None
    raw_response: str
    model_used: str
    timestamp: datetime
    parsed_json: Optional[Dict[str, Any]] = None
    metrics: PerformanceMetrics
    success: bool
    error_message: Optional[str] = None