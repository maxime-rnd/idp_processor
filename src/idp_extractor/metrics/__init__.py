import time
from codecarbon import EmissionsTracker
from .models import PerformanceMetrics
from typing import Optional, List
from ..config import Settings


class PerformanceTracker:
    """Track performance metrics for extraction."""

    def __init__(self, settings: Optional[Settings] = None):
        self.tracker = EmissionsTracker()
        self._settings = settings or Settings()

    def start_tracking(self):
        self.start_time = time.time()
        self.tracker.start()

    def stop_tracking(self, tokens_used: int = 0, model: str = "gpt-4o", logprobs: Optional[List[float]] = None) -> PerformanceMetrics:
        """Stop tracking and return PerformanceMetrics with raw measurements."""
        end_time = time.time()
        processing_time = end_time - self.start_time

        # Stop the emissions tracker (for future use if needed)
        self.tracker.stop()

        return PerformanceMetrics.from_measurements(
            processing_time_seconds=processing_time,
            tokens_used=tokens_used,
            model=model,
            logprobs=logprobs,
            settings=self._settings
        )