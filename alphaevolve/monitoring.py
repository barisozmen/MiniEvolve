from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import time
import asyncio
from contextlib import asynccontextmanager

@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    model: str = ""
    
class MetricsCollector:
    """Base class for metrics collection."""
    def __init__(self):
        self._metrics: Dict[str, List] = {}
    
    def add_metric(self, name: str, value: any):
        """Add a metric to the collector."""
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(value)
    
    def get_metrics(self, name: str) -> List:
        """Get all values for a specific metric."""
        return self._metrics.get(name, [])
    
    def clear(self):
        """Clear all metrics."""
        self._metrics.clear()

class LLMMetricsCollector(MetricsCollector):
    """Specialized metrics collector for LLM operations."""
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.error_count = 0
        self.total_tokens = 0
        
    @asynccontextmanager
    async def track_call(self, model: str):
        """Context manager to track a single LLM call."""
        metrics = LLMCallMetrics(model=model)
        start_time = time.time()
        
        try:
            yield metrics
            
        finally:
            end_time = time.time()
            metrics.latency_seconds = end_time - start_time
            metrics.end_time = datetime.now()
            
            self.call_count += 1
            self.total_tokens += metrics.total_tokens
            if metrics.error:
                self.error_count += 1
            
            self.add_metric("calls", metrics)
    
    def get_average_latency(self) -> float:
        """Calculate average latency across all calls."""
        calls = self.get_metrics("calls")
        if not calls:
            return 0.0
        return sum(call.latency_seconds for call in calls) / len(calls)
    
    def get_success_rate(self) -> float:
        """Calculate the success rate of calls."""
        if self.call_count == 0:
            return 1.0
        return 1 - (self.error_count / self.call_count)
    
    def get_token_stats(self) -> Dict[str, float]:
        """Get statistics about token usage."""
        calls = self.get_metrics("calls")
        if not calls:
            return {"avg_tokens_per_call": 0.0, "total_tokens": 0}
        
        return {
            "avg_tokens_per_call": self.total_tokens / len(calls),
            "total_tokens": self.total_tokens
        } 