
from dataclasses import dataclass, field
from typing import List, Dict, Any
import time
import psutil
import numpy as np

from src.batch_forecasting_config import BatchPerformanceMetrics

class BatchPerformanceMonitor:
    """Monitors and records performance metrics for batch forecasting."""

    def __init__(self):
        self.all_metrics: List[BatchPerformanceMetrics] = []
        self._current_batch_start_time: float = 0.0
        self._current_batch_size: int = 0

    def start_batch(self, batch_size: int):
        """Call when a new batch processing starts."""
        self._current_batch_start_time = time.time()
        self._current_batch_size = batch_size

    def end_batch(self, model_used: str, feature_extraction_time: float, model_training_time: float, prediction_time: float, fallback_rate: float = 0.0):
        """Call when a batch has finished processing."""
        processing_time = time.time() - self._current_batch_start_time
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024) # in MB

        metrics = BatchPerformanceMetrics(
            batch_size=self._current_batch_size,
            processing_time=processing_time,
            memory_usage=memory_usage,
            model_used=model_used,
            feature_extraction_time=feature_extraction_time,
            model_training_time=model_training_time,
            prediction_time=prediction_time,
            fallback_rate=fallback_rate
        )
        self.all_metrics.append(metrics)

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of all recorded performance metrics."""
        if not self.all_metrics:
            return {"message": "No performance data recorded."}

        total_tickers = sum(m.batch_size for m in self.all_metrics)
        total_time = sum(m.processing_time for m in self.all_metrics)
        avg_time_per_ticker = total_time / total_tickers if total_tickers > 0 else 0
        avg_memory = np.mean([m.memory_usage for m in self.all_metrics])

        return {
            "total_batches": len(self.all_metrics),
            "total_tickers_processed": total_tickers,
            "total_processing_time": total_time,
            "average_time_per_ticker": avg_time_per_ticker,
            "average_batch_processing_time": np.mean([m.processing_time for m in self.all_metrics]),
            "average_memory_usage_mb": avg_memory,
            "metrics_per_batch": self.all_metrics
        }
