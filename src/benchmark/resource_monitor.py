"""
System and process resource monitor using psutil.
Measures process RSS memory, peak usage, and CPU times.
"""
import os
import psutil
from typing import Dict, Any

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_rss = 0
        self.peak_rss = 0

    def start(self) -> None:
        """Captures initial baseline memory."""
        self.initial_rss = self.process.memory_info().rss
        self.peak_rss = self.initial_rss

    def sample(self) -> None:
        """Updates peak memory tracking."""
        current_rss = self.process.memory_info().rss
        if current_rss > self.peak_rss:
            self.peak_rss = current_rss

    def get_summary(self) -> Dict[str, float]:
        """Returns memory metrics in Megabytes (MB)."""
        current_rss = self.process.memory_info().rss
        if current_rss > self.peak_rss:
            self.peak_rss = current_rss
        return {
            "initial_rss_mb": round(self.initial_rss / (1024 * 1024), 2),
            "peak_rss_mb": round(self.peak_rss / (1024 * 1024), 2),
            "delta_rss_mb": round(max(0, self.peak_rss - self.initial_rss) / (1024 * 1024), 2)
        }
