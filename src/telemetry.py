"""
Telemetry and performance monitoring for OCR pipeline.
Provides per-page timing, global kill-switches, and metrics collection.
"""
import time
import logging
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PageTiming:
    """Timing metrics for a single page."""
    page_id: str
    total_time: float = 0.0
    ocr_time: float = 0.0
    preprocessing_time: float = 0.0
    reading_order_time: float = 0.0
    language_detection_time: float = 0.0
    llm_correction_time: float = 0.0
    post_processing_time: float = 0.0
    lines_processed: int = 0
    characters_processed: int = 0
    corrections_made: int = 0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def lines_per_second(self) -> float:
        """Calculate lines processed per second."""
        return self.lines_processed / max(0.001, self.total_time)
    
    @property
    def characters_per_second(self) -> float:
        """Calculate characters processed per second."""
        return self.characters_processed / max(0.001, self.total_time)
    
    @property
    def ms_per_page(self) -> float:
        """Get total time in milliseconds."""
        return self.total_time * 1000

class GlobalKillSwitch:
    """Global kill-switch for pipeline operations."""
    
    def __init__(self):
        self._killed = threading.Event()
        self._llm_disabled = threading.Event()
        self._ocr_disabled = threading.Event()
        self._lock = threading.Lock()
        
        # Performance thresholds
        self.max_page_time = 300.0  # 5 minutes max per page
        self.max_llm_time = 60.0    # 1 minute max for LLM per page
        self.min_lines_per_second = 0.1  # Minimum processing speed
        
    def kill_all(self) -> None:
        """Stop all pipeline operations."""
        with self._lock:
            self._killed.set()
            logger.warning("Global kill switch activated - stopping all operations")
    
    def disable_llm(self) -> None:
        """Disable LLM corrections only."""
        with self._lock:
            self._llm_disabled.set()
            logger.warning("LLM disabled via kill switch")
    
    def disable_ocr(self) -> None:
        """Disable OCR processing only."""
        with self._lock:
            self._ocr_disabled.set()
            logger.warning("OCR disabled via kill switch")
    
    def enable_llm(self) -> None:
        """Re-enable LLM corrections."""
        with self._lock:
            self._llm_disabled.clear()
            logger.info("LLM re-enabled")
    
    def enable_ocr(self) -> None:
        """Re-enable OCR processing."""
        with self._lock:
            self._ocr_disabled.clear()
            logger.info("OCR re-enabled")
    
    def reset(self) -> None:
        """Reset all kill switches."""
        with self._lock:
            self._killed.clear()
            self._llm_disabled.clear()
            self._ocr_disabled.clear()
            logger.info("All kill switches reset")
    
    @property
    def is_killed(self) -> bool:
        """Check if global kill switch is active."""
        return self._killed.is_set()
    
    @property
    def is_llm_disabled(self) -> bool:
        """Check if LLM is disabled."""
        return self._llm_disabled.is_set()
    
    @property
    def is_ocr_disabled(self) -> bool:
        """Check if OCR is disabled."""
        return self._ocr_disabled.is_set()
    
    def check_page_timeout(self, start_time: float, page_id: str = "") -> bool:
        """Check if page processing has timed out."""
        elapsed = time.time() - start_time
        if elapsed > self.max_page_time:
            logger.error(f"Page {page_id} exceeded max time limit ({elapsed:.1f}s)")
            self.kill_all()
            return True
        return False
    
    def check_llm_timeout(self, start_time: float, page_id: str = "") -> bool:
        """Check if LLM processing has timed out."""
        elapsed = time.time() - start_time
        if elapsed > self.max_llm_time:
            logger.warning(f"LLM timeout on page {page_id} ({elapsed:.1f}s) - disabling LLM")
            self.disable_llm()
            return True
        return False

class TelemetryCollector:
    """Centralized telemetry and performance metrics collector."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.page_timings: List[PageTiming] = []
        self.global_stats: Dict[str, Any] = defaultdict(list)
        self.kill_switch = GlobalKillSwitch()
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        # Recent performance window for adaptive thresholds
        self.recent_timings = deque(maxlen=50)
        
    def start_page_timing(self, page_id: str) -> PageTiming:
        """Start timing for a new page."""
        return PageTiming(page_id=page_id)
    
    def record_page_timing(self, timing: PageTiming) -> None:
        """Record completed page timing."""
        with self._lock:
            self.page_timings.append(timing)
            self.recent_timings.append(timing)
            
            # Trim history if needed
            if len(self.page_timings) > self.max_history:
                self.page_timings = self.page_timings[-self.max_history:]
            
            # Check for performance issues
            self._check_performance_alerts(timing)
            
            logger.info(f"Page {timing.page_id} completed: {timing.ms_per_page:.1f}ms, "
                       f"{timing.lines_per_second:.2f} lines/sec, "
                       f"{timing.corrections_made} corrections")
    
    def _check_performance_alerts(self, timing: PageTiming) -> None:
        """Check for performance issues and trigger alerts."""
        # Check processing speed
        if timing.lines_per_second < self.kill_switch.min_lines_per_second:
            logger.warning(f"Slow processing detected on {timing.page_id}: "
                          f"{timing.lines_per_second:.3f} lines/sec")
        
        # Check LLM time ratio
        if timing.total_time > 0:
            llm_ratio = timing.llm_correction_time / timing.total_time
            if llm_ratio > 0.8:  # LLM taking more than 80% of time
                logger.warning(f"LLM bottleneck on {timing.page_id}: "
                              f"{llm_ratio:.1%} of processing time")
        
        # Adaptive threshold adjustment
        if len(self.recent_timings) >= 10:
            recent_times = [t.total_time for t in self.recent_timings]
            avg_time = statistics.mean(recent_times)
            std_time = statistics.stdev(recent_times) if len(recent_times) > 1 else 0
            
            # If this page is significantly slower than recent average
            if timing.total_time > avg_time + 3 * std_time:
                logger.warning(f"Page {timing.page_id} significantly slower than recent average: "
                              f"{timing.total_time:.1f}s vs {avg_time:.1f}s avg")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            if not self.page_timings:
                return {"status": "no_data"}
            
            # Calculate aggregate metrics
            total_pages = len(self.page_timings)
            total_time = sum(t.total_time for t in self.page_timings)
            total_lines = sum(t.lines_processed for t in self.page_timings)
            total_chars = sum(t.characters_processed for t in self.page_timings)
            total_corrections = sum(t.corrections_made for t in self.page_timings)
            
            # Performance metrics
            times = [t.total_time for t in self.page_timings]
            lines_per_sec = [t.lines_per_second for t in self.page_timings]
            chars_per_sec = [t.characters_per_second for t in self.page_timings]
            
            # Recent performance (last 10 pages)
            recent = self.page_timings[-10:] if len(self.page_timings) >= 10 else self.page_timings
            recent_avg_time = statistics.mean(t.total_time for t in recent)
            recent_avg_lines_sec = statistics.mean(t.lines_per_second for t in recent)
            
            return {
                "total_pages": total_pages,
                "total_processing_time": total_time,
                "total_pipeline_time": time.time() - self.start_time,
                "total_lines": total_lines,
                "total_characters": total_chars,
                "total_corrections": total_corrections,
                "avg_time_per_page": statistics.mean(times),
                "median_time_per_page": statistics.median(times),
                "min_time_per_page": min(times),
                "max_time_per_page": max(times),
                "std_time_per_page": statistics.stdev(times) if len(times) > 1 else 0,
                "avg_lines_per_second": statistics.mean(lines_per_sec),
                "avg_characters_per_second": statistics.mean(chars_per_sec),
                "recent_avg_time": recent_avg_time,
                "recent_avg_lines_per_second": recent_avg_lines_sec,
                "kill_switch_status": {
                    "global_killed": self.kill_switch.is_killed,
                    "llm_disabled": self.kill_switch.is_llm_disabled,
                    "ocr_disabled": self.kill_switch.is_ocr_disabled
                },
                "performance_breakdown": {
                    "avg_ocr_time": statistics.mean(t.ocr_time for t in self.page_timings),
                    "avg_preprocessing_time": statistics.mean(t.preprocessing_time for t in self.page_timings),
                    "avg_reading_order_time": statistics.mean(t.reading_order_time for t in self.page_timings),
                    "avg_language_detection_time": statistics.mean(t.language_detection_time for t in self.page_timings),
                    "avg_llm_correction_time": statistics.mean(t.llm_correction_time for t in self.page_timings),
                    "avg_post_processing_time": statistics.mean(t.post_processing_time for t in self.page_timings)
                }
            }
    
    def export_metrics(self, filepath: str) -> None:
        """Export all metrics to JSON file."""
        with self._lock:
            metrics = {
                "summary": self.get_performance_summary(),
                "page_timings": [
                    {
                        "page_id": t.page_id,
                        "total_time": t.total_time,
                        "ocr_time": t.ocr_time,
                        "preprocessing_time": t.preprocessing_time,
                        "reading_order_time": t.reading_order_time,
                        "language_detection_time": t.language_detection_time,
                        "llm_correction_time": t.llm_correction_time,
                        "post_processing_time": t.post_processing_time,
                        "lines_processed": t.lines_processed,
                        "characters_processed": t.characters_processed,
                        "corrections_made": t.corrections_made,
                        "lines_per_second": t.lines_per_second,
                        "characters_per_second": t.characters_per_second,
                        "ms_per_page": t.ms_per_page,
                        "timestamp": t.timestamp
                    }
                    for t in self.page_timings
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Exported metrics to {filepath}")
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time statistics for monitoring dashboards."""
        with self._lock:
            if not self.recent_timings:
                return {"status": "no_recent_data"}
            
            recent_times = [t.total_time for t in self.recent_timings]
            recent_lines_sec = [t.lines_per_second for t in self.recent_timings]
            
            return {
                "pages_in_window": len(self.recent_timings),
                "avg_time_recent": statistics.mean(recent_times),
                "avg_lines_per_sec_recent": statistics.mean(recent_lines_sec),
                "last_page_time": recent_times[-1] if recent_times else 0,
                "last_page_lines_sec": recent_lines_sec[-1] if recent_lines_sec else 0,
                "pipeline_uptime": time.time() - self.start_time,
                "kill_switch_active": self.kill_switch.is_killed,
                "llm_active": not self.kill_switch.is_llm_disabled,
                "ocr_active": not self.kill_switch.is_ocr_disabled
            }

# Global telemetry instance
_telemetry_collector = None

def get_telemetry() -> TelemetryCollector:
    """Get global telemetry collector instance."""
    global _telemetry_collector
    if _telemetry_collector is None:
        _telemetry_collector = TelemetryCollector()
    return _telemetry_collector

def reset_telemetry() -> None:
    """Reset global telemetry collector."""
    global _telemetry_collector
    _telemetry_collector = TelemetryCollector()