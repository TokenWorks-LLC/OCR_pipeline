"""
Structured logging schema for OCR pipeline metrics and analytics.
Provides consistent logging format for easy metrics extraction and plotting.
"""
import logging
import json
import time
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Define structured log event types
LOG_EVENTS = {
    'PIPELINE_START': 'pipeline.start',
    'PIPELINE_END': 'pipeline.end',
    'PAGE_START': 'page.start',
    'PAGE_END': 'page.end',
    'OCR_START': 'ocr.start',
    'OCR_END': 'ocr.end',
    'LLM_START': 'llm.start',
    'LLM_END': 'llm.end',
    'PREPROCESSING_START': 'preprocessing.start',
    'PREPROCESSING_END': 'preprocessing.end',
    'READING_ORDER_START': 'reading_order.start',
    'READING_ORDER_END': 'reading_order.end',
    'LANGUAGE_DETECTION_START': 'language_detection.start',
    'LANGUAGE_DETECTION_END': 'language_detection.end',
    'ERROR': 'error',
    'WARNING': 'warning',
    'PERFORMANCE': 'performance',
    'CACHE_HIT': 'cache.hit',
    'CACHE_MISS': 'cache.miss',
    'KILL_SWITCH': 'kill_switch',
    'METRIC': 'metric'
}

@dataclass
class LogEvent:
    """Structured log event."""
    event_type: str
    timestamp: float
    level: str
    message: str
    data: Dict[str, Any]
    session_id: str
    page_id: Optional[str] = None
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON logging."""
        return asdict(self)

class StructuredLogger:
    """Structured logger for OCR pipeline metrics."""
    
    def __init__(self, name: str, log_file: Optional[str] = None, session_id: Optional[str] = None):
        self.name = name
        self.session_id = session_id or f"session_{int(time.time())}"
        self.logger = logging.getLogger(name)
        
        # Set up structured JSON logging
        if log_file:
            self._setup_json_logging(log_file)
        
        # Performance tracking
        self.start_times: Dict[str, float] = {}
        self.event_buffer: List[LogEvent] = []
        
    def _setup_json_logging(self, log_file: str) -> None:
        """Set up JSON file logging."""
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Custom formatter for JSON output
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                if hasattr(record, 'structured_data'):
                    return json.dumps(record.structured_data)
                else:
                    # Fallback for regular log messages
                    return json.dumps({
                        'timestamp': record.created,
                        'level': record.levelname,
                        'message': record.getMessage(),
                        'logger': record.name
                    })
        
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_event(self, event_type: str, message: str, data: Dict[str, Any] = None, 
                  page_id: str = None, level: str = "INFO") -> None:
        """Log a structured event."""
        event = LogEvent(
            event_type=event_type,
            timestamp=time.time(),
            level=level,
            message=message,
            data=data or {},
            session_id=self.session_id,
            page_id=page_id
        )
        
        # Add to buffer
        self.event_buffer.append(event)
        
        # Log to structured logger
        log_record = self.logger.makeRecord(
            name=self.name,
            level=getattr(logging, level),
            fn='',
            lno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        log_record.structured_data = event.to_dict()
        self.logger.handle(log_record)
    
    def start_timing(self, operation: str, page_id: str = None) -> None:
        """Start timing an operation."""
        key = f"{page_id or 'global'}:{operation}"
        self.start_times[key] = time.time()
        
        self.log_event(
            event_type=f"{operation}.start",
            message=f"Started {operation}",
            data={'operation': operation},
            page_id=page_id
        )
    
    def end_timing(self, operation: str, page_id: str = None, extra_data: Dict[str, Any] = None) -> float:
        """End timing an operation and return duration."""
        key = f"{page_id or 'global'}:{operation}"
        start_time = self.start_times.get(key, time.time())
        duration_ms = (time.time() - start_time) * 1000
        
        data = {'operation': operation, 'duration_ms': duration_ms}
        if extra_data:
            data.update(extra_data)
        
        self.log_event(
            event_type=f"{operation}.end",
            message=f"Completed {operation} in {duration_ms:.1f}ms",
            data=data,
            page_id=page_id,
            duration_ms=duration_ms
        )
        
        # Clean up
        if key in self.start_times:
            del self.start_times[key]
        
        return duration_ms
    
    def log_performance(self, page_id: str, metrics: Dict[str, Any]) -> None:
        """Log performance metrics."""
        self.log_event(
            event_type=LOG_EVENTS['PERFORMANCE'],
            message=f"Performance metrics for {page_id}",
            data=metrics,
            page_id=page_id
        )
    
    def log_cache_event(self, hit: bool, text_preview: str, page_id: str = None) -> None:
        """Log cache hit/miss event."""
        event_type = LOG_EVENTS['CACHE_HIT'] if hit else LOG_EVENTS['CACHE_MISS']
        self.log_event(
            event_type=event_type,
            message=f"Cache {'hit' if hit else 'miss'}: {text_preview[:50]}...",
            data={'text_preview': text_preview[:100], 'cache_hit': hit},
            page_id=page_id
        )
    
    def log_kill_switch(self, switch_type: str, reason: str, page_id: str = None) -> None:
        """Log kill switch activation."""
        self.log_event(
            event_type=LOG_EVENTS['KILL_SWITCH'],
            message=f"Kill switch activated: {switch_type} - {reason}",
            data={'switch_type': switch_type, 'reason': reason},
            page_id=page_id,
            level="WARNING"
        )
    
    def log_metric(self, metric_name: str, value: Any, unit: str = None, 
                   page_id: str = None, tags: Dict[str, str] = None) -> None:
        """Log a specific metric."""
        data = {
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'tags': tags or {}
        }
        
        self.log_event(
            event_type=LOG_EVENTS['METRIC'],
            message=f"Metric {metric_name}: {value} {unit or ''}",
            data=data,
            page_id=page_id
        )
    
    def export_events(self, filepath: str) -> None:
        """Export all logged events to JSON file."""
        events_data = [event.to_dict() for event in self.event_buffer]
        
        with open(filepath, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'export_timestamp': time.time(),
                'total_events': len(events_data),
                'events': events_data
            }, f, indent=2)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current logging session."""
        events_by_type = {}
        for event in self.event_buffer:
            events_by_type[event.event_type] = events_by_type.get(event.event_type, 0) + 1
        
        performance_events = [e for e in self.event_buffer if e.event_type == LOG_EVENTS['PERFORMANCE']]
        
        return {
            'session_id': self.session_id,
            'total_events': len(self.event_buffer),
            'events_by_type': events_by_type,
            'performance_events_count': len(performance_events),
            'session_duration': max([e.timestamp for e in self.event_buffer], default=0) - 
                               min([e.timestamp for e in self.event_buffer], default=0) if self.event_buffer else 0
        }

class OCRMetricsCollector:
    """Specialized metrics collector for OCR pipeline."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.page_metrics: Dict[str, Dict[str, Any]] = {}
        
    def record_page_start(self, page_id: str, page_info: Dict[str, Any]) -> None:
        """Record page processing start."""
        self.logger.log_event(
            event_type=LOG_EVENTS['PAGE_START'],
            message=f"Started processing {page_id}",
            data=page_info,
            page_id=page_id
        )
        self.logger.start_timing('page_processing', page_id)
        
    def record_page_end(self, page_id: str, results: Dict[str, Any]) -> None:
        """Record page processing end."""
        duration = self.logger.end_timing('page_processing', page_id, results)
        
        # Calculate additional metrics
        lines_per_second = results.get('lines_processed', 0) / max(duration / 1000, 0.001)
        chars_per_second = results.get('characters_processed', 0) / max(duration / 1000, 0.001)
        
        metrics = {
            **results,
            'lines_per_second': lines_per_second,
            'characters_per_second': chars_per_second,
            'processing_duration_ms': duration
        }
        
        self.logger.log_performance(page_id, metrics)
        self.page_metrics[page_id] = metrics
        
    def record_ocr_metrics(self, page_id: str, ocr_results: Dict[str, Any]) -> None:
        """Record OCR-specific metrics."""
        self.logger.log_metric('ocr_confidence_avg', ocr_results.get('avg_confidence', 0), 
                              unit='confidence', page_id=page_id)
        self.logger.log_metric('ocr_elements_detected', ocr_results.get('elements_count', 0), 
                              unit='count', page_id=page_id)
        
    def record_llm_metrics(self, page_id: str, llm_results: Dict[str, Any]) -> None:
        """Record LLM correction metrics."""
        self.logger.log_metric('llm_corrections_made', llm_results.get('corrections_made', 0), 
                              unit='count', page_id=page_id)
        self.logger.log_metric('llm_processing_time', llm_results.get('processing_time', 0), 
                              unit='ms', page_id=page_id)
        
        cache_stats = llm_results.get('cache_stats', {})
        if cache_stats:
            self.logger.log_metric('llm_cache_hit_rate', cache_stats.get('hit_rate', 0), 
                                  unit='ratio', page_id=page_id)
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics across all pages."""
        if not self.page_metrics:
            return {}
        
        all_pages = list(self.page_metrics.values())
        
        return {
            'total_pages': len(all_pages),
            'avg_processing_time_ms': sum(p.get('processing_duration_ms', 0) for p in all_pages) / len(all_pages),
            'total_lines_processed': sum(p.get('lines_processed', 0) for p in all_pages),
            'total_characters_processed': sum(p.get('characters_processed', 0) for p in all_pages),
            'total_corrections_made': sum(p.get('corrections_made', 0) for p in all_pages),
            'avg_lines_per_second': sum(p.get('lines_per_second', 0) for p in all_pages) / len(all_pages),
            'avg_characters_per_second': sum(p.get('characters_per_second', 0) for p in all_pages) / len(all_pages)
        }

def setup_pipeline_logging(log_dir: str, session_id: str = None) -> tuple[StructuredLogger, OCRMetricsCollector]:
    """Set up structured logging for OCR pipeline."""
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True)
    
    # Create structured logger
    log_file = log_dir_path / f"pipeline_metrics_{session_id or int(time.time())}.jsonl"
    logger = StructuredLogger('ocr_pipeline', str(log_file), session_id)
    
    # Create metrics collector
    metrics_collector = OCRMetricsCollector(logger)
    
    # Log pipeline start
    logger.log_event(
        event_type=LOG_EVENTS['PIPELINE_START'],
        message="OCR Pipeline started",
        data={
            'python_version': sys.version,
            'log_file': str(log_file),
            'pid': os.getpid()
        }
    )
    
    return logger, metrics_collector

# Global logger instance
_pipeline_logger = None
_metrics_collector = None

def get_pipeline_logger() -> tuple[StructuredLogger, OCRMetricsCollector]:
    """Get global pipeline logger and metrics collector."""
    global _pipeline_logger, _metrics_collector
    if _pipeline_logger is None:
        _pipeline_logger, _metrics_collector = setup_pipeline_logging('./logs')
    return _pipeline_logger, _metrics_collector