"""
Cloud configuration management with environment variable support.
"""
import os
import json
from typing import Dict, Any

class CloudConfig:
    """Configuration for cloud deployment."""

    def __init__(self):
        # AWS Configuration
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.s3_bucket = os.getenv('S3_BUCKET', 'your-ocr-bucket')

        # Batch Queues
        self.standard_queue = os.getenv('STANDARD_QUEUE', 'ocr-standard-queue')
        self.high_memory_queue = os.getenv('HIGH_MEMORY_QUEUE', 'ocr-high-memory-queue')

        # Escalation Settings
        self.escalation_thresholds = {
            'min_confidence': float(os.getenv('ESCALATION_CONFIDENCE', '0.8')),
            'max_pages': int(os.getenv('ESCALATION_MAX_PAGES', '50')),
            'max_processing_time': int(os.getenv('ESCALATION_MAX_TIME', '600'))
        }

        # Retry Settings
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))

        # Notifications (placeholders for founder approval)
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL', 'PLACEHOLDER_DISCORD_WEBHOOK')
        self.email_smtp_server = os.getenv('EMAIL_SMTP_SERVER', 'PLACEHOLDER_SMTP_SERVER')
        self.email_recipients = os.getenv('EMAIL_RECIPIENTS', 'PLACEHOLDER_EMAIL_RECIPIENTS').split(',')

        # Cost optimization settings
        self.cost_optimization_mode = os.getenv('COST_OPTIMIZATION', 'false').lower() == 'true'
        if self.cost_optimization_mode:
            # Reduce resource allocation for cost savings
            self.escalation_thresholds['max_pages'] = 25  # Reduce threshold
            self.max_retries = 2  # Reduce retries

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return {
            'aws_region': self.aws_region,
            's3_bucket': self.s3_bucket,
            'standard_queue': self.standard_queue,
            'high_memory_queue': self.high_memory_queue,
            'escalation_thresholds': self.escalation_thresholds,
            'max_retries': self.max_retries,
            'cost_optimization_mode': self.cost_optimization_mode
        }
