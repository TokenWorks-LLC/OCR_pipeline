"""
AWS Lambda function for periodic supervisor execution.
Deploy this as a scheduled Lambda to run every 5-15 minutes.
"""
import json
import os
import sys
from pathlib import Path

# Add cloud module to path (package with deployment)
sys.path.insert(0, '/opt')

from cloud.config import CloudConfig
from cloud.supervisor.escalation_manager import EscalationManager

def lambda_handler(event, context):
    """Lambda handler for supervisor execution."""
    print("🚀 Starting OCR Supervisor Lambda")

    try:
        config = CloudConfig()
        manager = EscalationManager(config)
        result = manager.scan_and_process_escalations()

        # Log to CloudWatch (automatically handled) and S3
        import boto3
        s3_client = boto3.client('s3', region_name=config.aws_region)

        log_key = f"supervisor_logs/lambda_{result['timestamp'].replace(':', '-')}.json"
        s3_client.put_object(
            Bucket=config.s3_bucket,
            Key=log_key,
            Body=json.dumps(result, indent=2),
            ContentType='application/json'
        )

        return {
            'statusCode': 200,
            'body': json.dumps(result['stats'])
        }

    except Exception as e:
        print(f"❌ Supervisor Lambda failed: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
