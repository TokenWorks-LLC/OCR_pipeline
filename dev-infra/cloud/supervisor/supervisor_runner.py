"""
Supervisor runner script for periodic escalation checking.
Run this as a scheduled Lambda function or EC2 cron job.
"""
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cloud.config import CloudConfig
from cloud.supervisor.escalation_manager import EscalationManager

def main():
    """Main supervisor execution."""
    print("🚀 Starting OCR Pipeline Supervisor")

    config = CloudConfig()
    manager = EscalationManager(config)

    try:
        result = manager.scan_and_process_escalations()

        # Log results to S3 for dashboard
        import boto3
        s3_client = boto3.client('s3', region_name=config.aws_region)

        log_key = f"supervisor_logs/{result['timestamp'].replace(':', '-')}.json"
        s3_client.put_object(
            Bucket=config.s3_bucket,
            Key=log_key,
            Body=json.dumps(result, indent=2),
            ContentType='application/json'
        )

        print("✅ Supervisor run completed successfully")
        print(json.dumps(result['stats'], indent=2))

        return result

    except Exception as e:
        print(f"❌ Supervisor run failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

if __name__ == '__main__':
    main()
