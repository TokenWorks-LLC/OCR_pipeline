"""
Supervisor node for handling failed and heavy job escalation.
Automatically resubmits jobs to higher-resource queues and manages retries.
"""
import json
import time
import boto3
from typing import Dict, Any, List
from datetime import datetime, timedelta

class EscalationManager:
    """Manages job escalation and supervisor operations."""

    def __init__(self, config):
        self.config = config
        self.s3_client = boto3.client('s3', region_name=config.aws_region)
        self.batch_client = boto3.client('batch', region_name=config.aws_region)

        # Tracking for notifications
        self.stats = {
            'processed_this_run': 0,
            'escalated_this_run': 0,
            'failed_this_run': 0,
            'total_pending': 0,
            'total_manual_review': 0
        }

    def scan_and_process_escalations(self) -> Dict[str, Any]:
        """Main supervisor function - scan S3 folders and process escalations."""
        print(f"🔍 Scanning escalation folders in s3://{self.config.s3_bucket}")

        # Reset stats for this run
        self.stats = {k: 0 for k in self.stats.keys()}

        # Scan failed/ and heavy/ folders
        failed_jobs = self._scan_folder('failed/')
        heavy_jobs = self._scan_folder('heavy/')

        all_escalated_jobs = failed_jobs + heavy_jobs
        self.stats['total_pending'] = len(all_escalated_jobs)

        processed_jobs = []

        for job_data in all_escalated_jobs:
            result = self._process_escalated_job(job_data)
            processed_jobs.append(result)
            self.stats['processed_this_run'] += 1

            if result['action'] == 'escalated':
                self.stats['escalated_this_run'] += 1
            elif result['action'] == 'failed':
                self.stats['failed_this_run'] += 1

        # Send notifications if configured
        self._send_notifications()

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'stats': self.stats,
            'processed_jobs': processed_jobs
        }

    def _scan_folder(self, folder_prefix: str) -> List[Dict[str, Any]]:
        """Scan S3 folder for escalated jobs."""
        jobs = []

        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.config.s3_bucket,
                Prefix=folder_prefix
            )

            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['Key'].endswith('.json'):
                            job_data = self._load_job_from_s3(obj['Key'])
                            if job_data:
                                jobs.append(job_data)

        except Exception as e:
            print(f"❌ Error scanning {folder_prefix}: {e}")

        return jobs

    def _load_job_from_s3(self, key: str) -> Optional[Dict[str, Any]]:
        """Load job data from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.config.s3_bucket, Key=key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            print(f"❌ Error loading {key}: {e}")
            return None

    def _process_escalated_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single escalated job."""
        job_id = job_data['job_id']
        attempt_count = job_data.get('attempt_count', 1)

        print(f"🔄 Processing escalated job {job_id} (attempt {attempt_count})")

        # Check if we've exceeded max retries
        if attempt_count >= self.config.max_retries:
            return self._mark_for_manual_review(job_data)

        # Resubmit to high-memory queue
        success = self._resubmit_to_high_memory_queue(job_data)

        if success:
            # Move from escalation folder to processing
            self._move_job_to_processing(job_data)
            return {
                'job_id': job_id,
                'action': 'escalated',
                'attempt_count': attempt_count + 1,
                'target_queue': self.config.high_memory_queue
            }
        else:
            return {
                'job_id': job_id,
                'action': 'failed',
                'error': 'resubmission_failed',
                'attempt_count': attempt_count
            }

    def _resubmit_to_high_memory_queue(self, job_data: Dict[str, Any]) -> bool:
        """Resubmit job to high-memory queue."""
        try:
            # Prepare job data for resubmission
            resubmit_data = {
                's3_input_path': job_data['original_s3_path'],
                'job_id': f"{job_data['job_id']}_attempt_{job_data.get('attempt_count', 1) + 1}",
                'attempt_count': job_data.get('attempt_count', 1) + 1,
                'escalation_reason': job_data.get('escalation_reason'),
                'original_job_data': job_data
            }

            # Submit to AWS Batch
            response = self.batch_client.submit_job(
                jobName=f"ocr-escalated-{resubmit_data['job_id']}",
                jobQueue=self.config.high_memory_queue,
                jobDefinition="ocr-processing-job",  # Use your job definition name
                parameters={
                    'job_data': json.dumps(resubmit_data)
                }
            )

            print(f"✅ Resubmitted job {resubmit_data['job_id']} to {self.config.high_memory_queue}")
            return True

        except Exception as e:
            print(f"❌ Failed to resubmit job {job_data['job_id']}: {e}")
            return False

    def _mark_for_manual_review(self, job_data: Dict[str, Any]):
        """Mark job as requiring manual review."""
        job_id = job_data['job_id']

        # Move to manual review folder
        manual_review_key = f"manual_review/{job_id}.json"

        try:
            self.s3_client.copy_object(
                CopySource={'Bucket': self.config.s3_bucket, 'Key': f"failed/{job_id}.json"},
                Bucket=self.config.s3_bucket,
                Key=manual_review_key
            )

            # Update job data
            job_data['manual_review'] = True
            job_data['manual_review_timestamp'] = datetime.utcnow().isoformat()

            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=manual_review_key,
                Body=json.dumps(job_data, indent=2),
                ContentType='application/json'
            )

            print(f"📋 Marked job {job_id} for manual review")

        except Exception as e:
            print(f"❌ Error marking job {job_id} for manual review: {e}")

        return {
            'job_id': job_id,
            'action': 'manual_review',
            'attempt_count': job_data.get('attempt_count', 1)
        }

    def _move_job_to_processing(self, job_data: Dict[str, Any]):
        """Move job from escalation folder to processing tracking."""
        job_id = job_data['job_id']

        # Copy to processing folder and delete from escalation
        processing_key = f"processing/{job_id}.json"

        try:
            # Determine source folder
            escalation_reason = job_data.get('escalation_reason', '')
            source_folder = 'heavy/' if ('large_document' in escalation_reason or 'timeout' in escalation_reason) else 'failed/'
            source_key = f"{source_folder}{job_id}.json"

            self.s3_client.copy_object(
                CopySource={'Bucket': self.config.s3_bucket, 'Key': source_key},
                Bucket=self.config.s3_bucket,
                Key=processing_key
            )

            # Delete from escalation folder
            self.s3_client.delete_object(Bucket=self.config.s3_bucket, Key=source_key)

        except Exception as e:
            print(f"⚠️ Error moving job {job_id} to processing: {e}")

    def _send_notifications(self):
        """Send notifications about escalation status."""
        # Discord notification
        if self.config.discord_webhook and self.config.discord_webhook != 'PLACEHOLDER_DISCORD_WEBHOOK':
            self._send_discord_notification()

        # Email notification
        if self.config.email_smtp_server and self.config.email_smtp_server != 'PLACEHOLDER_SMTP_SERVER':
            self._send_email_notification()

    def _send_discord_notification(self):
        """Send Discord webhook notification."""
        import requests

        embed = {
            "title": "OCR Pipeline Escalation Report",
            "description": f"Supervisor run completed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "color": 16776960,  # Yellow for warnings
            "fields": [
                {"name": "Jobs Processed", "value": str(self.stats['processed_this_run']), "inline": True},
                {"name": "Escalated", "value": str(self.stats['escalated_this_run']), "inline": True},
                {"name": "Failed", "value": str(self.stats['failed_this_run']), "inline": True},
                {"name": "Pending Review", "value": str(self.stats['total_pending']), "inline": True}
            ]
        }

        payload = {"embeds": [embed]}

        try:
            requests.post(self.config.discord_webhook, json=payload, timeout=10)
        except Exception as e:
            print(f"⚠️ Discord notification failed: {e}")

    def _send_email_notification(self):
        """Send email notification."""
        import smtplib
        from email.mime.text import MIMEText

        subject = "OCR Pipeline Escalation Report"
        body = f"""
OCR Pipeline Escalation Report
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

Statistics:
- Jobs Processed: {self.stats['processed_this_run']}
- Escalated: {self.stats['escalated_this_run']}
- Failed: {self.stats['failed_this_run']}
- Pending Review: {self.stats['total_pending']}

This is an automated notification from the OCR pipeline supervisor.
        """

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = 'ocr-pipeline@tokenworks.com'
        msg['To'] = ', '.join(self.config.email_recipients)

        try:
            server = smtplib.SMTP(self.config.email_smtp_server)
            server.sendmail(msg['From'], self.config.email_recipients, msg.as_string())
            server.quit()
        except Exception as e:
            print(f"⚠️ Email notification failed: {e}")
