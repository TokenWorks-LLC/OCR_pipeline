"""
Main worker node for processing OCR jobs.
Handles file processing, escalation detection, and result upload.
"""
import sys
import os
import json
import time
import boto3
from pathlib import Path
from typing import Dict, Any, Optional

# Add paths to access the main OCR pipeline
# Adjust these paths based on your actual directory structure
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Up to OCR_pipeline root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))  # To src directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'production'))  # To production directory

from cloud.config import CloudConfig
from production.comprehensive_pipeline import ComprehensivePipeline, PipelineConfig

class JobProcessor:
    """Processes individual OCR jobs with intelligent escalation."""

    def __init__(self):
        self.config = CloudConfig()
        self.s3_client = boto3.client('s3', region_name=self.config.aws_region)
        self.batch_client = boto3.client('batch', region_name=self.config.aws_region)

    def process_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single OCR job."""
        start_time = time.time()
        s3_input_path = job_data['s3_input_path']
        job_id = job_data.get('job_id', f"job_{int(start_time)}")

        try:
            # Download file from S3
            local_file_path = self._download_from_s3(s3_input_path)

            # Process with existing pipeline
            result = self._process_file(local_file_path)

            # Check if escalation is needed
            escalation_reason = self._check_escalation_needed(result)

            if escalation_reason:
                # Move to failed/heavy folder for supervisor
                self._escalate_job(job_id, s3_input_path, result, escalation_reason, job_data)
                result['escalated'] = True
                result['escalation_reason'] = escalation_reason
            else:
                # Upload successful results to S3
                self._upload_results(job_id, result)
                result['escalated'] = False

            # Calculate processing time
            result['processing_time'] = time.time() - start_time
            result['job_id'] = job_id

            return result

        except Exception as e:
            error_result = {
                'error': str(e),
                'job_id': job_id,
                'processing_time': time.time() - start_time,
                'escalated': True,
                'escalation_reason': 'processing_error'
            }
            self._escalate_job(job_id, s3_input_path, error_result, 'processing_error', job_data)
            return error_result

    def _download_from_s3(self, s3_path: str) -> str:
        """Download file from S3 to local temp storage."""
        # Extract bucket and key from s3://bucket/key format
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]
        bucket, key = s3_path.split('/', 1)

        # Create temp directory if needed
        temp_dir = Path('/tmp/ocr_input')
        temp_dir.mkdir(exist_ok=True)

        local_path = temp_dir / Path(key).name
        self.s3_client.download_file(bucket, key, str(local_path))

        return str(local_path)

    def _process_file(self, file_path: str) -> Dict[str, Any]:
        """Process file using existing pipeline."""
        # Create pipeline config from existing config.json
        config_path = Path(__file__).parent.parent.parent / 'config.json'
        with open(config_path) as f:
            base_config = json.load(f)

        pipeline_config = PipelineConfig(
            llm_provider=base_config.get('llm', {}).get('provider', 'ollama'),
            llm_model=base_config.get('llm', {}).get('model', 'mistral:latest'),
            dpi=base_config.get('ocr', {}).get('dpi', 300),
            enable_akkadian_extraction=base_config.get('akkadian', {}).get('enable_extraction', True)
        )

        pipeline = ComprehensivePipeline(pipeline_config)

        if file_path.lower().endswith('.pdf'):
            result = pipeline.process_pdf(file_path, '/tmp/ocr_output')
        else:
            result = pipeline.process_image(file_path, '/tmp/ocr_output')

        return result

    def _check_escalation_needed(self, result: Dict[str, Any]) -> Optional[str]:
        """Check if job needs escalation based on configured thresholds."""
        # Check confidence
        avg_confidence = result.get('avg_confidence', 1.0)
        if avg_confidence < self.config.escalation_thresholds['min_confidence']:
            return f"low_confidence_{avg_confidence:.2f}"

        # Check page count
        pages_processed = result.get('pages_processed', 0)
        if pages_processed > self.config.escalation_thresholds['max_pages']:
            return f"large_document_{pages_processed}_pages"

        # Check processing time
        processing_time = result.get('processing_time', 0)
        if processing_time > self.config.escalation_thresholds['max_processing_time']:
            return f"timeout_{processing_time:.0f}s"

        return None

    def _escalate_job(self, job_id: str, s3_input_path: str, result: Dict[str, Any],
                     reason: str, original_job_data: Dict[str, Any]):
        """Move failed job to escalation folders for supervisor."""
        escalation_data = {
            'job_id': job_id,
            'original_s3_path': s3_input_path,
            'escalation_reason': reason,
            'result': result,
            'original_job_data': original_job_data,
            'timestamp': time.time(),
            'attempt_count': original_job_data.get('attempt_count', 0) + 1
        }

        # Determine folder based on reason
        if 'large_document' in reason or 'timeout' in reason:
            folder = 'heavy'
        else:
            folder = 'failed'

        # Upload escalation data to S3
        escalation_key = f"{folder}/{job_id}.json"
        self.s3_client.put_object(
            Bucket=self.config.s3_bucket,
            Key=escalation_key,
            Body=json.dumps(escalation_data, indent=2),
            ContentType='application/json'
        )

    def _upload_results(self, job_id: str, result: Dict[str, Any]):
        """Upload successful results to S3."""
        result_key = f"results/{job_id}.json"
        self.s3_client.put_object(
            Bucket=self.config.s3_bucket,
            Key=result_key,
            Body=json.dumps(result, indent=2),
            ContentType='application/json'
        )

def main():
    """Main entry point for AWS Batch job."""
    import argparse

    parser = argparse.ArgumentParser(description='OCR Job Processor')
    parser.add_argument('--job-data', required=True, help='JSON job data')
    args = parser.parse_args()

    job_data = json.loads(args.job_data)
    processor = JobProcessor()
    result = processor.process_job(job_data)

    # Output result for AWS Batch
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
