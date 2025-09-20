"""
Comprehensive health check system for OCR pipeline components.
"""
import logging
import os
import time
import tempfile
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import cv2
import numpy as np

from config import (
    TESSERACT_LANGUAGES, TARGET_LANGUAGES, LLM_PROVIDER, LLM_MODEL,
    PDF_DPI, OUTPUT_DIRS, CONFIDENCE_THRESHOLD
)
from preflight import PreflightChecker, PreflightResult
from orientation import detect_orientation, apply_rotation
from pdf_ingest import validate_pdf, pdf_to_images, get_pdf_page_count
from ocr_utils import ocr_ensemble, quick_ocr_conf
from llama_correction import get_corrector

logger = logging.getLogger(__name__)


class HealthChecker:
    """Comprehensive health check system."""
    
    def __init__(self):
        self.results = {}
        self.test_assets = {}
        
    def run_full_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check covering all pipeline components."""
        logger.info("Starting comprehensive health check...")
        start_time = time.time()
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'summary': {},
            'test_results': {}
        }
        
        # 1. Preflight validation
        self._run_preflight_check()
        
        # 2. OCR engine health
        self._check_ocr_engines()
        
        # 3. PDF processing health
        self._check_pdf_processing()
        
        # 4. Orientation detection health
        self._check_orientation_detection()
        
        # 5. LLaMA integration health
        self._check_llama_integration()
        
        # 6. Output system health
        self._check_output_systems()
        
        # 7. End-to-end pipeline test
        self._run_e2e_pipeline_test()
        
        # 8. Performance benchmarks
        self._run_performance_benchmarks()
        
        # Generate summary
        total_time = time.time() - start_time
        self._generate_summary(total_time)
        
        logger.info(f"Health check complete in {total_time:.2f}s")
        return self.results
    
    def _run_preflight_check(self):
        """Run preflight validation."""
        logger.info("Running preflight validation...")
        
        checker = PreflightChecker()
        passed, preflight_results = checker.run_all_checks()
        
        # Convert to dict format
        preflight_dict = {}
        for result in preflight_results:
            preflight_dict[result.component] = {
                'status': result.status,
                'message': result.message,
                'details': result.details
            }
        
        self.results['checks']['preflight'] = {
            'passed': passed,
            'results': preflight_dict,
            'error_count': len([r for r in preflight_results if r.status == 'error']),
            'warning_count': len([r for r in preflight_results if r.status == 'warning'])
        }
    
    def _check_ocr_engines(self):
        """Test OCR engine functionality with synthetic data."""
        logger.info("Testing OCR engines...")
        
        # Create test images
        test_results = {}
        
        # Test 1: Simple English text
        english_img = self._create_test_image("Hello World OCR Test", "en")
        english_result = self._test_ocr_on_image(english_img, "English Simple")
        test_results['english_simple'] = english_result
        
        # Test 2: Multi-language text
        multilang_img = self._create_test_image(
            "Français: Bonjour\\nDeutsch: Hallo\\nTürkçe: Merhaba", "multi"
        )
        multilang_result = self._test_ocr_on_image(multilang_img, "Multilingual")
        test_results['multilingual'] = multilang_result
        
        # Test 3: Noisy/degraded text
        noisy_img = self._create_noisy_test_image("Degraded Text Quality", 0.3)
        noisy_result = self._test_ocr_on_image(noisy_img, "Noisy Text")
        test_results['noisy_text'] = noisy_result
        
        # Test 4: Two-column layout
        two_col_img = self._create_two_column_test_image()
        two_col_result = self._test_ocr_on_image(two_col_img, "Two Column")
        test_results['two_column'] = two_col_result
        
        self.results['checks']['ocr_engines'] = test_results
    
    def _check_pdf_processing(self):
        """Test PDF processing functionality."""
        logger.info("Testing PDF processing...")
        
        pdf_results = {}
        
        try:
            # Create a minimal test PDF
            test_pdf_path = self._create_test_pdf()
            
            # Test PDF validation
            is_valid, error = validate_pdf(test_pdf_path)
            pdf_results['validation'] = {
                'passed': is_valid,
                'error': error
            }
            
            # Test page count
            try:
                page_count = get_pdf_page_count(test_pdf_path)
                pdf_results['page_count'] = {
                    'passed': page_count > 0,
                    'count': page_count
                }
            except Exception as e:
                pdf_results['page_count'] = {
                    'passed': False,
                    'error': str(e)
                }
            
            # Test PDF to image conversion
            try:
                images = list(pdf_to_images(test_pdf_path, dpi=PDF_DPI))
                pdf_results['image_conversion'] = {
                    'passed': len(images) > 0,
                    'image_count': len(images)
                }
                
                if images:
                    # Test first image properties
                    first_img = images[0]
                    pdf_results['image_properties'] = {
                        'shape': first_img.shape,
                        'dtype': str(first_img.dtype),
                        'dpi_test': first_img.shape[0] > 100 and first_img.shape[1] > 100
                    }
            
            except Exception as e:
                pdf_results['image_conversion'] = {
                    'passed': False,
                    'error': str(e)
                }
            
            # Cleanup
            try:
                os.unlink(test_pdf_path)
            except:
                pass
                
        except Exception as e:
            pdf_results['creation_failed'] = {
                'passed': False,
                'error': str(e)
            }
        
        self.results['checks']['pdf_processing'] = pdf_results
    
    def _check_orientation_detection(self):
        """Test orientation detection functionality."""
        logger.info("Testing orientation detection...")
        
        orientation_results = {}
        
        # Test different orientations
        orientations_to_test = [0, 90, 180, 270]
        base_img = self._create_test_image("Orientation Test Text", "en")
        
        for target_angle in orientations_to_test:
            # Rotate the image
            if target_angle == 0:
                rotated_img = base_img
            else:
                rotated_img = self._rotate_image(base_img, target_angle)
            
            try:
                # Test orientation detection
                detected_angle, detection_info = detect_orientation(rotated_img)
                
                # Check if detection is correct (within 10 degrees)
                angle_diff = abs(detected_angle - target_angle) % 360
                angle_diff = min(angle_diff, 360 - angle_diff)
                
                is_correct = angle_diff <= 10
                
                orientation_results[f'rotation_{target_angle}'] = {
                    'target_angle': target_angle,
                    'detected_angle': detected_angle,
                    'method': detection_info.get('method', 'unknown'),
                    'confidence': detection_info.get('confidence', 0),
                    'correct': is_correct,
                    'angle_error': angle_diff
                }
                
            except Exception as e:
                orientation_results[f'rotation_{target_angle}'] = {
                    'target_angle': target_angle,
                    'error': str(e),
                    'correct': False
                }
        
        self.results['checks']['orientation_detection'] = orientation_results
    
    def _check_llama_integration(self):
        """Test LLaMA integration if enabled."""
        logger.info("Testing LLaMA integration...")
        
        llama_results = {}
        
        try:
            corrector = get_corrector()
            
            llama_results['availability'] = {
                'available': corrector.is_available(),
                'provider': corrector.provider,
                'model': corrector.model
            }
            
            if corrector.is_available():
                # Test correction functionality
                test_text = "Helo wrold with speling erors"
                
                try:
                    corrected = corrector.correct_single_line(test_text, 'en')
                    
                    llama_results['correction_test'] = {
                        'passed': corrected != test_text,  # Should be different
                        'original': test_text,
                        'corrected': corrected,
                        'changed': corrected != test_text
                    }
                    
                except Exception as e:
                    llama_results['correction_test'] = {
                        'passed': False,
                        'error': str(e)
                    }
            else:
                llama_results['correction_test'] = {
                    'skipped': 'LLaMA not available'
                }
                
        except Exception as e:
            llama_results['initialization_error'] = str(e)
        
        self.results['checks']['llama_integration'] = llama_results
    
    def _check_output_systems(self):
        """Test output directory and file generation systems."""
        logger.info("Testing output systems...")
        
        output_results = {}
        
        # Test directory creation and permissions
        for purpose, dirname in OUTPUT_DIRS.items():
            try:
                # Ensure directory exists
                os.makedirs(dirname, exist_ok=True)
                
                # Test write permissions
                test_file = os.path.join(dirname, f'health_test_{purpose}.tmp')
                with open(test_file, 'w') as f:
                    f.write(f'Health check test for {purpose}')
                
                # Test read permissions
                with open(test_file, 'r') as f:
                    content = f.read()
                
                # Cleanup
                os.remove(test_file)
                
                output_results[purpose] = {
                    'directory': dirname,
                    'writable': True,
                    'readable': True
                }
                
            except Exception as e:
                output_results[purpose] = {
                    'directory': dirname,
                    'error': str(e),
                    'writable': False
                }
        
        self.results['checks']['output_systems'] = output_results
    
    def _run_e2e_pipeline_test(self):
        """Run end-to-end pipeline test with synthetic data."""
        logger.info("Running end-to-end pipeline test...")
        
        e2e_results = {}
        
        try:
            # Create test image
            test_img = self._create_two_column_test_image()
            
            # Test full OCR pipeline
            start_time = time.time()
            
            # 1. Orientation detection
            angle, orient_info = detect_orientation(test_img)
            oriented_img = apply_rotation(test_img, angle)
            
            # 2. OCR processing
            ocr_lines = ocr_ensemble(oriented_img)
            
            # 3. LLaMA correction (if available)
            corrector = get_corrector()
            if corrector.is_available():
                corrected_lines, correction_stats = corrector.correct_lines(ocr_lines)
            else:
                corrected_lines = ocr_lines
                correction_stats = {'available': False}
            
            processing_time = time.time() - start_time
            
            e2e_results = {
                'passed': len(corrected_lines) > 0,
                'processing_time': processing_time,
                'orientation': {
                    'detected_angle': angle,
                    'method': orient_info.get('method', 'unknown')
                },
                'ocr': {
                    'lines_detected': len(ocr_lines),
                    'avg_confidence': np.mean([line.conf for line in ocr_lines]) if ocr_lines else 0
                },
                'correction': correction_stats,
                'total_output_lines': len(corrected_lines)
            }
            
        except Exception as e:
            e2e_results = {
                'passed': False,
                'error': str(e)
            }
        
        self.results['checks']['e2e_pipeline'] = e2e_results
    
    def _run_performance_benchmarks(self):
        """Run performance benchmarks."""
        logger.info("Running performance benchmarks...")
        
        perf_results = {}
        
        # Benchmark OCR performance on different image sizes
        sizes_to_test = [(800, 600), (1600, 1200), (2400, 1800)]
        
        for width, height in sizes_to_test:
            try:
                # Create test image of specified size
                test_img = self._create_test_image_size(width, height)
                
                # Benchmark OCR
                start_time = time.time()
                lines = ocr_ensemble(test_img)
                ocr_time = time.time() - start_time
                
                perf_results[f'{width}x{height}'] = {
                    'ocr_time': ocr_time,
                    'lines_detected': len(lines),
                    'pixels_per_second': (width * height) / ocr_time if ocr_time > 0 else 0
                }
                
            except Exception as e:
                perf_results[f'{width}x{height}'] = {
                    'error': str(e)
                }
        
        self.results['checks']['performance'] = perf_results
    
    def _generate_summary(self, total_time: float):
        """Generate health check summary."""
        summary = {
            'total_time': total_time,
            'overall_status': 'healthy',
            'critical_failures': [],
            'warnings': [],
            'passed_checks': 0,
            'total_checks': 0
        }
        
        # Analyze results
        for check_category, check_results in self.results['checks'].items():
            if isinstance(check_results, dict):
                if check_category == 'preflight':
                    summary['total_checks'] += len(check_results['results'])
                    if check_results['passed']:
                        summary['passed_checks'] += len(check_results['results']) - check_results['error_count']
                    
                    if check_results['error_count'] > 0:
                        summary['critical_failures'].append(f"Preflight: {check_results['error_count']} errors")
                        summary['overall_status'] = 'unhealthy'
                    
                    if check_results['warning_count'] > 0:
                        summary['warnings'].append(f"Preflight: {check_results['warning_count']} warnings")
                
                else:
                    # Count passed/failed in other categories
                    for test_name, test_result in check_results.items():
                        if isinstance(test_result, dict):
                            summary['total_checks'] += 1
                            
                            if test_result.get('passed', test_result.get('correct', False)):
                                summary['passed_checks'] += 1
                            else:
                                if test_result.get('error'):
                                    summary['critical_failures'].append(f"{check_category}.{test_name}: {test_result['error']}")
                                    if summary['overall_status'] == 'healthy':
                                        summary['overall_status'] = 'degraded'
        
        # Set final status
        if summary['critical_failures']:
            if len(summary['critical_failures']) > 3:
                summary['overall_status'] = 'unhealthy'
            else:
                summary['overall_status'] = 'degraded'
        elif summary['warnings']:
            if summary['overall_status'] == 'healthy':
                summary['overall_status'] = 'healthy_with_warnings'
        
        summary['success_rate'] = summary['passed_checks'] / summary['total_checks'] if summary['total_checks'] > 0 else 0
        
        self.results['summary'] = summary
    
    def _create_test_image(self, text: str, lang_hint: str = "en") -> np.ndarray:
        """Create a test image with specified text."""
        # Estimate image size based on text length
        char_width = 12
        line_height = 30
        margin = 20
        
        lines = text.split('\\n')
        max_line_length = max(len(line) for line in lines)
        
        width = max(300, max_line_length * char_width + 2 * margin)
        height = max(100, len(lines) * line_height + 2 * margin)
        
        # Create white background
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add text
        for i, line in enumerate(lines):
            y_pos = margin + (i + 1) * line_height
            cv2.putText(img, line, (margin, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return img
    
    def _create_test_image_size(self, width: int, height: int) -> np.ndarray:
        """Create a test image of specified size with sample text."""
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add some text at different positions
        texts = [
            "Performance Test Image",
            "Multiple lines of text",
            "For benchmarking OCR",
            "System performance"
        ]
        
        line_height = height // (len(texts) + 2)
        font_scale = min(2.0, width / 800)
        
        for i, text in enumerate(texts):
            y_pos = (i + 1) * line_height
            cv2.putText(img, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
        
        return img
    
    def _create_noisy_test_image(self, text: str, noise_level: float) -> np.ndarray:
        """Create a test image with noise."""
        img = self._create_test_image(text, "en")
        
        # Add noise
        noise = np.random.randint(0, int(255 * noise_level), img.shape, dtype=np.uint8)
        noisy_img = cv2.addWeighted(img, 1 - noise_level, noise, noise_level, 0)
        
        return noisy_img
    
    def _create_two_column_test_image(self) -> np.ndarray:
        """Create a two-column test image."""
        width, height = 800, 600
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Left column text
        left_texts = [
            "Left Column Title",
            "This is the first column",
            "of text in our test",
            "document for checking",
            "column detection."
        ]
        
        # Right column text
        right_texts = [
            "Right Column Title", 
            "This is the second column",
            "which should be detected",
            "as separate from the",
            "left column text."
        ]
        
        col_width = width // 2 - 40
        left_x = 20
        right_x = width // 2 + 20
        
        # Draw left column
        for i, text in enumerate(left_texts):
            y_pos = 50 + i * 30
            cv2.putText(img, text, (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw right column
        for i, text in enumerate(right_texts):
            y_pos = 50 + i * 30
            cv2.putText(img, text, (right_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return img
    
    def _create_test_pdf(self) -> str:
        """Create a simple test PDF file."""
        try:
            import fitz  # PyMuPDF
            
            # Create a simple PDF
            doc = fitz.open()
            page = doc.new_page()
            
            # Add some text
            text = "Test PDF Document\\nFor health check validation\\nPage 1"
            page.insert_text((50, 50), text, fontsize=12)
            
            # Save to temporary file
            temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            doc.save(temp_pdf.name)
            doc.close()
            
            return temp_pdf.name
            
        except ImportError:
            # Fallback: create empty file (will fail validation as expected)
            temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            temp_pdf.write(b'%PDF-1.4\\nfake pdf content')
            temp_pdf.close()
            return temp_pdf.name
    
    def _rotate_image(self, img: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by specified angle."""
        if angle == 0:
            return img
        elif angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # Custom angle
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(img, matrix, (w, h))
    
    def _test_ocr_on_image(self, img: np.ndarray, test_name: str) -> Dict[str, Any]:
        """Test OCR on a specific image."""
        try:
            start_time = time.time()
            lines = ocr_ensemble(img)
            ocr_time = time.time() - start_time
            
            # Analyze results
            if lines:
                confidences = [line.conf for line in lines]
                avg_conf = np.mean(confidences)
                min_conf = np.min(confidences)
                max_conf = np.max(confidences)
                total_text = ' '.join([line.text for line in lines])
            else:
                avg_conf = min_conf = max_conf = 0.0
                total_text = ""
            
            return {
                'passed': len(lines) > 0,
                'processing_time': ocr_time,
                'lines_detected': len(lines),
                'avg_confidence': avg_conf,
                'min_confidence': min_conf,
                'max_confidence': max_conf,
                'total_characters': len(total_text),
                'sample_text': total_text[:100] + ('...' if len(total_text) > 100 else '')
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }


def run_health_check() -> Dict[str, Any]:
    """Run comprehensive health check."""
    checker = HealthChecker()
    return checker.run_full_health_check()


def print_health_report(health_results: Dict[str, Any]):
    """Print formatted health check report."""
    print("\\n" + "="*70)
    print("OCR PIPELINE HEALTH CHECK REPORT")
    print("="*70)
    
    summary = health_results.get('summary', {})
    
    # Overall status
    status = summary.get('overall_status', 'unknown')
    print(f"\nOVERALL STATUS: {status.upper()}")
    print(f"   Success Rate: {summary.get('success_rate', 0)*100:.1f}% ({summary.get('passed_checks', 0)}/{summary.get('total_checks', 0)} checks passed)")
    print(f"   Total Time: {summary.get('total_time', 0):.2f}s")
    
    # Critical failures
    failures = summary.get('critical_failures', [])
    if failures:
        print(f"\nCRITICAL ISSUES ({len(failures)}):")
        for failure in failures:
            print(f"   • {failure}")
    
    # Warnings
    warnings = summary.get('warnings', [])
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"   • {warning}")
    
    # Key metrics
    checks = health_results.get('checks', {})
    
    print("\\n" + "-"*70)
    print("COMPONENT STATUS:")
    
    # Preflight
    preflight = checks.get('preflight', {})
    if preflight.get('passed'):
        print("   Preflight: All dependencies OK")
    else:
        print(f"   Preflight: {preflight.get('error_count', 0)} errors, {preflight.get('warning_count', 0)} warnings")
    
    # OCR Engines
    ocr = checks.get('ocr_engines', {})
    if ocr:
        passed_tests = len([t for t in ocr.values() if isinstance(t, dict) and t.get('passed')])
        total_tests = len([t for t in ocr.values() if isinstance(t, dict)])
        if passed_tests == total_tests:
            print(f"   OCR Engines: {passed_tests}/{total_tests} tests passed")
        else:
            print(f"   🟡 OCR Engines: {passed_tests}/{total_tests} tests passed")
    
    # PDF Processing
    pdf = checks.get('pdf_processing', {})
    if pdf:
        pdf_tests = [k for k, v in pdf.items() if isinstance(v, dict) and v.get('passed')]
        if len(pdf_tests) > 0:
            print(f"   PDF Processing: {len(pdf_tests)} tests passed")
        else:
            print("   ❌ PDF Processing: Tests failed")
    
    # LLaMA Integration
    llama = checks.get('llama_integration', {})
    if llama and llama.get('availability', {}).get('available'):
        if llama.get('correction_test', {}).get('passed'):
            print("   LLaMA Integration: Available and functional")
        else:
            print("   🟡 LLaMA Integration: Available but test failed")
    else:
        print("   ⚪ LLaMA Integration: Not available/disabled")
    
    # E2E Pipeline
    e2e = checks.get('e2e_pipeline', {})
    if e2e and e2e.get('passed'):
        print(f"   End-to-End Pipeline: Working ({e2e.get('processing_time', 0):.2f}s)")
    else:
        print("   ❌ End-to-End Pipeline: Failed")
    
    # Performance
    perf = checks.get('performance', {})
    if perf:
        avg_time = np.mean([v.get('ocr_time', 0) for v in perf.values() if isinstance(v, dict) and 'ocr_time' in v])
        print(f"   📊 Performance: Avg OCR time {avg_time:.2f}s")
    
    print("="*70 + "\\n")
