"""
Enhanced preflight validation system for OCR pipeline.
Validates dependencies, models, and system readiness before processing.
"""
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from config import (
    TESSERACT_LANGUAGES, PDF_DPI, LLM_PROVIDER, LLM_MODEL, 
    LLM_BASE_URL, LLM_TIMEOUT, OUTPUT_DIRS
)

logger = logging.getLogger(__name__)


@dataclass
class PreflightResult:
    """Result of a preflight check."""
    component: str
    status: str  # 'ok', 'warning', 'error'
    message: str
    details: Optional[Dict[str, Any]] = None


class PreflightChecker:
    """Comprehensive preflight validation system."""
    
    def __init__(self):
        self.results: List[PreflightResult] = []
    
    def run_all_checks(self) -> Tuple[bool, List[PreflightResult]]:
        """Run all preflight checks."""
        self.results = []
        
        logger.info("Starting preflight validation...")
        
        # Core dependency checks
        self._check_opencv()
        self._check_tesseract()
        self._check_paddle()
        self._check_optional_ocr()
        
        # System checks
        self._check_output_directories()
        self._check_pdf_libraries()
        self._check_system_resources()
        
        # Model health checks
        self._check_llama_integration()
        
        # Performance validation
        self._check_ocr_performance()
        
        # Summary
        errors = [r for r in self.results if r.status == 'error']
        warnings = [r for r in self.results if r.status == 'warning']
        
        logger.info(f"Preflight complete: {len(errors)} errors, {len(warnings)} warnings")
        
        return len(errors) == 0, self.results
    
    def _check_opencv(self):
        """Validate OpenCV installation and capabilities."""
        try:
            import cv2
            version = cv2.__version__
            
            # Test basic operations
            test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # Test color conversion
            gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            
            # Test morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            _ = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Test rotation
            _ = cv2.rotate(test_img, cv2.ROTATE_90_CLOCKWISE)
            
            self.results.append(PreflightResult(
                component="OpenCV",
                status="ok",
                message=f"OpenCV {version} working correctly",
                details={"version": version, "tests_passed": ["color_conversion", "morphology", "rotation"]}
            ))
            
        except Exception as e:
            self.results.append(PreflightResult(
                component="OpenCV",
                status="error",
                message=f"OpenCV validation failed: {e}"
            ))
    
    def _check_tesseract(self):
        """Validate Tesseract installation and language support."""
        try:
            import pytesseract
            
            # Check Tesseract version
            version = pytesseract.get_tesseract_version()
            
            # Check language support
            available_langs = pytesseract.get_languages()
            required_langs = TESSERACT_LANGUAGES.split('+')
            missing_langs = [lang for lang in required_langs if lang not in available_langs]
            
            # Test OCR functionality
            test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
            cv2.putText(test_img, "Test Text", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            try:
                result = pytesseract.image_to_string(test_img, config='--oem 1 --psm 6')
                ocr_works = "Test" in result or "Text" in result
            except:
                ocr_works = False
            
            if missing_langs:
                self.results.append(PreflightResult(
                    component="Tesseract",
                    status="warning",
                    message=f"Missing languages: {', '.join(missing_langs)}",
                    details={
                        "version": str(version),
                        "available_langs": available_langs,
                        "missing_langs": missing_langs,
                        "ocr_functional": ocr_works
                    }
                ))
            elif not ocr_works:
                self.results.append(PreflightResult(
                    component="Tesseract",
                    status="error",
                    message="Tesseract OCR test failed"
                ))
            else:
                self.results.append(PreflightResult(
                    component="Tesseract",
                    status="ok",
                    message=f"Tesseract {version} with all required languages",
                    details={
                        "version": str(version),
                        "languages": available_langs,
                        "ocr_test": "passed"
                    }
                ))
                
        except ImportError:
            self.results.append(PreflightResult(
                component="Tesseract",
                status="error",
                message="pytesseract not installed"
            ))
        except Exception as e:
            self.results.append(PreflightResult(
                component="Tesseract",
                status="error",
                message=f"Tesseract validation failed: {e}"
            ))
    
    def _check_paddle(self):
        """Validate PaddleOCR installation and functionality."""
        try:
            from paddleocr import PaddleOCR
            
            # Test PaddleOCR initialization
            try:
                paddle = PaddleOCR(use_textline_orientation=True, lang='en')
            
            # Test OCR functionality
            test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
            cv2.putText(test_img, "Test Text", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            results = paddle.ocr(test_img)
            ocr_works = results and len(results[0]) > 0 if results[0] else False
            
            if ocr_works:
                detected_text = results[0][0][1][0] if results[0] else ""
                confidence = results[0][0][1][1] if results[0] else 0
                
                self.results.append(PreflightResult(
                    component="PaddleOCR",
                    status="ok",
                    message="PaddleOCR working correctly",
                    details={
                        "test_result": detected_text,
                        "confidence": confidence,
                        "model_loaded": True
                    }
                ))
            else:
                self.results.append(PreflightResult(
                    component="PaddleOCR",
                    status="warning",
                    message="PaddleOCR test produced no results"
                ))
                
        except ImportError:
            self.results.append(PreflightResult(
                component="PaddleOCR",
                status="error",
                message="PaddleOCR not installed"
            ))
        except Exception as e:
            self.results.append(PreflightResult(
                component="PaddleOCR",
                status="error",
                message=f"PaddleOCR validation failed: {e}"
            ))
    
    def _check_optional_ocr(self):
        """Check optional OCR engines."""
        # Check EasyOCR
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            self.results.append(PreflightResult(
                component="EasyOCR",
                status="ok",
                message="EasyOCR available (optional)",
                details={"languages": reader.lang_list}
            ))
        except ImportError:
            self.results.append(PreflightResult(
                component="EasyOCR",
                status="ok",
                message="EasyOCR not installed (optional)"
            ))
        except Exception as e:
            self.results.append(PreflightResult(
                component="EasyOCR",
                status="warning",
                message=f"EasyOCR available but failed test: {e}"
            ))
        
        # Check TrOCR
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.results.append(PreflightResult(
                component="TrOCR",
                status="ok",
                message="TrOCR transformers available (optional)"
            ))
        except ImportError:
            self.results.append(PreflightResult(
                component="TrOCR",
                status="ok",
                message="TrOCR not installed (optional)"
            ))
    
    def _check_output_directories(self):
        """Validate output directory structure."""
        try:
            created_dirs = []
            for purpose, dirname in OUTPUT_DIRS.items():
                if not os.path.exists(dirname):
                    os.makedirs(dirname, exist_ok=True)
                    created_dirs.append(dirname)
                
                # Test write permissions
                test_file = os.path.join(dirname, 'preflight_test.tmp')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            
            message = "Output directories validated"
            if created_dirs:
                message += f" (created: {', '.join(created_dirs)})"
            
            self.results.append(PreflightResult(
                component="Output Directories",
                status="ok",
                message=message,
                details={"directories": OUTPUT_DIRS, "created": created_dirs}
            ))
            
        except Exception as e:
            self.results.append(PreflightResult(
                component="Output Directories",
                status="error",
                message=f"Directory validation failed: {e}"
            ))
    
    def _check_pdf_libraries(self):
        """Validate PDF processing libraries."""
        try:
            import fitz  # PyMuPDF
            version = fitz.version
            
            # Test PDF capabilities
            test_works = hasattr(fitz, 'Document')
            
            self.results.append(PreflightResult(
                component="PyMuPDF",
                status="ok",
                message=f"PyMuPDF {version[0]} working correctly",
                details={"version": version, "document_support": test_works}
            ))
            
        except ImportError:
            self.results.append(PreflightResult(
                component="PyMuPDF",
                status="error",
                message="PyMuPDF (fitz) not installed"
            ))
        except Exception as e:
            self.results.append(PreflightResult(
                component="PyMuPDF",
                status="error",
                message=f"PyMuPDF validation failed: {e}"
            ))
    
    def _check_system_resources(self):
        """Check system resources and performance."""
        try:
            import psutil
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            # Disk space check
            disk = psutil.disk_usage('.')
            disk_free_gb = disk.free / (1024**3)
            
            # CPU check
            cpu_count = psutil.cpu_count()
            
            warnings = []
            if memory_gb < 4:
                warnings.append(f"Low memory: {memory_gb:.1f}GB (recommend 4GB+)")
            if disk_free_gb < 1:
                warnings.append(f"Low disk space: {disk_free_gb:.1f}GB")
            if cpu_count < 2:
                warnings.append(f"Low CPU cores: {cpu_count}")
            
            status = "warning" if warnings else "ok"
            message = "; ".join(warnings) if warnings else f"Resources OK: {memory_gb:.1f}GB RAM, {cpu_count} cores"
            
            self.results.append(PreflightResult(
                component="System Resources",
                status=status,
                message=message,
                details={
                    "memory_gb": memory_gb,
                    "disk_free_gb": disk_free_gb,
                    "cpu_cores": cpu_count
                }
            ))
            
        except ImportError:
            self.results.append(PreflightResult(
                component="System Resources",
                status="warning",
                message="psutil not available for resource monitoring"
            ))
        except Exception as e:
            self.results.append(PreflightResult(
                component="System Resources",
                status="warning",
                message=f"Resource check failed: {e}"
            ))
    
    def _check_llama_integration(self):
        """Validate LLaMA integration if configured."""
        if LLM_PROVIDER == 'none':
            self.results.append(PreflightResult(
                component="LLaMA Integration",
                status="ok",
                message="LLaMA correction disabled"
            ))
            return
        
        if LLM_PROVIDER == 'ollama':
            self._check_ollama()
        elif LLM_PROVIDER == 'llamacpp':
            self._check_llamacpp()
        else:
            self.results.append(PreflightResult(
                component="LLaMA Integration",
                status="error",
                message=f"Unknown LLM provider: {LLM_PROVIDER}"
            ))
    
    def _check_ollama(self):
        """Check Ollama availability and model."""
        try:
            import requests
            
            # Check Ollama server
            response = requests.get(f"{LLM_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                if LLM_MODEL in model_names:
                    # Test model inference
                    test_payload = {
                        "model": LLM_MODEL,
                        "prompt": "Test: correct 'helo' -> ",
                        "stream": False,
                        "options": {"temperature": 0.1}
                    }
                    
                    test_response = requests.post(
                        f"{LLM_BASE_URL}/api/generate",
                        json=test_payload,
                        timeout=LLM_TIMEOUT
                    )
                    
                    if test_response.status_code == 200:
                        self.results.append(PreflightResult(
                            component="Ollama LLaMA",
                            status="ok",
                            message=f"Ollama + {LLM_MODEL} working correctly",
                            details={
                                "server": LLM_BASE_URL,
                                "model": LLM_MODEL,
                                "available_models": model_names
                            }
                        ))
                    else:
                        self.results.append(PreflightResult(
                            component="Ollama LLaMA",
                            status="error",
                            message=f"Model inference failed: {test_response.status_code}"
                        ))
                else:
                    self.results.append(PreflightResult(
                        component="Ollama LLaMA",
                        status="error",
                        message=f"Model {LLM_MODEL} not found. Available: {', '.join(model_names)}"
                    ))
            else:
                self.results.append(PreflightResult(
                    component="Ollama LLaMA",
                    status="error",
                    message=f"Ollama server unreachable: {LLM_BASE_URL}"
                ))
                
        except ImportError:
            self.results.append(PreflightResult(
                component="Ollama LLaMA",
                status="error",
                message="requests library required for Ollama"
            ))
        except Exception as e:
            self.results.append(PreflightResult(
                component="Ollama LLaMA",
                status="error",
                message=f"Ollama validation failed: {e}"
            ))
    
    def _check_llamacpp(self):
        """Check llama-cpp-python availability."""
        try:
            from llama_cpp import Llama
            self.results.append(PreflightResult(
                component="llama-cpp-python",
                status="warning",
                message="llama-cpp-python available but not tested (model path required)"
            ))
        except ImportError:
            self.results.append(PreflightResult(
                component="llama-cpp-python",
                status="error",
                message="llama-cpp-python not installed"
            ))
    
    def _check_ocr_performance(self):
        """Test OCR performance with a sample image."""
        try:
            # Create a test image with text
            test_img = np.ones((200, 600, 3), dtype=np.uint8) * 255
            texts = [
                "Performance Test",
                "Français: Bonjour le monde",
                "Deutsch: Hallo Welt",
                "Türkçe: Merhaba dünya"
            ]
            
            for i, text in enumerate(texts):
                cv2.putText(test_img, text, (10, 50 + i * 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Time OCR operations
            start_time = time.time()
            
            # Test both engines if available
            results = []
            
            try:
                import pytesseract
                tess_result = pytesseract.image_to_string(test_img, lang=TESSERACT_LANGUAGES)
                tess_time = time.time() - start_time
                results.append(("Tesseract", tess_time, len(tess_result.strip())))
            except:
                pass
            
            try:
                from paddleocr import PaddleOCR
                paddle = PaddleOCR(use_textline_orientation=False, lang='en')
                paddle_result = paddle.ocr(test_img)
                paddle_time = time.time() - start_time
                paddle_text_len = len(' '.join([r[1][0] for r in paddle_result[0]]) if paddle_result[0] else "")
                results.append(("PaddleOCR", paddle_time, paddle_text_len))
            except:
                pass
            
            total_time = time.time() - start_time
            
            if results:
                details = {
                    "total_time": total_time,
                    "engine_results": results
                }
                
                status = "warning" if total_time > 10 else "ok"
                message = f"OCR performance: {total_time:.2f}s total"
                if total_time > 10:
                    message += " (slower than expected)"
                
                self.results.append(PreflightResult(
                    component="OCR Performance",
                    status=status,
                    message=message,
                    details=details
                ))
            else:
                self.results.append(PreflightResult(
                    component="OCR Performance",
                    status="error",
                    message="No OCR engines available for performance test"
                ))
                
        except Exception as e:
            self.results.append(PreflightResult(
                component="OCR Performance",
                status="warning",
                message=f"Performance test failed: {e}"
            ))


def run_preflight() -> Tuple[bool, List[PreflightResult]]:
    """Run comprehensive preflight validation."""
    checker = PreflightChecker()
    return checker.run_all_checks()


def print_preflight_report(results: List[PreflightResult]):
    """Print formatted preflight report."""
    print("\n" + "="*60)
    print("OCR PIPELINE PREFLIGHT REPORT")
    print("="*60)
    
    # Group by status
    errors = [r for r in results if r.status == 'error']
    warnings = [r for r in results if r.status == 'warning']
    success = [r for r in results if r.status == 'ok']
    
    # Print errors first
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for result in errors:
            print(f"  • {result.component}: {result.message}")
    
    # Then warnings
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for result in warnings:
            print(f"  • {result.component}: {result.message}")
    
    # Then successes
    if success:
        print(f"\nOK ({len(success)}):")
        for result in success:
            print(f"  • {result.component}: {result.message}")
    
    # Summary
    print("\n" + "-"*60)
    if errors:
        print(f"PIPELINE NOT READY - {len(errors)} critical errors")
        print("   Please resolve errors before processing")
    elif warnings:
        print(f"PIPELINE READY WITH WARNINGS - {len(warnings)} warnings")
        print("   Consider resolving warnings for optimal performance")
    else:
        print("PIPELINE FULLY READY - All checks passed!")
    
    print("="*60 + "\n")
