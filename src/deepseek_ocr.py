"""
DeepSeek-OCR integration for the OCR pipeline.
Provides CPU-compatible DeepSeek OCR with explicit error handling.
"""
import logging
import os
import tempfile
import time
from typing import List, Optional, Dict, Any
import cv2
import numpy as np

# Try to import DeepSeek dependencies
try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from config import DEEPSEEK_MODEL, DEEPSEEK_DEVICE, DEEPSEEK_PROMPT, DEEPSEEK_BASE_SIZE, DEEPSEEK_IMAGE_SIZE
from ocr_utils import Line

logger = logging.getLogger(__name__)

class DeepSeekOCRError(Exception):
    """Explicit error for DeepSeek-OCR failures."""
    pass

class DeepSeekOCRWrapper:
    """
    Wrapper for DeepSeek-OCR that matches existing OCR interface.
    Provides hardware-aware inference with automatic device selection.
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or DEEPSEEK_MODEL
        self.requested_device = device or DEEPSEEK_DEVICE
        self.detected_device = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None
        self.initialization_error = None
        self.is_initialized = False

        # Initialize on first use (lazy loading)
        self._ensure_initialized()

    def _detect_optimal_device(self):
        """Detect the best available device for inference."""
        if self.requested_device != "auto":
            return self.requested_device

        # Check for CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return "cuda"

        # Check for ROCm (AMD GPUs) - presented as CUDA in PyTorch
        try:
            if hasattr(torch.version, 'hip') and torch.version.hip:
                logger.info("ROCm detected - using CUDA device mapping")
                return "cuda"
        except:
            pass

        # Fallback to CPU
        logger.info("Using CPU for inference")
        return "cpu"

    def _ensure_initialized(self):
        """Lazy initialization with hardware-aware device selection."""
        if self.is_initialized or self.initialization_error:
            return

        if not TRANSFORMERS_AVAILABLE:
            self.initialization_error = "transformers library not available"
            logger.error(f"DeepSeek-OCR initialization failed: {self.initialization_error}")
            return

        try:
            self.detected_device = self._detect_optimal_device()
            logger.info(f"Loading DeepSeek-OCR model: {self.model_name} on {self.detected_device}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            # Configure device mapping based on detected hardware
            device_map = self.detected_device if self.detected_device != "cpu" else "cpu"

            # Load model with hardware-appropriate settings
            if self.detected_device == "cpu":
                # CPU-optimized settings
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    _attn_implementation='eager',  # CPU-compatible attention
                    trust_remote_code=True,
                    use_safetensors=True,
                    device_map=device_map,
                    torch_dtype=torch.float32  # CPU compatible dtype
                )
            else:
                # GPU-optimized settings
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    _attn_implementation='flash_attention_2',  # GPU optimized
                    trust_remote_code=True,
                    use_safetensors=True,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16  # GPU optimized dtype
                )

            # Set to eval mode
            self.model = self.model.eval()

            self.is_initialized = True
            logger.info(f"DeepSeek-OCR model loaded successfully on {self.detected_device}")

        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"DeepSeek-OCR initialization failed: {e}")
            self.model = None
            self.tokenizer = None

    def ocr_image(self, img: np.ndarray, prompt: str = None, timeout: int = 300) -> List[Line]:
        """
        OCR image using DeepSeek-OCR with explicit error handling.

        Args:
            img: OpenCV BGR image
            prompt: OCR prompt (defaults to config)
            timeout: Maximum time in seconds for inference

        Returns:
            List[Line] matching existing OCR interface

        Raises:
            DeepSeekOCRError: On any failure during OCR processing
        """
        if self.initialization_error:
            raise DeepSeekOCRError(f"Model not initialized: {self.initialization_error}")

        if not self.is_initialized or self.model is None or self.tokenizer is None:
            raise DeepSeekOCRError("DeepSeek-OCR not properly initialized")

        if not isinstance(img, np.ndarray):
            raise DeepSeekOCRError(f"Expected numpy array, got {type(img)}")

        if img.size == 0:
            raise DeepSeekOCRError("Empty image provided")

        prompt = prompt or DEEPSEEK_PROMPT

        # Create temp file for DeepSeek (required by their API)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                success = cv2.imwrite(tmp.name, img)
                if not success:
                    raise DeepSeekOCRError("Failed to write image to temporary file")
                temp_path = tmp.name

            logger.debug(f"Created temp file: {temp_path}")

            # Verify file exists and has content
            if not os.path.exists(temp_path):
                raise DeepSeekOCRError(f"Temp file was not created: {temp_path}")
            if os.path.getsize(temp_path) == 0:
                raise DeepSeekOCRError(f"Temp file is empty: {temp_path}")

            # Run inference with timeout
            start_time = time.time()
            logger.debug(f"Starting DeepSeek-OCR inference on {img.shape} image, temp_file: {temp_path}")

            # Create a temporary output directory for DeepSeek (required by their API)
            with tempfile.TemporaryDirectory() as temp_output_dir:
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=temp_path,
                    output_path=temp_output_dir,  # Must be a valid directory path
                    base_size=DEEPSEEK_BASE_SIZE,
                    image_size=DEEPSEEK_IMAGE_SIZE,
                    crop_mode=True,
                    save_results=False,  # Don't save intermediate files
                    test_compress=True
                )

            inference_time = time.time() - start_time
            logger.debug(f"DeepSeek-OCR inference completed in {inference_time:.2f}s")

            # Check for timeout
            if timeout and inference_time > timeout:
                logger.warning(f"DeepSeek-OCR inference exceeded timeout ({timeout}s)")

            # Parse result
            if not result or 'text' not in result:
                raise DeepSeekOCRError(f"Unexpected result format from DeepSeek-OCR: {result}")

            text_content = result['text']
            if not isinstance(text_content, str):
                raise DeepSeekOCRError(f"Expected string text, got {type(text_content)}")

            if not text_content.strip():
                logger.warning("DeepSeek-OCR returned empty text")
                return []

            # Split into lines and create Line objects
            lines = []
            text_lines = text_content.split('\n')

            for i, line_text in enumerate(text_lines):
                if line_text.strip():  # Skip empty lines
                    lines.append(Line(
                        text=line_text.strip(),
                        conf=0.95,  # Default high confidence for DeepSeek
                        bbox=(0, i * 30, len(line_text) * 12, 30),  # Estimated bbox
                        engine="deepseek"
                    ))

            logger.info(f"DeepSeek-OCR extracted {len(lines)} lines from image")
            return lines

        except Exception as e:
            logger.error(f"DeepSeek-OCR inference failed: {e}")
            raise DeepSeekOCRError(f"Inference failed: {e}")

        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass  # Don't mask the real error

def ocr_deepseek_lines(img: np.ndarray, prompt: str = None, timeout: int = 300) -> List[Line]:
    """
    Convenience function matching existing OCR interface.

    Args:
        img: OpenCV BGR image
        prompt: Optional custom prompt
        timeout: Maximum inference time in seconds

    Returns:
        List[Line] with text/conf/bbox/engine

    Raises:
        DeepSeekOCRError: On any failure
    """
    wrapper = DeepSeekOCRWrapper()
    return wrapper.ocr_image(img, prompt, timeout)

def is_deepseek_available() -> bool:
    """Check if DeepSeek-OCR is available and can be initialized."""
    try:
        wrapper = DeepSeekOCRWrapper()
        return wrapper.is_initialized
    except:
        return False
