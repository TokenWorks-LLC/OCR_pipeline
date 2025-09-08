"""
PDF processing utilities for extracting images from PDF files.
Handles both single-page and multi-page PDFs containing scanned images.
"""
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union
import tempfile

import numpy as np

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdf2image import convert_from_path, convert_from_bytes
    from pdf2image.exceptions import (
        PDFInfoNotInstalledError,
        PDFPageCountError,
        PDFSyntaxError
    )
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    convert_from_path = None
    convert_from_bytes = None

logger = logging.getLogger(__name__)


def is_pdf_file(filepath: Union[str, Path]) -> bool:
    """Check if file is a PDF."""
    filepath = Path(filepath)
    return filepath.suffix.lower() == '.pdf'


def extract_images_pymupdf(pdf_path: Union[str, Path], dpi: int = 300) -> List[Tuple[np.ndarray, str]]:
    """
    Extract images from PDF using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
        dpi: DPI for image extraction
    
    Returns:
        List of (image_array, page_id) tuples
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF not available. Install with: pip install PyMuPDF")
    
    pdf_path = Path(pdf_path)
    images = []
    
    try:
        doc = fitz.open(str(pdf_path))
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Convert page to image
            mat = fitz.Matrix(dpi/72, dpi/72)  # 72 is default DPI
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to numpy array
            img_data = pix.tobytes("png")
            
            # Convert PNG bytes to numpy array
            import cv2
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Sanitize the filename for page_id generation
                import re
                sanitized_name = re.sub(r'[<>:"/\\|?*]', '_', pdf_path.stem)
                # Replace Turkish characters 
                replacements = {
                    'ç': 'c', 'Ç': 'C', 'ğ': 'g', 'Ğ': 'G', 'ı': 'i', 'İ': 'I',
                    'ö': 'o', 'Ö': 'O', 'ş': 's', 'Ş': 'S', 'ü': 'u', 'Ü': 'U'
                }
                for turkish, replacement in replacements.items():
                    sanitized_name = sanitized_name.replace(turkish, replacement)
                sanitized_name = re.sub(r'\s+', '_', sanitized_name.strip())
                sanitized_name = re.sub(r'_+', '_', sanitized_name)
                # Limit length
                if len(sanitized_name) > 50:
                    sanitized_name = sanitized_name[:50].rstrip('_')
                
                page_id = f"{sanitized_name}_page_{page_num+1:03d}"
                images.append((img, page_id))
                logger.debug(f"Extracted page {page_num+1} from {pdf_path}")
            else:
                logger.warning(f"Failed to extract page {page_num+1} from {pdf_path}")
        
        doc.close()
        logger.info(f"Extracted {len(images)} pages from {pdf_path}")
        
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
        return []
    
    return images


def extract_images_pdf2image(pdf_path: Union[str, Path], dpi: int = 300) -> List[Tuple[np.ndarray, str]]:
    """
    Extract images from PDF using pdf2image.
    
    Args:
        pdf_path: Path to PDF file
        dpi: DPI for image extraction
    
    Returns:
        List of (image_array, page_id) tuples
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError("pdf2image not available. Install with: pip install pdf2image")
    
    pdf_path = Path(pdf_path)
    images = []
    
    try:
        # Convert PDF pages to PIL images
        pil_images = convert_from_path(str(pdf_path), dpi=dpi)
        
        for page_num, pil_img in enumerate(pil_images):
            # Convert PIL to OpenCV format
            import cv2
            img_array = np.array(pil_img)
            
            # Convert RGB to BGR (OpenCV format)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = img_array
            
            # Sanitize the filename for page_id generation
            import re
            sanitized_name = re.sub(r'[<>:"/\\|?*]', '_', pdf_path.stem)
            # Replace Turkish characters 
            replacements = {
                'ç': 'c', 'Ç': 'C', 'ğ': 'g', 'Ğ': 'G', 'ı': 'i', 'İ': 'I',
                'ö': 'o', 'Ö': 'O', 'ş': 's', 'Ş': 'S', 'ü': 'u', 'Ü': 'U'
            }
            for turkish, replacement in replacements.items():
                sanitized_name = sanitized_name.replace(turkish, replacement)
            sanitized_name = re.sub(r'\s+', '_', sanitized_name.strip())
            sanitized_name = re.sub(r'_+', '_', sanitized_name)
            # Limit length
            if len(sanitized_name) > 50:
                sanitized_name = sanitized_name[:50].rstrip('_')
                
            page_id = f"{sanitized_name}_page_{page_num+1:03d}"
            images.append((img, page_id))
            logger.debug(f"Extracted page {page_num+1} from {pdf_path}")
        
        logger.info(f"Extracted {len(images)} pages from {pdf_path}")
        
    except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
        logger.error(f"pdf2image extraction failed for {pdf_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error extracting from {pdf_path}: {e}")
        return []
    
    return images


def extract_images_from_pdf(pdf_path: Union[str, Path], dpi: int = 300, 
                          method: str = 'auto') -> List[Tuple[np.ndarray, str]]:
    """
    Extract images from PDF using available libraries.
    
    Args:
        pdf_path: Path to PDF file
        dpi: DPI for image extraction
        method: Extraction method ('auto', 'pymupdf', 'pdf2image')
    
    Returns:
        List of (image_array, page_id) tuples
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not is_pdf_file(pdf_path):
        raise ValueError(f"File is not a PDF: {pdf_path}")
    
    if method == 'auto':
        # Try PyMuPDF first (usually faster), then pdf2image
        if PYMUPDF_AVAILABLE:
            method = 'pymupdf'
        elif PDF2IMAGE_AVAILABLE:
            method = 'pdf2image'
        else:
            raise ImportError("No PDF processing library available. Install PyMuPDF or pdf2image.")
    
    if method == 'pymupdf':
        return extract_images_pymupdf(pdf_path, dpi)
    elif method == 'pdf2image':
        return extract_images_pdf2image(pdf_path, dpi)
    else:
        raise ValueError(f"Unknown extraction method: {method}")


def get_pdf_info(pdf_path: Union[str, Path]) -> dict:
    """Get basic information about a PDF file."""
    pdf_path = Path(pdf_path)
    info = {
        'filepath': str(pdf_path),
        'filename': pdf_path.name,
        'size_bytes': pdf_path.stat().st_size if pdf_path.exists() else 0,
        'pages': 0,
        'method': None,
        'error': None
    }
    
    try:
        if PYMUPDF_AVAILABLE:
            doc = fitz.open(str(pdf_path))
            info['pages'] = len(doc)
            info['method'] = 'pymupdf'
            doc.close()
        elif PDF2IMAGE_AVAILABLE:
            from pdf2image import pdfinfo_from_path
            pdf_info = pdfinfo_from_path(str(pdf_path))
            info['pages'] = pdf_info.get('Pages', 0)
            info['method'] = 'pdf2image'
        else:
            info['error'] = 'No PDF processing library available'
    
    except Exception as e:
        info['error'] = str(e)
        logger.error(f"Failed to get PDF info for {pdf_path}: {e}")
    
    return info


def batch_extract_from_pdfs(pdf_paths: List[Union[str, Path]], 
                           dpi: int = 300) -> List[Tuple[np.ndarray, str, str]]:
    """
    Extract images from multiple PDF files.
    
    Args:
        pdf_paths: List of PDF file paths
        dpi: DPI for image extraction
    
    Returns:
        List of (image_array, page_id, source_pdf) tuples
    """
    all_images = []
    
    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            images = extract_images_from_pdf(pdf_path, dpi)
            
            for img, page_id in images:
                all_images.append((img, page_id, str(pdf_path)))
                
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            continue
    
    logger.info(f"Extracted {len(all_images)} total images from {len(pdf_paths)} PDFs")
    return all_images


def save_extracted_images(images: List[Tuple[np.ndarray, str]], 
                         output_dir: Union[str, Path]) -> List[Path]:
    """
    Save extracted images to directory.
    
    Args:
        images: List of (image_array, page_id) tuples
        output_dir: Output directory path
    
    Returns:
        List of saved image file paths
    """
    import cv2
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for img, page_id in images:
        output_path = output_dir / f"{page_id}.png"
        
        try:
            success = cv2.imwrite(str(output_path), img)
            if success:
                saved_paths.append(output_path)
                logger.debug(f"Saved image: {output_path}")
            else:
                logger.warning(f"Failed to save image: {output_path}")
        except Exception as e:
            logger.error(f"Error saving {output_path}: {e}")
    
    logger.info(f"Saved {len(saved_paths)} images to {output_dir}")
    return saved_paths


def check_pdf_libraries() -> dict:
    """Check availability of PDF processing libraries."""
    status = {
        'pymupdf': PYMUPDF_AVAILABLE,
        'pdf2image': PDF2IMAGE_AVAILABLE,
        'any_available': PYMUPDF_AVAILABLE or PDF2IMAGE_AVAILABLE
    }
    
    if PYMUPDF_AVAILABLE:
        status['pymupdf_version'] = fitz.version[0]
    
    if PDF2IMAGE_AVAILABLE:
        try:
            import pdf2image
            status['pdf2image_version'] = getattr(pdf2image, '__version__', 'unknown')
        except:
            status['pdf2image_version'] = 'unknown'
    
    return status
