#!/usr/bin/env python3
"""
Detection audit mode for Akkadian detection tuning.
OCR + blockification + detection only (no LLM/pairing).
"""

import csv
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def audit_page(
    pdf_path: str,
    page_no: int,
    detection_config: Dict,
    ocr_engines: Dict,
    ensemble=None
) -> List[Dict]:
    """
    Audit a single page: OCR + blockification + detection only.
    
    Returns:
        List of audit records (one per detected block)
    """
    import fitz
    import cv2
    import numpy as np
    from blockification import TextBlockifier
    from lang_and_akkadian import is_akkadian_transliteration
    
    pdf_name = Path(pdf_path).name
    audit_records = []
    
    try:
        # Open PDF and render page
        doc = fitz.open(str(pdf_path))
        page = doc.load_page(page_no - 1)  # 0-based indexing
        
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Try extracting text first (for PDFs with text layer)
        extracted_text = page.get_text()
        
        # If PDF has text layer, use it directly; otherwise OCR the image
        if extracted_text and len(extracted_text.strip()) > 50:
            logger.info(f"Using embedded text layer ({len(extracted_text)} chars)")
            
            # Create simple lines from extracted text (one line per paragraph/block)
            lines = []
            text_blocks = page.get_text("dict")["blocks"]
            doc.close()  # Close after extracting text
            
            for block in text_blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        line_text = " ".join([span["text"] for span in line.get("spans", [])])
                        if line_text.strip():
                            lines.append({
                                'text': line_text,
                                'confidence': 1.0,  # Text layer is 100% confident
                                'bbox': line.get("bbox", [0, 0, page_width, 10]),
                                'engine': 'text_layer'
                            })
        else:
            # No text layer or insufficient text - use OCR
            logger.info("No text layer found, using OCR")
            
            # Render to image
            mat = fitz.Matrix(2, 2)  # 2x scaling for better OCR
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            doc.close()
            
            # Run OCR (single or ensemble)
            if ensemble and len(ocr_engines) > 1:
                # Multi-engine with fusion
                engine_results = {}
                for engine_name, engine in ocr_engines.items():
                    result = engine.predict(img)
                    engine_results[engine_name] = result
                
                lines = ensemble.combine_results(engine_results)
            else:
                # Single engine
                engine_name = list(ocr_engines.keys())[0]
                engine = ocr_engines[engine_name]
                ocr_result = engine.predict(img)
                
                # Convert PaddleOCR format to standard lines
                lines = []
                if ocr_result and len(ocr_result) > 0:
                    for line_data in ocr_result[0]:
                        try:
                            # Validate line_data structure
                            if not isinstance(line_data, (list, tuple)) or len(line_data) < 2:
                                logger.debug(f"Skipping invalid line_data: {line_data}")
                                continue
                            
                            bbox = line_data[0]
                            
                            # Handle different PaddleOCR result formats with validation
                            text = ""
                            conf = 0.0
                            
                            if isinstance(line_data[1], (list, tuple)):
                                if len(line_data[1]) >= 2:
                                    text, conf = line_data[1][0], line_data[1][1]
                                elif len(line_data[1]) == 1:
                                    text = line_data[1][0]
                                # Empty list/tuple case
                                else:
                                    logger.debug(f"Empty text container in line_data[1]: {line_data}")
                                    continue
                            else:
                                # Direct text value
                                text = str(line_data[1])
                            
                            # Ensure text is valid string
                            if not text or not isinstance(text, str):
                                text = str(text) if text is not None else ""
                            
                            # Skip empty text
                            if not text.strip():
                                continue
                            
                            lines.append({
                                'text': text,
                                'confidence': float(conf),
                                'bbox': bbox,
                                'engine': engine_name
                            })
                        except Exception as e:
                            logger.debug(f"Skipping malformed OCR line: {e}")
                            continue
        
        # Blockify
        blockifier = TextBlockifier()
        blocks = blockifier.blockify(
            lines,
            page_num=page_no,
            page_width=page_width,
            page_height=page_height
        )
        
        # Detect Akkadian for each block
        for i, block in enumerate(blocks):
            try:
                # Ensure block has text
                if not hasattr(block, 'text') or not block.text:
                    continue
                
                block_text = str(block.text).strip()
                
                # Skip empty blocks after stripping
                if not block_text:
                    continue
                
                is_akk, akk_score = is_akkadian_transliteration(block_text, config=detection_config)
                
                # Count diacritics
                DIACRITIC_CHARS = "āēīūŠšṢṣṬṭḪḫáéíóúàèìù"
                num_diacritics = sum(1 for char in block_text if char in DIACRITIC_CHARS)
                
                # Check for markers
                MARKERS = {"DUMU", "LUGAL", "KÙ.BABBAR", "KUBABBAR", "KU.BABBAR", "URU", "É", "É.GAL", "KUR", "LU₂", "LÚ", "MUNUS", "MÍ"}
                text_upper = block_text.upper()
                has_marker = any(m in text_upper for m in MARKERS)
                
                # Calculate syllabic metrics (same logic as is_akkadian_transliteration)
                import re
                syllabic_pattern = re.compile(r'\b[a-zšṣṭḫāēīū]{1,4}(?:[-—][a-zšṣṭḫāēīū]{1,4}){2,}\b', re.IGNORECASE)
                syllabic_matches = syllabic_pattern.findall(block_text)
                all_tokens = re.findall(r'\b[a-zšṣṭḫāēīū]+(?:[-—][a-zšṣṭḫāēīū]+)*\b', block_text, re.IGNORECASE)
                num_syllabic_tokens = len(syllabic_matches)
                total_tokens = len(all_tokens)
                syllabic_ratio = num_syllabic_tokens / total_tokens if total_tokens > 0 else 0.0
                
                # Safe text truncation
                text_preview = block_text[:200] if len(block_text) > 200 else block_text
                
                audit_records.append({
                    'pdf': pdf_name,
                    'page': page_no,
                    'block_id': f"p{page_no}_b{i}",
                    'text': text_preview,
                    'is_akkadian': is_akk,
                    'score': akk_score,
                    'num_diacritics': num_diacritics,
                    'has_marker': has_marker,
                    'num_syllabic_tokens': num_syllabic_tokens,
                    'syllabic_ratio': round(syllabic_ratio, 3),
                    'block_length': len(block_text)
                })
            except Exception as e:
                logger.debug(f"Error processing block {i} on page {page_no}: {e}")
                continue
        
    except Exception as e:
        logger.error(f"Audit failed for {pdf_name} page {page_no}: {e}")
        audit_records.append({
            'pdf': pdf_name,
            'page': page_no,
            'block_id': 'ERROR',
            'text': str(e),
            'is_akkadian': False,
            'score': 0.0,
            'num_diacritics': 0,
            'has_marker': False,
            'num_syllabic_tokens': 0,
            'syllabic_ratio': 0.0,
            'block_length': 0
        })
    
    return audit_records


def run_audit(
    manifest: List[Tuple[str, int]],
    detection_config: Dict,
    ocr_engines: Dict,
    ensemble,
    output_csv: Path,
    sample_size: int = None
) -> None:
    """
    Run detection audit on manifest.
    
    Args:
        manifest: List of (pdf_path, page_no) tuples
        detection_config: Detection configuration dict
        ocr_engines: Dictionary of OCR engines
        ensemble: Ensemble fusion object (or None)
        output_csv: Path to output audit CSV
        sample_size: If set, only process first N pages
    """
    from tqdm import tqdm
    
    # Limit manifest if sample_size specified
    if sample_size:
        manifest = manifest[:sample_size]
        logger.info(f"Audit mode: processing first {sample_size} pages")
    
    # Initialize audit CSV
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pdf', 'page', 'block_id', 'text', 'is_akkadian', 
            'score', 'num_diacritics', 'has_marker', 'num_syllabic_tokens', 
            'syllabic_ratio', 'block_length'
        ])
        writer.writeheader()
    
    all_records = []
    
    logger.info(f"Starting detection audit on {len(manifest)} pages")
    logger.info(f"Output CSV: {output_csv}")
    
    for pdf_path, page_no in tqdm(manifest, desc="Auditing pages", unit="page"):
        records = audit_page(
            pdf_path=pdf_path,
            page_no=page_no,
            detection_config=detection_config,
            ocr_engines=ocr_engines,
            ensemble=ensemble
        )
        
        all_records.extend(records)
        
        # Append to CSV immediately
        with open(output_csv, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'pdf', 'page', 'block_id', 'text', 'is_akkadian', 
                'score', 'num_diacritics', 'has_marker', 'num_syllabic_tokens', 
                'syllabic_ratio', 'block_length'
            ])
            for record in records:
                writer.writerow(record)
    
    # Summary statistics
    total_blocks = len(all_records)
    detected_akk = sum(1 for r in all_records if r['is_akkadian'])
    
    logger.info("=" * 60)
    logger.info("AUDIT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total blocks analyzed: {total_blocks}")
    
    if total_blocks > 0:
        logger.info(f"Detected as Akkadian: {detected_akk} ({100*detected_akk/total_blocks:.1f}%)")
    else:
        logger.warning("NO BLOCKS FOUND! OCR may have failed or PDFs may be image-only/corrupted.")
        logger.warning("Check that PDFs in manifest are valid and contain extractable text/images.")
    
    logger.info(f"Audit CSV: {output_csv}")
    logger.info("=" * 60)
    
    if total_blocks > 0:
        logger.info("NEXT STEPS:")
        logger.info("1. Manually review 10-15 blocks where detected_akkadian=true")
        logger.info("2. Calculate false positive rate: FP% = (false positives / detected_akk) * 100")
        logger.info("3. Target: FP ≤ 5%")
        logger.info("4. If FP > 5%, raise threshold to 0.55-0.60 and re-run audit")
    else:
        logger.info("TROUBLESHOOTING:")
        logger.info("1. Check manifest file - verify PDFs exist and paths are correct")
        logger.info("2. Try opening a few PDFs manually to verify they contain readable content")
        logger.info("3. Check PaddleOCR initialization - may need different model or config")
    
    logger.info("=" * 60)
