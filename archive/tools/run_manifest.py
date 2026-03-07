#!/usr/bin/env python3
"""
Streaming manifest processor for large-scale PDF processing.
Supports live progress, resume capability, continuous CSV append, and RAM throttling.
"""

import sys
import csv
import json
import time
import re
import psutil
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
import logging

# Set environment variable BEFORE importing modules that use it
if 'AKKADIAN_LM_PATH' not in os.environ:
    default_lm_path = Path(__file__).parent.parent / 'models' / 'akkadian_char_lm.json'
    if default_lm_path.exists():
        os.environ['AKKADIAN_LM_PATH'] = str(default_lm_path)
        print(f"Auto-detected Akkadian LM: {default_lm_path}")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> List[Tuple[str, int]]:
    """Load manifest TSV file."""
    logger.info(f"Loading manifest: {manifest_path}")
    
    entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            pdf_path = row['pdf_path']
            page_no = int(row['page_no'])
            entries.append((pdf_path, page_no))
    
    logger.info(f"Loaded {len(entries)} manifest entries")
    return entries


def load_completed_pages(progress_file: Path) -> Set[Tuple[str, int]]:
    """Load set of completed (pdf_path, page_no) from progress file."""
    if not progress_file.exists():
        logger.info("No progress file found, starting fresh")
        return set()
    
    completed = set()
    with open(progress_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdf_path = row['pdf_path']
            page_no = int(row['page_no'])
            completed.add((pdf_path, page_no))
    
    logger.info(f"Loaded {len(completed)} completed pages from {progress_file}")
    return completed


def append_to_client_csv(
    csv_path: Path,
    pdf_name: str,
    page_no: int,
    akkadian_text: str,
    translation: str,
    confidence: float,
    status: str,
    write_header: bool = False
):
    """
    Append a single row to client CSV with UTF-8 BOM for Excel compatibility.
    
    Args:
        csv_path: Path to client CSV
        pdf_name: PDF filename
        page_no: Page number
        akkadian_text: Akkadian text block
        translation: Translation text
        confidence: Pairing confidence score
        status: Processing status (OK, ERROR, etc.)
        write_header: If True, write BOM and header first
    """
    # Ensure output directory exists
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    mode = 'w' if write_header else 'a'
    
    with open(csv_path, mode, encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        
        if write_header:
            writer.writerow(['pdf_name', 'page_no', 'akkadian_text', 'translation', 'confidence', 'status'])
        
        writer.writerow([pdf_name, page_no, akkadian_text, translation, f"{confidence:.4f}", status])


def append_to_progress_csv(
    csv_path: Path,
    pdf_path: str,
    page_no: int,
    success: bool,
    processing_time: float,
    error: Optional[str] = None,
    write_header: bool = False
):
    """Append processing result to progress CSV."""
    # Ensure output directory exists
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    mode = 'w' if write_header else 'a'
    
    with open(csv_path, mode, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        if write_header:
            writer.writerow(['pdf_path', 'page_no', 'success', 'processing_time', 'error', 'timestamp'])
        
        timestamp = datetime.now().isoformat()
        writer.writerow([pdf_path, page_no, success, f"{processing_time:.2f}", error or '', timestamp])


def get_memory_usage() -> float:
    """Get current RSS memory usage as fraction (0.0 to 1.0)."""
    return psutil.virtual_memory().percent / 100.0


def process_page(
    pdf_path: str,
    page_no: int,
    detection_config: Dict,
    pairing_config: Dict,
    output_dir: Path,
    ocr_engines: Dict,
    ensemble=None,
    llm_corrector=None,
    client_csv: Optional[Path] = None
) -> Dict:
    """
    Process a single page: OCR → Detection → Pairing → Translation.
    
    Returns:
        Result dict with success status, metrics, and any errors
    """
    import fitz
    import cv2
    import numpy as np
    from blockification import TextBlockifier
    from translation_pairing import TranslationPairer, PairingConfig
    from lang_and_akkadian import is_akkadian_transliteration, detect_language
    from block_splitter import split_blocks
    from block_roles import tag_block_roles, filter_blocks_by_role
    
    pdf_name = Path(pdf_path).name
    pdf_stem = Path(pdf_path).stem
    
    result = {
        'pdf_path': pdf_path,
        'pdf_name': pdf_name,
        'page': page_no,
        'success': False,
        'ocr_lines': 0,
        'akkadian_blocks': 0,
        'translation_blocks': 0,
        'pairs_created': 0,
        'avg_confidence': 0.0,
        'processing_time': 0.0,
        'error': None
    }
    
    start_time = time.time()
    
    try:
        # Open PDF and check for embedded text
        doc = fitz.open(str(pdf_path))
        page = doc.load_page(page_no - 1)  # 0-based indexing
        
        # Try to extract embedded text first (FAST PATH)
        extracted_text = page.get_text()
        use_text_layer = False
        
        print(f"[DEBUG] {pdf_name} p{page_no}: extracted {len(extracted_text)} chars")
        
        if extracted_text and len(extracted_text.strip()) > 50:
            # PDF has embedded text - use it directly (35x faster!)
            logger.info(f"✓ [{pdf_name} p{page_no}] Using embedded text layer ({len(extracted_text)} chars)")
            print(f"[TEXT_LAYER] {pdf_name} p{page_no}: Using text layer!")
            use_text_layer = True
            
            # Get page dimensions from PDF
            rect = page.rect
            page_width = rect.width
            page_height = rect.height
            
            # Extract text blocks with bounding boxes
            text_blocks = page.get_text("dict")["blocks"]
            lines = []
            
            for block in text_blocks:
                if block.get("type") == 0:  # Text block
                    bbox = block.get("bbox", [0, 0, page_width, page_height])
                    block_text_lines = []
                    
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        if line_text.strip():
                            block_text_lines.append(line_text.strip())
                    
                    if block_text_lines:
                        combined_text = " ".join(block_text_lines)
                        lines.append({
                            "text": combined_text,
                            "bbox": list(bbox),
                            "confidence": 1.0  # Embedded text has perfect confidence
                        })
            
            doc.close()
            engine_results = {'text_layer': lines}
            
        else:
            # No embedded text - fall back to OCR (SLOW PATH)
            logger.info(f"✗ [{pdf_name} p{page_no}] No text layer - using OCR")
            print(f"[OCR_FALLBACK] {pdf_name} p{page_no}: No text layer, using OCR")
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better OCR
            img = cv2.imdecode(np.frombuffer(pix.tobytes("ppm"), np.uint8), cv2.IMREAD_COLOR)
            
            # Get page dimensions
            page_width = img.shape[1]
            page_height = img.shape[0]
            
            doc.close()
            
            # Run OCR with all engines
            engine_results = {}
        
        # Run OCR engines only if we don't have text layer
        if not use_text_layer:
            for engine_name, engine in ocr_engines.items():
                try:
                    if engine_name == 'paddle':
                        ocr_result = engine.predict(img)
                        engine_lines = []
                        
                        if ocr_result and len(ocr_result) > 0:
                            result_obj = ocr_result[0]
                            
                            # Handle new dict format
                            if isinstance(result_obj, dict):
                                texts = result_obj.get('rec_texts', [])
                                scores = result_obj.get('rec_scores', [])
                                polys = result_obj.get('rec_polys', [])
                                
                                for i, text in enumerate(texts):
                                    if not text:
                                        continue
                                    poly = polys[i] if i < len(polys) else None
                                    conf = scores[i] if i < len(scores) else 0.5
                                    
                                    if poly is not None:
                                        if hasattr(poly, 'tolist'):
                                            poly = poly.tolist()
                                        xs = [p[0] for p in poly]
                                        ys = [p[1] for p in poly]
                                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                                    else:
                                        bbox = [0, i*25, 200, (i+1)*25]
                                    
                                    engine_lines.append({"text": text, "bbox": bbox, "confidence": float(conf)})
                            else:
                                # Handle old tuple format
                                for item in result_obj:
                                    if len(item) >= 2:
                                        bbox_pts = item[0]
                                        text_info = item[1]
                                        if isinstance(text_info, tuple) and len(text_info) >= 2:
                                            text, conf = text_info[0], text_info[1]
                                        else:
                                            text, conf = str(text_info), 0.5
                                        
                                        xs = [p[0] for p in bbox_pts]
                                        ys = [p[1] for p in bbox_pts]
                                        engine_lines.append({
                                            "text": text,
                                            "bbox": [min(xs), min(ys), max(xs), max(ys)],
                                            "confidence": float(conf)
                                        })
                        
                        engine_results['paddle'] = engine_lines
                    else:
                        # Other engines (doctr, mmocr, kraken) - use their predict method
                        engine_result = engine.predict(img)
                        if engine_result:
                            engine_results[engine_name] = engine_result
                except Exception as e:
                    logger.warning(f"Engine {engine_name} failed: {e}")
        
        # Combine results using ensemble if multiple engines
        if ensemble and len(engine_results) > 1:
            combined = ensemble.combine_results(engine_results)
            lines = combined
            logger.debug(f"Ensemble combined {sum(len(r) for r in engine_results.values())} detections from {len(engine_results)} engines into {len(lines)} lines")
        elif engine_results:
            # Single engine or no ensemble - use first available
            lines = list(engine_results.values())[0]
        else:
            lines = []
        
        result['ocr_lines'] = len(lines)
        
        if not lines:
            result['success'] = True
            result['processing_time'] = time.time() - start_time
            return result
        
        # Apply LLM correction if available
        if llm_corrector:
            try:
                corrected_lines = []
                for i, line in enumerate(lines):
                    # Detect if Akkadian for appropriate edit budget
                    from lang_and_akkadian import is_akkadian_transliteration
                    is_akk, _ = is_akkadian_transliteration(line['text'], config=detection_config)
                    
                    # Determine language for LLM correction
                    lang = 'akk' if is_akk else 'en'  # Default to 'en' for non-Akkadian
                    
                    # Get context from adjacent lines
                    prev_line = lines[i-1]['text'] if i > 0 else None
                    next_line = lines[i+1]['text'] if i < len(lines)-1 else None
                    
                    correction_result = llm_corrector.correct_line(
                        text=line['text'],
                        lang=lang,
                        confidence=line.get('confidence', 0.0),
                        prev_line=prev_line,
                        next_line=next_line,
                        span_id=f"{pdf_name}_p{page_no}_l{i}"
                    )
                    
                    if correction_result.applied:
                        line['text'] = correction_result.corrected_text
                        line['llm_corrected'] = True
                        line['edit_ratio'] = correction_result.edit_ratio
                    else:
                        line['llm_corrected'] = False
                        line['edit_ratio'] = 0.0
                    
                    corrected_lines.append(line)
                
                lines = corrected_lines
                logger.debug(f"LLM correction applied to {len(lines)} lines")
            except Exception as e:
                logger.warning(f"LLM correction failed: {e}, using original OCR")
        
        # Blockify
        blockifier = TextBlockifier()
        blocks = blockifier.blockify(
            lines,
            page_num=page_no,
            page_width=page_width,
            page_height=page_height
        )
        
        # PHASE 1: Split mixed-content blocks (NEW)
        block_clean_config = detection_config.get('block_clean', {})
        if block_clean_config.get('split_enabled', True):
            # Convert blocks to dict format for splitting
            block_dicts = []
            for i, block in enumerate(blocks):
                block_dicts.append({
                    'block_id': f"p{page_no}_c{block.column_index}_b{i}",
                    'text': block.text,
                    'bbox': block.bbox,
                    'column': block.column_index,
                    'original_block': block
                })
            
            # Split blocks on structural boundaries
            split_block_dicts = split_blocks(block_dicts, config=block_clean_config)
            
            # Reconstruct block objects from split fragments
            from blockification import TextBlock
            blocks = []
            for split_dict in split_block_dicts:
                original = split_dict.get('original_block')
                # Create new TextBlock with all required fields
                new_block = TextBlock(
                    block_id=split_dict['block_id'],
                    page=page_no,
                    text=split_dict['text'],
                    bbox=split_dict.get('bbox', original.bbox if original else (0, 0, 100, 100)),
                    mean_conf=original.mean_conf if original else 1.0,
                    lines=original.lines if original else [],
                    lang=original.lang if original else '',
                    is_akk=original.is_akk if original else False,
                    akk_conf=original.akk_conf if original else 0.0,
                    column_index=split_dict.get('column', original.column_index if original else 0),
                    reading_order=original.reading_order if original else 0
                )
                new_block.split_reason = split_dict.get('split_reason', 'original')
                blocks.append(new_block)
            
            logger.debug(f"Block splitting: {len(block_dicts)} → {len(blocks)} fragments")
        
        # PHASE 2: Tag blocks with semantic roles (NEW)
        if block_clean_config.get('role_tagging', True):
            # Convert blocks to dicts for tagging
            block_dicts = []
            for block in blocks:
                block_dicts.append({
                    'block_id': getattr(block, 'block_id', 'unknown'),
                    'text': block.text,
                    'bbox': block.bbox,
                    'column': block.column_index
                })
            
            # Tag with roles
            tagged = tag_block_roles(block_dicts, config=block_clean_config)
            
            # Apply roles back to block objects
            for i, block in enumerate(blocks):
                if i < len(tagged):
                    block.role = tagged[i].get('role', 'other')
                    block.role_confidence = tagged[i].get('role_confidence', 0.0)
                    block.role_reasons = tagged[i].get('role_reasons', [])
            
            # DIAGNOSTIC: Log role distribution per page
            role_counts = {}
            for b in blocks:
                r = getattr(b, "role", "unassigned")
                role_counts[r] = role_counts.get(r, 0) + 1
            logger.info(f"[roles] page={page_no} dist={role_counts}")
            
            logger.debug(f"Role tagging: {len(blocks)} blocks tagged")
        
        # Detect Akkadian using config (with any-line aggregation if enabled)
        agg_mode = detection_config.get('aggregation_mode', 'block-level')
        agg_qual_lines_min = detection_config.get('aggregation_qual_lines_min', 3)
        agg_qual_ratio_min = detection_config.get('aggregation_qual_ratio_min', 0.25)
        
        for block in blocks:
            if agg_mode == 'any-line':
                # Line-by-line detection with aggregation
                lines = block.text.split('\n')
                qual_lines = 0
                total_lines = len([line for line in lines if line.strip()])
                
                for line in lines:
                    if not line.strip():
                        continue
                    is_akk, akk_conf = is_akkadian_transliteration(line, config=detection_config)
                    if is_akk:
                        qual_lines += 1
                
                # Aggregate: block is Akkadian if (qual_lines >= N) OR (qual_ratio >= M)
                qual_ratio = qual_lines / total_lines if total_lines > 0 else 0.0
                block_is_akkadian = (qual_lines >= agg_qual_lines_min) or (qual_ratio >= agg_qual_ratio_min)
                
                if block_is_akkadian:
                    block.lang = "akkadian"
                    block.is_akk = True
                    block.akk_conf = qual_ratio  # Use qualified ratio as confidence
                    result['akkadian_blocks'] += 1
                else:
                    # Fall back to general language detection
                    lang_result = detect_language(block.text)
                    if isinstance(lang_result, tuple):
                        lang, conf = lang_result
                    else:
                        lang = lang_result
                        conf = 0.5
                    block.lang = lang
                    block.is_akk = False
                    block.akk_conf = 0.0
                    result['translation_blocks'] += 1
            else:
                # Original block-level detection
                is_akk, akk_conf = is_akkadian_transliteration(block.text, config=detection_config)
                
                if is_akk:
                    block.lang = "akkadian"
                    block.is_akk = True
                    block.akk_conf = akk_conf
                    result['akkadian_blocks'] += 1
                else:
                    # Fall back to general language detection
                    lang_result = detect_language(block.text)
                    if isinstance(lang_result, tuple):
                        lang, conf = lang_result
                    else:
                        lang = lang_result
                        conf = 0.5
                    block.lang = lang
                    block.is_akk = False
                    block.akk_conf = 0.0
                    result['translation_blocks'] += 1
        
        # PHASE 3: Filter blocks for pairing (NEW)
        # Remove reference_meta, header_footer, figure_caption from translation candidates
        exclude_roles = block_clean_config.get('exclude_roles_in_pairing', [
            'reference_meta', 'header_footer', 'figure_caption'
        ])
        
        # Count before filtering for metrics
        total_blocks_before = len(blocks)
        excluded_blocks = [b for b in blocks if hasattr(b, 'role') and b.role.value in exclude_roles]
        
        # Filter blocks for pairing (keep akkadian + valid translation candidates)
        pairing_blocks = [b for b in blocks 
                         if not hasattr(b, 'role') or b.role.value not in exclude_roles]
        
        # DIAGNOSTIC: Log filter effect with block IDs
        excluded_ids = [getattr(b, "block_id", None) for b in excluded_blocks]
        logger.info(f"[filter] page={page_no} kept={len(pairing_blocks)} "
                   f"excluded={len(excluded_blocks)} by_roles={list(exclude_roles)} "
                   f"excluded_ids={excluded_ids[:5]}...")  # Show first 5 IDs
        
        logger.info(f"Pairing filter: {total_blocks_before} blocks → {len(pairing_blocks)} candidates "
                   f"(excluded {len(excluded_blocks)} as {', '.join(exclude_roles)})")
        
        # Use filtered blocks for pairing
        blocks_for_pairing = pairing_blocks
        
        # Pair translations - load config from profile
        config = PairingConfig.from_dict(pairing_config) if pairing_config else PairingConfig()
        
        pairer = TranslationPairer(config)
        pairs = pairer.pair_blocks(blocks_for_pairing, page=page_no, pdf_id=pdf_stem)
        
        result['pairs_created'] = len(pairs)
        
        if pairs:
            result['avg_confidence'] = sum(p.score for p in pairs) / len(pairs)
            
            # Save per-PDF translations CSV
            pdf_out = output_dir / pdf_stem
            pdf_out.mkdir(parents=True, exist_ok=True)
            pairer.save_pairs_csv(pairs, pdf_out / "translations.csv")
            
            # Append to client CSV if requested
            if client_csv:
                for pair in pairs:
                    append_to_client_csv(
                        client_csv,
                        pdf_name=pdf_name,
                        page_no=page_no,
                        akkadian_text=pair.akk_text,
                        translation=pair.trans_text,
                        confidence=pair.score,
                        status='OK'
                    )
        
        result['success'] = True
        result['processing_time'] = time.time() - start_time
        
    except Exception as e:
        logger.error(f"Error processing {pdf_name} page {page_no}: {e}")
        result['error'] = str(e)
        result['processing_time'] = time.time() - start_time
        
        # Write error to client CSV if requested
        if client_csv:
            append_to_client_csv(
                client_csv,
                pdf_name=pdf_name,
                page_no=page_no,
                akkadian_text='',
                translation='',
                confidence=0.0,
                status=f'ERROR: {str(e)}'
            )
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Process PDF manifest with streaming and live progress')
    parser.add_argument('--manifest', required=True, help='Manifest TSV file (pdf_path, page_no)')
    parser.add_argument('--profile', default='profiles/akkadian_strict.json', help='Detection config JSON')
    parser.add_argument('--engines', default='paddle', help='OCR engines (comma-separated)')
    parser.add_argument('--client-csv', help='Output client CSV path (UTF-8 BOM, continuous append)')
    parser.add_argument('--progress-csv', help='Progress tracking CSV (for resume)')
    parser.add_argument('--output-dir', default='reports/manifest_run', help='Output directory')
    parser.add_argument('--resume', action='store_true', help='Resume from progress CSV')
    parser.add_argument('--live-progress', action='store_true', help='Show live progress bar with tqdm')
    parser.add_argument('--max-pages-in-mem', type=int, default=1, help='Max pages to hold in memory')
    parser.add_argument('--throttle-mem', type=float, default=0.80, help='Throttle if RSS > this fraction')
    parser.add_argument('--limit', type=int, help='Limit number of pages to process (for testing)')
    
    # Detection tuning overrides
    parser.add_argument('--akkadian-threshold', type=float, help='Override Akkadian detection threshold (0.0-1.0)')
    parser.add_argument('--require-diacritic-or-marker', action='store_true', help='Require diacritics or markers for Akkadian detection')
    parser.add_argument('--min-diacritics', type=int, help='Minimum diacritics per line for Akkadian detection')
    parser.add_argument('--min-syllabic-tokens', type=int, help='Minimum syllabic tokens for Akkadian detection')
    parser.add_argument('--min-syllabic-ratio', type=float, help='Minimum syllabic ratio (0.0-1.0) for Akkadian detection')
    parser.add_argument('--markers-strict', action='store_true', help='Use strict logogram markers only (DUMU, LUGAL, KÙ.BABBAR, etc.)')
    
    # Audit mode
    parser.add_argument('--detect-only', action='store_true', help='Detection audit mode: OCR + detection only, no LLM/pairing')
    parser.add_argument('--sample', type=int, help='Process only first N pages (for audit/testing)')
    
    # LLM control
    parser.add_argument('--llm-off', action='store_true', help='Disable LLM correction (heuristic pairing only)')
    parser.add_argument('--llm-on', action='store_true', help='Force enable LLM correction')
    parser.add_argument('--llm-json-strict', action='store_true', help='Use strict JSON mode for LLM')
    parser.add_argument('--llm-max-retries', type=int, help='Max LLM retry attempts')
    
    # Pairing control
    parser.add_argument('--pairing', choices=['heuristic', 'standard'], default='standard', help='Pairing method (heuristic=no LLM, standard=with LLM)')
    parser.add_argument('--output-root', help='Alias for --output-dir')
    
    # Production flags
    parser.add_argument('--prefer-text-layer', action='store_true', default=True, help='Extract embedded text first, OCR only if needed (default: True)')
    parser.add_argument('--resume-safe', action='store_true', help='Alias for --resume')
    parser.add_argument('--skip-completed', action='store_true', help='Skip pages in progress CSV (same as --resume)')
    parser.add_argument('--status-bar', action='store_true', help='Alias for --live-progress')
    parser.add_argument('--only-unpaired', action='store_true', help='Only process pages with no existing translation pairs')
    
    # Fail-fast gates
    parser.add_argument('--fail-fast-check-every', type=int, default=25, help='Check client CSV row count every N pages (default: 25)')
    parser.add_argument('--fail-fast-min-rows', type=int, default=5, help='Abort if client CSV has fewer than M rows after checkpoint (default: 5)')
    
    args = parser.parse_args()
    
    # Handle aliases
    if args.output_root:
        output_dir = Path(args.output_root)
    else:
        output_dir = Path(args.output_dir)
    
    if args.resume_safe or args.skip_completed:
        args.resume = True
    
    if args.status_bar:
        args.live_progress = True
    
    # Setup paths
    manifest_path = Path(args.manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default progress CSV if not specified
    progress_csv = Path(args.progress_csv) if args.progress_csv else output_dir / 'progress.csv'
    
    # Load detection config
    config_path = Path(args.profile)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        detection_config = profile.get("akkadian_detection", {})
        logger.info(f"Loaded detection config: {config_path}")
    else:
        logger.warning(f"Config not found: {config_path}, using defaults")
        detection_config = {
            'threshold': 0.50,
            'require_diacritic_or_marker': True,
            'min_diacritics_per_line': 1
        }
    
    # Apply CLI overrides
    if args.akkadian_threshold is not None:
        detection_config['threshold'] = args.akkadian_threshold
        logger.info(f"CLI override: threshold={args.akkadian_threshold}")
    if args.require_diacritic_or_marker:
        detection_config['require_diacritic_or_marker'] = True
        logger.info(f"CLI override: require_diacritic_or_marker=True")
    if args.min_diacritics is not None:
        detection_config['min_diacritics_per_line'] = args.min_diacritics
        logger.info(f"CLI override: min_diacritics_per_line={args.min_diacritics}")
    if args.min_syllabic_tokens is not None:
        detection_config['min_syllabic_tokens'] = args.min_syllabic_tokens
        logger.info(f"CLI override: min_syllabic_tokens={args.min_syllabic_tokens}")
    if args.min_syllabic_ratio is not None:
        detection_config['min_syllabic_ratio'] = args.min_syllabic_ratio
        logger.info(f"CLI override: min_syllabic_ratio={args.min_syllabic_ratio}")
    if args.markers_strict:
        detection_config['markers_strict'] = True
        logger.info(f"CLI override: markers_strict=True")
    
    # Log final detection config
    logger.info(f"[detect] using config: {detection_config}")
    logger.info(f"  threshold={detection_config.get('threshold', 0.60)}")
    logger.info(f"  require_diacritic_or_marker={detection_config.get('require_diacritic_or_marker', True)}")
    logger.info(f"  min_diacritics_per_line={detection_config.get('min_diacritics_per_line', 1)}")
    logger.info(f"  min_syllabic_tokens={detection_config.get('min_syllabic_tokens', 3)}")
    logger.info(f"  min_syllabic_ratio={detection_config.get('min_syllabic_ratio', 0.5)}")
    logger.info(f"  markers_strict={detection_config.get('markers_strict', True)}")
    
    # Extract pairing config from profile
    pairing_config = profile.get("pairing", {})
    logger.info(f"[pairing] Loaded config with {len(pairing_config)} keys")
    if 'strategy' in pairing_config:
        logger.info(f"  Tiered strategy enabled: {list(pairing_config['strategy'].keys())}")
    
    # Initialize OCR engines
    from paddleocr import PaddleOCR
    from ensemble import OCREnsemble
    
    engines_list = [e.strip() for e in args.engines.split(',')]
    logger.info(f"Initializing OCR engines: {engines_list}")
    
    ocr_engines = {}
    
    # Initialize each requested engine
    for engine_name in engines_list:
        try:
            if engine_name.lower() == 'paddle':
                logger.info("Initializing PaddleOCR...")
                ocr_engines['paddle'] = PaddleOCR(
                    use_textline_orientation=True,
                    lang='en'
                )
                logger.info("PaddleOCR initialized")
            elif engine_name.lower() in ['doctr', 'mmocr', 'kraken']:
                # Try to initialize other engines
                try:
                    from engines import create_engine, ENGINE_AVAILABILITY
                    if ENGINE_AVAILABILITY.get(engine_name.lower(), False):
                        logger.info(f"Initializing {engine_name}...")
                        ocr_engines[engine_name.lower()] = create_engine(
                            engine_name.lower(),
                            profile='balanced',
                            device='auto'
                        )
                        logger.info(f"{engine_name} initialized")
                    else:
                        logger.warning(f"{engine_name} not available, skipping")
                except Exception as e:
                    logger.warning(f"Failed to initialize {engine_name}: {e}")
            else:
                logger.warning(f"Unknown engine: {engine_name}, skipping")
        except Exception as e:
            logger.error(f"Failed to initialize {engine_name}: {e}")
    
    if not ocr_engines:
        logger.error("No OCR engines initialized!")
        return 1
    
    logger.info(f"Active OCR engines: {list(ocr_engines.keys())}")
    
    # Initialize ensemble if multiple engines
    ensemble = None
    if len(ocr_engines) > 1:
        fusion_config = profile.get("fusion", {})
        ensemble_config = {
            'voting_method': fusion_config.get('method', 'weighted'),
            'overlap_threshold': 0.5,
            'min_confidence': profile.get('engines', {}).get('min_confidence', 0.55),
            'engine_weights': fusion_config.get('weights', {})
        }
        ensemble = OCREnsemble(ensemble_config)
        logger.info(f"Ensemble initialized with method: {ensemble_config['voting_method']}")
    else:
        logger.info("Single engine mode (no ensemble fusion)")
    
    # Initialize LLM corrector if enabled
    llm_corrector = None
    llm_config = profile.get("llm", {})
    guardrails_config = profile.get("guardrails", {})
    
    # Determine if LLM should be enabled
    llm_enabled = llm_config.get("enabled", False)
    if args.llm_off or args.pairing == 'heuristic':
        llm_enabled = False
        logger.info("LLM correction DISABLED (--llm-off or --pairing=heuristic)")
    elif args.llm_on:
        llm_enabled = True
        logger.info("LLM correction FORCED ON (--llm-on)")
    
    # Apply LLM overrides to config
    if args.llm_max_retries is not None:
        llm_config['max_retries'] = args.llm_max_retries
        logger.info(f"CLI override: llm_max_retries={args.llm_max_retries}")
    
    if args.llm_json_strict:
        llm_config['json_mode'] = True
        logger.info(f"CLI override: json_mode=True (strict JSON)")
    
    if llm_enabled:
        try:
            from llm.corrector import LLMCorrector
            from llm.clients.ollama_client import OllamaConfig
            
            logger.info("Initializing LLM corrector with guardrails...")
            logger.info(f"  Model: {llm_config.get('model', 'qwen2.5:7b-instruct')}")
            logger.info(f"  Akkadian edit budget: {guardrails_config.get('edit_budget_akkadian', 0.03):.1%}")
            logger.info(f"  Non-Akkadian edit budget: {guardrails_config.get('edit_budget_non_akkadian', 0.12):.1%}")
            
            # Create Ollama config
            ollama_config = OllamaConfig(
                base_url=llm_config.get("base_url", "http://localhost:11434"),
                model_id=llm_config.get("model", "qwen2.5:7b-instruct"),
                max_tokens=llm_config.get("max_tokens", 512),
                temperature=llm_config.get("temperature", 0.3),
                timeout_s=llm_config.get("timeout_s", 120),
                retries=llm_config.get("max_retries", 1)
            )
            
            # Initialize corrector with config
            llm_corrector = LLMCorrector(
                ollama_config=ollama_config,
                cache={},  # Empty cache to start
                enable_telemetry=True
            )
            
            # Store guardrails in corrector for use during correction
            llm_corrector.edit_budget_akkadian = guardrails_config.get('edit_budget_akkadian', 0.03)
            llm_corrector.edit_budget_non_akkadian = guardrails_config.get('edit_budget_non_akkadian', 0.12)
            llm_corrector.preserve_diacritics = guardrails_config.get('preserve_diacritics', True)
            llm_corrector.preserve_superscripts = guardrails_config.get('preserve_superscripts', True)
            llm_corrector.preserve_brackets = guardrails_config.get('preserve_brackets', True)
            
            logger.info("LLM corrector initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM corrector: {e}")
            logger.warning("Continuing without LLM correction")
            llm_corrector = None
    else:
        logger.info("LLM correction disabled in profile")
    
    # Load manifest
    manifest = load_manifest(manifest_path)
    
    # Apply limit if specified
    if args.limit:
        manifest = manifest[:args.limit]
        logger.info(f"Limited to first {args.limit} pages")
    
    # Resume handling
    completed = set()
    if args.resume:
        completed = load_completed_pages(progress_csv)
        manifest = [(p, n) for p, n in manifest if (p, n) not in completed]
        logger.info(f"Resuming: {len(manifest)} pages remaining after skipping {len(completed)} completed")
    
    if not manifest:
        logger.info("No pages to process!")
        return 0
    
    # DETECTION AUDIT MODE: Run audit and exit
    if args.detect_only:
        from detect_audit import run_audit
        
        audit_csv = output_dir / "detection_audit.csv"
        
        logger.info("=" * 60)
        logger.info("DETECTION AUDIT MODE")
        logger.info("=" * 60)
        logger.info(f"Total pages in manifest: {len(manifest)}")
        logger.info(f"Sample size: {args.sample or 'ALL'}")
        logger.info(f"Detection threshold: {detection_config.get('threshold', 0.50)}")
        logger.info(f"Require diacritics/markers: {detection_config.get('require_diacritic_or_marker', True)}")
        logger.info(f"Min diacritics per line: {detection_config.get('min_diacritics_per_line', 1)}")
        logger.info(f"Output audit CSV: {audit_csv}")
        logger.info("=" * 60)
        
        run_audit(
            manifest=manifest,
            detection_config=detection_config,
            ocr_engines=ocr_engines,
            ensemble=ensemble,
            output_csv=audit_csv,
            sample_size=args.sample
        )
        
        return 0
    
    # Initialize client CSV with header if needed
    client_csv_path = Path(args.client_csv) if args.client_csv else None
    if client_csv_path and not client_csv_path.exists():
        append_to_client_csv(
            client_csv_path,
            pdf_name='', page_no=0, akkadian_text='', translation='', confidence=0.0, status='',
            write_header=True
        )
        logger.info(f"Initialized client CSV: {client_csv_path}")
    
    # Initialize progress CSV with header if needed
    if not progress_csv.exists():
        append_to_progress_csv(
            progress_csv,
            pdf_path='', page_no=0, success=False, processing_time=0.0,
            write_header=True
        )
        logger.info(f"Initialized progress CSV: {progress_csv}")
    
    # Process manifest
    logger.info("=" * 60)
    logger.info("STARTING MANIFEST PROCESSING")
    logger.info("=" * 60)
    logger.info(f"Total pages to process: {len(manifest)}")
    logger.info(f"Detection config: {args.profile}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Client CSV: {client_csv_path or 'None'}")
    logger.info(f"Progress CSV: {progress_csv}")
    logger.info(f"Resume mode: {args.resume}")
    logger.info(f"Live progress: {args.live_progress}")
    logger.info("=" * 60)
    
    results = []
    start_time = time.time()
    
    # Progress bar setup
    if args.live_progress:
        pbar = tqdm(manifest, desc="Processing pages", unit="page")
    else:
        pbar = manifest
    
    pages_processed = 0
    
    for pdf_path, page_no in pbar:
        # Check memory usage
        mem_usage = get_memory_usage()
        if mem_usage > args.throttle_mem:
            logger.warning(f"Memory usage high: {mem_usage:.1%} > {args.throttle_mem:.1%}, throttling...")
            time.sleep(1)  # Brief pause to allow GC
        
        # Process page
        result = process_page(
            pdf_path=pdf_path,
            page_no=page_no,
            detection_config=detection_config,
            pairing_config=pairing_config,
            output_dir=output_dir,
            ocr_engines=ocr_engines,
            ensemble=ensemble,
            llm_corrector=llm_corrector,
            client_csv=client_csv_path
        )
        
        results.append(result)
        pages_processed += 1
        
        # Append to progress CSV immediately
        append_to_progress_csv(
            progress_csv,
            pdf_path=pdf_path,
            page_no=page_no,
            success=result['success'],
            processing_time=result['processing_time'],
            error=result['error']
        )
        
        # Fail-fast gate: Check client CSV row count every N pages
        if client_csv_path and hasattr(args, 'fail_fast_check_every') and hasattr(args, 'fail_fast_min_rows'):
            if pages_processed % args.fail_fast_check_every == 0:
                # Count rows in client CSV (excluding header)
                try:
                    if client_csv_path.exists():
                        with open(client_csv_path, 'r', encoding='utf-8-sig') as f:
                            rows = list(csv.DictReader(f))
                            row_count = len(rows)
                        
                        logger.info(f"[FAIL-FAST:QUANTITY] Checkpoint at {pages_processed} pages: {row_count} translation pairs in client CSV")
                        
                        if row_count < args.fail_fast_min_rows:
                            logger.error(f"[FAIL-FAST:QUANTITY] ABORTING: Only {row_count} translation pairs after {pages_processed} pages (minimum required: {args.fail_fast_min_rows})")
                            logger.error(f"[FAIL-FAST] Detection threshold may be too strict or PDFs lack Akkadian content")
                            logger.error(f"[FAIL-FAST] Review detection config: {args.profile}")
                            
                            # Close progress bar before exit
                            if args.live_progress:
                                pbar.close()
                            
                            return 1  # Exit with error code
                        else:
                            logger.info(f"[FAIL-FAST:QUANTITY] ✅ Gate passed: {row_count} >= {args.fail_fast_min_rows} pairs")
                        
                        # NEW: Quality gate - check for reference metadata leakage
                        REF_PAT = re.compile(
                            r"(HW\s+s\.)|(Kt\s+[a-z]/k)|(\bM[üu]ze\s+env\.)"
                            r"|(Env\.\s*Nr\.)|(vgl\.)|(Bkz\.)|(\bsceau\b)",
                            re.IGNORECASE
                        )
                        recent_rows = rows[-300:] if len(rows) > 300 else rows
                        ref_hits = sum(1 for r in recent_rows if REF_PAT.search(r.get('translation', '')))
                        ratio = ref_hits / max(1, len(recent_rows))
                        
                        logger.info(f"[FAIL-FAST:QUALITY] Reference metadata in last {len(recent_rows)} pairs: {ref_hits} ({ratio:.1%})")
                        
                        if ratio > 0.10:  # >10% reference metadata
                            logger.error(f"[FAIL-FAST:QUALITY] ABORTING: {ratio:.1%} reference metadata in recent pairs (threshold: 10%)")
                            logger.error(f"[FAIL-FAST:QUALITY] Role filtering not working - check block_clean config and src/block_roles.py")
                            logger.error(f"[FAIL-FAST:QUALITY] Examples: {[r.get('translation','')[:50] for r in recent_rows if REF_PAT.search(r.get('translation',''))][:3]}")
                            
                            # Close progress bar before exit
                            if args.live_progress:
                                pbar.close()
                            
                            return 1  # Exit with error code
                        else:
                            logger.info(f"[FAIL-FAST:QUALITY] ✅ Gate passed: {ratio:.1%} <= 10% reference leakage")
                        
                except Exception as e:
                    logger.warning(f"[FAIL-FAST] Failed to check client CSV: {e}")
        
        # Update progress bar postfix
        if args.live_progress:
            elapsed = time.time() - start_time
            pages_done = len(results)
            rate = pages_done / elapsed if elapsed > 0 else 0
            eta_seconds = (len(manifest) - pages_done) / rate if rate > 0 else 0
            eta_str = f"{eta_seconds/60:.1f}min" if eta_seconds > 0 else "N/A"
            
            pbar.set_postfix({
                'rate': f'{rate:.1f}p/min',
                'ETA': eta_str,
                'mem': f'{mem_usage:.1%}'
            })
    
    # Close progress bar
    if args.live_progress:
        pbar.close()
    
    # Generate summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    total_akk = sum(r['akkadian_blocks'] for r in successful)
    total_pairs = sum(r['pairs_created'] for r in successful)
    
    avg_time = sum(r['processing_time'] for r in successful) / len(successful) if successful else 0
    throughput = len(successful) / total_time if total_time > 0 else 0
    
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total pages: {len(manifest)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Akkadian blocks detected: {total_akk}")
    logger.info(f"Translation pairs created: {total_pairs}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Avg time per page: {avg_time:.2f}s")
    logger.info(f"Throughput: {throughput:.1f} pages/min")
    logger.info("=" * 60)
    
    # Write summary to file
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("MANIFEST PROCESSING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Manifest: {manifest_path}\n")
        f.write(f"Detection config: {args.profile}\n")
        f.write(f"\nRESULTS:\n")
        f.write(f"  Total pages: {len(manifest)}\n")
        f.write(f"  Successful: {len(successful)}\n")
        f.write(f"  Failed: {len(failed)}\n")
        f.write(f"  Akkadian blocks: {total_akk}\n")
        f.write(f"  Translation pairs: {total_pairs}\n")
        f.write(f"\nPERFORMANCE:\n")
        f.write(f"  Total time: {total_time/60:.1f} minutes\n")
        f.write(f"  Avg time per page: {avg_time:.2f}s\n")
        f.write(f"  Throughput: {throughput:.1f} pages/min\n")
        f.write(f"\nOUTPUTS:\n")
        f.write(f"  Client CSV: {client_csv_path or 'None'}\n")
        f.write(f"  Progress CSV: {progress_csv}\n")
        f.write(f"  Output directory: {output_dir}\n")
        f.write("=" * 60 + "\n")
    
    logger.info(f"Summary written to: {summary_path}")
    
    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
