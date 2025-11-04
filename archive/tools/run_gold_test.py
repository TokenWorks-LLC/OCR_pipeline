#!/usr/bin/env python3
import sys, json, logging, argparse, time, csv, os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

# Set environment variable BEFORE importing modules that use it
if 'AKKADIAN_LM_PATH' not in os.environ:
    default_lm_path = Path(__file__).parent.parent / 'models' / 'akkadian_char_lm.json'
    if default_lm_path.exists():
        os.environ['AKKADIAN_LM_PATH'] = str(default_lm_path)
        print(f"Auto-detected Akkadian LM: {default_lm_path}")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from paddleocr import PaddleOCR
from blockification import TextBlockifier
from translation_pairing import TranslationPairer
from lang_and_akkadian import detect_language, is_akkadian_transliteration

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Global detection config (set in main, used in process_page)
DETECTION_CONFIG = {}

@dataclass
class ManifestEntry:
    pdf_path: str
    page_no: int

@dataclass  
class PageResult:
    pdf_id: str
    success: bool
    time: float
    ocr_lines: int = 0
    blocks: int = 0
    akk_blocks: int = 0
    trans_blocks: int = 0
    pairs: int = 0
    score: float = 0.0
    error: str = None

def load_manifest(path):
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, row in enumerate(csv.reader(f, delimiter='\t')):
            if i == 0 or len(row) < 2:  # Skip header and empty rows
                continue
            entries.append(ManifestEntry(row[0], int(row[1])))
    return entries

def process_page(pdf_path, page_no, paddle, blockifier, pairer, out_dir):
    import fitz, cv2, numpy as np
    t0 = time.time()
    pdf_path = Path(pdf_path)
    result = PageResult(pdf_path.stem, False, 0.0)
    
    try:
        doc = fitz.open(str(pdf_path))
        page = doc.load_page(page_no - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = cv2.imdecode(np.frombuffer(pix.tobytes("ppm"), np.uint8), cv2.IMREAD_COLOR)
        
        # Get page dimensions
        page_width = img.shape[1]
        page_height = img.shape[0]
        
        doc.close()
        
        ocr_result = paddle.predict(img)
        lines = []
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
                    
                    lines.append({"text": text, "bbox": bbox, "confidence": float(conf)})
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
                        lines.append({"text": text, "bbox": [min(xs),min(ys),max(xs),max(ys)], "confidence": float(conf)})
        
        result.ocr_lines = len(lines)
        if not lines:
            result.success = True
            result.time = time.time() - t0
            return result
        
        blocks = blockifier.blockify(lines, page_num=page_no, page_width=page_width, page_height=page_height)
        result.blocks = len(blocks)
        
        for b in blocks:
            # First check if Akkadian using dedicated detector with config
            is_akk_result, akk_conf = is_akkadian_transliteration(b.text, config=DETECTION_CONFIG)
            
            if is_akk_result:
                b.lang = "akkadian"  # Use correct attribute name
                b.is_akk = True      # Use correct attribute name
                b.akk_conf = akk_conf
                result.akk_blocks += 1
            else:
                # Fall back to general language detection
                lang_result = detect_language(b.text)
                # Handle both tuple and single value returns
                if isinstance(lang_result, tuple):
                    lang, conf = lang_result
                else:
                    lang = lang_result
                    conf = 0.5
                b.lang = lang        # Use correct attribute name
                b.is_akk = False     # Use correct attribute name
                b.akk_conf = 0.0
                result.trans_blocks += 1
        
        pairs = pairer.pair_blocks(blocks, page=page_no, pdf_id=pdf_path.stem)
        result.pairs = len(pairs)
        
        # Debug: log pairing results
        if len(akk_blocks := [b for b in blocks if b.is_akk]) > 0:
            trans_blocks = [b for b in blocks if not b.is_akk]
            logger.debug(
                f"Page {page_no}: {len(akk_blocks)} Akkadian, "
                f"{len(trans_blocks)} translation, {len(pairs)} pairs created"
            )
        
        if pairs:
            result.score = sum(p.score for p in pairs) / len(pairs)
        
        pdf_out = out_dir / pdf_path.stem
        pdf_out.mkdir(parents=True, exist_ok=True)
        from pathlib import Path as PathLib
        pairer.save_pairs_csv(pairs, PathLib(pdf_out / "translations.csv"))
        
        result.success = True
    except Exception as e:
        logger.error(f"Error: {e}")
        result.error = str(e)
    
    result.time = time.time() - t0
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile", default="profiles/akkadian_strict.json")
    parser.add_argument("--akkadian-threshold", type=float, default=None,
                       help="Override Akkadian detection threshold (default from profile)")
    parser.add_argument("--require-diacritic-or-marker", action="store_true",
                       help="Require diacritic or marker for Akkadian detection")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--engines", default="paddle", help="Comma-separated engine list")
    args = parser.parse_args()
    
    # Load profile configuration
    profile_path = Path(args.profile)
    if profile_path.exists():
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        logger.info(f"Loaded profile: {profile_path}")
    else:
        logger.warning(f"Profile not found: {profile_path}, using defaults")
        profile = {}
    
    # Get Akkadian detection config from profile
    det_cfg = profile.get("akkadian_detection", {})
    
    # CLI overrides take precedence
    if args.akkadian_threshold is not None:
        det_cfg["threshold"] = args.akkadian_threshold
    if args.require_diacritic_or_marker:
        det_cfg["require_diacritic_or_marker"] = True
    
    # Set defaults if not specified
    if "threshold" not in det_cfg:
        det_cfg["threshold"] = 0.65
    if "require_diacritic_or_marker" not in det_cfg:
        det_cfg["require_diacritic_or_marker"] = True
    
    # Log the actual detection config being used
    logger.info(f"[detect] using config: {det_cfg}")
    print(f"\n{'='*60}")
    print(f"AKKADIAN DETECTION CONFIG")
    print(f"{'='*60}")
    print(f"Threshold: {det_cfg.get('threshold', 0.65)}")
    print(f"Require diacritic/marker: {det_cfg.get('require_diacritic_or_marker', True)}")
    print(f"{'='*60}\n")
    
    # Store config globally so process_page can access it
    global DETECTION_CONFIG
    DETECTION_CONFIG = det_cfg
    
    out_root = Path(args.output_root or f"reports/gold_full_{datetime.now().strftime('%Y%m%d_%H%M')}")
    out_root.mkdir(parents=True, exist_ok=True)
    
    entries = load_manifest(args.manifest)
    if args.limit:
        entries = entries[:args.limit]
    
    logger.info(f"Processing {len(entries)} pages")
    paddle = PaddleOCR(use_textline_orientation=True, lang="en")
    blockifier = TextBlockifier()
    pairer = TranslationPairer()
    
    results = []
    for i, e in enumerate(entries, 1):
        logger.info(f"[{i}/{len(entries)}] {Path(e.pdf_path).name}")
        r = process_page(e.pdf_path, e.page_no, paddle, blockifier, pairer, out_root / "outputs")
        results.append(r)
    
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    
    summary = {
        "total": len(results),
        "success": len(ok),
        "failed": len(fail),
        "ocr_lines": sum(r.ocr_lines for r in ok),
        "blocks": sum(r.blocks for r in ok),
        "akk": sum(r.akk_blocks for r in ok),
        "trans": sum(r.trans_blocks for r in ok),
        "pairs": sum(r.pairs for r in ok),
        "avg_time": sum(r.time for r in ok) / len(ok) if ok else 0
    }
    
    with open(out_root / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    report = f"""=== GOLD VALIDATION REPORT ===
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Processed: {summary["success"]}/{summary["total"]}
OCR Lines: {summary["ocr_lines"]}
Blocks: {summary["blocks"]} ({summary["akk"]} Akkadian, {summary["trans"]} translation)
Pairs: {summary["pairs"]}
Avg Time: {summary["avg_time"]:.2f}s/page
Status: {"PASS - All processed" if len(fail) == 0 else "FAIL - " + str(len(fail)) + " errors"}
"""
    
    with open(out_root / "ACCEPTANCE_REPORT.md", "w") as f:
        f.write(report)
    
    logger.info(f"Complete! Report: {out_root}/ACCEPTANCE_REPORT.md")
    print(f"\n{report}")

if __name__ == "__main__":
    main()