"""
CSV writer utilities for outputting translation results.
"""
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Dataclass representing a translation result for CSV output."""
    page_id: str
    lang: str
    text: str
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    engine: str
    conf: float


def dict_to_translation_result(translation_dict: Dict, page_id: str) -> TranslationResult:
    """Convert translation dictionary to TranslationResult dataclass."""
    bbox = translation_dict.get('bbox', (0, 0, 0, 0))
    
    # Handle both 'conf' and 'confidence' field names
    confidence = translation_dict.get('conf', translation_dict.get('confidence', 0.0))
    
    return TranslationResult(
        page_id=page_id,
        lang=translation_dict['lang'],
        text=translation_dict['text'],
        bbox_x=bbox[0],
        bbox_y=bbox[1],
        bbox_w=bbox[2],
        bbox_h=bbox[3],
        engine=translation_dict['engine'],
        conf=round(float(confidence), 3)  # Ensure it's a Python float
    )


def write_csv(filepath: Union[str, Path], translations: List[Union[Dict, TranslationResult]], page_id: str = None) -> None:
    """
    Write translations to CSV file.
    
    Args:
        filepath: Path to output CSV file
        translations: List of translation dictionaries or TranslationResult objects
        page_id: Page ID to use if not provided in translation objects
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dictionaries to TranslationResult objects if needed
    csv_rows = []
    for translation in translations:
        if isinstance(translation, dict):
            if page_id is None:
                raise ValueError("page_id must be provided when using translation dictionaries")
            csv_rows.append(dict_to_translation_result(translation, page_id))
        elif isinstance(translation, TranslationResult):
            csv_rows.append(translation)
        else:
            raise ValueError(f"Unsupported translation type: {type(translation)}")
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            # Define CSV headers
            fieldnames = [
                'page_id', 'lang', 'text', 
                'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                'engine', 'conf'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            
            # Write header
            writer.writeheader()
            
            # Write data rows
            for row in csv_rows:
                writer.writerow({
                    'page_id': row.page_id,
                    'lang': row.lang,
                    'text': row.text,
                    'bbox_x': row.bbox_x if row.bbox_x is not None else '',
                    'bbox_y': row.bbox_y if row.bbox_y is not None else '',
                    'bbox_w': row.bbox_w if row.bbox_w is not None else '',
                    'bbox_h': row.bbox_h if row.bbox_h is not None else '',
                    'engine': row.engine,
                    'conf': row.conf
                })
        
        logger.debug(f"Wrote {len(csv_rows)} translations to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to write CSV {filepath}: {e}")
        raise


def read_csv(filepath: Union[str, Path]) -> List[TranslationResult]:
    """Read translations from CSV file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    translations = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                # Handle empty bbox values
                def safe_int(value, default=0):
                    try:
                        return int(value) if value.strip() else default
                    except (ValueError, AttributeError):
                        return default
                
                def safe_float(value, default=0.0):
                    try:
                        return float(value) if value.strip() else default
                    except (ValueError, AttributeError):
                        return default
                
                translation = TranslationResult(
                    page_id=row['page_id'],
                    lang=row['lang'],
                    text=row['text'],
                    bbox_x=safe_int(row.get('bbox_x', 0)),
                    bbox_y=safe_int(row.get('bbox_y', 0)),
                    bbox_w=safe_int(row.get('bbox_w', 0)),
                    bbox_h=safe_int(row.get('bbox_h', 0)),
                    engine=row['engine'],
                    conf=safe_float(row['conf'], 0.0)
                )
                
                translations.append(translation)
        
        logger.debug(f"Read {len(translations)} translations from {filepath}")
        return translations
        
    except Exception as e:
        logger.error(f"Failed to read CSV {filepath}: {e}")
        raise


def validate_csv_output(filepath: Union[str, Path]) -> bool:
    """Validate that CSV file was written correctly and contains expected columns."""
    try:
        translations = read_csv(filepath)
        
        if not translations:
            logger.warning(f"CSV file is empty: {filepath}")
            return False
        
        # Check required fields
        for i, translation in enumerate(translations[:5]):  # Check first 5 rows
            if not translation.page_id or not translation.lang or not translation.text:
                logger.error(f"Missing required fields in row {i}: {translation}")
                return False
            
            if translation.lang not in {'fr', 'de', 'tr', 'en', 'it'}:
                logger.error(f"Invalid language code in row {i}: {translation.lang}")
                return False
            
            if not (0.0 <= translation.conf <= 1.0):
                logger.warning(f"Confidence out of range in row {i}: {translation.conf}")
        
        logger.info(f"CSV validation passed: {filepath} ({len(translations)} translations)")
        return True
        
    except Exception as e:
        logger.error(f"CSV validation failed for {filepath}: {e}")
        return False
