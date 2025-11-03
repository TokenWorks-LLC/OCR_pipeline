"""
Build research assets: character language model and lexicon.
Run this before enabling research features in the pipeline.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def build_character_lm(gold_csv: str, output_dir: str, order: int = 5):
    """Build character language model from gold data."""
    logger.info("=" * 60)
    logger.info("Building Character Language Model")
    logger.info("=" * 60)
    
    try:
        from build_char_lm import load_gold_texts, extract_charset, save_charset, prepare_training_data, build_kenlm, validate_lm
        
        # Load gold texts
        logger.info(f"Loading gold data from {gold_csv}")
        texts = load_gold_texts(gold_csv)
        logger.info(f"Loaded {len(texts)} gold texts")
        
        # Extract and save charset
        logger.info("Extracting character set")
        charset = extract_charset(texts)
        logger.info(f"Character set size: {len(charset)}")
        
        os.makedirs(output_dir, exist_ok=True)
        charset_path = os.path.join(output_dir, 'charset.txt')
        save_charset(charset, charset_path)
        logger.info(f"Saved charset to {charset_path}")
        
        # Prepare training data
        logger.info("Preparing training data")
        train_path = os.path.join(output_dir, 'train.txt')
        prepare_training_data(texts, train_path)
        logger.info(f"Saved training data to {train_path}")
        
        # Build KenLM
        logger.info(f"Building {order}-gram language model")
        arpa_path = os.path.join(output_dir, 'char.arpa')
        klm_path = os.path.join(output_dir, 'char.klm')
        
        build_kenlm(train_path, arpa_path, klm_path, order=order)
        logger.info(f"Built language model: {klm_path}")
        
        # Validate
        logger.info("Validating language model")
        validate_lm(klm_path, texts, charset_path)
        
        logger.info("✓ Character LM built successfully")
        logger.info(f"  Files:")
        logger.info(f"    - {charset_path}")
        logger.info(f"    - {train_path}")
        logger.info(f"    - {arpa_path}")
        logger.info(f"    - {klm_path}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import build_char_lm: {e}")
        logger.error("Make sure tools/build_char_lm.py exists")
        return False
    except Exception as e:
        logger.error(f"Failed to build character LM: {e}")
        return False


def build_lexicon(gold_csv: str, output_dir: str):
    """Build lexicon from gold data and default word lists."""
    logger.info("=" * 60)
    logger.info("Building Lexicon")
    logger.info("=" * 60)
    
    try:
        from lexicon_bias import LexiconBias
        import pandas as pd
        
        # Initialize lexicon
        lexicon = LexiconBias(min_freq=2, bias=0.95, max_boost_per_token=2.0)
        
        # Load default lexicons
        logger.info("Loading default lexicons")
        lexicon.load_sumerograms()
        logger.info(f"  Loaded Sumerograms")
        
        lexicon.load_akkadian_morphemes()
        logger.info("  Loaded Akkadian morphemes")
        
        lexicon.load_function_words()
        logger.info("  Loaded function words")
        
        # Load from gold data
        if os.path.exists(gold_csv):
            logger.info(f"Loading gold data from {gold_csv}")
            df = pd.read_csv(gold_csv, encoding='utf-8')
            
            # Try different column names
            text_column = None
            for col in ['HANDTYPED', 'text', 'gold_text', 'reference']:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column:
                gold_texts = df[text_column].dropna().tolist()
                logger.info(f"Extracted {len(gold_texts)} gold texts from column '{text_column}'")
                
                lexicon.load_from_gold(gold_texts)
                logger.info("  Added terms from gold data")
            else:
                logger.warning(f"No text column found in {gold_csv}")
                logger.warning(f"Available columns: {df.columns.tolist()}")
        
        # Save lexicon
        os.makedirs(output_dir, exist_ok=True)
        lex_path = os.path.join(output_dir, 'akkadian_lexicon.json')
        lexicon.save(lex_path)
        
        logger.info("✓ Lexicon built successfully")
        logger.info(f"  File: {lex_path}")
        # Don't count nodes as it's complex with multiple tries
        logger.info(f"  Lexicon saved with Sumerograms, morphemes, and function words")
        
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import lexicon_bias: {e}")
        logger.error("Make sure src/lexicon_bias.py exists")
        return False
    except Exception as e:
        logger.error(f"Failed to build lexicon: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Build research assets for OCR pipeline')
    parser.add_argument('--gold', type=str, required=True,
                      help='Path to gold CSV file')
    parser.add_argument('--output', type=str, default='reports/research_assets',
                      help='Output directory for assets (default: reports/research_assets)')
    parser.add_argument('--lm-order', type=int, default=5,
                      help='N-gram order for language model (default: 5)')
    parser.add_argument('--skip-lm', action='store_true',
                      help='Skip building character language model')
    parser.add_argument('--skip-lexicon', action='store_true',
                      help='Skip building lexicon')
    
    args = parser.parse_args()
    
    # Validate gold file
    if not os.path.exists(args.gold):
        logger.error(f"Gold file not found: {args.gold}")
        return 1
    
    logger.info("Research Assets Builder")
    logger.info(f"Gold CSV: {args.gold}")
    logger.info(f"Output directory: {args.output}")
    logger.info("")
    
    success = True
    
    # Build character LM
    if not args.skip_lm:
        lm_dir = os.path.join(args.output, 'lm')
        if not build_character_lm(args.gold, lm_dir, args.lm_order):
            success = False
            logger.error("Character LM build failed")
        print()
    
    # Build lexicon
    if not args.skip_lexicon:
        lex_dir = os.path.join(args.output, 'lexicon')
        if not build_lexicon(args.gold, lex_dir):
            success = False
            logger.error("Lexicon build failed")
        print()
    
    if success:
        logger.info("=" * 60)
        logger.info("✓ All research assets built successfully")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Update your config JSON to enable research features:")
        logger.info('   "preserve_transliteration": {"enable": true}')
        logger.info('   "research": {')
        logger.info('     "enable_char_lm_decoding": true,')
        logger.info('     "enable_rover_ensemble": true,')
        logger.info('     "enable_lexicon_bias": true,')
        logger.info('     "char_lm": {')
        logger.info(f'       "model_path": "{os.path.join(args.output, "lm", "char.klm")}",')
        logger.info(f'       "charset_path": "{os.path.join(args.output, "lm", "charset.txt")}"')
        logger.info('     },')
        logger.info('     "lexicon": {')
        logger.info(f'       "lexicon_file": "{os.path.join(args.output, "lexicon", "akkadian_lexicon.json")}"')
        logger.info('     }')
        logger.info('   }')
        logger.info("")
        logger.info("2. Run pipeline with research config:")
        logger.info("   python run_pipeline.py --config your_research_config.json")
        return 0
    else:
        logger.error("Some assets failed to build")
        return 1


if __name__ == '__main__':
    sys.exit(main())
