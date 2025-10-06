#!/usr/bin/env python3
"""
Simplified Gold Standard Evaluation Script

This script evaluates the OCR pipeline performance against gold standard data
by directly using the comprehensive pipeline rather than the command line interface.
"""

import pandas as pd
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
import re

# Add src and production to path for imports
sys.path.append('src')
sys.path.append('production')

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simple_gold_evaluation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_gold_data(gold_data_path):
    """Load and clean the gold standard data"""
    logger = logging.getLogger(__name__)
    
    try:
        for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(gold_data_path, encoding=encoding)
                logger.info(f"Successfully loaded gold data with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not decode the CSV file with any common encoding")
        
        df_clean = df.dropna()
        logger.info(f"Loaded {len(df)} total rows, {len(df_clean)} clean rows")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error loading gold data: {e}")
        return None

def find_matching_pdfs(gold_df, input_pdfs_dir):
    """Find PDFs that exist in both gold data and input directory"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(input_pdfs_dir):
        logger.error(f"Input PDFs directory not found: {input_pdfs_dir}")
        return []
    
    available_pdfs = set(os.listdir(input_pdfs_dir))
    gold_pdfs = set(gold_df['PDF LINK'].unique())
    matching_pdfs = gold_pdfs.intersection(available_pdfs)
    
    logger.info(f"Found {len(matching_pdfs)} matching PDFs out of {len(gold_pdfs)} in gold data")
    
    tasks = []
    for pdf in matching_pdfs:
        pdf_rows = gold_df[gold_df['PDF LINK'] == pdf]
        for _, row in pdf_rows.iterrows():
            page_str = str(row['PAGE']).strip()
            
            if '-' in page_str:
                try:
                    page_num = int(page_str.split('-')[0])
                    logger.info(f"Page range detected '{page_str}', using first page: {page_num}")
                except ValueError:
                    logger.warning(f"Could not parse page range '{page_str}' for {pdf}, skipping")
                    continue
            else:
                try:
                    page_num = int(page_str)
                except ValueError:
                    logger.warning(f"Could not parse page number '{page_str}' for {pdf}, skipping")
                    continue
            
            tasks.append({
                'pdf_name': pdf,
                'page': page_num,
                'gold_text': str(row['HANDTYPED']),
                'pdf_path': os.path.join(input_pdfs_dir, pdf),
                'original_page_str': page_str
            })
    
    logger.info(f"Created {len(tasks)} evaluation tasks")
    return tasks

def calculate_similarity(text1, text2):
    """Calculate text similarity metrics"""
    def normalize_text(text):
        text = str(text).lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)
    
    similarity_ratio = SequenceMatcher(None, norm_text1, norm_text2).ratio()
    
    total_chars = max(len(norm_text1), len(norm_text2))
    char_errors = sum(c1 != c2 for c1, c2 in zip(norm_text1, norm_text2))
    char_errors += abs(len(norm_text1) - len(norm_text2))
    char_accuracy = max(0, 1 - (char_errors / total_chars)) if total_chars > 0 else 0
    
    words1 = norm_text1.split()
    words2 = norm_text2.split()
    word_matcher = SequenceMatcher(None, words1, words2)
    word_similarity = word_matcher.ratio()
    
    return {
        'similarity_ratio': similarity_ratio,
        'character_accuracy': char_accuracy,
        'word_similarity': word_similarity,
        'gold_length': len(text1),
        'ocr_length': len(text2)
    }

def run_pipeline_on_page(pdf_path, page_num, output_dir):
    """Run the OCR pipeline on a specific page using direct imports"""
    logger = logging.getLogger(__name__)
    
    try:
        from comprehensive_pipeline import ComprehensivePipeline
        
        # Create output directory for this specific run
        page_output_dir = os.path.join(output_dir, f"{Path(pdf_path).stem}_page_{page_num}")
        os.makedirs(page_output_dir, exist_ok=True)
        
        # Create a basic pipeline configuration
        pipeline_config = {
            'input_file': pdf_path,
            'output_directory': page_output_dir,
            'dpi': 300,
            'confidence_threshold': 0.5,
            'enable_llm_correction': False,
            'enable_akkadian_extraction': False,
            'create_subdirectories': False,
            'specific_pages': [page_num]
        }
        
        logger.info(f"Processing {pdf_path} page {page_num}")
        
        # Initialize and run pipeline
        pipeline = ComprehensivePipeline(pipeline_config)
        result = pipeline.process_file()
        
        if result and 'comprehensive_results.csv' in result:
            csv_path = result['comprehensive_results.csv']
            if os.path.exists(csv_path):
                # Read the CSV and extract text
                df = pd.read_csv(csv_path, encoding='utf-8')
                
                if 'corrected_text' in df.columns:
                    texts = df['corrected_text'].dropna().tolist()
                elif 'text' in df.columns:
                    texts = df['text'].dropna().tolist()
                else:
                    logger.error(f"No text column found in {csv_path}")
                    return None
                
                combined_text = ' '.join(str(t) for t in texts if str(t).strip())
                return combined_text
            else:
                logger.error(f"Expected CSV file not found: {csv_path}")
                return None
        else:
            logger.error(f"Pipeline did not return expected result for {pdf_path} page {page_num}")
            return None
            
    except Exception as e:
        logger.error(f"Error running pipeline on {pdf_path} page {page_num}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_report(results, output_dir):
    """Generate evaluation report"""
    logger = logging.getLogger(__name__)
    
    if not results:
        logger.warning("No results to report")
        return
    
    similarities = [r['similarity_ratio'] for r in results]
    char_accuracies = [r['character_accuracy'] for r in results]
    word_similarities = [r['word_similarity'] for r in results]
    
    overall_stats = {
        'total_tasks': len(results),
        'avg_similarity': sum(similarities) / len(similarities),
        'avg_char_accuracy': sum(char_accuracies) / len(char_accuracies),
        'avg_word_similarity': sum(word_similarities) / len(word_similarities),
        'min_similarity': min(similarities),
        'max_similarity': max(similarities),
        'evaluation_date': datetime.now().isoformat()
    }
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'detailed_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save summary statistics
    stats_file = os.path.join(output_dir, 'evaluation_summary.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, indent=2)
    
    # Generate markdown report
    report_file = os.path.join(output_dir, 'evaluation_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# OCR Pipeline Evaluation Report\\n\\n")
        f.write(f"**Evaluation Date:** {overall_stats['evaluation_date']}\\n\\n")
        f.write(f"**Total Tasks Evaluated:** {overall_stats['total_tasks']}\\n\\n")
        
        f.write("## Overall Performance\\n\\n")
        f.write(f"- **Average Similarity:** {overall_stats['avg_similarity']:.3f}\\n")
        f.write(f"- **Average Character Accuracy:** {overall_stats['avg_char_accuracy']:.3f}\\n")
        f.write(f"- **Average Word Similarity:** {overall_stats['avg_word_similarity']:.3f}\\n")
        f.write(f"- **Range:** {overall_stats['min_similarity']:.3f} - {overall_stats['max_similarity']:.3f}\\n\\n")
        
        f.write("## Detailed Results\\n\\n")
        f.write("| PDF | Page | Similarity | Char Accuracy | Word Similarity |\\n")
        f.write("|-----|------|------------|---------------|------------------|\\n")
        
        for result in sorted(results, key=lambda x: x['similarity_ratio'], reverse=True):
            f.write(f"| {result['pdf_name']} | {result['page']} | {result['similarity_ratio']:.3f} | {result['character_accuracy']:.3f} | {result['word_similarity']:.3f} |\\n")
    
    logger.info(f"Report saved to {report_file}")
    
    # Print summary to console
    print("\\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total tasks evaluated: {overall_stats['total_tasks']}")
    print(f"Average similarity: {overall_stats['avg_similarity']:.3f}")
    print(f"Average character accuracy: {overall_stats['avg_char_accuracy']:.3f}")
    print(f"Average word similarity: {overall_stats['avg_word_similarity']:.3f}")
    print(f"Performance range: {overall_stats['min_similarity']:.3f} - {overall_stats['max_similarity']:.3f}")
    print("="*50)

def main():
    """Main evaluation function"""
    logger = setup_logging()
    logger.info("Starting simplified gold standard evaluation")
    
    # Configuration
    gold_data_path = "data/gold_data/gold_pages.csv"
    input_pdfs_dir = "data/input_pdfs"
    output_dir = "data/simple_evaluation_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load gold data
    logger.info("Loading gold standard data...")
    gold_df = load_gold_data(gold_data_path)
    if gold_df is None:
        logger.error("Failed to load gold data")
        return
    
    # Find matching PDFs and create tasks  
    logger.info("Finding matching PDFs...")
    tasks = find_matching_pdfs(gold_df, input_pdfs_dir)
    if not tasks:
        logger.error("No matching PDFs found")
        return
    
    # Limit to first 3 tasks for testing
    tasks = tasks[:3]
    logger.info(f"Processing first {len(tasks)} tasks for testing...")
    
    results = []
    
    for i, task in enumerate(tasks, 1):
        logger.info(f"Processing task {i}/{len(tasks)}: {task['pdf_name']} page {task['page']}")
        
        # Run pipeline
        ocr_text = run_pipeline_on_page(task['pdf_path'], task['page'], output_dir)
        
        if ocr_text is None:
            logger.warning(f"Failed to get OCR result for {task['pdf_name']} page {task['page']}")
            continue
        
        # Calculate metrics
        metrics = calculate_similarity(task['gold_text'], ocr_text)
        
        result = {
            'pdf_name': task['pdf_name'],
            'page': task['page'],
            'gold_text_preview': task['gold_text'][:100] + "..." if len(task['gold_text']) > 100 else task['gold_text'],
            'ocr_text_preview': ocr_text[:100] + "..." if len(ocr_text) > 100 else ocr_text,
            **metrics
        }
        
        results.append(result)
        logger.info(f"Similarity: {metrics['similarity_ratio']:.3f}, Char accuracy: {metrics['character_accuracy']:.3f}")
    
    # Generate report
    logger.info("Generating evaluation report...")
    generate_report(results, output_dir)
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()