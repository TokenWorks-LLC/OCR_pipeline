#!/usr/bin/env python3
"""
Incremental Evaluation Runner for OCR Pipeline
Processes PDFs page by page with real-time progress and incremental saving
"""

import json
import sys
import os
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# Enhanced monitoring dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# GPU monitoring
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# Add necessary directories to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'production'))

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        sys.exit(1)

def get_pdf_page_count(pdf_path: str) -> int:
    """Get total number of pages in PDF."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()
        return page_count
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return 0

def get_gpu_usage() -> Dict[str, Any]:
    """Get GPU usage metrics if available."""
    if not GPUTIL_AVAILABLE:
        return {
            'utilization': 0.0,
            'memory_used_mb': 0.0,
            'memory_total_mb': 0.0,
            'temperature': 0.0,
            'available': False
        }
    
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # First GPU
            return {
                'utilization': gpu.load * 100,  # Convert to percentage
                'memory_used_mb': gpu.memoryUsed,
                'memory_total_mb': gpu.memoryTotal,
                'temperature': gpu.temperature,
                'available': True
            }
    except Exception as e:
        print(f"⚠️  GPU monitoring failed: {e}")
    
    return {
        'utilization': 0.0,
        'memory_used_mb': 0.0,
        'memory_total_mb': 0.0,
        'temperature': 0.0,
        'available': False
    }

def get_process_tree_resources() -> Dict[str, Any]:
    """Get resource usage for OCR process tree (main + children)."""
    if not PSUTIL_AVAILABLE:
        return {
            'total_cpu_percent': 0.0,
            'total_memory_mb': 0.0,
            'process_count': 0,
            'available': False
        }
    
    try:
        # Get current process and all children
        main_process = psutil.Process()
        children = main_process.children(recursive=True)
        all_processes = [main_process] + children
        
        # Aggregate CPU and memory across all processes
        total_cpu = 0.0
        total_memory = 0.0
        
        for proc in all_processes:
            try:
                # CPU percentage (non-blocking)
                cpu_percent = proc.cpu_percent()
                total_cpu += cpu_percent
                
                # Memory usage
                memory_info = proc.memory_info()
                total_memory += memory_info.rss / 1024 / 1024  # Convert to MB
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process might have terminated or we don't have access
                continue
        
        return {
            'total_cpu_percent': total_cpu,
            'total_memory_mb': total_memory,
            'process_count': len(all_processes),
            'available': True
        }
        
    except Exception as e:
        print(f"⚠️  Process tree monitoring failed: {e}")
        return {
            'total_cpu_percent': 0.0,
            'total_memory_mb': 0.0,
            'process_count': 0,
            'available': False
        }

def get_io_counters() -> Dict[str, Any]:
    """Get I/O operation counters."""
    if not PSUTIL_AVAILABLE:
        return {
            'read_mb': 0.0,
            'write_mb': 0.0,
            'available': False
        }
    
    try:
        io_counters = psutil.disk_io_counters()
        if io_counters:
            return {
                'read_mb': io_counters.read_bytes / 1024 / 1024,
                'write_mb': io_counters.write_bytes / 1024 / 1024,
                'available': True
            }
    except Exception as e:
        print(f"⚠️  I/O monitoring failed: {e}")
    
    return {
        'read_mb': 0.0,
        'write_mb': 0.0,
        'available': False
    }

def get_enhanced_monitoring() -> Dict[str, Any]:
    """Get comprehensive monitoring data including process tree, GPU, and I/O."""
    return {
        'timestamp': time.time(),
        'process_tree': get_process_tree_resources(),
        'gpu': get_gpu_usage(),
        'io': get_io_counters()
    }

def save_incremental_progress(output_dir: str, page_num: int, total_pages: int, 
                            page_result, processing_stats: Dict, 
                            cumulative_stats: Dict, config: Dict):
    """Save incremental progress in the same format as final eval output."""
    try:
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Update cumulative stats
        cumulative_stats['pages_processed'] = page_num
        cumulative_stats['total_pages'] = total_pages
        cumulative_stats['progress_percent'] = (page_num / total_pages) * 100
        cumulative_stats['last_updated'] = datetime.now().isoformat()
        
        # Add current page stats
        if 'page_statistics' not in cumulative_stats:
            cumulative_stats['page_statistics'] = []
        cumulative_stats['page_statistics'].append(processing_stats)
        
        # Update aggregated counts
        if page_result:
            cumulative_stats['total_text_elements'] = cumulative_stats.get('total_text_elements', 0) + processing_stats.get('text_elements', 0)
            cumulative_stats['total_corrections'] = cumulative_stats.get('total_corrections', 0) + processing_stats.get('corrections_made', 0)
            # Add new cost of compute metrics
            cumulative_stats['total_word_count'] = cumulative_stats.get('total_word_count', 0) + processing_stats.get('word_count', 0)
            cumulative_stats['total_token_count'] = cumulative_stats.get('total_token_count', 0) + processing_stats.get('token_count', 0)
            
            # Aggregate enhanced monitoring data
            if 'enhanced_monitoring' in processing_stats:
                enhanced = processing_stats['enhanced_monitoring']
                
                # Initialize enhanced monitoring aggregates if not present
                if 'enhanced_monitoring_aggregates' not in cumulative_stats:
                    cumulative_stats['enhanced_monitoring_aggregates'] = {
                        'total_cpu_delta': 0.0,
                        'total_memory_delta_mb': 0.0,
                        'max_gpu_utilization': 0.0,
                        'max_gpu_memory_mb': 0.0,
                        'total_io_read_delta_mb': 0.0,
                        'total_io_write_delta_mb': 0.0,
                        'pages_with_gpu_usage': 0,
                        'pages_with_cpu_usage': 0
                    }
                
                # Aggregate enhanced monitoring metrics
                agg = cumulative_stats['enhanced_monitoring_aggregates']
                agg['total_cpu_delta'] += enhanced.get('cpu_delta', 0.0)
                agg['total_memory_delta_mb'] += enhanced.get('memory_delta_mb', 0.0)
                agg['max_gpu_utilization'] = max(agg['max_gpu_utilization'], enhanced.get('gpu_utilization', 0.0))
                agg['max_gpu_memory_mb'] = max(agg['max_gpu_memory_mb'], enhanced.get('gpu_memory_mb', 0.0))
                agg['total_io_read_delta_mb'] += enhanced.get('io_read_delta_mb', 0.0)
                agg['total_io_write_delta_mb'] += enhanced.get('io_write_delta_mb', 0.0)
                
                if enhanced.get('gpu_utilization', 0.0) > 0:
                    agg['pages_with_gpu_usage'] += 1
                if enhanced.get('cpu_delta', 0.0) > 0:
                    agg['pages_with_cpu_usage'] += 1
        
        # Save incremental progress
        progress_file = os.path.join(output_dir, "comprehensive_report.json")
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(cumulative_stats, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Progress saved: {progress_file}")
        
    except Exception as e:
        print(f"   ⚠️  Could not save progress: {e}")

def process_pdf_incremental(pdf_path: str, config: Dict[str, Any], pipeline, logger) -> Dict[str, Any]:
    """Process PDF page by page with incremental saving."""
    start_time = time.time()
    
    # Get PDF info
    total_pages = get_pdf_page_count(pdf_path)
    if total_pages == 0:
        return {'error': 'Could not read PDF or PDF has no pages'}
    
    print(f"📄 PDF: {os.path.basename(pdf_path)}")
    print(f"📊 Total pages: {total_pages}")
    print(f"⏱️  Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    # Initialize cumulative stats
    cumulative_stats = {
        'pdf_path': pdf_path,
        'total_pages': total_pages,
        'pages_processed': 0,
        'progress_percent': 0.0,
        'total_processing_time': 0.0,
        'total_text_elements': 0,
        'total_corrections': 0,
        'successful_pages': 0,
        'failed_pages': 0,
        'config': config,
        'page_statistics': [],
        'start_time': start_time,
        'last_updated': datetime.now().isoformat()
    }
    
    # Determine output directory
    output_dir = config['output']['output_directory']
    if config['output'].get('create_subdirectories', True):
        file_stem = Path(pdf_path).stem
        output_dir = os.path.join(output_dir, file_stem)
    
    # Process each page
    for page_num in range(1, total_pages + 1):
        print(f"📄 Processing page {page_num}/{total_pages} ({page_num/total_pages*100:.1f}%)")
        
        # Get enhanced monitoring data before processing
        pre_monitoring = get_enhanced_monitoring()
        
        try:
            # Process single page
            page_result, processing_stats = pipeline.process_single_page(
                pdf_path, page_num, output_dir, Path(pdf_path).stem
            )
            
            # Get enhanced monitoring data after processing
            post_monitoring = get_enhanced_monitoring()
            
            # Calculate resource usage during this page
            page_resource_usage = {
                'pre_monitoring': pre_monitoring,
                'post_monitoring': post_monitoring,
                'cpu_delta': post_monitoring['process_tree']['total_cpu_percent'] - pre_monitoring['process_tree']['total_cpu_percent'],
                'memory_delta_mb': post_monitoring['process_tree']['total_memory_mb'] - pre_monitoring['process_tree']['total_memory_mb'],
                'gpu_utilization': post_monitoring['gpu']['utilization'],
                'gpu_memory_mb': post_monitoring['gpu']['memory_used_mb'],
                'io_read_delta_mb': post_monitoring['io']['read_mb'] - pre_monitoring['io']['read_mb'],
                'io_write_delta_mb': post_monitoring['io']['write_mb'] - pre_monitoring['io']['write_mb']
            }
            
            # Add enhanced monitoring to processing stats
            processing_stats['enhanced_monitoring'] = page_resource_usage
            
            if page_result:
                cumulative_stats['successful_pages'] += 1
                print(f"   ✅ Success: {processing_stats.get('text_elements', 0)} elements, "
                      f"lang={processing_stats.get('language', 'unknown')}")
                
                # Print enhanced monitoring info
                if page_resource_usage['cpu_delta'] > 0:
                    print(f"   📊 CPU: {page_resource_usage['cpu_delta']:.1f}%, "
                          f"Memory: {page_resource_usage['memory_delta_mb']:.1f}MB")
                if page_resource_usage['gpu_utilization'] > 0:
                    print(f"   🎮 GPU: {page_resource_usage['gpu_utilization']:.1f}%, "
                          f"VRAM: {page_resource_usage['gpu_memory_mb']:.1f}MB")
            else:
                cumulative_stats['failed_pages'] += 1
                print(f"   ⚠️  No text detected on page {page_num}")
            
            # Save incremental progress
            save_incremental_progress(output_dir, page_num, total_pages, 
                                    page_result, processing_stats, cumulative_stats, config)
            
        except KeyboardInterrupt:
            print(f"\n🛑 Processing interrupted by user at page {page_num}")
            cumulative_stats['interrupted'] = True
            cumulative_stats['interrupted_at_page'] = page_num
            break
        except Exception as e:
            print(f"   ❌ Error processing page {page_num}: {e}")
            cumulative_stats['failed_pages'] += 1
            processing_stats = {'page_num': page_num, 'error': str(e)}
            save_incremental_progress(output_dir, page_num, total_pages, 
                                    None, processing_stats, cumulative_stats, config)
    
    # Finalize stats
    end_time = time.time()
    cumulative_stats['total_processing_time'] = end_time - start_time
    cumulative_stats['avg_time_per_page'] = cumulative_stats['total_processing_time'] / cumulative_stats['pages_processed'] if cumulative_stats['pages_processed'] > 0 else 0
    cumulative_stats['completion_time'] = datetime.now().isoformat()
    
    # Save final report
    final_report_path = os.path.join(output_dir, "comprehensive_report.json")
    with open(final_report_path, 'w', encoding='utf-8') as f:
        json.dump(cumulative_stats, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"📄 Pages processed: {cumulative_stats['pages_processed']}/{total_pages}")
    print(f"✅ Successful: {cumulative_stats['successful_pages']}")
    print(f"❌ Failed: {cumulative_stats['failed_pages']}")
    print(f"⏱️  Total time: {cumulative_stats['total_processing_time']:.1f}s")
    print(f"📝 Text elements: {cumulative_stats['total_text_elements']}")
    print(f"🔧 Corrections: {cumulative_stats['total_corrections']}")
    print(f"📖 Word count: {cumulative_stats.get('total_word_count', 0)}")
    print(f"🎯 Token count: {cumulative_stats.get('total_token_count', 0)}")
    
    # Enhanced monitoring summary
    if 'enhanced_monitoring_aggregates' in cumulative_stats:
        agg = cumulative_stats['enhanced_monitoring_aggregates']
        print("\n🔍 ENHANCED MONITORING")
        print("-" * 30)
        print(f"💻 Total CPU delta: {agg['total_cpu_delta']:.1f}%")
        print(f"🧠 Total memory delta: {agg['total_memory_delta_mb']:.1f}MB")
        print(f"🎮 Max GPU utilization: {agg['max_gpu_utilization']:.1f}%")
        print(f"🎮 Max GPU memory: {agg['max_gpu_memory_mb']:.1f}MB")
        print(f"💾 Total I/O read: {agg['total_io_read_delta_mb']:.1f}MB")
        print(f"💾 Total I/O write: {agg['total_io_write_delta_mb']:.1f}MB")
        print(f"🎮 Pages with GPU usage: {agg['pages_with_gpu_usage']}")
        print(f"💻 Pages with CPU usage: {agg['pages_with_cpu_usage']}")
    
    print(f"📋 Report saved: {final_report_path}")
    
    return cumulative_stats

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Incremental OCR Pipeline Evaluation")
    parser.add_argument("-c", "--config", default="config_eval_basic.json", 
                       help="Path to evaluation config file")
    parser.add_argument("--validate-only", action="store_true", 
                       help="Only validate configuration and exit")
    
    args = parser.parse_args()
    
    print("🚀 OCR Pipeline - Incremental Evaluation Runner")
    print("=" * 60)
    
    # Load configuration
    print(f"📄 Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Validate evaluation config
    if 'evaluation' not in config:
        print("❌ This is not an evaluation config file")
        print("   Please use config_eval_*.json files")
        sys.exit(1)
    
    if args.validate_only:
        print("✅ Configuration valid for incremental evaluation")
        return 0
    
    # Setup logging
    log_level = logging.INFO if config.get('processing', {}).get('verbose', True) else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    
    # Get files to process
    input_dir = config['input']['input_directory']
    supported_formats = config['input']['supported_formats']
    
    files_to_process = []
    for format_ext in supported_formats:
        pattern = os.path.join(input_dir, f"*{format_ext}")
        files_to_process.extend([f for f in os.listdir(input_dir) if f.endswith(format_ext)])
    
    if not files_to_process:
        print("❌ No files found to process")
        return 1
    
    print(f"📁 Found {len(files_to_process)} file(s) to process")
    
    # Initialize pipeline
    print("\n🔧 Initializing OCR pipeline...")
    try:
        from comprehensive_pipeline import ComprehensivePipeline, PipelineConfig
        
        # Create pipeline config
        pipeline_config = PipelineConfig(
            llm_provider=config.get('llm', {}).get('provider', 'none'),
            llm_model=config.get('llm', {}).get('model', ''),
            llm_base_url=config.get('llm', {}).get('base_url', ''),
            llm_timeout=config.get('llm', {}).get('timeout', 30),
            dpi=config.get('ocr', {}).get('dpi', 300),
            enable_reading_order=config.get('ocr', {}).get('enable_reading_order', True),
            enable_llm_correction=config.get('llm', {}).get('enable_correction', False),
            max_concurrent_corrections=config.get('llm', {}).get('max_concurrent_corrections', 1),
            enable_akkadian_extraction=config.get('akkadian', {}).get('enable_extraction', False),
            generate_translations_pdf=config.get('akkadian', {}).get('generate_pdf_report', False),
            akkadian_confidence_threshold=config.get('akkadian', {}).get('confidence_threshold', 0.8),
            output_csv_metadata=config.get('csv_output', {}).get('include_metadata', True),
            create_html_overlay=config.get('processing', {}).get('create_html_overlay', False),
            create_visualization=config.get('processing', {}).get('create_visualizations', False)
        )
        
        pipeline = ComprehensivePipeline(pipeline_config)
        print("✅ Pipeline initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return 1
    
    # Process files
    results = []
    for i, file_path in enumerate(files_to_process, 1):
        full_path = os.path.join(input_dir, file_path)
        print(f"\n📄 Processing file {i}/{len(files_to_process)}: {file_path}")
        
        result = process_pdf_incremental(full_path, config, pipeline, logger)
        results.append(result)
    
    print(f"\n🎉 Evaluation completed! Processed {len(results)} file(s)")
    return 0

if __name__ == "__main__":
    exit(main())





