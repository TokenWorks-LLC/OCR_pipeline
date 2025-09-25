#!/usr/bin/env python3
"""
OCR Pipeline - Quick Start Examples

This file contains simple examples of how to use the OCR pipeline
with different configurations.
"""

import json
import subprocess
import sys
from pathlib import Path

def run_example(config_file: str, description: str):
    """Run an example with a specific configuration."""
    print(f"\n{'='*50}")
    print(f"Running Example: {description}")
    print('='*50)
    
    try:
        result = subprocess.run([
            sys.executable, "run_pipeline.py", "-c", config_file
        ], check=True, capture_output=False)
        
        print(f"✅ Example completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Example failed with exit code {e.returncode}")
    except FileNotFoundError:
        print(f"❌ Could not find run_pipeline.py or Python interpreter")

def create_example_configs():
    """Create example configuration files."""
    
    # Example 1: Basic PDF processing
    basic_config = {
        "input": {
            "input_directory": "./data/input_pdfs",
            "supported_formats": [".pdf"],
            "process_all_files": True,
            "specific_file": "",
            "recursive_search": False
        },
        "output": {
            "output_directory": "./data/output",
            "create_subdirectories": True,
            "timestamp_outputs": False
        },
        "ocr": {
            "engine": "paddleocr",
            "dpi": 300,
            "languages": ["en"],
            "enable_text_correction": False,
            "enable_reading_order": True,
            "confidence_threshold": 0.5
        },
        "llm": {
            "enable_correction": False,
            "provider": "none",
            "model": "",
            "base_url": "",
            "timeout": 30,
            "max_concurrent_corrections": 1
        },
        "akkadian": {
            "enable_extraction": False,
            "confidence_threshold": 0.8,
            "generate_pdf_report": False,
            "translation_languages": ["english"]
        },
        "processing": {
            "batch_size": 10,
            "max_workers": 2,
            "skip_existing": True,
            "create_visualizations": False,
            "create_html_overlay": False,
            "verbose": True
        },
        "csv_output": {
            "format": "aggregated",
            "include_metadata": True,
            "include_confidence_scores": True,
            "separate_akkadian_csv": False
        }
    }
    
    # Example 2: High-quality processing with AI correction
    advanced_config = basic_config.copy()
    advanced_config.update({
        "ocr": {
            "engine": "paddleocr",
            "dpi": 600,
            "languages": ["en", "tr", "de", "fr"],
            "enable_text_correction": True,
            "enable_reading_order": True,
            "confidence_threshold": 0.7
        },
        "llm": {
            "enable_correction": True,
            "provider": "ollama",
            "model": "mistral:latest",
            "base_url": "http://localhost:11434",
            "timeout": 60,
            "max_concurrent_corrections": 3
        },
        "processing": {
            "batch_size": 5,
            "max_workers": 2,
            "skip_existing": True,
            "create_visualizations": True,
            "create_html_overlay": True,
            "verbose": True
        }
    })
    
    # Example 3: Akkadian research processing
    akkadian_config = advanced_config.copy()
    akkadian_config.update({
        "input": {
            "input_directory": "./data/input",
            "supported_formats": [".pdf", ".png", ".jpg", ".jpeg"],
            "process_all_files": True,
            "specific_file": "",
            "recursive_search": True
        },
        "akkadian": {
            "enable_extraction": True,
            "confidence_threshold": 0.8,
            "generate_pdf_report": True,
            "translation_languages": ["english", "german", "french", "turkish"]
        },
        "csv_output": {
            "format": "aggregated",
            "include_metadata": True,
            "include_confidence_scores": True,
            "separate_akkadian_csv": True
        }
    })
    
    # Example 4: Single image processing
    single_image_config = {
        "input": {
            "input_directory": "./data/samples",
            "supported_formats": [".png", ".jpg", ".jpeg"],
            "process_all_files": False,
            "specific_file": "./data/samples/test_document.jpg",
            "recursive_search": False
        },
        "output": {
            "output_directory": "./data/output/single_image_test",
            "create_subdirectories": False,
            "timestamp_outputs": True
        },
        "ocr": {
            "engine": "paddleocr",
            "dpi": 300,
            "languages": ["en"],
            "enable_text_correction": True,
            "enable_reading_order": True,
            "confidence_threshold": 0.6
        },
        "llm": {
            "enable_correction": True,
            "provider": "ollama",
            "model": "mistral:latest",
            "base_url": "http://localhost:11434",
            "timeout": 30,
            "max_concurrent_corrections": 1
        },
        "akkadian": {
            "enable_extraction": False,
            "confidence_threshold": 0.8,
            "generate_pdf_report": False,
            "translation_languages": ["english"]
        },
        "processing": {
            "batch_size": 1,
            "max_workers": 1,
            "skip_existing": False,
            "create_visualizations": True,
            "create_html_overlay": True,
            "verbose": True
        },
        "csv_output": {
            "format": "aggregated",
            "include_metadata": True,
            "include_confidence_scores": True,
            "separate_akkadian_csv": False
        }
    }
    
    # Save example configurations
    configs = [
        (basic_config, "config_basic.json", "Basic PDF Processing"),
        (advanced_config, "config_advanced.json", "Advanced AI-Enhanced Processing"),
        (akkadian_config, "config_akkadian.json", "Academic Akkadian Research"),
        (single_image_config, "config_single_image.json", "Single Image Processing")
    ]
    
    for config, filename, description in configs:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"✅ Created example config: {filename} - {description}")
        except Exception as e:
            print(f"❌ Failed to create {filename}: {e}")
    
    return configs

def main():
    """Main function to run examples."""
    print("🚀 OCR Pipeline - Quick Start Examples")
    print("="*50)
    
    # Check if run_pipeline.py exists
    if not Path("run_pipeline.py").exists():
        print("❌ Error: run_pipeline.py not found in current directory")
        print("Please make sure you're running this from the OCR_pipeline root directory")
        sys.exit(1)
    
    # Create example configurations
    print("\n📄 Creating example configuration files...")
    configs = create_example_configs()
    
    print(f"\n📋 Available Examples:")
    for i, (_, filename, description) in enumerate(configs, 1):
        print(f"   {i}. {description} ({filename})")
    
    print(f"\n🔧 Usage Instructions:")
    print(f"   1. Choose an example configuration that matches your needs")
    print(f"   2. Edit the chosen config file to point to your input files")
    print(f"   3. Run: python run_pipeline.py -c <config_file>")
    print(f"   4. Check the output directory for results")
    
    print(f"\n💡 Quick Commands:")
    print(f"   Basic processing:    python run_pipeline.py -c config_basic.json")
    print(f"   Advanced processing: python run_pipeline.py -c config_advanced.json")
    print(f"   Akkadian research:   python run_pipeline.py -c config_akkadian.json")
    print(f"   Single image:        python run_pipeline.py -c config_single_image.json")
    
    print(f"\n🔍 Other useful commands:")
    print(f"   Validate config:     python run_pipeline.py -c config.json --validate-only")
    print(f"   Dry run:             python run_pipeline.py -c config.json --dry-run")
    print(f"   Help:                python run_pipeline.py --help")
    
    print(f"\n📊 Analysis Commands:")
    print(f"   Summary analysis:    python run_analysis_menu.py")
    print(f"   Compare modes:       python src/summary_analysis.py <eval_dir1> <eval_dir2>")
    
    # Interactive mode
    while True:
        print(f"\n" + "="*50)
        print("Options:")
        print("1-4. Run example configurations")
        print("5. 📊 Run Summary Analysis")
        print("q. Quit")
        choice = input("Select option (1-5, or 'q' to quit): ").strip().lower()
        
        if choice == 'q' or choice == 'quit':
            print("👋 Goodbye!")
            break
        
        elif choice == '5':
            # Run summary analysis
            print("\n📊 Starting Summary Analysis...")
            try:
                import subprocess
                result = subprocess.run([sys.executable, "run_analysis_menu.py"], check=True)
                print("✅ Analysis completed!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Analysis failed with exit code {e.returncode}")
            except FileNotFoundError:
                print("❌ Analysis menu not found. Please ensure run_analysis_menu.py exists.")
            except Exception as e:
                print(f"❌ Error running analysis: {e}")
            continue
        
        try:
            example_num = int(choice)
            if 1 <= example_num <= len(configs):
                config, filename, description = configs[example_num - 1]
                
                # Check if input directory exists for this config
                input_dir = config['input']['input_directory']
                if not Path(input_dir).exists():
                    print(f"⚠️  Warning: Input directory not found: {input_dir}")
                    create_dir = input(f"Create directory? (y/n): ").strip().lower()
                    if create_dir == 'y':
                        Path(input_dir).mkdir(parents=True, exist_ok=True)
                        print(f"✅ Created directory: {input_dir}")
                        print(f"📄 Please add your files to this directory and run the example again")
                        continue
                    else:
                        print(f"❌ Cannot run example without input directory")
                        continue
                
                run_example(filename, description)
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(configs)} or 'q'")
        
        except ValueError:
            print("❌ Invalid input. Please enter a number 1-4 or 'q'")
        except KeyboardInterrupt:
            print(f"\n\n👋 Interrupted by user. Goodbye!")
            break

if __name__ == "__main__":
    main()
