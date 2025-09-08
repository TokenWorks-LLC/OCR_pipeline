#!/usr/bin/env python3
"""
OCR Pipeline Setup Script

This script helps users set up the OCR pipeline environment
and verify that all dependencies are working correctly.
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path
import json

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8 or higher")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        ('cv2', 'opencv-python'),
        ('paddleocr', 'paddleocr'),
        ('fitz', 'PyMuPDF'),
        ('numpy', 'numpy'),
        ('PIL', 'Pillow'),
        ('requests', 'requests')
    ]
    
    missing_packages = []
    installed_packages = []
    
    for import_name, package_name in required_packages:
        try:
            importlib.import_module(import_name)
            installed_packages.append(package_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"   ❌ {package_name} - Missing")
    
    return installed_packages, missing_packages

def install_missing_packages(missing_packages):
    """Install missing packages using pip."""
    if not missing_packages:
        return True
    
    print(f"\n📥 Installing missing packages...")
    
    for package in missing_packages:
        print(f"   Installing {package}...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True, text=True)
            print(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
            return False
    
    return True

def create_directory_structure():
    """Create necessary directory structure."""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "data",
        "data/input",
        "data/input_pdfs", 
        "data/output",
        "data/samples",
        "logs"
    ]
    
    created_dirs = []
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory)
            print(f"   ✅ Created: {directory}")
        else:
            print(f"   📁 Exists: {directory}")
    
    return created_dirs

def create_default_config():
    """Create a default configuration file if it doesn't exist."""
    print("\n⚙️ Setting up configuration...")
    
    config_path = Path("config.json")
    
    if config_path.exists():
        print("   📄 config.json already exists")
        return True
    
    default_config = {
        "input": {
            "input_directory": "./data/input",
            "supported_formats": [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
            "process_all_files": True,
            "specific_file": "",
            "recursive_search": False
        },
        "output": {
            "output_directory": "./data/output",
            "create_subdirectories": True,
            "timestamp_outputs": True
        },
        "ocr": {
            "engine": "paddleocr",
            "dpi": 300,
            "languages": ["en", "tr", "de", "fr", "it"],
            "enable_text_correction": True,
            "enable_reading_order": True,
            "confidence_threshold": 0.5
        },
        "llm": {
            "enable_correction": True,
            "provider": "ollama",
            "model": "mistral:latest",
            "base_url": "http://localhost:11434",
            "timeout": 30,
            "max_concurrent_corrections": 3
        },
        "akkadian": {
            "enable_extraction": True,
            "confidence_threshold": 0.8,
            "generate_pdf_report": True,
            "translation_languages": ["english", "german", "french", "turkish"]
        },
        "processing": {
            "batch_size": 10,
            "max_workers": 4,
            "skip_existing": True,
            "create_visualizations": True,
            "create_html_overlay": True,
            "verbose": True
        },
        "csv_output": {
            "format": "aggregated",
            "include_metadata": True,
            "include_confidence_scores": True,
            "separate_akkadian_csv": True
        }
    }
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        print("   ✅ Created default config.json")
        return True
    except Exception as e:
        print(f"   ❌ Failed to create config.json: {e}")
        return False

def test_ocr_functionality():
    """Test if OCR functionality is working."""
    print("\n🔍 Testing OCR functionality...")
    
    try:
        # Test PaddleOCR import
        from paddleocr import PaddleOCR
        print("   ✅ PaddleOCR import successful")
        
        # Try to initialize PaddleOCR (this may take a while on first run)
        print("   📥 Initializing PaddleOCR (this may take a while on first run)...")
        ocr = PaddleOCR(use_textline_orientation=True, lang='en', show_log=False)
        print("   ✅ PaddleOCR initialization successful")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ PaddleOCR import failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ PaddleOCR initialization failed: {e}")
        return False

def check_llm_service():
    """Check if LLM service (Ollama) is running."""
    print("\n🧠 Checking LLM service...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            print("   ✅ Ollama service is running")
            
            # Check available models
            try:
                models = response.json().get('models', [])
                if models:
                    print("   📋 Available models:")
                    for model in models[:3]:  # Show first 3 models
                        print(f"      - {model.get('name', 'Unknown')}")
                    if len(models) > 3:
                        print(f"      ... and {len(models) - 3} more")
                else:
                    print("   ⚠️  No models found. You may need to pull a model:")
                    print("      ollama pull mistral:latest")
                
            except Exception as e:
                print(f"   ⚠️  Could not parse model list: {e}")
            
            return True
            
        else:
            print(f"   ❌ Ollama service not responding (status: {response.status_code})")
            return False
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Ollama service not running")
        print("      To install Ollama: https://ollama.com/download")
        print("      To start: ollama serve")
        return False
    except Exception as e:
        print(f"   ❌ Error checking Ollama service: {e}")
        return False

def create_sample_files():
    """Create sample files for testing."""
    print("\n📄 Creating sample files...")
    
    # Create a simple README for the samples directory
    sample_readme = """# Sample Files Directory

This directory is for sample input files to test the OCR pipeline.

## Supported File Types:
- PDF files (.pdf)
- Image files (.png, .jpg, .jpeg, .tiff, .bmp)

## Usage:
1. Place your test files in this directory
2. Update config.json to point to this directory
3. Run the pipeline: python run_pipeline.py

## Example Configuration:
```json
{
  "input": {
    "input_directory": "./data/samples",
    "process_all_files": true
  }
}
```
"""
    
    sample_readme_path = Path("data/samples/README.md")
    try:
        with open(sample_readme_path, 'w', encoding='utf-8') as f:
            f.write(sample_readme)
        print("   ✅ Created sample directory README")
    except Exception as e:
        print(f"   ⚠️  Could not create sample README: {e}")

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE")
    print("="*60)
    print("\n📋 Next Steps:")
    print("   1. Add your PDF/image files to ./data/input/ directory")
    print("   2. Edit config.json if needed to customize processing")
    print("   3. Run the pipeline:")
    print("      python run_pipeline.py")
    print("\n🔧 Advanced Usage:")
    print("   - Run examples: python quick_start.py")
    print("   - Validate config: python run_pipeline.py --validate-only")
    print("   - Dry run: python run_pipeline.py --dry-run")
    print("   - Custom config: python run_pipeline.py -c my_config.json")
    print("\n📚 Documentation:")
    print("   - Main documentation: README.md")
    print("   - Configuration options: Check config.json comments")
    print("\n❓ Troubleshooting:")
    print("   - Check logs in pipeline.log")
    print("   - Ensure input files are in correct directory")
    print("   - For LLM features, ensure Ollama is running")

def main():
    """Main setup function."""
    print("🚀 OCR Pipeline Setup")
    print("="*50)
    
    setup_success = True
    
    # Check Python version
    if not check_python_version():
        setup_success = False
        print("\n❌ Setup cannot continue with incompatible Python version")
        return
    
    # Check and install dependencies
    installed, missing = check_dependencies()
    
    if missing:
        print(f"\n📦 Found {len(missing)} missing packages")
        install_choice = input("Would you like to install them automatically? (y/n): ").strip().lower()
        
        if install_choice in ['y', 'yes']:
            if not install_missing_packages(missing):
                setup_success = False
                print("\n❌ Some packages failed to install")
        else:
            print("\n⚠️  Setup continuing without installing missing packages")
            print("   You may need to install them manually:")
            for package in missing:
                print(f"      pip install {package}")
            setup_success = False
    
    # Create directory structure
    create_directory_structure()
    
    # Create default configuration
    create_default_config()
    
    # Test OCR functionality
    if not missing or installed:  # Only test if we have the dependencies
        ocr_test = input("\nWould you like to test OCR functionality? (y/n): ").strip().lower()
        if ocr_test in ['y', 'yes']:
            if not test_ocr_functionality():
                setup_success = False
    
    # Check LLM service (optional)
    llm_check = input("\nWould you like to check LLM service availability? (y/n): ").strip().lower()
    if llm_check in ['y', 'yes']:
        check_llm_service()  # This is optional, so don't fail setup if it's not available
    
    # Create sample files
    create_sample_files()
    
    # Print final status and next steps
    if setup_success:
        print_next_steps()
    else:
        print("\n" + "="*60)
        print("⚠️  SETUP COMPLETED WITH WARNINGS")
        print("="*60)
        print("\n🔧 Some components may not work correctly.")
        print("Please review the warnings above and install missing dependencies.")
        print("\n📋 To install all dependencies manually:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n👋 Setup interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
