#!/usr/bin/env python3
"""
Requirements Validation Script for OCR Pipeline
Run this after installing requirements.txt to verify everything works
"""

import sys
import importlib
from typing import List, Tuple, Dict, Any

def check_package_import(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Check if a package can be imported and return status with version info.
    
    Args:
        package_name: Display name of the package
        import_name: Actual import name (if different from package name)
    
    Returns:
        (success, message) tuple
    """
    if import_name is None:
        import_name = package_name
        
    try:
        module = importlib.import_module(import_name)
        
        # Try to get version information
        version = "unknown"
        for attr in ['__version__', 'version', 'VERSION']:
            if hasattr(module, attr):
                version = getattr(module, attr)
                if callable(version):
                    version = version()
                break
                
        return True, f"{package_name} v{version}"
    except ImportError as e:
        return False, f"{package_name}: {str(e)}"
    except Exception as e:
        return True, f"{package_name}: imported but version check failed ({str(e)})"

def main():
    """Main validation function."""
    print("🔍 OCR Pipeline Requirements Validation")
    print("=" * 50)
    
    # Define core packages with their import names
    core_packages = [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("Pillow", "PIL"),
        ("PyMuPDF", "fitz"),
        ("PDF2Image", "pdf2image"),
        ("Requests", "requests"),
        ("Pandas", "pandas"),
        ("Scikit-learn", "sklearn"),
        ("SciPy", "scipy"),
        ("TQDM", "tqdm"),
        ("ReportLab", "reportlab"),
    ]
    
    # OCR-specific packages  
    ocr_packages = [
        ("PaddleOCR", "paddleocr"),
        ("Pytesseract", "pytesseract"),
    ]
    
    # Optional packages
    optional_packages = [
        ("PSUtil", "psutil"),
        ("PaddlePaddle", "paddle"),
    ]
    
    # Track results
    results = {
        "core_success": 0,
        "core_total": len(core_packages),
        "ocr_success": 0, 
        "ocr_total": len(ocr_packages),
        "optional_success": 0,
        "optional_total": len(optional_packages),
    }
    
    # Check core packages
    print("\n📦 Core Packages:")
    for package_name, import_name in core_packages:
        success, message = check_package_import(package_name, import_name)
        status = "✅" if success else "❌"
        print(f"  {status} {message}")
        if success:
            results["core_success"] += 1
    
    # Check OCR packages  
    print("\n🔍 OCR Packages:")
    for package_name, import_name in ocr_packages:
        success, message = check_package_import(package_name, import_name)
        status = "✅" if success else "❌"
        print(f"  {status} {message}")
        if success:
            results["ocr_success"] += 1
            
    # Check optional packages
    print("\n🔧 Optional Packages:")
    for package_name, import_name in optional_packages:
        success, message = check_package_import(package_name, import_name)
        status = "✅" if success else "⚠️ "
        print(f"  {status} {message}")
        if success:
            results["optional_success"] += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Validation Summary:")
    print(f"  Core packages:     {results['core_success']}/{results['core_total']}")
    print(f"  OCR packages:      {results['ocr_success']}/{results['ocr_total']}")
    print(f"  Optional packages: {results['optional_success']}/{results['optional_total']}")
    
    # Overall status
    core_ok = results["core_success"] == results["core_total"]
    ocr_ok = results["ocr_success"] == results["ocr_total"]
    
    if core_ok and ocr_ok:
        print("\n🎉 All required packages are installed and working!")
        print("   You're ready to run the OCR pipeline!")
        print("\n   Next steps:")
        print("   1. Edit config.json with your settings")
        print("   2. Add files to data/input/ directory")  
        print("   3. Run: python run_pipeline.py")
        return 0
    else:
        print("\n⚠️  Some required packages are missing or have issues.")
        print("\n   To fix this:")
        print("   1. Make sure you're in a virtual environment:")
        print("      python -m venv .env && source .env/bin/activate")
        print("   2. Install requirements:")
        print("      pip install -r requirements.txt")
        print("   3. Run this validation again:")
        print("      python validate_requirements.py")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
