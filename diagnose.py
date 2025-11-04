#!/usr/bin/env python3
"""
Diagnostic script to check model files and environment.
Run this on Render or locally to diagnose issues.

Usage:
    python diagnose.py
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check environment variables and setup."""
    print("="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)
    
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"CWD: {os.getcwd()}")
    
    env_vars = [
        "PORT", "FORCE_CPU", "DISEASE_CKPT", "DISEASE_CLASSES",
        "XR_CKPT", "XR_CLASSES", "RENDER"
    ]
    
    print("\nEnvironment Variables:")
    for var in env_vars:
        val = os.environ.get(var, "<not set>")
        print(f"  {var}: {val}")
    
    print()

def check_dependencies():
    """Check if required packages are installed."""
    print("="*60)
    print("DEPENDENCY CHECK")
    print("="*60)
    
    packages = [
        "fastapi", "uvicorn", "torch", "torchvision", "timm",
        "transformers", "PIL", "requests", "gdown"
    ]
    
    for pkg in packages:
        try:
            if pkg == "PIL":
                import PIL
                print(f"✅ {pkg:20} {PIL.__version__}")
            else:
                mod = __import__(pkg)
                version = getattr(mod, "__version__", "unknown")
                print(f"✅ {pkg:20} {version}")
        except ImportError:
            print(f"❌ {pkg:20} NOT INSTALLED")
    
    print()

def check_model_files():
    """Check if model files exist and are valid."""
    print("="*60)
    print("MODEL FILE CHECK")
    print("="*60)
    
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print(f"❌ outputs/ directory does not exist")
        outputs_dir.mkdir()
        print(f"✅ Created outputs/ directory")
    
    expected_files = [
        "derm_best.pt",
        "melanoma_best.pt",
        "best.pt",
        "xray_best.pt",
        "derm_classes.json",
        "mel_classes.json",
        "classes.json",
        "xray_classes.json",
        "calibration.json"
    ]
    
    found_files = []
    for filename in expected_files:
        filepath = outputs_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            
            # Check if it's HTML
            with open(filepath, 'rb') as f:
                first_byte = f.read(1)
            
            is_html = (first_byte == b'<')
            size_mb = size / 1024 / 1024
            
            if is_html:
                status = f"❌ HTML ({size} bytes)"
            elif size < 10000:  # Less than 10KB is suspicious
                status = f"⚠️  Too small ({size} bytes)"
            else:
                status = f"✅ OK ({size_mb:.2f} MB)"
                found_files.append(filename)
            
            print(f"  {filename:25} {status}")
        else:
            print(f"  {filename:25} ❌ Not found")
    
    print(f"\n✅ Found {len(found_files)} valid files")
    print()
    
    return found_files

def check_model_loading():
    """Try to load models and show errors."""
    print("="*60)
    print("MODEL LOADING CHECK")
    print("="*60)
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        # Try loading disease model
        disease_paths = [
            "outputs/derm_best.pt",
            "outputs/melanoma_best.pt",
            "outputs/best.pt"
        ]
        
        disease_loaded = False
        for path in disease_paths:
            if Path(path).exists():
                print(f"\nTrying to load: {path}")
                try:
                    checkpoint = torch.load(path, map_location=device, weights_only=False)
                    print(f"✅ Loaded checkpoint successfully")
                    
                    # Check structure
                    if isinstance(checkpoint, dict):
                        keys = list(checkpoint.keys())
                        print(f"   Keys: {keys[:5]}...")
                        
                        if "model" in checkpoint:
                            model_keys = list(checkpoint["model"].keys())
                            print(f"   Model keys: {model_keys[:5]}...")
                    
                    disease_loaded = True
                    break
                except Exception as e:
                    print(f"❌ Failed to load: {e}")
        
        if not disease_loaded:
            print("❌ Could not load disease model")
        
        # Try loading xray model
        xray_path = "outputs/xray_best.pt"
        if Path(xray_path).exists():
            print(f"\nTrying to load: {xray_path}")
            try:
                checkpoint = torch.load(xray_path, map_location=device, weights_only=False)
                print(f"✅ Loaded checkpoint successfully")
            except Exception as e:
                print(f"❌ Failed to load: {e}")
        else:
            print(f"\n❌ {xray_path} not found")
        
    except Exception as e:
        print(f"❌ Error during loading: {e}")
    
    print()

def check_gdrive_download():
    """Test Google Drive download capability."""
    print("="*60)
    print("GOOGLE DRIVE DOWNLOAD TEST")
    print("="*60)
    
    try:
        import gdown
        print("✅ gdown is available")
        
        # Test with a small public file
        test_url = "https://drive.google.com/file/d/1NzMkIVjQ25D4j5cSih8khm8NV5ZPfxgY/view?usp=sharing"
        test_output = "test_download.tmp"
        
        print(f"Testing download from Google Drive...")
        print(f"URL: {test_url}")
        
        try:
            gdown.download(test_url, test_output, quiet=False, fuzzy=True)
            
            if os.path.exists(test_output):
                size = os.path.getsize(test_output)
                with open(test_output, 'rb') as f:
                    first_byte = f.read(1)
                
                if first_byte == b'<':
                    print(f"❌ Downloaded HTML ({size} bytes)")
                    print(f"🔧 FIX: Ensure Google Drive file has 'Anyone with the link' sharing")
                else:
                    print(f"✅ Downloaded successfully ({size} bytes)")
                
                os.remove(test_output)
            else:
                print(f"❌ File not created")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            
    except ImportError:
        print("❌ gdown not installed")
        print("   Install with: pip install gdown")
    
    print()

def main():
    """Run all diagnostic checks."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "DISEASELLM DIAGNOSTICS" + " "*21 + "║")
    print("╚" + "="*58 + "╝")
    print()
    
    check_environment()
    check_dependencies()
    found_files = check_model_files()
    check_model_loading()
    check_gdrive_download()
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    if len(found_files) >= 2:  # At least disease + xray
        print("✅ Model files look good")
    else:
        print("❌ Missing or corrupted model files")
        print("   Run: python gdrive_config.py")
    
    print("\nNext steps:")
    print("1. Fix any ❌ issues above")
    print("2. Ensure Google Drive files have proper sharing")
    print("3. Run: python app.py")
    print()

if __name__ == "__main__":
    main()

