#!/usr/bin/env python3
"""
Quick setup script for Google Drive integration.
This script helps you configure Google Drive for model storage.
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check if required environment variables are set."""
    print("🔍 Checking environment setup...")
    
    # No Google API key needed anymore!
    print("✅ No Google API key required!")
    print("   Direct download URLs are configured in gdrive_config.py")
    return True

def check_model_files():
    """Check if model files exist locally."""
    print("\n📁 Checking local model files...")
    
    outputs_dir = Path("outputs")
    model_files = ["best.pt", "xray_best.pt", "classes.json", "xray_classes.json", "calibration.json"]
    
    existing_files = []
    missing_files = []
    
    for filename in model_files:
        file_path = outputs_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            existing_files.append((filename, size_mb))
            print(f"✅ {filename} ({size_mb:.1f} MB)")
        else:
            missing_files.append(filename)
            print(f"❌ {filename} (missing)")
    
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} files missing locally")
        print("   These will be downloaded from Google Drive on first run")
    
    return len(existing_files) > 0

def test_gdrive_connection():
    """Test Google Drive direct download configuration."""
    print("\n🌐 Testing Google Drive configuration...")
    
    try:
        from gdrive_config import get_model_status, MODEL_FILES
        
        # Check if URLs are configured
        configured_files = []
        unconfigured_files = []
        
        for filename, url in MODEL_FILES.items():
            if "YOUR_" in url:
                unconfigured_files.append(filename)
            else:
                configured_files.append(filename)
        
        if unconfigured_files:
            print(f"⚠️  {len(unconfigured_files)} files not configured:")
            for filename in unconfigured_files:
                print(f"   - {filename}")
            print("\n🔧 To fix:")
            print("1. Run: python get_gdrive_links.py")
            print("2. Update URLs in gdrive_config.py")
            return False
        
        # Test connection by getting model status
        status = get_model_status()
        print("✅ Google Drive configuration looks good")
        
        # Show status
        print("\n📊 Model files status:")
        for filename, info in status.items():
            status_icon = "✅" if info["exists"] else "❌"
            print(f"   {status_icon} {filename} ({info['size_mb']} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Google Drive configuration check failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check direct download URLs in gdrive_config.py")
        print("2. Ensure files are shared with 'Anyone with link can view'")
        print("3. Test URLs in browser")
        return False

def main():
    """Main setup function."""
    print("🚀 DiseaseLLM Google Drive Setup")
    print("=" * 40)
    
    # Check environment
    env_ok = check_environment()
    
    # Check local files
    local_files_ok = check_model_files()
    
    # Test Google Drive connection
    if env_ok:
        gdrive_ok = test_gdrive_connection()
    else:
        gdrive_ok = False
    
    print("\n" + "=" * 40)
    print("📋 Setup Summary")
    print("=" * 40)
    
    if env_ok and gdrive_ok:
        print("🎉 Setup complete! Your Google Drive integration is ready.")
        print("\n🚀 Next steps:")
        print("1. Run your app: python app.py")
        print("2. Models will be downloaded automatically on first run")
        print("3. Deploy to Vercel when ready")
    else:
        print("⚠️  Setup incomplete. Please fix the issues above.")
        
        if not env_ok:
            print("\n🔧 Environment setup needed:")
            print("1. Upload model files to Google Drive")
            print("2. Share files with 'Anyone with link can view'")
            print("3. Run: python get_gdrive_links.py")
            print("4. Update URLs in gdrive_config.py")
        
        if not gdrive_ok and env_ok:
            print("\n🔧 Google Drive setup needed:")
            print("1. Upload model files to Google Drive")
            print("2. Share files with 'Anyone with link can view'")
            print("3. Update file IDs in gdrive_config.py")
    
    print("\n📖 For detailed instructions, see:")
    print("   - deploy_setup.md")
    print("   - DEPLOYMENT_CHECKLIST.md")

if __name__ == "__main__":
    main()
