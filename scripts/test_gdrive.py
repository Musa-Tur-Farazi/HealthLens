#!/usr/bin/env python3
"""
Quick test script to diagnose Google Drive download issues.
Run this locally to verify your Google Drive links work correctly.

Usage:
    python test_gdrive.py
"""

import os
from pathlib import Path

def test_single_file(file_id: str, filename: str):
    """Test downloading a single file from Google Drive."""
    print(f"\n{'='*60}")
    print(f"Testing: {filename}")
    print(f"File ID: {file_id}")
    print(f"{'='*60}")
    
    url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    output_path = f"test_downloads/{filename}"
    
    # Create test directory
    Path("test_downloads").mkdir(exist_ok=True)
    
    try:
        # Try with gdown first
        import gdown
        print("✅ gdown is available")
        print(f"Downloading to: {output_path}")
        
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        
        # Check result
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"✅ File downloaded: {size:,} bytes ({size/1024/1024:.2f} MB)")
            
            # Check if it's HTML
            with open(output_path, 'rb') as f:
                first_bytes = f.read(100)
                
            if first_bytes[:1] == b'<':
                print(f"❌ ERROR: File is HTML, not binary!")
                print(f"First 100 bytes: {first_bytes[:100]}")
                print(f"\n🔧 FIX: Right-click file in Google Drive → Share → 'Anyone with the link' → Viewer")
                return False
            else:
                print(f"✅ File appears to be binary (first byte: {first_bytes[:1]})")
                return True
        else:
            print(f"❌ ERROR: File was not created")
            return False
            
    except ImportError:
        print("❌ gdown not installed. Install with: pip install gdown")
        print("Falling back to requests...")
        
        try:
            import requests
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            print(f"Using URL: {download_url}")
            
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            # Check if we got HTML
            content_type = response.headers.get('content-type', '')
            print(f"Content-Type: {content_type}")
            
            if 'text/html' in content_type.lower():
                print(f"❌ ERROR: Received HTML instead of file")
                print(f"First 200 chars: {response.content[:200]}")
                return False
            
            # Download
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            size = os.path.getsize(output_path)
            print(f"✅ Downloaded: {size:,} bytes")
            
            # Verify not HTML
            with open(output_path, 'rb') as f:
                if f.read(1) == b'<':
                    print(f"❌ ERROR: File is HTML")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return False
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Test all model files."""
    print("="*60)
    print("Google Drive Download Tester")
    print("="*60)
    
    # Your file IDs from gdrive_config.py
    files = {
        "best.pt": "1NzMkIVjQ25D4j5cSih8khm8NV5ZPfxgY",
        "xray_best.pt": "1tE91s0B8m8OoxOXEdZA-dJBwKAkEveaA",
        "derm_classes.json": "1NsfARLKkJBuGzmDSYfu079yvR2dXZJlP",
        "xray_classes.json": "1mTcEOH53um9OVDar1EsMK3LvvIz-T_-O",
        "calibration.json": "1RaLHku-kVhX4LMfVsrhqevw56wxYI7xh",
    }
    
    results = {}
    for filename, file_id in files.items():
        results[filename] = test_single_file(file_id, filename)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for filename, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status:15} {filename}")
    
    failed = [f for f, s in results.items() if not s]
    if failed:
        print(f"\n❌ {len(failed)} file(s) failed to download correctly:")
        for f in failed:
            print(f"   - {f}")
        print(f"\n🔧 NEXT STEPS:")
        print(f"   1. Open Google Drive in your browser")
        print(f"   2. For each failed file:")
        print(f"      - Right-click → Share")
        print(f"      - Under 'General access' → 'Anyone with the link'")
        print(f"      - Permission: Viewer")
        print(f"      - Click 'Done'")
        print(f"   3. Re-run this script to verify")
    else:
        print(f"\n✅ All files downloaded successfully!")
        print(f"   You're ready to deploy to Render.")
    
    print(f"\nTest files saved to: test_downloads/")
    print(f"You can delete this directory after testing.")

if __name__ == "__main__":
    main()

