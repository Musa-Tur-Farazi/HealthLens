"""
Google Drive configuration for model storage and retrieval.
This module handles downloading model files from Google Drive using direct links.
No API key required - just public share links!
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional

try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    import requests

# Model file mappings - Update these with your actual Google Drive direct download URLs
MODEL_FILES = {
    "best.pt": "https://drive.google.com/file/d/1NzMkIVjQ25D4j5cSih8khm8NV5ZPfxgY/view?usp=sharing",
    "xray_best.pt": "https://drive.google.com/file/d/1tE91s0B8m8OoxOXEdZA-dJBwKAkEveaA/view?usp=sharing", 
    "classes.json": "https://drive.google.com/file/d/1NsfARLKkJBuGzmDSYfu079yvR2dXZJlP/view?usp=sharing",
    "xray_classes.json": "https://drive.google.com/file/d/1mTcEOH53um9OVDar1EsMK3LvvIz-T_-O/view?usp=sharing",
    "calibration.json": "https://drive.google.com/file/d/1RaLHku-kVhX4LMfVsrhqevw56wxYI7xh/view?usp=sharing"
}

def download_file_from_google_drive(download_url: str, local_path: str, expected_size: Optional[int] = None) -> bool:
    """
    Download a file from Google Drive using gdown (preferred) or requests fallback.
    
    Args:
        download_url: Direct Google Drive download URL or sharing link
        local_path: Local path to save the file
        expected_size: Expected file size for validation (optional)
    
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        print(f"Downloading from Google Drive...")
        print(f"URL: {download_url}")
        
        # Create directory if it doesn't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Try gdown first (more reliable for Google Drive)
        if GDOWN_AVAILABLE:
            print("Using gdown for reliable Google Drive download...")
            try:
                gdown.download(download_url, local_path, quiet=False, fuzzy=True)
                
                # Validate the downloaded file
                if not os.path.exists(local_path):
                    print(f"❌ Download failed - file not created")
                    return False
                
                file_size = os.path.getsize(local_path)
                print(f"✅ Downloaded {local_path} ({file_size} bytes)")
                
                # Check if file is HTML (failed download)
                with open(local_path, 'rb') as f:
                    first_byte = f.read(1)
                    if first_byte == b'<':
                        print(f"❌ Error: Downloaded file is HTML, not a valid model file.")
                        print(f"   Make sure the Google Drive file has 'Anyone with the link' sharing enabled.")
                        os.remove(local_path)
                        return False
                
                return True
            except Exception as e:
                print(f"gdown failed: {e}, falling back to requests...")
        
        # Fallback to requests method
        import requests
        
        # Convert view/sharing URLs to direct download format
        if "/file/d/" in download_url and "/view" in download_url:
            import re
            match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', download_url)
            if match:
                file_id = match.group(1)
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                print(f"Converted to direct download URL: {download_url}")
        
        # Handle Google Drive's virus scan warning for large files
        session = requests.Session()
        
        # First request to get the actual download URL (for large files)
        response = session.get(download_url, stream=True, allow_redirects=True)
        
        # Check if we got HTML instead of the file (common for large files)
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type or response.content[:1] == b'<':
            # Extract confirmation token for large files
            import re
            content_str = response.content.decode('utf-8', errors='ignore')
            
            # Look for the confirmation form
            match = re.search(r'action="([^"]+)"', content_str)
            if match:
                confirm_url = match.group(1).replace('&amp;', '&')
                if not confirm_url.startswith('http'):
                    confirm_url = "https://drive.google.com" + confirm_url
                print("Large file detected, confirming download...")
                response = session.get(confirm_url, stream=True, allow_redirects=True)
            else:
                # Try the confirm parameter approach
                if 'id=' in download_url:
                    file_id = download_url.split('id=')[1].split('&')[0]
                    confirm_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
                    print("Trying with confirm parameter...")
                    response = session.get(confirm_url, stream=True, allow_redirects=True)
        
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        first_chunk = True
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    # Validate first chunk is not HTML
                    if first_chunk:
                        if chunk[:1] == b'<':
                            print(f"\n❌ Error: Received HTML instead of file. Check sharing permissions.")
                            print(f"   First bytes: {chunk[:100]}")
                            return False
                        first_chunk = False
                    
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end="", flush=True)
        
        print(f"\nDownloaded {local_path} ({downloaded} bytes)")
        
        # Final validation - check file is not HTML
        with open(local_path, 'rb') as f:
            first_byte = f.read(1)
            if first_byte == b'<':
                print(f"❌ Error: Downloaded file is HTML, not a valid model file.")
                os.remove(local_path)
                return False
        
        # Validate file size if expected size provided
        if expected_size and downloaded != expected_size:
            print(f"Warning: File size mismatch. Expected {expected_size}, got {downloaded}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error downloading from {download_url}: {e}")
        return False

def ensure_models_downloaded() -> Dict[str, bool]:
    """
    Ensure all required model files are downloaded from Google Drive.
    
    Returns:
        Dict mapping file names to download success status
    """
    results = {}
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    for filename, download_url in MODEL_FILES.items():
        if "YOUR_" in download_url:
            print(f"Warning: {filename} download URL not configured in MODEL_FILES")
            results[filename] = False
            continue
            
        local_path = outputs_dir / filename
        
        # Check if file already exists and is valid
        if local_path.exists():
            file_size = local_path.stat().st_size
            # Skip if file is reasonably sized (not empty or corrupted)
            if file_size > 1000:  # At least 1KB
                print(f"{filename} already exists ({file_size} bytes)")
                results[filename] = True
                continue
        
        # Download from Google Drive
        print(f"Downloading {filename}...")
        success = download_file_from_google_drive(download_url, str(local_path))
        results[filename] = success
    
    return results

def get_model_status() -> Dict[str, Dict]:
    """Get status of all model files."""
    status = {}
    outputs_dir = Path("outputs")
    
    for filename in MODEL_FILES.keys():
        file_path = outputs_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            status[filename] = {
                "exists": True,
                "size": size,
                "size_mb": round(size / (1024 * 1024), 2)
            }
        else:
            status[filename] = {
                "exists": False,
                "size": 0,
                "size_mb": 0
            }
    
    return status

if __name__ == "__main__":
    # Test the download functionality
    print("Model files status:")
    status = get_model_status()
    for filename, info in status.items():
        print(f"  {filename}: {'✓' if info['exists'] else '✗'} ({info['size_mb']} MB)")
    
    print("\nEnsuring models are downloaded...")
    results = ensure_models_downloaded()
    
    print("\nDownload results:")
    for filename, success in results.items():
        print(f"  {filename}: {'✓' if success else '✗'}")
