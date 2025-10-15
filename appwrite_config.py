"""
Google Drive configuration for model storage and retrieval.
This module handles downloading model files from Google Drive.
"""

import os
import requests
import json
from pathlib import Path
from typing import Dict, Optional

# Google Drive configuration
GOOGLE_DRIVE_FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Model file mappings - Update these with your actual Google Drive file IDs
MODEL_FILES = {
    "best.pt": "your_disease_model_file_id",
    "xray_best.pt": "your_xray_model_file_id", 
    "classes.json": "your_classes_file_id",
    "xray_classes.json": "your_xray_classes_file_id",
    "calibration.json": "your_calibration_file_id"
}

def get_google_drive_file_url(file_id: str) -> str:
    """Get the download URL for a file from Google Drive."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY must be set")
    
    # First get file info to get the download URL
    info_url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    params = {"key": GOOGLE_API_KEY}
    
    try:
        response = requests.get(info_url, params=params)
        response.raise_for_status()
        file_info = response.json()
        
        # For large files, we need to use the downloadUrl or webContentLink
        download_url = file_info.get("downloadUrl") or file_info.get("webContentLink")
        if not download_url:
            raise ValueError(f"No download URL found for file {file_id}")
        
        return download_url
    except Exception as e:
        raise ValueError(f"Failed to get file info for {file_id}: {e}")

def download_file_from_google_drive(file_id: str, local_path: str, expected_size: Optional[int] = None) -> bool:
    """
    Download a file from Google Drive.
    
    Args:
        file_id: The Google Drive file ID
        local_path: Local path to save the file
        expected_size: Expected file size for validation (optional)
    
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        url = get_google_drive_file_url(file_id)
        
        print(f"Downloading {file_id} from Google Drive...")
        
        # For Google Drive, we need to add the API key as a parameter
        if GOOGLE_API_KEY and "?" not in url:
            url += f"?key={GOOGLE_API_KEY}"
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end="", flush=True)
        
        print(f"\nDownloaded {local_path} ({downloaded} bytes)")
        
        # Validate file size if expected size provided
        if expected_size and downloaded != expected_size:
            print(f"Warning: File size mismatch. Expected {expected_size}, got {downloaded}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error downloading {file_id}: {e}")
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
    
    # Check if we have Google Drive configuration
    if not GOOGLE_API_KEY:
        print("Warning: Google Drive not configured. Models will not be downloaded.")
        return {file: False for file in MODEL_FILES.keys()}
    
    for filename, file_id in MODEL_FILES.items():
        if file_id == f"your_{filename.replace('.', '_')}_file_id":
            print(f"Warning: {filename} file ID not configured in MODEL_FILES")
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
        success = download_file_from_google_drive(file_id, str(local_path))
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
