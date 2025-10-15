#!/usr/bin/env python3
"""
Simple script to help you get Google Drive direct download links.
No API key required - just share your files and get the direct download URLs!
"""

def get_direct_download_url(share_url: str) -> str:
    """
    Convert Google Drive share URL to direct download URL.
    
    Args:
        share_url: Google Drive share URL like:
                  https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    
    Returns:
        Direct download URL
    """
    import re
    
    # Extract file ID from various Google Drive URL formats
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)'
    ]
    
    file_id = None
    for pattern in patterns:
        match = re.search(pattern, share_url)
        if match:
            file_id = match.group(1)
            break
    
    if not file_id:
        raise ValueError("Could not extract file ID from URL. Please check the URL format.")
    
    # Return direct download URL
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def main():
    """Interactive helper to get direct download URLs."""
    print("🚀 Google Drive Direct Download Link Generator")
    print("=" * 50)
    print()
    print("This script helps you convert Google Drive share links")
    print("to direct download URLs for your model files.")
    print()
    print("📋 Steps:")
    print("1. Upload your model files to Google Drive")
    print("2. Right-click each file → 'Get link' → 'Anyone with the link can view'")
    print("3. Copy the share URL")
    print("4. Paste it here to get the direct download URL")
    print()
    
    model_files = {
        "best.pt": "Disease Classification Model",
        "xray_best.pt": "X-Ray Classification Model",
        "classes.json": "Disease Classes",
        "xray_classes.json": "X-Ray Classes", 
        "calibration.json": "Model Calibration"
    }
    
    print("📝 Generated configuration for gdrive_config.py:")
    print("MODEL_FILES = {")
    
    for filename, description in model_files.items():
        print(f"\n# {description}")
        share_url = input(f"Enter share URL for {filename}: ").strip()
        
        if not share_url:
            print(f'    "{filename}": "https://drive.google.com/uc?export=download&id=YOUR_{filename.upper().replace(".", "_")}_FILE_ID",')
            continue
        
        try:
            direct_url = get_direct_download_url(share_url)
            print(f'    "{filename}": "{direct_url}",')
        except ValueError as e:
            print(f"    # Error: {e}")
            print(f'    "{filename}": "https://drive.google.com/uc?export=download&id=YOUR_{filename.upper().replace(".", "_")}_FILE_ID",')
    
    print("}")
    
    print("\n🎉 Done! Copy the above configuration into your gdrive_config.py file.")
    print("\n💡 Tips:")
    print("- Make sure your files are shared with 'Anyone with the link can view'")
    print("- Large files (>25MB) may show a virus scan warning - that's normal")
    print("- Test the direct URLs by opening them in a browser")

if __name__ == "__main__":
    main()
