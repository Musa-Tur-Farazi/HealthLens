# Google Drive Setup for Model Files

## Problem: "invalid load key, '<'"

This error means the downloaded file is HTML (a Google Drive error/preview page) instead of the actual model file.

## Root Cause

Your Google Drive links are in "view" format but the files may not have proper sharing permissions:
- ❌ Current: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
- The gdrive_config.py converts these to: `https://drive.google.com/uc?export=download&id=FILE_ID`
- **BUT** this only works if the file has correct sharing permissions!

## Solution: Fix Google Drive Sharing Permissions

### For EACH model file in your Google Drive:

1. **Right-click the file** → `Share`
2. **Under "General access"**, click `Change`
3. **Select "Anyone with the link"**
4. **Set permission to "Viewer"**
5. **Click "Done"**

### Files that need this setup:
- `best.pt` (or `derm_best.pt`)
- `xray_best.pt`
- `classes.json` (or `derm_classes.json`)
- `xray_classes.json`
- `calibration.json`

## How to Test Locally

```bash
python gdrive_config.py
```

This will:
- Show which files exist
- Attempt to download missing files
- Show first 100 bytes if HTML is received (helps debug)

## Expected Output (Success)

```
Downloading best.pt...
Converting to direct download URL: https://drive.google.com/uc?export=download&id=...
Progress: 100.0%
Downloaded outputs/best.pt (XXX bytes)
✅ Downloaded successfully
```

## Expected Output (Failure - needs permission fix)

```
Downloading best.pt...
❌ Error: Received HTML instead of file. Check sharing permissions.
   First bytes: b'<!DOCTYPE html><html>...'
```

## Alternative: Use Direct Download Services

If Google Drive permissions continue to fail, consider:

1. **Hugging Face Hub** (recommended for ML models):
   ```python
   from huggingface_hub import hf_hub_download
   model_path = hf_hub_download(repo_id="your-username/your-repo", filename="best.pt")
   ```

2. **GitHub Releases** (for files < 100MB):
   - Upload to GitHub release
   - Use raw release asset URL

3. **Dropbox/OneDrive** with direct links

## Render Deployment

On Render, make sure:
- Start command: `python app.py` (NOT `python -m uvicorn app:app`)
- Environment variables are NOT needed if using gdrive_config.py
- The app will auto-download on first startup (takes ~2-3 minutes)

## Port Binding (Render)

The app now:
- ✅ Binds to `0.0.0.0` (listens on all interfaces)
- ✅ Uses `$PORT` environment variable (Render sets this)
- ✅ Has both GET and HEAD on "/" route (for health checks)

Render should detect the port within 60 seconds after "Application startup complete."

