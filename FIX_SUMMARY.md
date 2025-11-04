# Deployment Fix Summary

## Issues Fixed

### 1. ❌ HEAD / returns 405 Method Not Allowed
**Root Cause:** Render health checks use `HEAD /` but route only had `@app.get("/")`

**Fix in `app.py`:**
```python
@app.get("/")
@app.head("/")  # ← Added this
def root():
    return {"status": "ok", ...}
```

### 2. ❌ invalid load key, '<'
**Root Cause:** Google Drive files downloading as HTML error pages instead of binary .pt files

**Multiple fixes applied:**

#### a) `gdrive_config.py` - Added `gdown` library support
```python
import gdown  # More reliable than raw requests for Google Drive
gdown.download(download_url, local_path, quiet=False, fuzzy=True)
```

#### b) `gdrive_config.py` - Better error detection
```python
# Check if downloaded file is HTML
with open(local_path, 'rb') as f:
    if f.read(1) == b'<':
        print("❌ Error: Downloaded file is HTML")
        print("   Make sure sharing is 'Anyone with the link'")
```

#### c) `app.py` - Pre-load validation
```python
# Check checkpoint file before loading
with open(ckpt_path, "rb") as fh:
    if fh.read(1) == b"<":
        raise RuntimeError("Checkpoint looks like HTML. Provide valid .pt file.")
```

#### d) `app.py` - Fixed default model paths
```python
# Changed from non-existent paths to actual files:
DISEASE["ckpt"] = "outputs/derm_best.pt"  # was "outputs/best.pt"
DISEASE["classes"] = "outputs/derm_classes.json"  # was "outputs/classes.json"

# Added fallback logic to find first existing file
```

### 3. 🔧 Port Binding
**Enhanced in `app.py`:**
```python
port = int(os.environ.get("PORT", 8000))
print(f"Starting server on 0.0.0.0:{port}")
uvicorn.run("app:app", host="0.0.0.0", port=port)
```

## Files Modified

### Core Changes
1. **app.py**
   - Added `@app.head("/")` decorator
   - Fixed DISEASE model default paths
   - Added local file fallback logic
   - Added checkpoint validation (HTML detection)
   - Enhanced port binding with logging

2. **gdrive_config.py**
   - Integrated `gdown` library (preferred method)
   - Fallback to requests if `gdown` unavailable
   - Better HTML detection and error messages
   - Auto-conversion of view URLs to download URLs

3. **requirements.txt**
   - Added `gdown>=5.0.0` for reliable Google Drive downloads

### Documentation Added
4. **GDRIVE_SETUP.md** - How to configure Google Drive sharing permissions
5. **RENDER_DEPLOY.md** - Complete Render deployment guide
6. **FIX_SUMMARY.md** - This file

## Root Cause Analysis

The "invalid load key, '<'" error happens when:
1. Google Drive URL points to a file
2. File doesn't have proper sharing permissions (or exceeded quota)
3. Instead of the binary file, Google Drive returns an HTML error/login page
4. PyTorch tries to load HTML as a .pt file
5. HTML starts with `<!DOCTYPE html>` → first character is '<'
6. PyTorch's unpickler sees '<' instead of expected pickle header → error

## Solution Path

### Immediate Fix (Recommended)
1. Install `gdown`: `pip install gdown>=5.0.0` ✅ (already in requirements.txt)
2. Ensure Google Drive files have "Anyone with the link" sharing
3. Test locally: `python gdrive_config.py`
4. Deploy to Render with start command: `python app.py`

### Alternative Fix (If Google Drive fails)
1. Download model files locally
2. Upload to Render Disk or commit to Git (if < 100MB)
3. Set environment variables to local paths:
   ```bash
   DISEASE_CKPT=outputs/derm_best.pt
   XR_CKPT=outputs/xray_best.pt
   ```

## Verification Steps

### Local Testing
```bash
# 1. Test Google Drive downloads
python gdrive_config.py

# Expected: "✅ Downloaded outputs/*.pt"
# If fails: "❌ Error: Downloaded file is HTML"

# 2. Start server
python app.py

# Expected:
# ✅ Loaded disease: XX classes on cpu
# ✅ Loaded xray: XX classes on cpu
# Starting server on 0.0.0.0:8000
```

### Render Testing
```bash
# 1. Check deployment logs for:
==> Running 'python app.py'
Starting server on 0.0.0.0:10000
✅ Loaded disease: XX classes
✅ Loaded xray: XX classes
INFO: Uvicorn running on http://0.0.0.0:10000
==> Your service is live

# 2. Test health endpoint:
curl https://your-app.onrender.com/health

# Should return JSON with:
# "vision_models_loaded": ["disease", "xray"]

# 3. Test root endpoint (Render health check):
curl -I https://your-app.onrender.com/

# Should return: HTTP/1.1 200 OK
```

## Key Changes in Start Command

**Old (❌ Won't work):**
```bash
python -m uvicorn app:app
```
- Doesn't bind to 0.0.0.0
- Doesn't use PORT env var

**New (✅ Works):**
```bash
python app.py
```
- Uses `if __name__ == "__main__"` block
- Binds to `0.0.0.0:$PORT`
- Shows startup message

## Estimated Deploy Time

- Build: ~2-3 minutes
- Model download (first time): ~2-3 minutes
- Total first deploy: ~5 minutes
- Subsequent deploys (cached): ~1-2 minutes

## Success Indicators

✅ Build logs show: "✅ Loaded disease" and "✅ Loaded xray"
✅ Render dashboard shows service as "Live"
✅ `/health` endpoint returns 200 with model info
✅ `/` endpoint returns 200 (not 404 or 405)
✅ No "invalid load key" errors in logs

## If Still Failing

1. **Check Google Drive sharing:**
   - Each file must be "Anyone with the link" → Viewer
   - Test download in incognito browser window
   
2. **Check downloaded file sizes:**
   - If ~100KB: likely HTML error page
   - Should be: derm_best.pt (~50-200MB), xray_best.pt (~50-200MB)

3. **Manual intervention:**
   - Download files locally
   - Re-upload to better hosting (Hugging Face Hub recommended)
   - Or commit directly to Git if < 100MB each

