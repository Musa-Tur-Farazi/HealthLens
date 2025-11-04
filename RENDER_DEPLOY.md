# Render Deployment Guide

## Quick Fix Summary

### Issues Fixed:
1. ✅ **HEAD / returns 405** → Added `@app.head("/")` decorator
2. ✅ **"invalid load key, '<'"** → Files downloading as HTML instead of binary
3. ✅ **Port binding** → Server binds to `0.0.0.0:$PORT`

## Render Configuration

### 1. Build Command
```bash
pip install -r requirements.txt
```

### 2. Start Command
**Use this:**
```bash
python app.py
```

**NOT this** (old, won't bind correctly):
```bash
python -m uvicorn app:app
```

### 3. Environment Variables
**Not required** if using `gdrive_config.py` auto-download.

Optional overrides:
```bash
FORCE_CPU=1                    # Use CPU (free tier has no GPU)
PORT=10000                     # Auto-set by Render
DISEASE_CKPT=outputs/derm_best.pt
DISEASE_CLASSES=outputs/derm_classes.json
XR_CKPT=outputs/xray_best.pt
XR_CLASSES=outputs/xray_classes.json
```

### 4. Health Check Endpoints
- **Root:** `GET/HEAD /` → Returns 200 OK
- **Health:** `GET /health` → Returns model status

## Fixing Google Drive Downloads

### Current Issue
Your files are downloading as HTML because Google Drive URLs return preview pages.

### Solution 1: Use `gdown` (Recommended)
✅ Already added to `requirements.txt`
✅ `gdrive_config.py` now uses `gdown` automatically

**Just ensure Google Drive sharing is correct:**
1. Right-click each file → Share
2. "Anyone with the link" → Viewer
3. Done

Test locally:
```bash
pip install gdown
python gdrive_config.py
```

### Solution 2: Manual File Upload
If Google Drive continues to fail:

1. Download model files locally
2. Upload to Render via:
   - Git (if < 100MB per file)
   - Build script that downloads from another host
   - Use Render Disk (persistent storage)

## Expected Deployment Flow

```
==> Build successful 🎉
==> Deploying...
==> Running 'python app.py'

Starting server on 0.0.0.0:10000
Checking model files...
Missing model files: ['best.pt', 'xray_best.pt']
Attempting to download from Google Drive...

Using gdown for reliable Google Drive download...
✅ Downloaded outputs/best.pt (XX MB)
✅ Downloaded outputs/xray_best.pt (XX MB)
✅ All model files downloaded successfully!
✅ Loaded disease: 23 classes on cpu
✅ Loaded xray: 15 classes on cpu

INFO: Uvicorn running on http://0.0.0.0:10000
==> Your service is live at https://your-app.onrender.com
```

## Troubleshooting

### "invalid load key, '<'"
**Cause:** Downloaded file is HTML, not binary .pt file

**Fix:**
1. Check Google Drive sharing permissions (see above)
2. Run `python gdrive_config.py` locally to test
3. Look for "❌ Error: Downloaded file is HTML" message
4. If using `gdown`, it will show the actual error

### "No open ports detected"
**Cause:** Server not binding to `0.0.0.0` or not using `$PORT`

**Fix:**
✅ Already fixed in `app.py`:
```python
port = int(os.environ.get("PORT", 8000))
uvicorn.run("app:app", host="0.0.0.0", port=port)
```

### Files downloading as 100-108KB (too small)
**Cause:** Downloading Google Drive error page HTML

**Fix:** Use `gdown` (already added) or fix sharing permissions

## Testing Locally First

```bash
# Install dependencies
pip install -r requirements.txt

# Test Google Drive downloads
python gdrive_config.py

# Start server
python app.py

# Should see:
# ✅ Loaded disease: X classes
# ✅ Loaded xray: X classes
# Starting server on 0.0.0.0:8000
```

## Next Steps After Deploy

1. Check Render logs for "✅ Loaded disease/xray" messages
2. Test health endpoint: `https://your-app.onrender.com/health`
3. Should return:
```json
{
  "status": "ok",
  "device": "cpu",
  "vision_models_loaded": ["disease", "xray"],
  "llm_ready": false,
  "model_files_status": {...}
}
```

## Frontend Configuration

Update `frontend/.env.local`:
```bash
NEXT_PUBLIC_BACKEND_URL=https://your-app.onrender.com
```

Redeploy frontend (Vercel/Netlify/etc.)

