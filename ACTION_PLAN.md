# 🚀 Action Plan to Fix Render Deployment

## What Was Wrong

1. **HEAD / returned 405** → Render health check failed
2. **"invalid load key, '<'"** → Downloaded HTML instead of .pt files
3. **Port not detected** → Server wasn't binding correctly

## What Was Fixed

✅ All issues fixed in code
✅ Added `gdown` for reliable Google Drive downloads
✅ Better error messages and validation
✅ Proper port binding for Render

## What YOU Need to Do

### Step 1: Fix Google Drive Permissions (CRITICAL)

Your model files are currently returning HTML error pages instead of the actual files.

**For EACH file in Google Drive:**

1. Go to Google Drive
2. Find these files:
   - `best.pt` (or `derm_best.pt`)
   - `xray_best.pt`
   - `classes.json` (or `derm_classes.json`)
   - `xray_classes.json`
   - `calibration.json`

3. **For each file:**
   - Right-click → **Share**
   - Under "General access" click **Change**
   - Select **"Anyone with the link"**
   - Set permission to **"Viewer"**
   - Click **Done**

### Step 2: Test Locally (Optional but Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Test Google Drive downloads
python test_gdrive.py

# If all pass, test the server
python app.py
```

**Expected output:**
```
✅ All files downloaded successfully!
✅ Loaded disease: XX classes on cpu
✅ Loaded xray: XX classes on cpu
Starting server on 0.0.0.0:8000
```

### Step 3: Deploy to Render

#### In Render Dashboard:

1. **Start Command:** (change to this)
   ```
   python app.py
   ```

2. **Environment Variables:** (optional, auto-detected if not set)
   ```
   FORCE_CPU=1
   ```

3. **Save and Deploy**

### Step 4: Verify Deployment

Watch the Render logs for:

```
✅ Loaded disease: XX classes on cpu
✅ Loaded xray: XX classes on cpu
Starting server on 0.0.0.0:10000
INFO: Uvicorn running on http://0.0.0.0:10000
==> Your service is live at https://...
```

Test the endpoints:
```bash
# Health check
curl https://your-app.onrender.com/health

# Should return JSON with "vision_models_loaded": ["disease", "xray"]
```

## If It Still Fails

### A) Files still downloading as HTML (100KB size)

**Cause:** Google Drive permissions not set correctly

**Fix:**
1. Try downloading in incognito browser: `https://drive.google.com/uc?export=download&id=YOUR_FILE_ID`
2. If you see login page → permissions are wrong
3. Re-check sharing settings for each file

### B) Files too large / quota exceeded

**Alternative:** Use Hugging Face Hub (better for ML models)

```bash
# Install huggingface_hub
pip install huggingface_hub

# Upload models
huggingface-cli upload your-username/diseasellm outputs/ --repo-type=model

# Update app.py to download from HF instead
```

### C) Google Drive completely blocked

**Last resort:** Commit small models to Git

```bash
# If models are < 100MB each
git lfs track "*.pt"
git add outputs/*.pt outputs/*.json
git commit -m "Add model files"
git push
```

## Expected Timeline

- ✅ Code fixes: **Done** (already committed)
- 🔧 Fix Google Drive: **5 minutes** (you need to do this)
- 🚀 Redeploy Render: **2 minutes** (save + deploy)
- ⏳ First deploy: **5-7 minutes** (download models)
- ✅ Service live: **Ready!**

## Key Files to Review

- **RENDER_DEPLOY.md** → Full deployment guide
- **GDRIVE_SETUP.md** → Google Drive configuration details
- **FIX_SUMMARY.md** → Technical details of all fixes
- **test_gdrive.py** → Test script to verify downloads

## Quick Commands

```bash
# Test downloads
python test_gdrive.py

# Start server locally
python app.py

# Deploy to Render (in dashboard)
Start Command: python app.py
```

## Success Checklist

- [ ] Google Drive files have "Anyone with the link" sharing
- [ ] `test_gdrive.py` shows all ✅ (optional)
- [ ] Render start command is `python app.py`
- [ ] Render logs show "✅ Loaded disease" and "✅ Loaded xray"
- [ ] `/health` endpoint returns 200 with model info
- [ ] Frontend can connect to backend

---

**TLDR: Fix Google Drive sharing permissions (Step 1), then redeploy with start command `python app.py`**

