# 📚 Documentation Index

## Start Here

### 🚨 **SOLUTION_SUMMARY.md** ⭐ READ THIS FIRST
Complete fix for Render OOM with step-by-step deployment instructions.

### 🔥 **QUICK_FIX.md**
3 solutions to OOM in 5-10 minutes each.

---

## Deployment Guides

### By Platform

| File | Platform | Time | Difficulty |
|------|----------|------|------------|
| **README_DEPLOY.md** | All platforms | - | ⭐⭐⭐ |
| **LOW_MEMORY_DEPLOY.md** | Render (lite) | 5 min | ⭐⭐⭐⭐⭐ |
| **Dockerfile.hf** | HF Spaces | 10 min | ⭐⭐⭐⭐ |
| **render.yaml** | Render (auto) | 5 min | ⭐⭐⭐⭐⭐ |

### By Goal

**I need it working NOW:**
→ `QUICK_FIX.md` → Option A (Render Lite)

**I want all features for FREE:**
→ `QUICK_FIX.md` → Option B (HF Spaces)

**I want to understand the problem:**
→ `MEMORY_BUDGET.md`

**I need step-by-step commands:**
→ Run `bash DEPLOY_COMMANDS.sh`

---

## Technical Documentation

### **MEMORY_BUDGET.md**
- Memory usage breakdown
- Full vs Lite mode comparison
- Performance analysis
- Optimization techniques

### **FIX_SUMMARY.md** (Original Issue)
- Details of the HEAD / 405 error
- Google Drive "invalid load key '<'" fix
- Port binding issues
- All code changes explained

### **GDRIVE_SETUP.md** (Original Issue)
- How to fix Google Drive sharing
- Why downloads fail
- Alternative hosting for models

### **RENDER_DEPLOY.md** (Original Issue)
- Render-specific configuration
- Start commands
- Environment variables
- Troubleshooting

---

## Helper Scripts

### **DEPLOY_COMMANDS.sh** 🚀
Interactive deployment wizard. Just run:
```bash
bash DEPLOY_COMMANDS.sh
```

### **test_gdrive.py**
Test if Google Drive model downloads work:
```bash
python test_gdrive.py
```

### **diagnose.py**
Check environment, dependencies, and model files:
```bash
python diagnose.py
```

---

## Configuration Files

### For Render (Lite Mode)
- `requirements-lite.txt` - Minimal dependencies (~450MB)
- `render.yaml` - Auto-configuration

### For Hugging Face Spaces
- `Dockerfile.hf` - Full features (~3.5GB)
- `.dockerignore` - Build optimization

### For All Platforms
- `requirements.txt` - Full dependencies
- `app.py` - Core application (with memory toggles)
- `gdrive_config.py` - Model downloader

---

## Quick Reference

### Decision Tree

```
Do you have >512MB RAM?
├─ YES → Use requirements.txt (full features)
│        Platform: HF Spaces, Railway, GCP Run
│
└─ NO → Use requirements-lite.txt (lite mode)
         Platform: Render (free)
```

### Environment Variables

```bash
# Lite Mode (512MB)
ENABLE_LLM=0
ENABLE_CLIP=0
ENABLE_GRADCAM=0
DISEASE_IMG_SIZE=224
XR_IMG_SIZE=224

# Full Mode (3.5GB)
ENABLE_LLM=1
ENABLE_CLIP=1
ENABLE_GRADCAM=1
DISEASE_IMG_SIZE=384
XR_IMG_SIZE=384
```

### Build Commands

```bash
# Lite (Render)
pip install -r requirements-lite.txt

# Full (HF Spaces, Railway)
pip install -r requirements.txt
```

### Start Commands

```bash
# All platforms
python app.py
```

---

## Reading Order (Recommended)

### If You Just Want It Fixed:
1. `SOLUTION_SUMMARY.md` (2 min read)
2. Run `bash DEPLOY_COMMANDS.sh` (5-10 min)
3. Done!

### If You Want to Understand:
1. `SOLUTION_SUMMARY.md` - What happened
2. `MEMORY_BUDGET.md` - Why it happened
3. `LOW_MEMORY_DEPLOY.md` - How to fix it
4. `README_DEPLOY.md` - All deployment options

### If You're Still Having Issues:
1. `diagnose.py` - Check your setup
2. `test_gdrive.py` - Test downloads
3. `FIX_SUMMARY.md` - Original fixes
4. `GDRIVE_SETUP.md` - Fix model downloads
5. `RENDER_DEPLOY.md` - Render specifics

---

## File Categories

### 🚨 Critical (Read First)
- `SOLUTION_SUMMARY.md` ⭐⭐⭐⭐⭐
- `QUICK_FIX.md` ⭐⭐⭐⭐⭐

### 🚀 Deployment
- `README_DEPLOY.md` - All platforms
- `LOW_MEMORY_DEPLOY.md` - Render lite
- `DEPLOY_COMMANDS.sh` - Interactive wizard

### 🔧 Technical
- `MEMORY_BUDGET.md` - Memory analysis
- `FIX_SUMMARY.md` - Code changes
- `GDRIVE_SETUP.md` - Model downloads
- `RENDER_DEPLOY.md` - Render details

### 🧪 Testing
- `test_gdrive.py` - Test downloads
- `diagnose.py` - Debug environment

### ⚙️ Configuration
- `requirements-lite.txt` - Lite deps
- `requirements.txt` - Full deps
- `render.yaml` - Render config
- `Dockerfile.hf` - HF Spaces config
- `.dockerignore` - Docker optimization

---

## By Use Case

### "I just deployed and got OOM"
→ `SOLUTION_SUMMARY.md` + `QUICK_FIX.md`

### "I want to deploy from scratch"
→ `README_DEPLOY.md` + `DEPLOY_COMMANDS.sh`

### "Google Drive downloads are broken"
→ `GDRIVE_SETUP.md` + run `python test_gdrive.py`

### "I want to optimize memory"
→ `MEMORY_BUDGET.md` + `LOW_MEMORY_DEPLOY.md`

### "I need to understand what changed"
→ `FIX_SUMMARY.md` (git diff style)

### "I want to compare platforms"
→ `README_DEPLOY.md` (comparison tables)

### "Something's not working"
→ Run `python diagnose.py`

---

## Dependencies

### Lite Mode (requirements-lite.txt)
```
Core: FastAPI, Uvicorn, PyTorch CPU, TIMM
Size: ~300MB total
Boots: ~2-3 minutes (first time)
RAM: ~450MB
Features: Vision classification only
```

### Full Mode (requirements.txt)
```
All of Lite Mode +
Transformers (LLM + CLIP)
GradCAM
Sentencepiece
Size: ~2GB total
Boots: ~5-10 minutes (first time)
RAM: ~3.5GB
Features: Everything
```

---

## Platform Comparison

| Platform | RAM | Best Doc | Setup Script |
|----------|-----|----------|--------------|
| **Render** | 512MB | `LOW_MEMORY_DEPLOY.md` | `DEPLOY_COMMANDS.sh` option 1 |
| **HF Spaces** | 16GB | `README_DEPLOY.md` | `DEPLOY_COMMANDS.sh` option 2 |
| **Railway** | 8GB | `README_DEPLOY.md` | `DEPLOY_COMMANDS.sh` option 3 |
| **Fly.io** | 1GB | `README_DEPLOY.md` | Manual |
| **GCP Run** | 4GB | `README_DEPLOY.md` | Manual |

---

## FAQ Document Mapping

**Q: Render says "out of memory"**
→ `SOLUTION_SUMMARY.md`, `QUICK_FIX.md`

**Q: Models not downloading / "invalid load key"**
→ `GDRIVE_SETUP.md`, `FIX_SUMMARY.md`

**Q: HEAD / returns 405**
→ `FIX_SUMMARY.md` (already fixed in code)

**Q: Which platform should I use?**
→ `README_DEPLOY.md` (comparison table)

**Q: How much memory does each feature use?**
→ `MEMORY_BUDGET.md`

**Q: Can I run full features on free hosting?**
→ Yes! `README_DEPLOY.md` → Hugging Face Spaces

**Q: What's the fastest way to deploy?**
→ Run `bash DEPLOY_COMMANDS.sh` and choose option

---

## Git History

All these changes fix the issues reported in:
1. Initial deploy: HEAD / → 404
2. Model loading: "invalid load key, '<'"
3. Port binding: "No open ports detected"
4. Memory: "Ran out of memory (512MB)"

---

## Next Steps

1. Read `SOLUTION_SUMMARY.md` (2 min)
2. Choose deployment:
   - Quick → Render lite
   - Best → HF Spaces
3. Run `bash DEPLOY_COMMANDS.sh`
4. Deploy!
5. Test with your frontend

---

## Need More Help?

**Check Files:**
- `SOLUTION_SUMMARY.md` - Complete overview
- `QUICK_FIX.md` - 3 fast solutions
- `README_DEPLOY.md` - All deployment details

**Run Scripts:**
```bash
bash DEPLOY_COMMANDS.sh  # Guided deployment
python diagnose.py        # Check your setup
python test_gdrive.py     # Test model downloads
```

**Still Stuck?**
Check logs for specific error messages, then:
- OOM → `MEMORY_BUDGET.md`
- Download fail → `GDRIVE_SETUP.md`
- Port issue → `RENDER_DEPLOY.md`
- Other → `FIX_SUMMARY.md`

---

**Total Documentation:** 15 files covering all aspects of deployment and troubleshooting!

