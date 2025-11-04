# Deployment Options Summary

## 🚨 Problem: Render OOM

Render free tier has only 512MB RAM. Your app needs ~3-4GB for full features.

## ✅ Solutions

### 1. Render (Lite Mode) - **FASTEST FIX**
**Time:** 5 minutes  
**Cost:** Free  
**RAM:** 512MB  

```bash
# In Render dashboard
Build Command: pip install -r requirements-lite.txt
Start Command: python app.py

# Environment Variables:
ENABLE_LLM=0
ENABLE_CLIP=0
ENABLE_GRADCAM=0
DISEASE_IMG_SIZE=224
XR_IMG_SIZE=224
```

**Features:** Vision classification only (no AI reports, no heatmaps)

---

### 2. Hugging Face Spaces - **BEST FREE OPTION** ⭐⭐⭐⭐⭐
**Time:** 10 minutes  
**Cost:** FREE  
**RAM:** 16GB + optional GPU  

```bash
# 1. Create Space at huggingface.co/spaces
# 2. Type: Docker

# 3. Add Dockerfile (use Dockerfile.hf)
# 4. Push your repo:
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/diseasellm
git push hf main

# 5. Done! Full features work
```

**URL:** `https://YOUR_USERNAME-diseasellm.hf.space`

**Features:** ALL features enabled (LLM, CLIP, GradCAM, etc.)

---

### 3. Railway - **PAID BUT SIMPLE**
**Time:** 5 minutes  
**Cost:** ~$5-10/month  
**RAM:** 8GB  

```bash
# 1. Go to railway.app
# 2. Connect GitHub repo
# 3. Deploy (auto-detected)
# 4. Set env vars in dashboard:
ENABLE_LLM=1
ENABLE_CLIP=1
ENABLE_GRADCAM=1
```

**Features:** All features

---

### 4. Fly.io
**Time:** 10 minutes  
**Cost:** ~$2-5/month  
**RAM:** 1GB  

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Deploy
fly launch --no-deploy

# Edit fly.toml to set memory = 1024MB
# Then:
fly deploy
```

**Features:** All features (tight fit)

---

### 5. Google Cloud Run
**Time:** 15 minutes  
**Cost:** Pay per request (generous free tier)  
**RAM:** Up to 4GB  

```bash
gcloud run deploy diseasellm \
  --source . \
  --memory 4Gi \
  --set-env-vars ENABLE_LLM=1,ENABLE_CLIP=1,ENABLE_GRADCAM=1
```

**Features:** All features, auto-scales to zero (cold starts)

---

## Comparison Table

| Platform | Setup | Monthly Cost | RAM | GPU | Features | Cold Start | Best For |
|----------|-------|--------------|-----|-----|----------|------------|----------|
| **Render (lite)** | ⭐⭐⭐⭐⭐ | Free | 512MB | ❌ | Vision only | No | Quick test |
| **HF Spaces** | ⭐⭐⭐⭐⭐ | Free | 16GB | ✅ | **All** | No | **Production** |
| **Railway** | ⭐⭐⭐⭐⭐ | $5-10 | 8GB | ❌ | All | No | Paid simple |
| **Fly.io** | ⭐⭐⭐⭐ | $2-5 | 1GB | ❌ | All | No | Budget |
| **GCP Run** | ⭐⭐⭐ | ~$0-5 | 4GB | ❌ | All | Yes | Serverless |

---

## Recommendation by Use Case

### Just Testing / Demo
→ **Render (lite mode)** - Get something working in 5 minutes

### Production / Serious Project  
→ **Hugging Face Spaces** - Free, 16GB RAM, built for ML

### Need Custom Domain / Enterprise
→ **Railway** or **GCP Run**

### Budget Conscious
→ **Hugging Face Spaces** (free) or **Fly.io** ($2/mo)

---

## Files You Need

### For Render (Lite):
- ✅ `requirements-lite.txt` (created)
- ✅ Set env vars in dashboard

### For Hugging Face Spaces:
- ✅ `Dockerfile.hf` (created, rename to `Dockerfile`)
- ✅ `requirements.txt` (existing)
- ✅ Push to HF repo

### For Railway:
- ✅ `requirements.txt` (existing)
- ✅ Set env vars in dashboard
- ✅ That's it!

---

## Quick Start Commands

### Test Lite Mode Locally:
```bash
pip install -r requirements-lite.txt
export ENABLE_LLM=0 ENABLE_CLIP=0 ENABLE_GRADCAM=0
python app.py
# Check RAM usage: should be ~300-400MB
```

### Test Full Mode Locally:
```bash
pip install -r requirements.txt
export ENABLE_LLM=1 ENABLE_CLIP=1 ENABLE_GRADCAM=1
python app.py
# Will use ~3-4GB RAM
```

---

## Step-by-Step: Hugging Face Spaces (Recommended)

1. **Create Account:** Go to [huggingface.co](https://huggingface.co) and sign up

2. **Create Space:**
   - Click "Spaces" → "Create new Space"
   - Name: `diseasellm`
   - License: MIT
   - SDK: **Docker** (important!)
   - Hardware: CPU (free) or GPU (also free!)
   - Click "Create Space"

3. **Prepare Files:**
   ```bash
   # Rename Dockerfile for HF
   cp Dockerfile.hf Dockerfile
   
   # Commit
   git add Dockerfile .dockerignore
   git commit -m "Add HF Spaces Docker config"
   ```

4. **Push to HF:**
   ```bash
   # Add HF remote (replace YOUR_USERNAME)
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/diseasellm
   
   # Push
   git push hf main
   ```

5. **Wait for Build:** (~5-10 minutes first time)

6. **Test:** Visit `https://huggingface.co/spaces/YOUR_USERNAME/diseasellm`

7. **Update Frontend:**
   ```bash
   # In frontend/.env.local
   NEXT_PUBLIC_BACKEND_URL=https://YOUR_USERNAME-diseasellm.hf.space
   ```

8. **Done!** 🎉

---

## Troubleshooting

### Render: Still OOM in lite mode
- Reduce image size further: `DISEASE_IMG_SIZE=192`
- Ensure transformers not installed: check `requirements-lite.txt`

### HF Spaces: Build fails
- Check Dockerfile syntax
- Ensure Dockerfile is in repo root
- Check Space logs for error details

### Railway: High costs
- Enable "Sleep on Idle" in settings
- Set memory limit to 2GB (enough for full features)

### All: Models not loading
- Check Google Drive sharing permissions
- Run `python gdrive_config.py` locally to test downloads
- Ensure `outputs/` directory exists

---

## Need More Help?

- **QUICK_FIX.md** - Immediate OOM solutions
- **LOW_MEMORY_DEPLOY.md** - Detailed low-memory guide
- **RENDER_DEPLOY.md** - Render-specific instructions
- **GDRIVE_SETUP.md** - Fix model download issues

---

## TL;DR

1. **Right now:** Render lite mode (5 min) → something works
2. **This weekend:** HF Spaces (10 min) → full features, free forever

**Best choice:** Hugging Face Spaces (free 16GB + GPU, built for ML)

