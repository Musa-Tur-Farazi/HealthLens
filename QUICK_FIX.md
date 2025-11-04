# 🚨 QUICK FIX: Render OOM (Out of Memory)

## The Problem
```
Ran out of memory (used over 512MB)
```

## The Solution (Choose ONE)

### ⚡ Option A: Lite Mode on Render (5 minutes)

**Use this if:** You want to keep using Render free tier

**Steps:**
1. Update build command to:
   ```bash
   pip install -r requirements-lite.txt
   ```

2. Add environment variables:
   ```bash
   ENABLE_LLM=0
   ENABLE_CLIP=0
   ENABLE_GRADCAM=0
   DISEASE_IMG_SIZE=224
   XR_IMG_SIZE=224
   ```

3. Redeploy

**What you lose:**
- ❌ AI-generated reports (simple reports from hints instead)
- ❌ GradCAM heatmaps
- ❌ Image type auto-detection

**What you keep:**
- ✅ Disease classification (core feature)
- ✅ X-ray classification (core feature)
- ✅ Symptom matching
- ✅ Top-3 predictions with probabilities

---

### 🚀 Option B: Hugging Face Spaces (10 minutes) **RECOMMENDED**

**Use this if:** You want ALL features for FREE

**Why:** Free 16GB RAM + optional GPU!

**Steps:**

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Name: `diseasellm`
4. Type: **Docker**
5. Click "Create Space"

6. Create `Dockerfile` in your repo:
   ```dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   
   # Copy files
   COPY requirements.txt .
   COPY app.py .
   COPY gdrive_config.py .
   COPY outputs/ outputs/
   
   # Install dependencies
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Expose port (HF uses 7860 by default)
   ENV PORT=7860
   EXPOSE 7860
   
   # Enable all features
   ENV ENABLE_LLM=1
   ENV ENABLE_CLIP=1
   ENV ENABLE_GRADCAM=1
   ENV FORCE_CPU=0
   
   # Start app
   CMD ["python", "app.py"]
   ```

7. Push to Hugging Face:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/diseasellm
   git push hf main
   ```

8. **Done!** Your app will have:
   - ✅ 16GB RAM (no OOM!)
   - ✅ Optional free GPU
   - ✅ All features enabled
   - ✅ Public URL: `https://YOUR_USERNAME-diseasellm.hf.space`

**Update frontend `.env.local`:**
```bash
NEXT_PUBLIC_BACKEND_URL=https://YOUR_USERNAME-diseasellm.hf.space
```

---

### 💰 Option C: Railway (5 minutes)

**Use this if:** You're OK paying ~$5/month

**Why:** 8GB RAM, easy setup, always-on

**Steps:**

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repo
5. Railway auto-detects Python

6. Add environment variables in dashboard:
   ```bash
   ENABLE_LLM=1
   ENABLE_CLIP=1
   ENABLE_GRADCAM=1
   ```

7. **Done!** Railway gives you a public URL

**Cost:** ~$5-10/month (500 hours free, then pay)

---

## Comparison

| Option | Time | Cost | RAM | Features | URL |
|--------|------|------|-----|----------|-----|
| **A: Render Lite** | 5 min | Free | 512MB | Vision only | ⭐⭐⭐ |
| **B: HF Spaces** | 10 min | Free | 16GB | **ALL + GPU** | ⭐⭐⭐⭐⭐ |
| **C: Railway** | 5 min | $5/mo | 8GB | All | ⭐⭐⭐⭐ |

---

## My Recommendation

1. **Right now:** Deploy Option A (Render Lite) to get something working
2. **This weekend:** Migrate to Option B (HF Spaces) for full features

**Why?** Hugging Face Spaces is:
- Free forever
- Built for ML apps
- 16GB RAM (32x more than Render)
- Optional GPU acceleration
- No cold starts
- Better ML community support

---

## Testing Each Option Locally

### Test Lite Mode:
```bash
pip install -r requirements-lite.txt
export ENABLE_LLM=0 ENABLE_CLIP=0 ENABLE_GRADCAM=0
python app.py
# Should use ~300-400MB RAM
```

### Test Full Mode:
```bash
pip install -r requirements.txt
export ENABLE_LLM=1 ENABLE_CLIP=1 ENABLE_GRADCAM=1
python app.py
# Will use ~3-4GB RAM (needs better hosting)
```

---

## Need Help?

Read these guides:
- **LOW_MEMORY_DEPLOY.md** - Complete low-memory guide
- **RENDER_DEPLOY.md** - Render-specific instructions
- **ACTION_PLAN.md** - General deployment steps

---

## TL;DR

**Immediate fix:** Change Render build command to `pip install -r requirements-lite.txt` and add env vars `ENABLE_LLM=0 ENABLE_CLIP=0 ENABLE_GRADCAM=0`

**Best solution:** Deploy to Hugging Face Spaces (free 16GB RAM, all features work)

