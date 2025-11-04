# 🚀 Low-Memory Deployment Guide (512MB)

## Problem: Render Free Tier OOM (Out of Memory)

Render's free tier provides only **512MB RAM**, but your app with full features needs:
- PyTorch models: ~150-300MB
- Transformers (LLM): ~2-4GB ❌ TOO MUCH
- CLIP: ~500MB ❌ TOO MUCH  
- Python + dependencies: ~100MB
- **Total: 3-5GB** → Doesn't fit!

## Solution: Lite Mode (Vision Only)

Run with **only core vision models**, disable heavy features:

### What Gets Disabled
- ❌ LLM (Phi-3) - No AI-generated reports
- ❌ CLIP - No image type auto-detection  
- ❌ GradCAM - No heatmap visualizations
- ✅ Vision models - **KEPT** (disease + xray classification)
- ✅ Symptom matching - **KEPT** (text-based, lightweight)

### Memory Usage (Lite Mode)
- PyTorch CPU models: ~200MB
- Core dependencies: ~100MB
- Python runtime: ~100MB
- **Total: ~400MB** ✅ Fits in 512MB!

---

## Quick Deploy to Render (Lite Mode)

### Option 1: Using render.yaml (Recommended)

1. **Commit the new files:**
   ```bash
   git add render.yaml requirements-lite.txt app.py
   git commit -m "Add low-memory deployment config"
   git push
   ```

2. **In Render Dashboard:**
   - Create New Web Service
   - Connect your repo
   - Render will auto-detect `render.yaml` ✅
   - Click "Create Web Service"

3. **Done!** Render will use lite settings automatically.

### Option 2: Manual Configuration

If not using `render.yaml`:

1. **Build Command:**
   ```bash
   pip install -r requirements-lite.txt
   ```

2. **Start Command:**
   ```bash
   python app.py
   ```

3. **Environment Variables:**
   ```bash
   ENABLE_LLM=0
   ENABLE_CLIP=0
   ENABLE_GRADCAM=0
   FORCE_CPU=1
   DISEASE_IMG_SIZE=224
   XR_IMG_SIZE=224
   ```

---

## What Changes in Lite Mode?

### API Response Changes

**Full Mode (with LLM):**
```json
{
  "topk": [...],
  "report": {
    "impression": "AI-generated clinical summary",
    "findings": ["Finding 1", "Finding 2"],
    "disease_summary": "Detailed explanation...",
    "red_flags": ["Warning 1", "Warning 2"],
    "next_steps": ["Step 1", "Step 2"],
    "disclaimer": "..."
  },
  "cam_b64": "data:image/png;base64,...",  // GradCAM heatmap
  "router": {
    "clip": {"clinical_skin": 0.8, ...}    // Image type detection
  }
}
```

**Lite Mode (no LLM/CLIP/GradCAM):**
```json
{
  "topk": [
    {"label": "Acne", "prob": 0.85},
    {"label": "Rosacea", "prob": 0.10}
  ],
  "report": {
    "impression": "Findings suggestive of Acne",
    "findings": ["Features align with Acne."],
    "disease_summary": "Inflammatory pilosebaceous disease...",  // From hints
    "red_flags": ["rapidly worsening cysts"],                   // From hints
    "next_steps": ["Monitor symptoms", "Seek clinician input"],
    "disclaimer": "Research demo; not medical advice."
  },
  "cam_b64": null,                    // No heatmap
  "router": {
    "clip": {},                       // No CLIP scores
    "stats": {...}
  }
}
```

### Frontend Impact

Your frontend will still work! It already handles:
- `cam_b64` can be null (no heatmap shown)
- `report` fields are always present (fallback to hints)
- CLIP scores can be empty object

**No frontend changes needed** ✅

---

## Alternative Hosting (For Full Features)

If you need LLM/CLIP/GradCAM, use platforms with more RAM:

### 1. Railway (Recommended)
- **Free tier:** 500 execution hours/month
- **RAM:** Up to 8GB on free tier
- **Pricing:** ~$5-10/month for always-on
- **Deploy:** 
  ```bash
  # Install Railway CLI
  npm install -g @railway/cli
  
  # Login and deploy
  railway login
  railway up
  ```
- **Config:** Use full `requirements.txt`, set `ENABLE_LLM=1`

### 2. Fly.io
- **Free tier:** 256MB RAM (still not enough)
- **Paid:** ~$2-5/month for 1GB RAM
- **Deploy:**
  ```bash
  # Install flyctl
  curl -L https://fly.io/install.sh | sh
  
  # Deploy
  fly launch
  fly deploy
  ```

### 3. Hugging Face Spaces (Best for ML)
- **Free tier:** 16GB RAM ✅ + free GPU!
- **Perfect for:** ML model hosting
- **Deploy:**
  1. Create Space at huggingface.co/spaces
  2. Select "Gradio" or "Docker"
  3. Push your code
  4. Set secrets: `ENABLE_LLM=1`, etc.

### 4. Google Cloud Run
- **Free tier:** 2 million requests/month
- **RAM:** Up to 4GB on free tier
- **Pricing:** Pay per request
- **Deploy:**
  ```bash
  gcloud run deploy diseasellm \
    --source . \
    --memory 4Gi \
    --set-env-vars ENABLE_LLM=1
  ```

### 5. AWS Lambda + API Gateway
- **Free tier:** 1 million requests/month
- **RAM:** Up to 10GB
- **Tradeoff:** Cold starts (slower)
- **Better for:** Infrequent usage

---

## Comparison Table

| Platform | Free RAM | Free GPU | Easy Deploy | Best For |
|----------|----------|----------|-------------|----------|
| **Render** | 512MB | ❌ | ⭐⭐⭐⭐⭐ | Lite mode only |
| **Railway** | 8GB | ❌ | ⭐⭐⭐⭐ | Full features |
| **Fly.io** | 256MB→1GB | ❌ | ⭐⭐⭐⭐ | Scaling apps |
| **HF Spaces** | 16GB | ✅ | ⭐⭐⭐⭐⭐ | **Best for ML** |
| **GCP Run** | 4GB | ❌ | ⭐⭐⭐ | Pay-per-use |
| **AWS Lambda** | 10GB | ❌ | ⭐⭐ | Serverless |

---

## Testing Lite Mode Locally

```bash
# Use lite requirements
pip install -r requirements-lite.txt

# Set environment variables
export ENABLE_LLM=0
export ENABLE_CLIP=0
export ENABLE_GRADCAM=0
export DISEASE_IMG_SIZE=224

# Run server
python app.py
```

**Expected output:**
```
ENABLE_LLM: 0 (LLM disabled)
ENABLE_CLIP: 0 (CLIP disabled)
ENABLE_GRADCAM: 0 (GradCAM disabled)
✅ Loaded disease: 23 classes on cpu (img_size=224)
✅ Loaded xray: 15 classes on cpu (img_size=224)
Starting server on 0.0.0.0:8000
```

Test endpoint:
```bash
# Health check should show disabled features
curl http://localhost:8000/health

{
  "status": "ok",
  "device": "cpu",
  "vision_models_loaded": ["disease", "xray"],
  "llm_ready": false,          // ← Disabled
  "clip_enabled": false,       // ← Disabled
  "gradcam_enabled": false     // ← Disabled
}
```

---

## Hybrid Approach (Separate Services)

Deploy vision and LLM separately:

### Backend 1: Render (Lite) - Vision models only
- Classification endpoint `/v1/diag`
- Returns predictions + basic hints
- Always fast, no OOM

### Backend 2: Hugging Face Spaces - LLM only  
- Report generation endpoint `/v1/report`
- Takes predictions, returns detailed report
- GPU-accelerated, free

### Frontend
- Calls Backend 1 first (fast classification)
- Calls Backend 2 async (detailed report loads later)
- Progressive enhancement UX

---

## Migration Path

### Now (Render Lite)
```
User → Frontend → Render (vision only) → Basic predictions
                                       → Hint-based reports
```

### Later (Full Features)
```
User → Frontend → HF Spaces (vision + LLM) → AI reports
                                            → GradCAM
                                            → CLIP routing
```

---

## FAQ

**Q: Will my frontend break?**  
A: No, it gracefully handles missing features (null CAM, empty CLIP).

**Q: Are predictions less accurate?**  
A: No! Vision model accuracy is identical. Only reporting is simpler.

**Q: Can I enable just LLM, not CLIP?**  
A: Yes, set `ENABLE_LLM=1 ENABLE_CLIP=0`. But ~2GB LLM still too big for 512MB.

**Q: What about quantized models?**  
A: Could reduce LLM to ~1GB with INT8, but still tight. Use HF Spaces instead.

**Q: Can I use Render's paid tier?**  
A: Yes! $7/month gets you 2GB RAM. Enough for vision + CLIP, but not full LLM.

---

## Recommended: Hugging Face Spaces (Free!)

**Best option for full features with no cost:**

1. Create account at huggingface.co
2. Create new Space: "Docker" type
3. Add `Dockerfile`:
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   CMD ["python", "app.py"]
   ```
4. Push your repo
5. Set environment in Space settings:
   ```
   ENABLE_LLM=1
   ENABLE_CLIP=1
   ENABLE_GRADCAM=1
   ```
6. **Free 16GB RAM + GPU!** ✅

---

## Summary

| Deployment | RAM | Cost | Features | Recommendation |
|------------|-----|------|----------|----------------|
| **Render (lite)** | 512MB | Free | Vision only | ⭐⭐⭐ Quick test |
| **Railway** | 8GB | $5-10/mo | Full | ⭐⭐⭐⭐ Production |
| **HF Spaces** | 16GB | Free | Full + GPU | ⭐⭐⭐⭐⭐ **BEST** |

**TL;DR:** Use Render lite mode for testing, migrate to Hugging Face Spaces for production.

