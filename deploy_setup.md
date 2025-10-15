# DiseaseLLM Deployment Guide

This guide explains how to deploy your DiseaseLLM application to Vercel with Google Drive for model storage.

## Prerequisites

1. **Google Account**: With Google Drive access
2. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
3. **GitHub Account**: For code repository

## Step 1: Setup Google Drive (No API Key Required!)

### 1.1 Upload Model Files to Google Drive
1. Go to [Google Drive](https://drive.google.com)
2. Create a new folder named `DiseaseLLM-Models` (optional)
3. Upload these files to your Google Drive:

```
DiseaseLLM-Models/
├── best.pt (disease model ~80MB)
├── xray_best.pt (xray model ~80MB)
├── classes.json
├── xray_classes.json
└── calibration.json
```

### 1.2 Get Direct Download Links
1. Right-click each file → **Get link** → **Anyone with the link can view**
2. Copy the share URL for each file
3. Use the helper script to convert to direct download URLs:

```bash
python get_gdrive_links.py
```

### 1.3 Update Configuration
Update the `MODEL_FILES` mapping in `gdrive_config.py` with the direct download URLs:

```python
MODEL_FILES = {
    "best.pt": "https://drive.google.com/uc?export=download&id=YOUR_DISEASE_MODEL_FILE_ID",
    "xray_best.pt": "https://drive.google.com/uc?export=download&id=YOUR_XRAY_MODEL_FILE_ID", 
    "classes.json": "https://drive.google.com/uc?export=download&id=YOUR_CLASSES_FILE_ID",
    "xray_classes.json": "https://drive.google.com/uc?export=download&id=YOUR_XRAY_CLASSES_FILE_ID",
    "calibration.json": "https://drive.google.com/uc?export=download&id=YOUR_CALIBRATION_FILE_ID"
}
```

### 1.4 Manual URL Conversion
If you prefer to do it manually:
1. Take your share URL: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
2. Convert to direct download URL: `https://drive.google.com/uc?export=download&id=FILE_ID`

## Step 2: Setup Environment Variables

### 2.1 Local Development
Create a `.env` file in your project root:

```bash
# Backend Configuration
PORT=8000
FORCE_CPU=0

# Model Configuration
DISEASE_MODEL=tf_efficientnetv2_s
DISEASE_IMG_SIZE=384
XR_MODEL=tf_efficientnetv2_s
XR_IMG_SIZE=384

# Frontend Configuration
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

### 2.2 Vercel Environment Variables
In your Vercel dashboard, add these environment variables:

```
FORCE_CPU=1
NEXT_PUBLIC_BACKEND_URL=https://your-backend-url.vercel.app
```

**Note**: No Google API key needed! The direct download URLs are configured in `gdrive_config.py`.

## Step 3: Deploy to Vercel

### 3.1 Push to GitHub
```bash
git add .
git commit -m "Add Appwrite integration and Vercel config"
git push origin main
```

### 3.2 Connect to Vercel
1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repository
4. Configure:
   - **Framework Preset**: Other
   - **Root Directory**: Leave empty
   - **Build Command**: Leave empty
   - **Output Directory**: Leave empty

### 3.3 Deploy
1. Click "Deploy"
2. Wait for deployment to complete
3. Your app will be available at the provided URL

## Step 4: Test Deployment

### 4.1 Check Health Endpoint
Visit: `https://your-app-url.vercel.app/health`

You should see:
```json
{
  "status": "ok",
  "device": "cpu",
  "vision_models_loaded": ["disease", "xray"],
  "llm_ready": true,
  "gdrive_available": true,
  "model_files_status": {
    "best.pt": {"exists": true, "size": 81714656, "size_mb": 77.9},
    "xray_best.pt": {"exists": true, "size": 81692748, "size_mb": 77.9},
    "classes.json": {"exists": true, "size": 1272, "size_mb": 0.0},
    "xray_classes.json": {"exists": true, "size": 235, "size_mb": 0.0},
    "calibration.json": {"exists": true, "size": 59, "size_mb": 0.0}
  }
}
```

### 4.2 Test Frontend
Visit your app URL and test uploading an image for classification.

## Troubleshooting

### Models Not Loading
1. Check direct download URLs in `gdrive_config.py`
2. Ensure files are shared with "Anyone with the link can view"
3. Test direct URLs in browser (should start downloading)
4. Look at Vercel function logs for errors
5. Check if files are larger than 25MB (may need special handling)

### Cold Start Issues
- Vercel functions have a 60-second timeout
- First request may take longer due to model downloading
- Consider using Vercel Pro for longer timeouts

### Memory Issues
- Models are large (~80MB each)
- Vercel has memory limits
- Consider using smaller models or optimizing

## Alternative: Manual Model Upload

If Appwrite setup is complex, you can:

1. Remove `outputs/` from `.gitignore`
2. Upload models directly to your repository
3. Note: This will make your repository large and may hit GitHub limits

## Cost Considerations

- **Google Drive**: Free tier includes 15GB storage
- **Vercel**: Free tier includes 100GB bandwidth
- **Models**: ~160MB total, well within limits
- **Total Cost**: $0 (completely free!)

## Security Notes

- Files are publicly accessible (shared with "Anyone with link")
- No API keys or credentials needed
- Don't commit `.env` files to Git
- Model files are not sensitive (they're just ML models)
