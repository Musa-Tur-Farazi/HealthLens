# 🚀 DiseaseLLM Deployment Checklist

## Pre-Deployment Setup

### ✅ Google Drive Setup
- [ ] Upload model files to Google Drive
- [ ] Share files with "Anyone with link can view"
- [ ] Get share URLs for each file
- [ ] Convert to direct download URLs using `get_gdrive_links.py`
- [ ] Update direct download URLs in `gdrive_config.py`

### ✅ Environment Variables
- [ ] Set up `.env` file locally (no Google credentials needed!)
- [ ] Configure Vercel environment variables
- [ ] Update `NEXT_PUBLIC_BACKEND_URL` for production

### ✅ Code Preparation
- [ ] Models excluded from Git (`.gitignore` updated)
- [ ] Google Drive integration added to `app.py`
- [ ] Vercel configuration files created
- [ ] All dependencies in `requirements.txt`

## Deployment Steps

### ✅ GitHub Repository
- [ ] Push code to GitHub (models will be excluded)
- [ ] Verify repository size is reasonable (< 100MB)

### ✅ Vercel Deployment
- [ ] Connect GitHub repository to Vercel
- [ ] Configure environment variables in Vercel dashboard
- [ ] Deploy and monitor build logs

### ✅ Testing
- [ ] Check `/health` endpoint returns model status
- [ ] Test image upload and classification
- [ ] Verify models download from Google Drive correctly
- [ ] Test both disease and X-ray modalities

## Post-Deployment Verification

### ✅ Health Check
```bash
curl https://your-app.vercel.app/health
```
Expected response includes:
- `"status": "ok"`
- `"vision_models_loaded": ["disease", "xray"]`
- `"gdrive_available": true`
- `"model_files_status"` with all files present

### ✅ Frontend Testing
- [ ] Upload a test image
- [ ] Verify classification works
- [ ] Check Grad-CAM visualization
- [ ] Test both light and dark themes

### ✅ Performance
- [ ] Monitor cold start times
- [ ] Check memory usage in Vercel dashboard
- [ ] Verify response times are acceptable

## Troubleshooting

### ❌ Models Not Loading
1. Check direct download URLs in `gdrive_config.py`
2. Ensure files are shared with "Anyone with link can view"
3. Test direct URLs in browser (should start downloading)
4. Check if files are larger than 25MB (may need special handling)
5. Review Vercel function logs

### ❌ Cold Start Timeout
1. Consider using Vercel Pro for longer timeouts
2. Optimize model loading code
3. Use smaller models if possible

### ❌ Memory Issues
1. Monitor Vercel function memory usage
2. Consider model quantization
3. Use CPU-only inference if needed

## Cost Monitoring

### 💰 Google Drive Usage
- [ ] Monitor storage usage (models ~160MB)
- [ ] Verify within free tier limits (15GB free)

### 💰 Vercel Usage
- [ ] Monitor function invocations
- [ ] Check bandwidth usage
- [ ] Verify within free tier limits

## Security Checklist

### 🔒 Environment Variables
- [ ] No secrets committed to Git
- [ ] Production environment variables set
- [ ] API keys have minimal required permissions

### 🔒 Google Drive Security
- [ ] Files shared with "Anyone with link can view"
- [ ] No sensitive data in public files
- [ ] Model files are not sensitive (just ML models)

## Success Criteria

Your deployment is successful when:
- ✅ Health endpoint shows all models loaded
- ✅ Image classification works for both modalities
- ✅ Frontend loads and functions correctly
- ✅ No errors in Vercel function logs
- ✅ Response times are acceptable (< 30 seconds for first request)

## Next Steps After Deployment

1. **Monitor**: Set up monitoring for uptime and performance
2. **Optimize**: Consider model optimization for faster inference
3. **Scale**: Monitor usage and consider scaling if needed
4. **Backup**: Ensure model files are backed up in Google Drive
5. **Documentation**: Update any user-facing documentation with new URLs

---

**🎉 Congratulations!** Your DiseaseLLM application is now deployed and ready to serve users!
