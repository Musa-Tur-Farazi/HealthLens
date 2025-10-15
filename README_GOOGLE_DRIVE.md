# 🚀 DiseaseLLM with Google Drive (No API Key Required!)

This is the **simplest possible** way to deploy your DiseaseLLM app with model storage. No API keys, no paid services, no complex setup!

## ✨ What's New

- **🔓 No API Key Required** - Just use direct Google Drive links
- **💰 Completely Free** - No paid services needed
- **⚡ Super Simple** - Upload files, share them, done!
- **🌐 Works Everywhere** - Vercel, any hosting platform

## 🚀 Quick Setup (5 minutes)

### Step 1: Upload Models to Google Drive
1. Go to [Google Drive](https://drive.google.com)
2. Upload these files:
   - `best.pt` (disease model)
   - `xray_best.pt` (xray model)  
   - `classes.json`
   - `xray_classes.json`
   - `calibration.json`

### Step 2: Get Share Links
1. Right-click each file → **Get link** → **Anyone with the link can view**
2. Copy the share URL for each file

### Step 3: Convert to Direct Download URLs
```bash
python get_gdrive_links.py
```

This script will:
- Ask for each file's share URL
- Convert them to direct download URLs
- Generate the configuration for you

### Step 4: Update Configuration
Copy the generated configuration into `gdrive_config.py`:

```python
MODEL_FILES = {
    "best.pt": "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID",
    "xray_best.pt": "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID",
    # ... etc
}
```

### Step 5: Test Locally
```bash
python setup_gdrive.py  # Check configuration
python app.py           # Start the app
```

### Step 6: Deploy to Vercel
```bash
git add .
git commit -m "Add Google Drive integration"
git push origin main
```

Connect to Vercel and deploy! That's it! 🎉

## 🔧 How It Works

1. **Direct Downloads**: Uses Google Drive's public file sharing
2. **No Authentication**: Files are publicly accessible via direct URLs
3. **Automatic Download**: Models download automatically when the app starts
4. **Vercel Compatible**: Works perfectly with Vercel's serverless functions

## 📋 File Structure

```
DiseaseLLM/
├── gdrive_config.py          # Google Drive configuration
├── get_gdrive_links.py       # Helper to convert share URLs
├── setup_gdrive.py           # Setup and testing script
├── app.py                    # Main application
├── frontend/                 # React frontend
└── outputs/                  # Local model cache (auto-created)
```

## 🎯 Benefits

- **✅ Zero Cost** - Completely free
- **✅ No API Keys** - No credentials to manage
- **✅ Simple Setup** - Just upload and share files
- **✅ Reliable** - Google's infrastructure
- **✅ Fast** - Direct downloads from Google's CDN
- **✅ Scalable** - No bandwidth limits for public files

## 🔍 Troubleshooting

### Models Not Downloading?
1. Check URLs in `gdrive_config.py`
2. Ensure files are shared with "Anyone with the link can view"
3. Test URLs in browser (should start downloading)
4. Check Vercel function logs

### Large Files (>25MB)?
Google Drive may show a virus scan warning for large files. This is normal and handled automatically by the download script.

### Still Having Issues?
1. Run `python setup_gdrive.py` to check configuration
2. Test URLs manually in browser
3. Check Vercel deployment logs

## 🚀 Ready to Deploy?

Your app is now ready for deployment! The models will download automatically when users first access your app. No complex setup, no API keys, no paid services - just simple, reliable Google Drive file sharing! 

**Total setup time: ~5 minutes** ⏱️
**Total cost: $0** 💰
**Total complexity: Minimal** 😊
