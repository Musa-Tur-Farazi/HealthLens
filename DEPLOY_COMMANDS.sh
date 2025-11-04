#!/bin/bash
# Copy-paste deployment commands for each platform
# Choose ONE option and follow its commands

set -e  # Exit on error

echo "🚀 DiseaseLLM Deployment Helper"
echo ""
echo "Choose your deployment target:"
echo ""
echo "1. Render (Lite Mode) - Free, 512MB, Vision only"
echo "2. Hugging Face Spaces - Free, 16GB, All features (RECOMMENDED)"
echo "3. Railway - \$5-10/mo, 8GB, All features"
echo "4. Test Locally (Lite)"
echo "5. Test Locally (Full)"
echo ""
read -p "Enter option (1-5): " option

case $option in
  1)
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  RENDER DEPLOYMENT (Lite Mode)"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    echo "📋 COPY THESE SETTINGS TO RENDER DASHBOARD:"
    echo ""
    echo "Build Command:"
    echo "  pip install -r requirements-lite.txt"
    echo ""
    echo "Start Command:"
    echo "  python app.py"
    echo ""
    echo "Environment Variables:"
    echo "  ENABLE_LLM=0"
    echo "  ENABLE_CLIP=0"
    echo "  ENABLE_GRADCAM=0"
    echo "  FORCE_CPU=1"
    echo "  DISEASE_IMG_SIZE=224"
    echo "  XR_IMG_SIZE=224"
    echo ""
    echo "Then click 'Save' and wait for deployment."
    echo ""
    echo "✅ Features: Vision classification only"
    echo "❌ Missing: AI reports, heatmaps, CLIP"
    ;;

  2)
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  HUGGING FACE SPACES DEPLOYMENT (Full Features)"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    echo "Step 1: Create Space at huggingface.co"
    echo "  - Go to: https://huggingface.co/spaces"
    echo "  - Click 'Create new Space'"
    echo "  - Name: diseasellm"
    echo "  - SDK: Docker (IMPORTANT!)"
    echo "  - Hardware: CPU (free) or GPU (also free!)"
    echo "  - Click 'Create Space'"
    echo ""
    read -p "Press Enter when Space is created..."
    echo ""
    echo "Step 2: Prepare Dockerfile"
    cp Dockerfile.hf Dockerfile
    echo "✅ Copied Dockerfile.hf to Dockerfile"
    echo ""
    echo "Step 3: Get your Hugging Face username"
    read -p "Enter your HF username: " hf_user
    echo ""
    echo "Step 4: Push to Hugging Face"
    echo ""
    echo "Running commands..."
    git add Dockerfile .dockerignore
    git commit -m "Add HF Spaces configuration" || echo "(Already committed)"
    git remote add hf "https://huggingface.co/spaces/${hf_user}/diseasellm" 2>/dev/null || echo "(Remote already exists)"
    echo ""
    echo "Now pushing to HF Spaces..."
    echo "You may be asked for HF credentials."
    git push hf main
    echo ""
    echo "✅ Deployment initiated!"
    echo ""
    echo "Your app will be at:"
    echo "  https://huggingface.co/spaces/${hf_user}/diseasellm"
    echo ""
    echo "Update your frontend .env.local:"
    echo "  NEXT_PUBLIC_BACKEND_URL=https://${hf_user}-diseasellm.hf.space"
    ;;

  3)
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  RAILWAY DEPLOYMENT"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    echo "Option A: Web UI (Easier)"
    echo "  1. Go to: https://railway.app"
    echo "  2. Sign in with GitHub"
    echo "  3. Click 'New Project' → 'Deploy from GitHub repo'"
    echo "  4. Select this repo"
    echo "  5. Add these environment variables:"
    echo "     ENABLE_LLM=1"
    echo "     ENABLE_CLIP=1"
    echo "     ENABLE_GRADCAM=1"
    echo "     FORCE_CPU=1"
    echo "  6. Deploy!"
    echo ""
    echo "Option B: CLI (Faster)"
    echo ""
    read -p "Install Railway CLI? (y/n): " install_cli
    if [ "$install_cli" = "y" ]; then
      if command -v npm &> /dev/null; then
        npm install -g @railway/cli
        echo "✅ Railway CLI installed"
        echo ""
        railway login
        railway up
        echo ""
        echo "Set environment variables:"
        railway variables set ENABLE_LLM=1 ENABLE_CLIP=1 ENABLE_GRADCAM=1
        echo ""
        echo "✅ Deployed to Railway!"
        railway open
      else
        echo "❌ npm not found. Install Node.js first or use Web UI."
      fi
    fi
    ;;

  4)
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  LOCAL TEST (Lite Mode)"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    echo "Installing lite dependencies..."
    pip install -r requirements-lite.txt
    echo ""
    echo "Setting environment variables..."
    export ENABLE_LLM=0
    export ENABLE_CLIP=0
    export ENABLE_GRADCAM=0
    export DISEASE_IMG_SIZE=224
    export XR_IMG_SIZE=224
    export FORCE_CPU=1
    echo ""
    echo "Testing Google Drive downloads..."
    python test_gdrive.py || echo "(test_gdrive.py not found, skipping)"
    echo ""
    echo "Starting server..."
    echo "Expected memory usage: ~400-500MB"
    echo ""
    python app.py
    ;;

  5)
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  LOCAL TEST (Full Mode)"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    echo "⚠️  WARNING: This will download ~3GB of models"
    echo "   Make sure you have enough RAM (4GB+ recommended)"
    echo ""
    read -p "Continue? (y/n): " cont
    if [ "$cont" != "y" ]; then
      echo "Aborted."
      exit 0
    fi
    echo ""
    echo "Installing full dependencies..."
    pip install -r requirements.txt
    echo ""
    echo "Setting environment variables..."
    export ENABLE_LLM=1
    export ENABLE_CLIP=1
    export ENABLE_GRADCAM=1
    export FORCE_CPU=1
    echo ""
    echo "Testing Google Drive downloads..."
    python test_gdrive.py || echo "(test_gdrive.py not found, skipping)"
    echo ""
    echo "Starting server..."
    echo "Expected memory usage: ~3-4GB"
    echo ""
    python app.py
    ;;

  *)
    echo "Invalid option"
    exit 1
    ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Need help?"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "📚 Documentation:"
echo "  - QUICK_FIX.md          - Immediate OOM solutions"
echo "  - README_DEPLOY.md      - All deployment options"
echo "  - LOW_MEMORY_DEPLOY.md  - 512MB optimization guide"
echo "  - MEMORY_BUDGET.md      - Memory usage breakdown"
echo ""

