# HealthLens: Integrated Medical AI Diagnostic Platform

HealthLens is a sophisticated medical imaging analysis platform designed to assist in the detection and classification of various health conditions. By leveraging advanced deep learning models, the platform provides insights into dermatological diseases, pulmonary findings via radiographs, and histological cancer types.

## Core Capabilities

### Dermatological Diagnostics
The platform analyzes clinical skin photographs to identify a wide spectrum of conditions, including acne, malignant lesions, inflammatory diseases, and viral infections. It utilizes high-performance backbones like EfficientNetV2 and ResNet50 to ensure diagnostic accuracy.

### Pulmonary Analysis
HealthLens features a dedicated chest X-ray analysis module capable of detecting common findings such as:
- Atelectasis and Consolidations
- Cardiomegaly and Edema
- Pleural Effusions and Pneumothorax
- Pulmonary Nodules and Masses

### Oncological Histology
Medical professionals can analyze histopathology slides for various cancer types, including:
- Acute Lymphoblastic Leukemia (ALL)
- Brain and Breast Cancer
- Cervical and Kidney Cancer
- Lung and Colon Cancer

## Technical Highlights

- **Test Time Augmentation (TTA)**: Implements multi-view inference by averaging predictions across various image transformations, significantly improving model robustness and reliability.
- **Probability Calibration**: Uses temperature scaling to refine model output probabilities, ensuring that confidence scores accurately reflect diagnostic certainty.
- **Out-of-Distribution (OOD) Detection**: Integrates OpenAI's CLIP model to verify image relevance, warning users if a submitted image does not match the expected medical context (e.g., skin, X-ray, or microscopy).
- **Automated Medical Summaries**: Optionally integrates with Large Language Models (such as Phi-3) to generate human-readable summaries and clinical impressions based on model findings.

## Technology Stack

- **Machine Learning**: PyTorch, TIMM (PyTorch Image Models), Hugging Face Transformers.
- **Backend**: FastAPI (Python) for high-performance API serving.
- **Frontend**: Next.js 14 and Tailwind CSS for a modern, responsive user experience.
- **Data & Cloud**: Appwrite for database and backend-as-a-service, Google Drive API for secure model weight management.
- **Deployment**: containerized with Docker, optimized for Vercel, Render, and Hugging Face Spaces.

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher
- PyTorch and compatible CUDA drivers (optional for GPU acceleration)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Musa-Tur-Farazi/HealthLens.git
   cd HealthLens
   ```

2. **Backend Setup:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Frontend Setup:**
   ```bash
   cd frontend
   npm install
   ```

4. **Model Initialization:**
   Setup and diagnostic scripts are located in the `scripts/` directory to assist with Google Drive integration and model validation.

## Project Structure

For improved maintainability, the project has been reorganized:
- `src/`: Core application logic (Backend).
- `frontend/`: Next.js application.
- `docs/`: Comprehensive documentation, deployment guides, and deployment checklists.
- `scripts/`: Diagnostic and initialization scripts.
- `outputs/`: Local storage for model checkpoints and classification metadata.

## Disclaimer
HealthLens is a research and educational demonstration. It is not intended for clinical use or to provide definitive medical advice. All diagnostic results must be reviewed by qualified medical professionals.

---

Developed by [Musa-Tur-Farazi](https://github.com/Musa-Tur-Farazi)
