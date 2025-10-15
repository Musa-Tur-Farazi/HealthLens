# app.py — HealthLens API (Unified Disease model + Chest X-ray)
# -------------------------------------------------------------
# Key fixes:
# - Auto-detect ResNet50 vs timm from checkpoint keys (no env knob needed)
# - Build transforms from checkpoint "meta" (img_size/mean/std) when present
# - Clear load diagnostics (missing/unexpected keys)
# - CLIP adds CT/MRI bucket to warn about OOD in "disease" mode
# -------------------------------------------------------------

import os, io, base64, json
from typing import List, Optional, Literal, Tuple, Dict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np

import torch, torch.nn as nn
import torchvision.transforms as T
import torchvision.models
import timm

# Import Google Drive model loader
try:
    from gdrive_config import ensure_models_downloaded, get_model_status
    GDRIVE_AVAILABLE = True
except ImportError:
    print("Warning: gdrive_config not available. Models must be manually placed in outputs/")
    GDRIVE_AVAILABLE = False

# ----------------------------
# Runtime & device
# ----------------------------
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"
DEVICE = "cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

DISEASE_HINTS = {
    # --- DermNet buckets ---
    "derm_Acne and Rosacea Photos": {
        "aka": ["acne vulgaris", "comedones", "papulopustular rosacea", "telangiectasia"],
        "visual": ["comedones (black/whiteheads)", "inflammatory papules/pustules", "facial erythema/flushing", "telangiectasia", "chin/cheeks/forehead"],
        "summary": "Inflammatory pilosebaceous disease; rosacea shows central facial erythema, flushing, and visible vessels.",
        "red_flags": ["rapidly worsening cysts with systemic symptoms", "sudden onset in older adult"]
    },
    "derm_Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": {
        "aka": ["AK", "SCC in situ risk", "BCC", "non-melanoma skin cancer"],
        "visual": ["gritty scaly macules on sun-exposed skin", "pearly telangiectatic papule (BCC)", "non-healing ulcer"],
        "summary": "UV-related keratinocyte dysplasia; BCC is slow-growing malignant tumor from basal cells.",
        "red_flags": ["bleeding or ulcerating lesion", "rapid change or pain"]
    },
    "derm_Atopic Dermatitis Photos": {
        "aka": ["eczema (atopic)", "AD"],
        "visual": ["pruritic eczematous patches", "flexural distribution", "lichenification", "xerosis"],
        "summary": "Chronic pruritic dermatitis with barrier dysfunction; flares with irritants/allergens.",
        "red_flags": ["widespread secondary infection", "eczema herpeticum (painful vesicles, fever)"]
    },
    "derm_Bullous Disease Photos": {
        "aka": ["bullous pemphigoid", "pemphigus", "tense/fragile blisters"],
        "visual": ["tense bullae on normal/erythematous base", "erosions/crusts after rupture"],
        "summary": "Autoimmune blistering disorders; elderly predominance (BP).",
        "red_flags": ["mucosal involvement", "skin sloughing (SJS/TEN)"]
    },
    "derm_Cellulitis Impetigo and other Bacterial Infections": {
        "aka": ["cellulitis", "erysipelas", "impetigo"],
        "visual": ["warm tender erythema", "indistinct borders", "honey-colored crusts (impetigo)"],
        "summary": "Bacterial skin infection; strepto- or staphylococcal causes common.",
        "red_flags": ["rapid spread", "systemic toxicity", "necrotizing infection signs"]
    },
    "derm_Eczema Photos": {
        "aka": ["nummular eczema", "dyshidrotic eczema"],
        "visual": ["ill-defined erythema", "scale/lichenification", "intense itch"],
        "summary": "Non-atopic eczematous dermatitis variants.",
        "red_flags": ["secondary bacterial infection", "extensive fissuring/bleeding"]
    },
    "derm_Exanthems and Drug Eruptions": {
        "aka": ["morbilliform drug eruption", "TARGET lesions in EM"],
        "visual": ["diffuse maculopapular rash", "onset after new medication/infection"],
        "summary": "Hypersensitivity exanthem; review recent meds (antibiotics, anticonvulsants).",
        "red_flags": ["mucosal involvement", "blistering/denudation (SJS/TEN)"]
    },
    "derm_Hair Loss Photos Alopecia and other Hair Diseases": {
        "aka": ["alopecia areata", "androgenetic alopecia"],
        "visual": ["well-demarcated patches", "miniaturized hairs", "scarring vs non-scarring"],
        "summary": "Hair cycle disruption; autoimmune or androgen-mediated.",
        "red_flags": ["scarring alopecia signs", "sudden diffuse shedding with systemic disease"]
    },
    "derm_Herpes HPV and other STDs Photos": {
        "aka": ["HSV", "VZV", "HPV warts"],
        "visual": ["grouped vesicles on erythematous base", "umbilicated/filiform warts in anogenital area"],
        "summary": "Viral mucocutaneous infections; consider STI context.",
        "red_flags": ["immunosuppression with severe/widespread lesions", "ocular involvement"]
    },
    "derm_Light Diseases and Disorders of Pigmentation": {
        "aka": ["vitiligo", "melasma", "PIH"],
        "visual": ["depigmented macules/patches", "hyperpigmented malar patches", "photodistribution"],
        "summary": "Pigment loss/excess; autoimmune (vitiligo) or photo-hormonal.",
        "red_flags": ["sudden widespread pigment change with systemic symptoms"]
    },
    "derm_Lupus and other Connective Tissue diseases": {
        "aka": ["cutaneous lupus", "discoid lupus"],
        "visual": ["malar rash sparing nasolabial folds", "photosensitivity", "annular or discoid plaques"],
        "summary": "Autoimmune connective tissue disease with cutaneous manifestations.",
        "red_flags": ["systemic symptoms (fever, serositis)", "ulcers, vasculitic lesions"]
    },
    "derm_Melanoma Skin Cancer Nevi and Moles": {
        "aka": ["melanoma", "nevus"],
        "visual": ["ABCDE: asymmetry, border irregularity, color variegation, diameter >6mm, evolving"],
        "summary": "Melanocytic neoplasm spectrum; changing/pigmented lesions need dermoscopic review.",
        "red_flags": ["rapid change", "bleeding/ulceration", "satellite lesions"]
    },
    "derm_Nail Fungus and other Nail Disease": {
        "aka": ["onychomycosis", "psoriatic nail"],
        "visual": ["subungual debris", "onycholysis", "pitting/ridging"],
        "summary": "Infectious or inflammatory nail disorders.",
        "red_flags": ["painful paronychia", "sudden pigmented streak (nail melanoma concern)"]
    },
    "derm_Poison Ivy Photos and other Contact Dermatitis": {
        "aka": ["allergic contact dermatitis", "Rhus dermatitis"],
        "visual": ["linear vesicles", "well-demarcated plaques at exposure sites"],
        "summary": "Allergic or irritant dermatitis from contactant.",
        "red_flags": ["facial/eyelid swelling", "extensive blistering"]
    },
    "derm_Psoriasis pictures Lichen Planus and related diseases": {
        "aka": ["psoriasis vulgaris", "lichen planus"],
        "visual": ["well-demarcated erythematous plaques", "silvery scale on extensor areas", "violaceous flat-topped papules (LP)"],
        "summary": "Immune-mediated papulosquamous disorders.",
        "red_flags": ["erythroderma", "fever/arthralgia (psoriatic arthritis)"]
    },
    "derm_Scabies Lyme Disease and other Infestations and Bites": {
        "aka": ["scabies", "tick bite erythema migrans"],
        "visual": ["burrows in web spaces", "pruritus worse at night", "targetoid expanding patch (Lyme)"],
        "summary": "Parasitic infestation or arthropod-borne disease.",
        "red_flags": ["secondary bacterial infection", "systemic Lyme symptoms"]
    },
    "derm_Seborrheic Keratoses and other Benign Tumors": {
        "aka": ["SK", "benign epidermal tumor"],
        "visual": ["'stuck-on' waxy plaque", "horn cysts", "multifocal on trunk"],
        "summary": "Common benign keratinocyte neoplasms.",
        "red_flags": ["sudden eruptive SKs (Leser–Trélat sign)"]
    },
    "derm_Systemic Disease": {
        "aka": ["cutaneous signs of internal disease"],
        "visual": ["jaundice/cyanosis", "digital clubbing", "livedo/necrobiosis"],
        "summary": "Skin findings reflecting systemic pathology; correlate clinically.",
        "red_flags": ["painful purpura/necrosis", "rapid systemic decline"]
    },
    "derm_Tinea Ringworm Candidiasis and other Fungal Infections": {
        "aka": ["dermatophyte", "candidiasis"],
        "visual": ["annular plaques with central clearing", "satellite pustules (candida)"],
        "summary": "Superficial fungal infections of skin/folds.",
        "red_flags": ["immunocompromise with widespread lesions"]
    },
    "derm_Urticaria Hives": {
        "aka": ["hives", "wheals"],
        "visual": ["edematous transient wheals", "dermographism", "migratory lesions"],
        "summary": "Mast-cell mediated wheals; often fleeting and pruritic.",
        "red_flags": ["angioedema", "respiratory compromise (anaphylaxis)"]
    },
    "derm_Vascular Tumors": {
        "aka": ["hemangioma", "angioma"],
        "visual": ["blanchable red/blue vascular papules/plaques"],
        "summary": "Benign vascular proliferations; many regress with time.",
        "red_flags": ["ulceration/bleeding", "rapid growth in infant"]
    },
    "derm_Vasculitis Photos": {
        "aka": ["small-vessel vasculitis", "palpable purpura"],
        "visual": ["palpable purpura on legs", "petechiae", "necrotic ulcers"],
        "summary": "Inflammation of vessel walls; evaluate triggers and systemic involvement.",
        "red_flags": ["renal/GI involvement", "fever/arthralgia"]
    },
    "derm_Warts Molluscum and other Viral Infections": {
        "aka": ["verruca", "molluscum contagiosum"],
        "visual": ["verrucous papules with black dots", "pearly umbilicated papules"],
        "summary": "Epidermal viral proliferations; autoinoculation common.",
        "red_flags": ["disseminated lesions in immunosuppressed"]
    },

    # --- Multi-Cancer buckets (histology) ---
    "cancer_ALL": {
        "aka": ["acute lymphoblastic leukemia", "lymphoblasts"],
        "visual": ["high N:C ratio lymphoblasts", "scant cytoplasm", "mitotic figures"],
        "summary": "Malignancy of lymphoid precursors; typically marrow/blood origin.",
        "red_flags": ["fever, bleeding, infections, cytopenias"]
    },
    "cancer_Brain Cancer": {
        "aka": ["glioma", "meningioma (histology variable)"],
        "visual": ["atypical glial cells", "necrosis/pseudopalisading (high-grade)"],
        "summary": "Primary CNS neoplasms with varied histo-patterns.",
        "red_flags": ["rapid neuro deficits", "increased ICP symptoms"]
    },
    "cancer_Breast Cancer": {
        "aka": ["ductal carcinoma", "lobular carcinoma"],
        "visual": ["infiltrating ducts/lobules", "pleomorphic nuclei", "mitoses"],
        "summary": "Common epithelial malignancy; receptors drive management.",
        "red_flags": ["inflammatory breast changes", "skin dimpling/peau d’orange"]
    },
    "cancer_Cervical Cancer": {
        "aka": ["squamous cell carcinoma", "HPV-related"],
        "visual": ["dysplastic squamous epithelium", "keratin pearls (SCC)"],
        "summary": "HPV-associated anogenital malignancy.",
        "red_flags": ["heavy bleeding", "pelvic pain/weight loss"]
    },
    "cancer_Kidney Cancer": {
        "aka": ["renal cell carcinoma", "clear cell"],
        "visual": ["clear cytoplasm cells", "alveolar/nested architecture"],
        "summary": "Renal epithelial malignancy; clear-cell most common.",
        "red_flags": ["hematuria", "flank pain with systemic symptoms"]
    },
    "cancer_Lung and Colon Cancer": {
        "aka": ["adenocarcinoma", "squamous cell carcinoma"],
        "visual": ["gland formation/mucin (adeno)", "keratinization (SCC)"],
        "summary": "Common epithelial cancers; histology depends on site.",
        "red_flags": ["hemoptysis", "obstruction, weight loss"]
    },
    "cancer_Lymphoma": {
        "aka": ["non-Hodgkin/Hodgkin patterns"],
        "visual": ["sheets of atypical lymphocytes", "Reed–Sternberg cells (Hodgkin)"],
        "summary": "Lymphoid malignancies with nodal or extranodal disease.",
        "red_flags": ["B-symptoms (fever, night sweats, weight loss)"]
    },
    "cancer_Oral Cancer": {
        "aka": ["oral squamous cell carcinoma"],
        "visual": ["keratin pearls", "invasive nests from mucosa"],
        "summary": "Mucosal SCC related to tobacco/alcohol/HPV.",
        "red_flags": ["non-healing oral ulcer", "odynophagia, neck nodes"]
    },
}
XRAY_HINTS = {
    "Atelectasis": {
        "aka": ["lung collapse", "volume loss"],
        "visual": ["platelike/linear opacities", "fissure/mediastinal shift toward opacity", "elevated hemidiaphragm"],
        "summary": "Partial lung collapse with loss of aeration, often post-operative or due to mucus plugging.",
        "red_flags": ["worsening breathlessness", "fever with post-op cough", "hypoxia"]
    },
    "Cardiomegaly": {
        "aka": ["enlarged heart"],
        "visual": ["cardiothoracic ratio > 0.5 (PA)", "pulmonary venous congestion", "cephalization of flow"],
        "summary": "Cardiac silhouette enlargement that can reflect chamber dilation or pericardial effusion.",
        "red_flags": ["acute dyspnea/orthopnea", "new edema or weight gain", "chest pain"]
    },
    "Consolidation": {
        "aka": ["alveolar opacification", "airspace disease"],
        "visual": ["lobar/segmental opacity", "air bronchograms", "silhouette sign with adjacent borders"],
        "summary": "Filling of alveoli by fluid, pus, blood, or cells; pneumonia is a common cause.",
        "red_flags": ["high fever", "oxygen desaturation", "confusion in elderly"]
    },
    "Edema": {
        "aka": ["pulmonary edema", "congestive changes"],
        "visual": ["perihilar bat-wing pattern", "Kerley B lines", "pleural effusions"],
        "summary": "Interstitial/alveolar fluid from heart failure or noncardiogenic causes.",
        "red_flags": ["acute respiratory distress", "pink frothy sputum", "rapid weight gain"]
    },
    "Effusion": {
        "aka": ["pleural effusion", "fluid around lung"],
        "visual": ["blunted costophrenic angle", "meniscus sign", "layering on decubitus"],
        "summary": "Fluid in the pleural space from infection, malignancy, heart failure, or inflammation.",
        "red_flags": ["sudden dyspnea", "fever + pleuritic pain", "known cancer with new effusion"]
    },
    "Emphysema": {
        "aka": ["COPD changes", "hyperinflation"],
        "visual": ["flattened diaphragms", "hyperlucent lungs", "increased retrosternal airspace"],
        "summary": "Airspace enlargement and alveolar wall destruction leading to airflow limitation.",
        "red_flags": ["acute worsening dyspnea", "cyanosis", "confusion"]
    },
    "Fibrosis": {
        "aka": ["interstitial fibrosis", "scarring"],
        "visual": ["reticular markings", "traction bronchiectasis", "volume loss (upper or lower lobe pattern)"],
        "summary": "Chronic interstitial scarring with reduced compliance and gas exchange.",
        "red_flags": ["rapid progression", "resting hypoxia", "new hemoptysis"]
    },
    "Hernia": {
        "aka": ["hiatal hernia", "diaphragmatic hernia"],
        "visual": ["air-fluid level behind heart (hiatal)", "bowel loops above diaphragm"],
        "summary": "Herniation of stomach or abdominal contents into thorax; often incidental on CXR.",
        "red_flags": ["severe chest/abdominal pain", "vomiting with obstruction signs"]
    },
    "Infiltration": {
        "aka": ["non-specific interstitial/alveolar change"],
        "visual": ["patchy ill-defined opacities", "asymmetric distribution", "compare with prior films"],
        "summary": "Non-specific parenchymal opacity that can reflect edema, infection, hemorrhage, or inflammation.",
        "red_flags": ["fever + worsening cough", "hypoxia", "immunosuppression"]
    },
    "Mass": {
        "aka": ["pulmonary mass", "large nodule"],
        "visual": ["> 3 cm focal opacity", "spiculation/cavitation assessment", "check location and margins"],
        "summary": "A focal lung lesion >3 cm; differential includes primary malignancy and metastasis.",
        "red_flags": ["hemoptysis", "rapid interval growth", "systemic symptoms (weight loss)"]
    },
    "No Finding": {
        "aka": ["normal chest radiograph", "no acute disease"],
        "visual": ["clear lungs", "sharp costophrenic angles", "normal cardiac size"],
        "summary": "No radiographic abnormality detected; clinical symptoms still require correlation.",
        "red_flags": ["persistent/worsening symptoms despite normal film"]
    },
    "Nodule": {
        "aka": ["pulmonary nodule", "coin lesion"],
        "visual": ["≤ 3 cm round opacity", "calcification pattern", "margin/spiculation review"],
        "summary": "Solitary or multiple nodules with benign vs malignant features determined by size and morphology.",
        "red_flags": ["growth on serial imaging", "smoking history or risk factors"]
    },
    "Pleural Thickening": {
        "aka": ["pleural plaques", "pleural scarring"],
        "visual": ["linear/focal pleural opacities", "costal pleural plaques (asbestos exposure)"],
        "summary": "Fibrous thickening of pleura from prior inflammation or exposure.",
        "red_flags": ["progressive dyspnea", "new chest pain", "occupational exposure + symptoms"]
    },
    "Pleural_Thickening": {  # alias key, same content as above for label compatibility
        "aka": ["pleural plaques", "pleural scarring"],
        "visual": ["linear/focal pleural opacities", "costal pleural plaques (asbestos exposure)"],
        "summary": "Fibrous thickening of pleura from prior inflammation or exposure.",
        "red_flags": ["progressive dyspnea", "new chest pain", "occupational exposure + symptoms"]
    },
    "Pneumonia": {
        "aka": ["lung infection", "infectious consolidation"],
        "visual": ["lobar/segmental consolidation", "air bronchograms", "possible parapneumonic effusion"],
        "summary": "Infectious airspace process; organism and host factors guide therapy.",
        "red_flags": ["high fever", "hypoxia", "confusion or sepsis signs"]
    },
    "Pneumothorax": {
        "aka": ["collapsed lung (air)"],
        "visual": ["visceral pleural line", "absent peripheral lung markings", "mediastinal shift if tension"],
        "summary": "Air in pleural space causing partial or complete lung collapse.",
        "red_flags": ["any suspected tension pneumothorax", "severe dyspnea", "hemodynamic instability"]
    },
}


# ----------------------------
# Backbones
# ----------------------------
class Net(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        # was: pretrained=True
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="")
        nf = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(nf, num_classes)
    def forward(self, x):
        feat = self.backbone.forward_features(x)
        if isinstance(feat, torch.Tensor) and feat.dim() == 4:
            feat = self.pool(feat).flatten(1)
        logits = self.head(feat)
        return logits, feat

class TVResNet50(nn.Module):
    """Torchvision ResNet-50 with Dropout+Linear head (compat)."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = torchvision.models.resnet50(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, num_classes))

    def forward(self, x):
        m = self.model
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        feat_map = x
        x = m.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.model.fc(feat)
        return logits, feat_map

# ----------------------------
# Transforms
# ----------------------------
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD  = (0.229, 0.224, 0.225)

def make_tfms(img_size: int, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

# These will be replaced per-model if checkpoint meta is present
DISEASE_TFMS = make_tfms(int(os.environ.get("DISEASE_IMG_SIZE", 384)))
XR_TFMS      = make_tfms(int(os.environ.get("XR_IMG_SIZE", 384)))

# ----------------------------
# Utilities
# ----------------------------
def _load_classes_list(path: str) -> list:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            try:
                items = sorted(obj.items(), key=lambda kv: int(kv[0]))
            except Exception:
                items = sorted(obj.items(), key=lambda kv: kv[0])
            return [str(v) for _, v in items]
        elif isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return []

def _is_resnet_state(raw_keys: List[str]) -> bool:
    # Torchvision resnet states typically have 'layer1.', 'layer2.', 'layer3.', 'layer4.', 'fc.'
    joined = " ".join(raw_keys[:50])
    hints = ["layer1.", "layer2.", "layer3.", "layer4.", "fc."]
    return any(h in joined for h in hints)

def _infer_head_size(state: Dict[str, torch.Tensor], keys=("head.weight","fc.1.weight","fc.weight","classifier.weight")) -> int:
    for k in keys:
        w = state.get(k)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[0])
    # fallback: any linear-looking weight outside obvious conv/bn/backbone
    for k, v in state.items():
        if k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 2 and "conv" not in k and "bn" not in k:
            return int(v.shape[0])
    return -1

def _normalize_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map common classifier names to 'head.*' so we can load into Net(...)."""
    out = {}
    for k, v in state.items():
        nk = k
        if k.startswith("head_main."):    nk = k.replace("head_main.", "head.")
        elif k.startswith("head_mel."):   nk = k.replace("head_mel.", "head.")
        elif k.startswith("model.fc.1."): nk = k.replace("model.fc.1.", "head.")
        elif k.startswith("model.fc."):   nk = k.replace("model.fc.", "head.")
        elif k.startswith("fc.1."):       nk = k.replace("fc.1.", "head.")
        elif k.startswith("fc."):         nk = k.replace("fc.", "head.")
        out[nk] = v
    return out

def _build_tfms_from_meta(meta: dict, fallback_size: int) -> T.Compose:
    img_size = int(meta.get("img_size", fallback_size))
    mean = tuple(meta.get("mean", DEFAULT_MEAN))
    std  = tuple(meta.get("std",  DEFAULT_STD))
    return make_tfms(img_size, mean, std)


def _timm_add_backbone_prefix_if_needed(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """If the checkpoint is a raw timm model (conv_stem/blocks at top-level),
    prefix all backbone params with 'backbone.' and map classifier->head."""
    if any(k.startswith("backbone.") for k in state.keys()):
        return state  # already wrapped

    roots = ("conv_stem.", "bn1.", "blocks.", "act2.", "conv_head.", "bn2.", "stages.", "stem.", "patch_embed.", "norm.", "pre_logits.")
    is_raw_timm = any(k.startswith(r) for r in roots for k in state.keys())
    if not is_raw_timm:
        return state

    new_state = {}
    for k, v in state.items():
        if k.startswith("classifier."):                 # timm classifier
            new_state["head." + k.split(".", 1)[1]] = v # -> head.weight/bias
        elif k.startswith("head.") or k.startswith("fc."):
            new_state[k] = v                            # already a head
        else:
            new_state["backbone." + k] = v             # prefix backbone
    return new_state
# ----------------------------
# Robust checkpoint loader (auto-architecture)
# ----------------------------
def load_ckpt_any(ckpt_path: str, classes_json: str, model_name_hint: str, default_tfms: T.Compose) -> Tuple[nn.Module, List[str], T.Compose]:
    classes = _load_classes_list(classes_json)
    # Handle torch 2.6+ weights_only change
    try:
        sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except Exception:
        sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    meta = sd.get("meta", {}) if isinstance(sd, dict) else {}
    raw  = sd.get("model", sd) if isinstance(sd, dict) else sd

    # Decide architecture automatically
    sample_keys = list(raw.keys())
    is_resnet = _is_resnet_state(sample_keys)

    # --- NEW: normalize names for timm checkpoints ---
    if not is_resnet:
        raw = _normalize_state_dict_keys(raw)            # classifier->head, fc->head
        raw = _timm_add_backbone_prefix_if_needed(raw)   # conv_stem/blocks -> backbone.*

    n_classes = _infer_head_size(raw)
    if n_classes <= 0:
        raise RuntimeError(f"Cannot infer num_classes from checkpoint: {ckpt_path}")

    # Build transforms from meta if present
    tfms = _build_tfms_from_meta(meta, getattr(default_tfms.transforms[0], "size", (384,384))[0])

    # Build model & load
    if is_resnet:
        print("[loader] Detected torchvision ResNet-like checkpoint")
        model = TVResNet50(num_classes=n_classes).to(DEVICE)
    else:
        print(f"[loader] Detected timm-style checkpoint (backbone hint: {model_name_hint})")
        model = Net(model_name_hint, n_classes).to(DEVICE)

    missing, unexpected = model.load_state_dict(raw, strict=False)
    print(f"[loader] load_state_dict ⇒ missing={len(missing) if missing else 0}, unexpected={len(unexpected) if unexpected else 0}")
    if missing:    print(f"[loader]   missing: {missing[:10]}")
    if unexpected: print(f"[loader]   unexpected: {unexpected[:10]}")
    model.eval()
    return model, classes, tfms

# ----------------------------
# Model bundles (paths)
# ----------------------------
DISEASE = {
    "name": "disease",
    "model_name": os.environ.get("DISEASE_MODEL", "tf_efficientnetv2_s"),  # hint only now
    "ckpt":       os.environ.get("DISEASE_CKPT", "outputs/best.pt"),
    "classes":    os.environ.get("DISEASE_CLASSES", "outputs/classes.json"),
}
XRAY = {
    "name": "xray",
    "model_name": os.environ.get("XR_MODEL", "tf_efficientnetv2_s"),
    "ckpt":       os.environ.get("XR_CKPT", "outputs/xray_best.pt"),
    "classes":    os.environ.get("XR_CLASSES", "outputs/xray_classes.json"),
}

# ----------------------------
# Ensure models are downloaded from Google Drive
# ----------------------------
if GDRIVE_AVAILABLE:
    print("Checking model files...")
    model_status = get_model_status()
    
    # Check if any model files are missing
    missing_files = [f for f, info in model_status.items() if not info["exists"]]
    if missing_files:
        print(f"Missing model files: {missing_files}")
        print("Attempting to download from Google Drive...")
        download_results = ensure_models_downloaded()
        
        failed_downloads = [f for f, success in download_results.items() if not success]
        if failed_downloads:
            print(f"❌ Failed to download: {failed_downloads}")
            print("Please ensure Google Drive is configured correctly or place model files manually.")
        else:
            print("✅ All model files downloaded successfully!")
    else:
        print("✅ All model files present locally")
else:
    print("⚠️  Google Drive not available - assuming model files are present locally")

MODELS: Dict[str, nn.Module] = {}
LABELS: Dict[str, List[str]] = {}
TFMS:   Dict[str, T.Compose] = {}

for cfg in (DISEASE, XRAY):
    name = cfg["name"]
    try:
        base_tfms = DISEASE_TFMS if name == "disease" else XR_TFMS
        m, cls, tfms = load_ckpt_any(cfg["ckpt"], cfg["classes"], cfg["model_name"], base_tfms)
        MODELS[name] = m
        LABELS[name] = cls
        TFMS[name]   = tfms
        print(f"✅ Loaded {name}: {len(cls)} classes on {DEVICE} from {cfg['ckpt']} (img_size={getattr(tfms.transforms[0],'size',(None,None))[0]})")
    except Exception as e:
        print(f"❌ Warning: {name} not loaded → {e}")

# ----------------------------
# Optional temperature calibration
# ----------------------------
CALIB_PATH = os.environ.get("CALIB_PATH", "outputs/calibration.json")

def _load_calib(path:str):
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        assert isinstance(obj, dict)
        return obj
    except Exception:
        return {}

CALIB = _load_calib(CALIB_PATH)
def _temp_for(name:str) -> float:
    return float(CALIB.get(name, {}).get("T", 1.0))

# ----------------------------
# LLM (Phi-3 mini) — lazy
# ----------------------------
import threading as _threading
_llm_lock = _threading.Lock()
_tok = None
_llm = None
LLM_ID = os.environ.get("LLM_ID", "microsoft/Phi-3-mini-4k-instruct")

def get_llm():
    global _tok, _llm
    if _tok is not None and _llm is not None: return _tok, _llm
    with _llm_lock:
        if _tok is not None and _llm is not None: return _tok, _llm
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            _tok = AutoTokenizer.from_pretrained(LLM_ID, use_fast=True, trust_remote_code=True)
            _llm = AutoModelForCausalLM.from_pretrained(
                LLM_ID,
                device_map="auto" if DEVICE == "cuda" else None,
                torch_dtype=(torch.float16 if DEVICE == "cuda" else torch.float32),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(DEVICE)
            try:
                if hasattr(_llm, "config"): _llm.config.attn_implementation = "eager"
            except Exception: pass
            try:
                if hasattr(_llm, "generation_config"): _llm.generation_config.use_cache = False
            except Exception: pass
        except Exception as e:
            print(f"Warning: Could not init LLM '{LLM_ID}' → {e}")
            _tok, _llm = None, None
        return _tok, _llm

def chat_completion(messages, max_new_tokens=128, temperature=0.2, top_p=0.95):
    tok, llm = get_llm()
    if tok is None or llm is None:
        # fallback JSON-ish
        try:
            user = next(m for m in messages if m.get("role") == "user")
            txt = user.get("content", "")
            payload = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        except Exception:
            payload = {}
        topk = payload.get("topk", [])
        return json.dumps({
            "impression": f"Findings suggestive of {topk[0]['label']}" if topk else "Uncertain.",
            "findings": [t.get("label") for t in topk],
            "differentials": [t.get("label") for t in topk[1:3]],
            "red_flags": ["Rapid progression", "Systemic symptoms"],
            "next_steps": ["Monitor symptoms", "Use supportive care", "Seek clinician input if worse"],
            "disclaimer": "Research demo; not medical advice.",
        })
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        try:
            out = llm.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=top_p, eos_token_id=getattr(tok, "eos_token_id", None),
            )
        except AttributeError:
            out = llm.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=top_p, eos_token_id=getattr(tok, "eos_token_id", None),
                use_cache=False,
            )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

# ----------------------------
# CLIP hinting (xray / skin / microscopy / CT-MRI / everyday)
# ----------------------------
_CLIP_MODEL = None
_CLIP_PROC  = None
_CLIP_LOCK  = _threading.Lock()
CLIP_ID = os.environ.get("CLIP_ID", "openai/clip-vit-base-patch32")
CLIP_LABELS = [
    ("clinical_skin", "a close-up clinical skin photograph"),
    ("microscopy",    "a histopathology microscopy slide of tissue"),
    ("xray",          "a frontal chest x-ray radiograph"),
    ("ct_mri",        "a medical CT or MRI image of the human body"),
    ("everyday",      "an ordinary everyday photo"),
]

def get_clip():
    global _CLIP_MODEL, _CLIP_PROC
    if _CLIP_MODEL is not None: return _CLIP_MODEL, _CLIP_PROC
    with _CLIP_LOCK:
        if _CLIP_MODEL is not None: return _CLIP_MODEL, _CLIP_PROC
        try:
            from transformers import CLIPModel, CLIPProcessor
            _CLIP_MODEL = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE)
            _CLIP_PROC  = CLIPProcessor.from_pretrained(CLIP_ID)
        except Exception as e:
            print(f"Warning: CLIP not available → {e}")
            _CLIP_MODEL, _CLIP_PROC = None, None
        return _CLIP_MODEL, _CLIP_PROC

@torch.no_grad()
def clip_scores(img: Image.Image) -> Dict[str, float]:
    model, proc = get_clip()
    if model is None:
        return {}
    texts = [t for _, t in CLIP_LABELS]
    inputs = proc(text=texts, images=img, return_tensors="pt", padding=True).to(DEVICE)
    out = model(**inputs)
    s = out.logits_per_image.squeeze(0).softmax(dim=0).detach().cpu().numpy()
    return {CLIP_LABELS[i][0]: float(s[i]) for i in range(len(CLIP_LABELS))}

# ----------------------------
# Inference helpers
# ----------------------------
def decode_image(b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64.split(",")[-1], validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

@torch.no_grad()
def infer_one(model: nn.Module, tfms, labels: List[str], img: Image.Image, k: int,
              tta: bool=True, temp: float=1.0):
    variants = [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.transpose(Image.FLIP_TOP_BOTTOM),
        img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM),
    ] if tta else [img]

    batch = torch.stack([tfms(v) for v in variants]).to(DEVICE)
    logits, _ = model(batch)
    logits = logits / (temp if temp and temp > 0 else 1.0)

    probs_each = torch.softmax(logits, dim=1)
    probs = probs_each.mean(dim=0).detach().cpu().numpy()

    idx_sorted = probs.argsort()
    top1 = idx_sorted[-1]
    top2 = idx_sorted[-2] if len(idx_sorted) > 1 else top1
    p0   = float(probs[top1])
    gap  = float(p0 - float(probs[top2]))
    tta_std = float(np.std(probs_each[:, top1].detach().cpu().numpy()))
    entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1))).sum())

    idx = probs.argsort()[::-1][:k]
    topk = [(labels[i], float(probs[i])) for i in idx]
    return topk, probs, {"p0": p0, "gap": gap, "tta_std": tta_std, "entropy": entropy}

def make_cam_b64(model, img: Image.Image, tfms, target_index=None):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        rgb = img.resize((tfms.transforms[0].size[0], tfms.transforms[0].size[1]))
        rgb_np = (np.array(rgb).astype("float32") / 255.0)
        x = tfms(img).unsqueeze(0).to(DEVICE)

        if isinstance(model, TVResNet50):
            target_layers = [model.model.layer4[-1]]
        elif hasattr(model, "backbone"):
            if hasattr(model.backbone, "blocks"):
                target_layers = [model.backbone.blocks[-1]]
            else:
                target_layers = [list(model.backbone.children())[-1]]
        else:
            target_layers = [list(model.children())[-2]]

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=x, targets=None, eigen_smooth=True)[0]
        vis = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True, image_weight=0.6)
        buff = io.BytesIO()
        Image.fromarray(vis).save(buff, format="PNG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        return None
# def _derm_hints_for(label_lower: str) -> list:
#     H = []
#     L = label_lower

#     # order is important (first match wins); keep items short (LLM will elaborate)
#     if any(k in L for k in ["acne"]):
#         H = ["Clogged follicles; papules/pustules", "Often on face/back; teens or adults", "Triggers: hormones, occlusion, some meds"]
#     elif any(k in L for k in ["atopic", "eczema"]):
#         H = ["Itchy, dry, eczematous patches", "Flexural surfaces common", "History of allergies/asthma common"]
#     elif any(k in L for k in ["psoriasis", "lichen planus"]):
#         H = ["Well-demarcated erythematous plaques", "Silvery scale; extensor surfaces", "Nail changes possible (pitting)"]
#     elif any(k in L for k in ["seborrheic keratos", "benign tumor"]):
#         H = ["Stuck-on waxy papules/plaques", "Common benign lesion in older adults", "Usually asymptomatic; cosmetic concern"]
#     elif any(k in L for k in ["herpes", "hpv", "std", "molluscum", "warts"]):
#         H = ["Viral etiology; clustered vesicles or papules", "Contagious; may be sexually transmitted", "Consider antivirals or procedural care"]
#     elif any(k in L for k in ["bacterial", "cellulitis", "impetigo"]):
#         H = ["Bacterial skin infection", "Erythema, warmth, tenderness", "Consider antibiotics; watch for systemic signs"]
#     elif any(k in L for k in ["fung", "tinea", "dermatophyte", "candida"]):
#         H = ["Annular or scaling plaques", "KOH positive; branching hyphae", "Topical/systemic antifungals"]
#     elif "nail" in L:
#         H = ["Onychomycosis or nail dystrophy", "Thickened, brittle, discolored nails", "Fungal culture/KOH; antifungals if confirmed"]
#     elif any(k in L for k in ["lupus", "connective tissue"]):
#         H = ["Autoimmune/connective tissue", "Photosensitive rash possible", "Systemic involvement; rheumatology input"]
#     elif any(k in L for k in ["scabies", "infestation", "bites", "lyme"]):
#         H = ["Parasite/arthropod related", "Intense pruritus; burrows or bite pattern", "Treat contacts; environmental measures"]
#     elif any(k in L for k in ["rosacea"]):
#         H = ["Central facial flushing with papules/pustules", "Triggers: heat, spicy food, alcohol", "Topicals/systemics; avoid triggers"]
#     else:
#         H = ["Dermatologic condition (category-level)", "Assess distribution, scale, borders", "Correlate with symptoms & history"]
#     return H


# def _cancer_hints_for(label_lower: str) -> list:
#     H = []
#     L = label_lower
#     # Multi-cancer folders in your dataset:
#     # ALL, Brain Cancer, Breast Cancer, Cervical Cancer, Kidney Cancer,
#     # Lung and Colon Cancer, Lymphoma, Oral Cancer
#     if any(k in L for k in ["all", "acute lymphoblastic"]):
#         H = ["Hematologic malignancy (lymphoblasts)", "Bone marrow involvement common", "Look for cytopenias; chemo-based protocols"]
#     elif "brain" in L:
#         H = ["Primary brain tumor (glioma/meningioma etc.)", "Neuro deficits; MRI patterns", "Histology: cellular atypia; mitoses"]
#     elif "breast" in L:
#         H = ["Malignant epithelial tumor (ductal/lobular)", "Imaging: spiculated masses, microcalcifications", "IHC: ER/PR/HER2 may guide therapy"]
#     elif "cervical" in L:
#         H = ["Often HPV-associated dysplasia/carcinoma", "Pap/HPV screening impacts prognosis", "Histology: squamous cell features"]
#     elif "kidney" in L or "renal" in L:
#         H = ["Renal cell carcinoma (clear cell common)", "CT/MRI: enhancing renal mass", "Histology: clear cytoplasm; vascular network"]
#     elif "lung" in L and "colon" in L:
#         H = ["Adenocarcinoma variants (lung/colon)", "Lung: peripheral lesions; Colon: gland-forming, 'dirty necrosis'", "Staging and molecular markers guide therapy"]
#     elif "lung" in L:
#         H = ["Primary lung cancer (adeno/squamous/small cell)", "Smoking risk; cough/hemoptysis", "Imaging nodules; biopsy confirms"]
#     elif "colon" in L:
#         H = ["Colorectal adenocarcinoma", "Change in bowel habits; anemia", "Histology: glandular structures; staging critical"]
#     elif "lymphoma" in L:
#         H = ["Monoclonal lymphoid proliferation (Hodgkin/Non-Hodgkin)", "B-symptoms possible (fever, weight loss, night sweats)", "IHC/flow cytometry phenotype"]
#     elif "oral" in L:
#         H = ["Oral cavity squamous cell carcinoma", "Tobacco/alcohol risk; ulcerative masses", "Keratin pearls; local nodes"]
#     else:
#         H = ["Histopathology malignancy (unspecified site)", "Architecture and cytologic atypia", "Immunoprofile may aid classification"]
#     return H


# def _build_label_hints(pretty_topk: list) -> list:
#     """
#     pretty_topk: [{"label": <nice name>, "prob": <float 0-100>, "domain": <domain string>}]
#     Returns: [{"label": str, "hints": [str, ...]}]
#     """
#     out = []
#     for item in pretty_topk:
#         lab = item["label"]
#         L = lab.lower()
#         domain = item.get("domain", "")
#         hints = _cancer_hints_for(L) if "cancer" in L or "histopath" in domain else _derm_hints_for(L)
#         out.append({"label": lab, "hints": hints[:5]})
#     return out
# ----------------------------
# LLM report template
# ----------------------------
def llm_report(payload: dict) -> dict:
    top_label = ""
    try:
        if isinstance(payload.get("topk"), list) and payload["topk"]:
            top_label = payload["topk"][0].get("label", "")
    except Exception:
        pass

    # Pick the right hint table
    hint = {}
    if payload.get("modality") == "xray":
        hint = XRAY_HINTS.get(top_label, {})
    else:
        hint = DISEASE_HINTS.get(top_label, {})
	
    system = (
        "You are a cautious clinical assistant. Write patient-friendly, factual summaries. "
        "Use the provided 'hint' only as guidance. Do not copy hint text verbatim; "
        "paraphrase and synthesize with the model predictions and symptoms."
    )



    user = (
        "Return strictly a compact JSON object with keys:\n"
        "impression, findings, disease_summary, red_flags, next_steps, disclaimer.\n"
        "Use short sentences and bullet lists where appropriate. Consider symptoms and hint.\n\n"
        + json.dumps({
            "top_prediction": top_label,
            "topk": payload.get("topk", []),
            "symptoms": payload.get("symptoms", ""),
            "hint": hint
        }, ensure_ascii=False)
    )

    out = chat_completion(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        max_new_tokens=220, temperature=0.2, top_p=0.9
    )

    def _ensure_report(obj: dict) -> dict:
        def _as_list(v):
            if isinstance(v, list): return [str(x) for x in v]
            if not v: return []
            return [str(v)]
        return {
            "impression": obj.get("impression") or f"Findings are suggestive of {top_label}; clinical correlation advised.",
            "findings": _as_list(obj.get("findings")) or [f"Features align with {top_label}."],
            "disease_summary": obj.get("disease_summary") or (hint.get("summary") or "Condition requires clinical evaluation for confirmation and management."),
            "red_flags": _as_list(obj.get("red_flags")) or (hint.get("red_flags") or ["Rapid progression", "Systemic symptoms"]),
            "next_steps": _as_list(obj.get("next_steps")) or ["Monitor symptoms", "Seek clinician input if worse", "Bring this report to your provider"],
            "disclaimer": obj.get("disclaimer") or "Research demo; not medical advice.",
        }

    try:
        raw = out[out.find("{"): out.rfind("}") + 1]
        return _ensure_report(json.loads(raw))
    except Exception:
        # Fallback to hint-only narrative
        return _ensure_report({})
# ----------------------------
# Schemas
# ----------------------------
class DiagRequest(BaseModel):
    modality: Literal["disease", "xray"] = "disease"
    symptoms: Optional[str] = ""
    topk: int = 3
    include_cam: bool = False
    image_b64: str

class DiagResponse(BaseModel):
    topk: List[dict]
    uncertain: bool
    payload_used: dict
    report: dict
    cam_b64: Optional[str] = None
    needs_confirmation: bool = False
    router: Optional[dict] = None

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="HealthLens (Unified)", version="0.6.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    model_status = {}
    if GDRIVE_AVAILABLE:
        try:
            model_status = get_model_status()
        except Exception as e:
            model_status = {"error": str(e)}
    
    return {
        "status": "ok",
        "device": DEVICE,
        "vision_models_loaded": list(MODELS.keys()),
        "llm_ready": bool(_tok is not None and _llm is not None),
        "gdrive_available": GDRIVE_AVAILABLE,
        "model_files_status": model_status,
    }

# ----------------------------
# Endpoint
# ----------------------------
@app.post("/v1/diag", response_model=DiagResponse)
def diag(req: DiagRequest):
    if not req.image_b64:
        raise HTTPException(400, "image_b64 required")
    img = decode_image(req.image_b64)

    if req.modality not in MODELS:
        raise HTTPException(503, f"Requested modality '{req.modality}' not available on server.")

    model  = MODELS[req.modality]
    tfms   = TFMS[req.modality]
    labels = LABELS[req.modality]
    temp   = _temp_for(req.modality)

    topk, probs, stats = infer_one(model, tfms, labels, img, req.topk, tta=True, temp=temp)
    out = [{"label": l, "prob": p, "source": req.modality} for (l, p) in topk]

    uncertain = (stats["p0"] < 0.60) or (stats["gap"] < 0.12) or (stats["tta_std"] > 0.08) or (stats["entropy"] > 1.20)

    clip_s = clip_scores(img)
    # For "disease", accept clinical_skin or microscopy; warn if CT/MRI dominates
    if req.modality == "disease":
        target_prob = clip_s.get("clinical_skin", 0.0) + clip_s.get("microscopy", 0.0)
        ct_mri_prob = clip_s.get("ct_mri", 0.0)
        clip_min = float(os.environ.get("DISEASE_CLIP_MIN", "0.35"))
        needs_confirmation = bool(target_prob < clip_min or ct_mri_prob > 0.40)
    else:
        target_prob = clip_s.get("xray", 0.0)
        clip_min = float(os.environ.get("XRAY_CLIP_MIN", "0.50"))
        needs_confirmation = bool(target_prob < clip_min)

    payload = {
        "modality": req.modality, "labels": labels, "topk": out,
        "uncertain": bool(uncertain or needs_confirmation),
        "symptoms": (req.symptoms or "").strip(),
        "router": {"clip": clip_s, "stats": stats},
    }

    report  = llm_report(payload)
    cam_b64 = make_cam_b64(model, img, tfms) if req.include_cam else None

    return DiagResponse(
        topk=out,
        uncertain=payload["uncertain"],
        payload_used=payload,
        report=report,
        cam_b64=("data:image/png;base64," + cam_b64) if cam_b64 else None,
        needs_confirmation=needs_confirmation,
        router=payload["router"],
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))