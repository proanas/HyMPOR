"""
config.py — الإعدادات المركزية للنظام الهجين
"""
import os
from types import SimpleNamespace

# ══════════════════════════════════════════
#  المسارات الجذرية
# ══════════════════════════════════════════
ROOT      = "/content/drive/MyDrive/hybrid_project"
SAM2_ROOT = f"{ROOT}/modules/SAM2"   # ← حرف كبير S كما هو في Drive
AOT_ROOT  = f"{ROOT}/modules/AOT_GAN"
AOT_SRC   = f"{AOT_ROOT}/src"

# ══════════════════════════════════════════
#  مسارات البيانات
# ══════════════════════════════════════════
DATA_ROOT  = f"{ROOT}/dataset"
INPUT_DIR  = f"{DATA_ROOT}/input"
MASK_DIR   = f"{DATA_ROOT}/masks"
OUTPUT_DIR = f"{DATA_ROOT}/output"

# ══════════════════════════════════════════
#  أوزان AOT-GAN
# ══════════════════════════════════════════
WEIGHTS = {
    "places2":  f"{AOT_ROOT}/experiments/places2/G0000000.pt",
    "celebahq": f"{AOT_ROOT}/experiments/celebahq/G0000000.pt",
}

# ══════════════════════════════════════════
#  أوزان SAM2  ← يكتشف اسم الـ checkpoint تلقائياً
# ══════════════════════════════════════════
def _find_sam2_checkpoint():
    ckpt_dir = f"{SAM2_ROOT}/checkpoints"
    if not os.path.exists(ckpt_dir):
        return None
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pt"):
            return os.path.join(ckpt_dir, f)
    return None

SAM2_CHECKPOINT = _find_sam2_checkpoint() or f"{SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"

# ══════════════════════════════════════════
#  إعدادات AOT-GAN
# ══════════════════════════════════════════
AOT_ARGS = SimpleNamespace(
    model      = "aotgan",
    rates      = [1, 2, 4, 8],
    block_num  = 8,
    image_size = 512,
    gan_type   = "smgan",
    rec_loss   = {"L1": 1.0, "Style": 250.0, "Perceptual": 0.1},
    adv_weight = 0.01,
    lrg        = 1e-4,
    lrd        = 1e-4,
    beta1      = 0.5,
    beta2      = 0.999,
)

# ══════════════════════════════════════════
#  إعدادات SAM2
# ══════════════════════════════════════════
SAM2_ARGS = SimpleNamespace(
    points_per_side        = 32,
    pred_iou_thresh        = 0.88,
    stability_score_thresh = 0.95,
    min_mask_region_area   = 500,
)

# ══════════════════════════════════════════
#  إعدادات اختيار الأوزان
# ══════════════════════════════════════════
SELECTOR_ARGS = SimpleNamespace(
    face_overlap_threshold    = 0.05,
    face_detection_confidence = 0.3,
)

def ensure_dirs():
    for d in [INPUT_DIR, MASK_DIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)

def print_config():
    print(f"ROOT         : {ROOT}")
    print(f"SAM2_ROOT    : {SAM2_ROOT}  [{'✅' if os.path.exists(SAM2_ROOT) else '❌'}]")
    print(f"SAM2_CKPT    : {SAM2_CHECKPOINT}  [{'✅' if SAM2_CHECKPOINT and os.path.exists(SAM2_CHECKPOINT) else '❌'}]")
    print(f"AOT_ROOT     : {AOT_ROOT}  [{'✅' if os.path.exists(AOT_ROOT) else '❌'}]")
    print(f"places2 w    : {'✅' if os.path.exists(WEIGHTS['places2']) else '❌'}")
    print(f"celebahq w   : {'✅' if os.path.exists(WEIGHTS['celebahq']) else '❌'}")
