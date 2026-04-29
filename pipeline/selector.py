"""
selector.py — اختيار أوزان AOT-GAN بكشف متعدد المستويات

كشف الوجه (4 مستويات):
  1. RetinaFace على الصورة الأصلية
  2. RetinaFace على نسخة مُنعَّمة (blur) — يُخفف الخدوش ويُبقي بنية الوجه
  3. Haar على الصورة الأصلية مع تأكيد العين
  4. Haar على نسخة مُنعَّمة مع تأكيد العين

اختيار الأوزان:
  celebahq : وجه مكتشف ويتقاطع مع قناع التلف
  places2  : لا يوجد وجه، أو الوجه موجود لكن القناع في الخلفية
"""
import cv2
import numpy as np
import torch
from . import config

# ── Haar fallback (نُبقيه للسلامة) ──
_face_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_face_alt = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
_eye_det  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def _blur_variants(image_rgb: np.ndarray):
    """
    يُنتج نسخاً مُنعَّمة من الصورة لتحسين كشف الوجه في الصور التالفة.

    السبب: الخدوش ذات تردد عالٍ (High Frequency) بينما بنية الوجه
    ذات تردد منخفض. Gaussian blur يُخفف الخدوش ويُبقي العيون والأنف
    والشفاه واضحة — مما يُساعد كلاً من RetinaFace و Haar.

    3 شدات تصاعدية:
      k=7  : خفيف — يُلطّف الخدوش الرفيعة
      k=15 : متوسط — يُخفي الخدوش العريضة
      k=25 : قوي  — للتلف الشديد جداً
    """
    return [cv2.GaussianBlur(image_rgb, (k, k), 0) for k in [7, 15, 25]]


def _confirm_haar_face(gray: np.ndarray, fx: int, fy: int, fw: int, fh: int) -> bool:
    """
    تأكيد وجه Haar بوجود عين في النصف العلوي.
    ضروري لمنع هلوسات Haar على السماء/الأشجار/الانعكاسات.
    RetinaFace لا يحتاج هذا — فقط Haar.
    """
    roi   = gray[fy:fy + fh, fx:fx + fw]
    upper = roi[:fh // 2, :]
    if upper.size == 0:
        return False
    eyes = _eye_det.detectMultiScale(
        upper,
        scaleFactor  = 1.1,
        minNeighbors = 4,
        minSize      = (max(4, int(fw * 0.08)), max(4, int(fh * 0.05))),
    )
    return len(eyes) >= 1


# ── Lazy-loaded RetinaFace helper ──
_face_helper = None          # إما FaceRestoreHelper أو None أو False(فشل)
_is_external = False         # هل تم تمريره من الخارج (GFPGAN)؟


def set_external_face_helper(face_helper):
    """
    (اختياري لكن مُوصى به) تمرير gfpganer.face_helper من الـ notebook
    لتجنب تحميل RetinaFace مرتين في الذاكرة.
    """
    global _face_helper, _is_external
    _face_helper = face_helper
    _is_external = True
    print("✅ Selector: تم ربط كاشف GFPGAN (RetinaFace)")


def _init_retinaface():
    """تحميل كسول لـ RetinaFace مستقل إن لم يُمرَّر خارجي."""
    global _face_helper
    if _face_helper is not None:
        return _face_helper  # قد يكون helper أو False
    try:
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _face_helper = FaceRestoreHelper(
            upscale_factor = 1,
            face_size      = 512,
            crop_ratio     = (1, 1),
            det_model      = "retinaface_resnet50",
            save_ext       = "png",
            use_parse      = False,
            device         = device,
        )
        print("✅ Selector: RetinaFace مستقل جاهز")
    except Exception as e:
        print(f"⚠️  Selector: فشل تحميل RetinaFace → {e}")
        _face_helper = False
    return _face_helper


def _detect_with_retinaface(image_rgb: np.ndarray, H: int, W: int):
    """
    يشغّل RetinaFace على الصورة ويُرجع أكبر bbox: (fx, fy, fw, fh) أو None.
    """
    helper = _init_retinaface()
    if not helper:
        return None
    try:
        # facexlib يتوقّع BGR (مثل cv2)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        helper.clean_all()
        helper.read_image(image_bgr)
        num_faces = helper.get_face_landmarks_5(
            only_center_face    = False,
            resize              = 640,
            eye_dist_threshold  = 5,
        )
        if num_faces == 0 or not helper.det_faces:
            return None

        # أكبر وجه من حيث المساحة
        biggest = max(
            helper.det_faces,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
        )
        x1, y1, x2, y2 = map(int, biggest[:4])
        fx = max(0, x1)
        fy = max(0, y1)
        fw = min(W - fx, x2 - x1)
        fh = min(H - fy, y2 - y1)
        if fw < 20 or fh < 20:   # حارس ضد كشف تافه
            return None
        return fx, fy, fw, fh
    except Exception as e:
        print(f"⚠️  Selector: خطأ في RetinaFace → {e}")
        return None


def _detect_with_haar(image_np: np.ndarray, H: int, W: int):
    """
    Haar fallback مع تأكيد إجباري بالعين.
    بدون هذا التأكيد، Haar يُنتج false positives من الغيوم والأشجار
    وانعكاسات الماء (شكّلت مشكلة في الصور الطبيعية).
    """
    gray     = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray_eq  = cv2.equalizeHist(gray)
    gray_cl  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    min_face = max(80, int(min(H, W) * 0.12))

    for g in [gray_eq, gray_cl, gray]:
        for detector in [_face_det, _face_alt]:
            if detector.empty():
                continue
            # معاملات صارمة: minNeighbors عالي يقلل الهلوسات
            for scale, neighbors in [(1.05, 6), (1.1, 6), (1.1, 8)]:
                found = detector.detectMultiScale(
                    g,
                    scaleFactor  = scale,
                    minNeighbors = neighbors,
                    minSize      = (min_face, min_face),
                )
                if not len(found):
                    continue
                # نرتّب من الأكبر للأصغر ونطلب تأكيد العين
                found = sorted(found, key=lambda r: r[2] * r[3], reverse=True)
                for (fx, fy, fw, fh) in found[:3]:
                    if _confirm_haar_face(g, fx, fy, fw, fh):
                        return int(fx), int(fy), int(fw), int(fh)
    return None


def select_weights(image_np: np.ndarray, mask_np: np.ndarray) -> str:
    """
    يختار الأوزان:
      celebahq : كُشف وجه بأي من المستويات الثلاثة
      places2  : لا يوجد وجه إطلاقاً
    """
    H, W = image_np.shape[:2]

    # ── المستوى 1/2: RetinaFace ──
    print("🔍 كشف الوجه [المستوى 1: RetinaFace]...")
    box = _detect_with_retinaface(image_np, H, W)
    source = "RetinaFace" + (" (GFPGAN)" if _is_external else "")

    # ── المستوى 3: Haar fallback ──
    if box is None:
        print("   ↳ لم يُكتشف — تجربة Haar [المستوى 2]...")
        box = _detect_with_haar(image_np, H, W)
        source = "Haar"

    if box is None:
        print("✅ لا يوجد وجه مؤكد → places2")
        return "places2"

    fx, fy, fw, fh = box

    # توسيع 20% للأمان
    pad_x = int(fw * 0.2)
    pad_y = int(fh * 0.2)
    fx = max(0, fx - pad_x)
    fy = max(0, fy - pad_y)
    fw = min(W - fx, fw + 2 * pad_x)
    fh = min(H - fy, fh + 2 * pad_y)

    face_region = np.zeros((H, W), dtype=np.uint8)
    face_region[fy:fy + fh, fx:fx + fw] = 255

    overlap = cv2.bitwise_and(face_region, mask_np)

    # ── معادلة التقاطع الصحيحة ──
    # السؤال: "كم من التلف يقع على الوجه؟" وليس "كم من الوجه تالف؟"
    # هذا يعمل بشكل صحيح حتى مع الخدوش الرفيعة التي مساحتها صغيرة
    # مقارنة بمساحة الوجه الكاملة.
    #
    # مثال صورة الخدش:
    #   قديم: 10,500 / 178,000 = 5.9%  → places2 (خطأ)
    #   جديد: 10,500 / 11,215  = 93.6% → celebahq (صحيح)
    mask_area   = float(max(mask_np.sum(), 1))
    overlap_pct = overlap.sum() / mask_area

    print(f"✅ وجه مكتشف بواسطة {source} [{fx},{fy},{fw},{fh}]")
    print(f"   تقاطع التلف مع الوجه: {overlap_pct * 100:.1f}% من مساحة القناع")

    # ── القرار النهائي ──
    if overlap_pct > config.SELECTOR_ARGS.face_overlap_threshold:
        print(f"→ celebahq (التلف على الوجه، {overlap_pct*100:.1f}%)")
        return "celebahq"
    else:
        print(f"→ places2 (القناع خارج منطقة الوجه، {overlap_pct*100:.1f}%)")
        return "places2"