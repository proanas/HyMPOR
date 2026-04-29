"""
run.py — المنسق الرئيسي للنظام الهجين
SAM2 → اختيار الأوزان → AOT-GAN

التحسينات المُطبَّقة:
  1. Boundary-Aware Mask Dilation تكيّفي:
       celebahq → kernel 5×5  (2 بكسل) — حذر على الوجوه لحماية التفاصيل
       places2  → kernel 11×11 (5 بكسل) — عدواني للخلفيات والمناطق الواسعة
  2. الكشف عن الوجه يعتمد على القناع الأصلي (قبل الـ dilation)
     لضمان دقة selector — ثم يُطبَّق التوسّع حسب القرار
"""
import os
import base64
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

from . import config
from .segmentation import SAM2Segmentor
from .inpainting   import AOTInpainter
from .selector     import select_weights


class HybridPipeline:
    """
    الأنبوب الهجين الكامل:
      1. SAM2      → استخراج القناع من رسم المستخدم
      2. Selector  → اختيار الأوزان بناءً على القناع الأصلي (قبل dilation)
      3. Dilation  → توسيع تكيّفي حسب نوع المحتوى (وجه vs خلفية)
      4. AOT-GAN   → تعبئة المنطقة المُقنَّعة
    """

    # ── إعدادات الـ dilation التكيّفي ──
    # celebahq: 5×5 = توسّع 2 بكسل — يحمي تفاصيل الوجه (عيون، شفاه، حواف)
    # places2 : 11×11 = توسّع 5 بكسل — يغطي حواف ضبابية في الخلفيات والمناطق الواسعة
    _DILATION_KERNELS = {
        "celebahq": (5,  5),
        "places2":  (11, 11),
    }

    def __init__(self):
        config.ensure_dirs()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Device: {device}")

        self.device   = device
        self.sam2     = SAM2Segmentor(device)
        self.aot      = AOTInpainter(device)
        self.image_np = None
        self.img_path = None

    # ──────────────────────────────────────
    #  تحميل الصورة
    # ──────────────────────────────────────
    def load_image(self, image_path: str):
        """تحميل صورة وإرسالها لـ SAM2 مع الحفاظ على النسبة الأصلية"""
        assert os.path.exists(image_path), f"❌ الصورة غير موجودة: {image_path}"

        orig = Image.open(image_path).convert("RGB")
        orig_w, orig_h = orig.size

        # تصغير مع الحفاظ على النسبة
        max_size = config.AOT_ARGS.image_size
        ratio  = min(max_size / orig_w, max_size / orig_h)
        new_w  = int(orig_w * ratio)
        new_h  = int(orig_h * ratio)
        resized = orig.resize((new_w, new_h), Image.LANCZOS)

        self.image_np  = np.array(resized)
        self.img_path  = image_path
        self.orig_size = (orig_w, orig_h)

        self.sam2.set_image(self.image_np)
        print(f"✅ uploaded image: {os.path.basename(image_path)} "
              f"— {orig_w}×{orig_h} → {new_w}×{new_h}")
        return self.image_np

    # ──────────────────────────────────────
    #  Boundary-Aware Mask Dilation التكيّفي
    # ──────────────────────────────────────
    @staticmethod
    def _dilate_mask(mask: np.ndarray, model_key: str) -> np.ndarray:
        """
        يُوسّع القناع بحجم kernel يعتمد على نوع المحتوى:

        celebahq (وجوه):
            kernel 5×5 = توسّع 2 بكسل
            سبب الحذر: الخدش على حافة العين أو الشفة يمتد 2-3 بكسل فقط.
            توسّع أكبر يُخاطر بمحو تفاصيل سليمة (رمش، خط الشفاه).

        places2 (خلفيات/مناطق واسعة):
            kernel 11×11 = توسّع 5 بكسل
            سبب العدوانية: حواف التمزق والبقع في الخلفيات تكون ضبابية
            وتمتد أبعد — 5 بكسل يضمن تغطية كاملة بلا آثار على الحواف.
        """
        import cv2 as _cv2
        ksize   = HybridPipeline._DILATION_KERNELS.get(model_key, (7, 7))
        kernel  = np.ones(ksize, np.uint8)
        dilated = _cv2.dilate(mask, kernel, iterations=1)
        return dilated

    # ──────────────────────────────────────
    #  تشغيل الأنبوب الكامل
    # ──────────────────────────────────────
    def process_drawn_mask(self, drawn_mask_b64: str) -> dict:
        """
        الدالة الرئيسية — تُستدعى من واجهة Colab عند الضغط على "تأكيد"

        Args:
            drawn_mask_b64: القناع المرسوم بصيغة base64 PNG

        Returns:
            dict يحتوي: mask, mask_raw, result, model_key, dilation_kernel
        """
        assert self.image_np is not None, "❌ استدعِ load_image() أولاً"

        # ── ١. فك ترميز القناع المرسوم ──
        H, W  = self.image_np.shape[:2]
        raw   = base64.b64decode(drawn_mask_b64)
        drawn = np.array(
            Image.open(io.BytesIO(raw)).convert("L").resize((W, H))
        )

        if drawn.max() == 0:
            raise ValueError("⚠️  لم ترسم أي منطقة — جرب مجدداً")

        # ── ٢. SAM2: توليد القناع الدقيق ──
        print("\n[1/4] SAM2 يستخرج القناع...")
        mask_raw = self.sam2.predict_from_drawn_mask(drawn)
        px_raw   = int(mask_raw.sum() // 255)
        print(f"   Original Mask (before dilation): {px_raw:,} px²")

        # ── ٣. اختيار الأوزان — يعتمد على القناع الأصلي (قبل التوسّع) ──
        # مهم: نُعطي selector القناع الأصلي لأن التوسّع سيُكبّر المنطقة
        # ويُضلّل حساب التقاطع مع منطقة الوجه
        print("\n[2/4] اختيار النموذج المناسب...")
        model_key = select_weights(self.image_np, mask_raw)

        # ── ٤. Boundary-Aware Mask Dilation التكيّفي ──
        ksize      = self._DILATION_KERNELS.get(model_key, (7, 7))
        final_mask = self._dilate_mask(mask_raw, model_key)
        px_dilated = int(final_mask.sum() // 255)

        print(f"\n[3/4] Boundary-Aware Dilation ({model_key})...")
        print(f"      kernel: {ksize[0]}×{ksize[1]} "
              f"({'2 بكسل — حذر على الوجه' if model_key == 'celebahq' else '5 بكسل — عدواني للخلفية'})")
        print(f"      Mask after dilation: {px_dilated:,} px² "
              f"(+{px_dilated - px_raw:,} px²)")

        # ── ٥. AOT-GAN: تعبئة المنطقة ──
        print(f"\n[4/4] AOT-GAN ({model_key}) يعبّئ المنطقة...")
        result = self.aot.inpaint(self.image_np, final_mask, model_key)

        # ── ٦. حفظ النتائج ──
        fname    = os.path.splitext(os.path.basename(self.img_path))[0]
        out_img  = f"{config.OUTPUT_DIR}/{fname}_inpainted.png"
        out_mask = f"{config.MASK_DIR}/{fname}_mask.png"

        Image.fromarray(result).save(out_img)
        Image.fromarray(final_mask).save(out_mask)
        print(f"\n💾 محفوظة: {out_img}")

        return {
            "mask":            final_mask,   # القناع بعد dilation (ما أُرسل لـ AOT-GAN)
            "mask_raw":        mask_raw,      # القناع الأصلي من SAM2 (للمقارنة)
            "result":          result,
            "model_key":       model_key,
            "dilation_kernel": ksize,
            "out_img":         out_img,
            "out_mask":        out_mask,
        }

    # ──────────────────────────────────────
    #  عرض النتائج
    # ──────────────────────────────────────
    def show_result(self, output: dict):
        """عرض المقارنة الرباعية: الأصل / قناع SAM2 / قناع بعد dilation / النتيجة"""
        show_raw = "mask_raw" in output and output["mask_raw"] is not None

        n_plots = 4 if show_raw else 3
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

        axes[0].imshow(self.image_np)
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(output["mask_raw"] if show_raw else output["mask"], cmap="gray")
        axes[1].set_title("Mask (SAM2 — Before dilation)" if show_raw else "Mask (SAM2)",
                          fontsize=12)
        axes[1].axis("off")

        if show_raw:
            ksize = output.get("dilation_kernel", ("?", "?"))
            axes[2].imshow(output["mask"], cmap="gray")
            axes[2].set_title(
                f"Mask (After dilation {ksize[0]}×{ksize[1]})\n"
                f"model: {output['model_key']}", fontsize=11)
            axes[2].axis("off")

        axes[-1].imshow(output["result"])
        axes[-1].set_title(f"Result ({output['model_key']})", fontsize=12)
        axes[-1].axis("off")

        plt.tight_layout()
        plt.show()
