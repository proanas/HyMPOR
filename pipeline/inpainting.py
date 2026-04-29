"""
inpainting.py — واجهة AOT-GAN للاستدلال
يتعامل مع تحميل النموذج وتشغيل الاستدلال مباشرة
"""
import sys
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from . import config


def _ensure_aot_path():
    """إضافة مسار AOT-GAN src للـ Python path"""
    if config.AOT_SRC not in sys.path:
        sys.path.insert(0, config.AOT_SRC)


class AOTInpainter:
    """
    واجهة موحدة لتشغيل AOT-GAN
    تدعم نموذجين: places2 و celebahq
    """

    def __init__(self, device=None):
        _ensure_aot_path()
        from model.aotgan import InpaintGenerator

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._models = {}
        self._load_all()

    def _load_model(self, weight_key: str) -> torch.nn.Module:
        """تحميل نموذج واحد بناءً على المفتاح"""
        weight_path = config.WEIGHTS[weight_key]
        assert os.path.exists(weight_path), f"❌ الأوزان غير موجودة: {weight_path}"

        _ensure_aot_path()
        from model.aotgan import InpaintGenerator

        net = InpaintGenerator(config.AOT_ARGS).to(self.device)
        state = torch.load(weight_path, map_location=self.device)
        net.load_state_dict(state)
        net.eval()
        print(f"  ✅ {weight_key} محمّل من: {weight_path}")
        return net

    def _load_all(self):
        """تحميل جميع النماذج المتاحة"""
        print("⬇️  تحميل نماذج AOT-GAN...")
        for key in config.WEIGHTS:
            self._models[key] = self._load_model(key)
        print(f"✅ AOT-GAN جاهز — {list(self._models.keys())}")

    def inpaint(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        model_key: str = "places2",
    ) -> np.ndarray:
        """
        تشغيل الاستدلال على صورة وقناع

        Args:
            image_np  : صورة RGB  [H, W, 3]  uint8
            mask_np   : قناع ثنائي [H, W]     uint8  (255=منطقة التعبئة)
            model_key : "places2" أو "celebahq"

        Returns:
            نتيجة RGB [H, W, 3] uint8
        """
        assert model_key in self._models, f"❌ مفتاح غير معروف: {model_key}"
        size = config.AOT_ARGS.image_size
        net  = self._models[model_key]

        # ── دالة مساعدة: padding مربع مع حفظ النسبة ──
        def pad_to_square(pil_img, fill=0):
            w, h = pil_img.size
            s = max(w, h)
            new_img = Image.new(pil_img.mode, (s, s), fill)
            new_img.paste(pil_img, ((s - w) // 2, (s - h) // 2))
            return new_img, w, h, s

        # ── تحضير الصورة مع الحفاظ على النسبة ──
        orig_pil = Image.fromarray(image_np).convert("RGB")
        orig_w, orig_h = orig_pil.size

        img_sq, _, _, sq = pad_to_square(orig_pil)
        msk_sq, _, _, _  = pad_to_square(Image.fromarray(mask_np).convert("L"))

        img = img_sq.resize((size, size), Image.LANCZOS)
        msk = msk_sq.resize((size, size), Image.NEAREST)

        img_t  = TF.to_tensor(img).unsqueeze(0).to(self.device) * 2.0 - 1.0
        mask_t = TF.to_tensor(msk).unsqueeze(0).to(self.device)
        mask_t = (mask_t > 0.5).float()

        # + mask_t يملأ المنطقة بـ 1 كما في test.py الرسمي
        masked_img = img_t * (1.0 - mask_t) + mask_t

        with torch.no_grad():
            pred = net(masked_img, mask_t)

        # دمج: خارج القناع أصلي + داخله مولَّد
        comp = (1.0 - mask_t) * img_t + mask_t * pred
        comp = torch.clamp(comp, -1.0, 1.0)
        comp = ((comp + 1.0) / 2.0 * 255.0)
        comp = comp.cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8)

        # ── إزالة الـ padding وإعادة الحجم الأصلي ──
        comp_img = Image.fromarray(comp).resize((sq, sq), Image.LANCZOS)
        pad_x = (sq - orig_w) // 2
        pad_y = (sq - orig_h) // 2
        comp = np.array(comp_img.crop((pad_x, pad_y, pad_x + orig_w, pad_y + orig_h)))

        return comp

    def inpaint_and_save(
        self,
        image_np: np.ndarray,
        mask_np:  np.ndarray,
        model_key: str,
        out_path:  str,
    ) -> np.ndarray:
        """تشغيل الاستدلال وحفظ النتيجة مباشرة"""
        result = self.inpaint(image_np, mask_np, model_key)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(result).save(out_path)
        print(f"  💾 محفوظة: {out_path}")
        return result
