"""
segmentation.py — واجهة SAM2 لتوليد الأقنعة
يدعم وضعين: نقاط من رسم المستخدم أو تلقائي
"""
import sys
import os
import numpy as np
import torch
from PIL import Image

from . import config


def _ensure_sam2_path():
    """إضافة مسار SAM2 للـ Python path"""
    if config.SAM2_ROOT not in sys.path:
        sys.path.insert(0, config.SAM2_ROOT)


class SAM2Segmentor:
    """
    واجهة موحدة لتشغيل SAM2
    تدعم: نقاط يدوية، صندوق bounding box، تلقائي
    """

    def __init__(self, device=None):
        _ensure_sam2_path()
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        assert os.path.exists(config.SAM2_CHECKPOINT), \
            f"❌ أوزان SAM2 غير موجودة: {config.SAM2_CHECKPOINT}"

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True

        model = build_sam2(
            config.SAM2_CONFIG,
            config.SAM2_CHECKPOINT,
            device=self.device,
            apply_postprocessing=False,
        )
        self.predictor = SAM2ImagePredictor(model)
        self._current_image = None
        print(f"✅ SAM2 جاهز على: {self.device}")

    def set_image(self, image_np: np.ndarray):
        """تحميل الصورة في SAM2 (يجب استدعاؤه قبل predict)"""
        self.predictor.set_image(image_np)
        self._current_image = image_np

    def predict_from_drawn_mask(self, drawn_mask_np: np.ndarray) -> np.ndarray:
        """
        توليد قناع SAM2 من منطقة رُسمت يدوياً

        Args:
            drawn_mask_np: قناع الرسم اليدوي [H, W] uint8

        Returns:
            قناع SAM2 [H, W] uint8
        """
        assert self._current_image is not None, "❌ استدعِ set_image() أولاً"

        # استخراج نقاط من منطقة الرسم
        ys, xs = np.where(drawn_mask_np > 30)
        if len(xs) == 0:
            raise ValueError("⚠️  القناع المرسوم فارغ")

        n_pts  = min(10, len(xs))
        idx    = np.linspace(0, len(xs) - 1, n_pts, dtype=int)
        points = np.stack([xs[idx], ys[idx]], axis=1)
        labels = np.ones(len(points), dtype=int)

        masks, scores, _ = self.predictor.predict(
            point_coords   = points,
            point_labels   = labels,
            multimask_output = True,
        )

        best = masks[np.argmax(scores)]
        return (best * 255).astype(np.uint8)

    def predict_from_points(
        self,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        توليد قناع من نقاط محددة يدوياً

        Args:
            points: مصفوفة [N, 2] إحداثيات (x, y)
            labels: مصفوفة [N]    1=foreground, 0=background

        Returns:
            قناع SAM2 [H, W] uint8
        """
        assert self._current_image is not None, "❌ استدعِ set_image() أولاً"

        masks, scores, _ = self.predictor.predict(
            point_coords     = points,
            point_labels     = labels,
            multimask_output = True,
        )
        best = masks[np.argmax(scores)]
        return (best * 255).astype(np.uint8)

    def predict_from_box(self, box: list) -> np.ndarray:
        """
        توليد قناع من bounding box

        Args:
            box: [x1, y1, x2, y2]

        Returns:
            قناع SAM2 [H, W] uint8
        """
        assert self._current_image is not None, "❌ استدعِ set_image() أولاً"

        masks, scores, _ = self.predictor.predict(
            box              = np.array(box),
            multimask_output = True,
        )
        best = masks[np.argmax(scores)]
        return (best * 255).astype(np.uint8)
