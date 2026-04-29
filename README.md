# HyMPOR — Hybrid Multi-Stage Pipeline for Restoring Damaged and Occluded Old Photographs

<p align="center">
  <img src="https://img.shields.io/badge/Status-Under%20Review-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

> **Note:** This repository is currently **private** and will be made public upon paper acceptance.

---
  Colab file :  https://github.com/proanas/HyMPOR/blob/main/HyMPOR_GitHub.ipynb
## 📖 Abstract

Old photographs suffer from complex, co-occurring degradations — scratches, stains, fading, occlusions, and loss of fine detail — that no single restoration model can adequately address. We present **HyMPOR**, a hybrid multi-stage pipeline that sequentially applies state-of-the-art models for scratch detection and removal, face enhancement, object removal via segmentation-guided inpainting, and colorization. Evaluated on both synthetic and real damaged photographs, HyMPOR achieves superior quantitative scores and perceptually compelling results across all degradation types.

---

## 🏗️ Pipeline Architecture

```
Input (Damaged Old Photo)
        │
        ▼
┌─────────────────────────────────────┐
│  Stage 1 — Scratch & Damage Removal │  ← Bringing Old Photos Back to Life
│  Global restoration + face detection│
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Stage 2 — Face Enhancement         │  ← GFPGAN
│  Blind face restoration via GAN     │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Stage 3 — Object / Occlusion       │  ← SAM2 + AOT-GAN
│  Segmentation → Inpainting          │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Stage 4 — Colorization             │  ← DeOldify
│  Self-attention GAN colorization    │
└─────────────────┬───────────────────┘
                  │
                  ▼
        Output (Restored Photo)
```

---

## 🧩 Modules & Pre-trained Weights

| # | Module | Purpose | Official Repo | Weights |
|---|--------|---------|--------------|---------|
| 1 | **Bringing Old Photos Back to Life** | Scratch detection & global restoration | [microsoft/Bringing-Old-Photos-Back-to-Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life) | [Download](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life#model-zoo) |
| 2 | **GFPGAN** | Blind face enhancement | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | [GFPGANv1.4.pth](https://github.com/TencentARC/GFPGAN/releases/tag/v1.3.4) |
| 3 | **SAM2** | Object segmentation & mask generation | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | [sam2.1_hiera_large.pt](https://github.com/facebookresearch/sam2#model-description) |
| 4 | **AOT-GAN** | Inpainting of segmented regions | [researchmm/AOT-GAN-for-Inpainting](https://github.com/researchmm/AOT-GAN-for-Inpainting) | [G0000000.pt](https://github.com/researchmm/AOT-GAN-for-Inpainting#getting-started) |
| 5 | **DeOldify** | Automatic colorization | [jantic/DeOldify](https://github.com/jantic/DeOldify) | [ColorizeArtistic_gen.pth](https://github.com/jantic/DeOldify#pretrained-weights) |

> ⚠️ **Important:** Pre-trained weights are **not included** in this repository due to file size constraints. Please download each model's weights from the official links above and place them in the corresponding `modules/<ModelName>/models/` directory.

---

## 📂 Repository Structure

```
HyMPOR/
├── pipeline/                   # Core pipeline code
│   ├── run.py                  # Main entry point
│   ├── config.py               # Pipeline configuration
│   ├── segmentation.py         # SAM2 segmentation module
│   ├── inpainting.py           # AOT-GAN inpainting module
│   └── selector.py             # Stage selector logic
│
├── modules/
    ├── Bringing_Old_Photos_Back_to_Life/   # Stage 1
    ├── GFPGAN/                             # Stage 2
    ├── SAM2/                               # Stage 3a
    ├── AOT_GAN/                            # Stage 3b
    └── DeOldify/                           # Stage 4


 

---

## ⚙️ Installation

```bash
git clone https://github.com/proanas/HyMPOR.git
cd HyMPOR
pip install -r modules/Bringing_Old_Photos_Back_to_Life/requirements.txt
pip install -r modules/GFPGAN/requirements.txt
pip install -r modules/DeOldify/requirements.txt
pip install -r modules/AOT_GAN/environment.yml
```

---

## 🚀 Usage

```python
from pipeline.run import HyMPORPipeline

pipeline = HyMPORPipeline(config='configs/default.yaml')
result = pipeline.run(input_image='path/to/old_photo.jpg')
result.save('path/to/output.jpg')
```

Or run from the notebook:
```
pipeline/hybrid_pipeline.ipynb
```

---

## 📊 Evaluation Metrics

We evaluate using four complementary metrics:

| Metric | Type | Range | Better |
|--------|------|--------|--------|
| **PSNR** | Pixel-level fidelity | dB | ↑ Higher |
| **SSIM** | Structural similarity | [0, 1] | ↑ Higher |
| **LPIPS** | Perceptual quality | [0, 1] | ↓ Lower |
| **BRISQUE** | No-reference quality (spatial) | [0, 100] | ↓ Lower |
| **MANIQA** | No-reference quality (transformer) | [0, 1] | ↑ Higher |

---

## 🖼️ Qualitative Results

*Sample results will be added upon paper publication.*

| Input | Stage 1 | Stage 2 | Stage 3 | Final Output |
|-------|---------|---------|---------|--------------|
| ![](results/) | | | | |

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{HyMPOR2025,
  title   = {A Hybrid Multi-Stage Pipeline for Restoring Damaged and Occluded Old Photographs},
  author  = {Anonymous},
  journal = {Under Review},
  year    = {2025}
}
```

---

## 🙏 Acknowledgements

This work builds upon several outstanding open-source projects:

- [Bringing Old Photos Back to Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life) — Microsoft Research
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — Tencent ARC
- [SAM2](https://github.com/facebookresearch/sam2) — Meta AI Research
- [AOT-GAN](https://github.com/researchmm/AOT-GAN-for-Inpainting) — researchmm
- [DeOldify](https://github.com/jantic/DeOldify) — Jason Antic

---

## 📬 Contact

For questions or collaborations, please open an issue or contact via GitHub.
بشكل كاف
