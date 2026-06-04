# HyMPOR — Hybrid Multi-Stage Pipeline for Restoring Damaged and Occluded Old Photographs

![Status](https://img.shields.io/badge/Status-Under%20Review-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/proanas/HyMPOR/blob/main/HyMPOR.ipynb)

## 📖 Abstract

Old photographs suffer from complex, co-occurring degradations — scratches, stains, fading, occlusions, and loss of fine detail — that no single restoration model can adequately address. We present **HyMPOR**, a hybrid multi-stage pipeline that sequentially applies state-of-the-art models for scratch detection and removal, face enhancement, object removal via segmentation-guided inpainting, and colorization. Evaluated on both synthetic and real damaged photographs, HyMPOR achieves superior quantitative scores and perceptually compelling results across all degradation types.

---
![HyMPOR Pipeline](https://github.com/proanas/HyMPOR/raw/main/results/pipeline.PNG)
---
## 🚀 Quick Start

Everything runs in Google Colab — **no manual setup, no separate weight downloads.**

1. Click the **Open in Colab** badge above.
2. Set the runtime to GPU: **Runtime ▸ Change runtime type ▸ T4 GPU**.
3. Run the **first cell**. It automatically downloads the complete project (code + all pre-trained weights) from the latest [Release](https://github.com/proanas/HyMPOR/releases/latest) and extracts it.
4. Run the remaining cells in order to restore your photo.

> The full project (~10 GB) is split into parts in the Release and reassembled automatically by the first cell. The first run takes 10–20 minutes depending on connection speed; afterwards everything is cached for the session.

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
│  Stage 2 — Object / Occlusion       │  ← SAM2 + AOT-GAN
│  Segmentation → Inpainting          │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Stage 3 — Face Enhancement         │  ← GFPGAN
│  Blind face restoration via GAN     │
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

## 🧩 Modules

| # | Module | Purpose | Official Repo |
| --- | --- | --- | --- |
| 1 | **Bringing Old Photos Back to Life** | Scratch detection & global restoration | [microsoft/Bringing-Old-Photos-Back-to-Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life) |
| 2 | **SAM2** | Object segmentation & mask generation | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| 3 | **AOT-GAN** | Inpainting of segmented regions | [researchmm/AOT-GAN-for-Inpainting](https://github.com/researchmm/AOT-GAN-for-Inpainting) |
| 4 | **GFPGAN** | Blind face enhancement | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) |
| 5 | **DeOldify** | Automatic colorization | [jantic/DeOldify](https://github.com/jantic/DeOldify) |

> ✅ All pre-trained weights are bundled in the [Release](https://github.com/proanas/HyMPOR/releases/latest) and downloaded automatically by the notebook. There is nothing to download or place manually.

---

## 📂 Repository Structure

```
HyMPOR/
├── HyMPOR_GitHub.ipynb         # Main Colab notebook — start here
├── pipeline/                   # Core pipeline code
│   ├── run.py                  # Orchestrator (SAM2 → selector → AOT-GAN)
│   ├── config.py               # Central configuration
│   ├── segmentation.py         # SAM2 segmentation
│   ├── inpainting.py           # AOT-GAN inpainting
│   └── selector.py             # Content-aware weight selector
│
└── modules/
    ├── Bringing_Old_Photos_Back_to_Life/   # Stage 1
    ├── SAM2/                               # Stage 2a
    ├── AOT_GAN/                            # Stage 2b
    ├── GFPGAN/                             # Stage 3
    └── DeOldify/                           # Stage 4
```

---

## 📊 Evaluation Metrics

| Metric | Type | Range | Better |
| --- | --- | --- | --- |
| **PSNR** | Pixel-level fidelity | dB | ↑ Higher |
| **SSIM** | Structural similarity | [0, 1] | ↑ Higher |
| **LPIPS** | Perceptual quality | [0, 1] | ↓ Lower |
| **BRISQUE** | No-reference quality (spatial) | [0, 100] | ↓ Lower |
| **MANIQA** | No-reference quality (transformer) | [0, 1] | ↑ Higher |

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{HyMPOR2026,
  title   = {A Hybrid Multi-Stage Pipeline for Restoring Damaged and Occluded Old Photographs},
  author  = {Anas Hameed Ali},
  journal = {Under Review},
  year    = {2026}
}
```

---

## 🙏 Acknowledgements

This work builds upon several outstanding open-source projects:

- [Bringing Old Photos Back to Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life) — Microsoft Research
- [SAM2](https://github.com/facebookresearch/sam2) — Meta AI Research
- [AOT-GAN](https://github.com/researchmm/AOT-GAN-for-Inpainting) — Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang and Guo, Baining
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — Tencent ARC
- [DeOldify](https://github.com/jantic/DeOldify) — Jason Antic

---

## 📬 Contact

For questions or collaborations, please open an issue.

## 📜 License

Released under the [MIT License](LICENSE).
