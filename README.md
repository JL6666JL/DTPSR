# DTPSR: Disentangled Textual Priors for Diffusion-based Image Super-Resolution

Official implementation of **DTPSR** (Disentangled Textual Priors for Diffusion-based Image Super-Resolution), a diffusion-based SR framework that injects **disentangled textual priors** along:

- **Spatial hierarchy**: *global* vs. *local*
- **Frequency semantics**: *low-frequency* (structure/layout/color) vs. *high-frequency* (texture/edges/details)

DTPSR also introduces **multi-branch classifier-free guidance (CFG)** with frequency-aware negative prompts, and releases **DisText-SR**, a structured dataset with ~95k image–text groups.  
(See paper for full details.) :

---

## 🔥 Highlights

- **Disentangled textual priors**: global caption + per-segment LF/HF descriptions, enabling controllable, interpretable guidance. 
- **Progressive semantic injection** via specialized cross-attention modules:
  - GTCA (Global Text Cross-Attention)
  - LFCA (Local Low-Frequency Cross-Attention)
  - HFCA (Local High-Frequency Cross-Attention)
  - LRCA (LR Feature Cross-Attention for identity anchoring) :
- **DisText-SR dataset** (~95k): global + region-wise LF/HF descriptions generated using panoptic segmentation + MLLM prompting.

---

## 🧠 Method Overview

Given an LR image, DTPSR:

1. Encodes LR to latent space (VAE encoder)
2. Runs diffusion denoising with **sequential semantic guidance**:
   - **Global prior** guides scene-level layout (GTCA)
   - **Local LF priors** refine object-level structure (LFCA)
   - **Local HF priors** enhance textures and edges (HFCA)
3. Fuses LR visual features through **LRCA** to maintain image identity and consistency 

---

## 📦 DisText-SR Dataset

**DisText-SR** contains ~95,000 groups of structured annotations:

- Global description (`c_g`)
- For each segment `S_i` (from panoptic segmentation):
  - Low-frequency text (`c_lf^(i)`): shape/size/colors/orientation (no fine details)
  - High-frequency text (`c_hf^(i)`): texture/material/edges/subtle details

Texts are encoded by a frozen CLIP text encoder to form embeddings for guidance. 

> TODO: Provide dataset download link / license / checksum.

---
