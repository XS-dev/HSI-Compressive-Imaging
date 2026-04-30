# paper_reproduction

## 📂 分类

| 目录 | 任务类型 | 方法 |
|------|----------|------|
| **CASSI/** | Coded Aperture Snapshot Spectral Imaging | MST, CST, DAUHST, PADUT, RDLUF, S²-Transformer, DPU, SSR, GAP-Net, HDNet, BiSRNet, DGSMP, TSA-Net, ADMM-Net, λ-Net, BIRNAT |
| **Spectral_Recovery/** | RGB→HSI Spectral Reconstruction | MST++ (NTIRE 2022 Winner) |
| **Optical_Filter/** | Optical Filter-based HSI | SPECAT (CVPR 2024, 40.3dB PSNR) |
| **Video_SCI/** | Video-level Compressive Imaging | PG-SVRT + DynaSpec (CVPR 2026) |

## ✅ 已复现

| 论文 | 指标 | 结果 |
|------|------|------|
| MST++ (CVPRW 2022) | PSNR 34.32 | **34.32** ✅ |
| SPECAT (CVPR 2024) | PSNR 40.3 | **40.39** ✅ |
| MST-SCI Benchmark | 15+ 方法 | 9/18 跑通 |

## 🔧 环境

- PyTorch 2.5.1 + CUDA 12.1
- RTX 4090 24GB
- Conda env: `mstpp`
