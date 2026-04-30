# HSI Compressive Imaging

**高光谱压缩成像论文复现与算法研究平台**

复现、对比、并持续添加高光谱成像领域的前沿方法，包括光谱重建 (Spectral Recovery)、快照压缩成像 (CASSI)、光学滤波型 HSI、视频级压缩重建等任务。

---

## 📊 已复现方法

| 论文 | 会议 | 任务 | 指标 | 复现结果 | 状态 |
|------|------|------|------|----------|:--:|
| **MST++** | CVPRW 2022 | RGB→HSI 光谱重建 | PSNR 34.32 | **34.32** | ✅ |
| **SPECAT** | CVPR 2024 | 光学滤波 HSI 重建 | PSNR 40.3 | **40.39** | ✅ |
| **MST** | CVPR 2022 | CASSI 快照重建 | PSNR 35.18 | **34.80** | ✅ |
| **CST** | ECCV 2022 | CASSI 快照重建 | PSNR 35.85 | **35.44** | ✅ |
| **DAUHST** | NeurIPS 2022 | CASSI 快照重建 | PSNR 38.36 | — | 🔄 |
| **HDNet** | CVPR 2022 | CASSI 快照重建 | PSNR 34.97 | **34.66** | ✅ |
| **PG-SVRT** | CVPR 2026 | 视频级光谱重建 | — | ⏳ | 代码就绪 |

## 🗂 项目结构

```
├── paper_reproduction/
│   ├── CASSI/                       # Coded Aperture Snapshot Spectral Imaging
│   │   ├── MST-SCI/                 # MST/CST/DAUHST/HDNet/BiSRNet/TSA-Net... (15+)
│   │   ├── PADUT/                   # Pixel Adaptive Deep Unfolding Transformer (ICCV 2023)
│   │   ├── RDLUF_MixS2/             # Residual Degradation Learning Unfolding (CVPR 2023)
│   │   ├── S2-transformer-HSI/      # S²-Transformer (ECCV 2022)
│   │   ├── DPU-SCI/                 # Dual Prior Unfolding (CVPR 2024)
│   │   ├── SSR-SCI/                 # Spatial-Spectral Rectification (CVPR 2024)
│   │   └── GAP-net/                 # GAP-net for SCI (arXiv 2020)
│   ├── Spectral_Recovery/           # RGB → HSI Spectral Reconstruction
│   │   └── MST-plus-plus/           # NTIRE 2022 Challenge Winner
│   ├── Optical_Filter/              # Optical Filter-based HSI Systems
│   │   └── SPECAT/                  # CVPR 2024, Tsinghua University
│   └── Video_SCI/                   # Video-level Compressive Imaging
│       └── DynaSpec/                # PG-SVRT + DynaSpec Dataset (CVPR 2026)
├── downloaded_papers/               # 原始论文 PDF（已 gitignore）
└── reports/                         # 下载/复现报告
```

## 🔧 环境

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| GPU | RTX 4090 24GB |
| Conda | `mstpp` |

```bash
conda activate mstpp
```

## 🚀 快速开始

### MST++ — 光谱重建

```bash
cd paper_reproduction/Spectral_Recovery/MST-plus-plus
python run.py --mode eval    # 验证集评估
python run.py --mode train   # 从头训练
```

### SPECAT — 光学滤波 HSI

```bash
cd paper_reproduction/Optical_Filter/SPECAT
python test.py --data_root ./dataset/ --outf ./exp/
```

### MST-SCI 工具箱 — CASSI 对比

```bash
cd paper_reproduction/CASSI/MST-SCI/simulation/test_code
python test.py --template mst_s --method mst_s \
    --pretrained_model_path ./model_zoo/mst/mst_s.pth
```

## 📐 任务分类

| 类别 | 前向模型 | 输入 | 输出 | 代表方法 |
|------|----------|------|------|----------|
| **光谱重建** | RGB 相机响应 | RGB (3ch) | HSI (31ch) | MST++ |
| **CASSI** | 编码孔径 + 色散 | 2D 压缩测量 | HSI (28ch) | MST, CST, DAUHST |
| **光学滤波** | Fabry-Pérot 滤波 | 3D 编码测量 | HSI (28ch) | SPECAT |
| **视频 SCI** | 时变编码孔径 | 2D 多帧测量 | 视频 HSI | PG-SVRT |

## 📝 数据集

| 数据集 | 用途 | 大小 |
|--------|------|------|
| NTIRE 2022 ARAD_1K | 光谱重建训练/测试 | 950 场景 (~21GB) |
| CAVE (cave_1024_28) | CASSI 仿真训练 | 205 场景 |
| KAIST | CASSI 仿真/真实测试 | 30 场景 |
| TSA_simu_data | CASSI 仿真测试基准 | 10 场景 |
| DynaSpec | 视频级动态 HSI | 30 序列 |

> 数据集下载链接见各方法目录下的 README

## 🎯 计划

- [ ] SPECAT CASSI 适配版训练
- [ ] PG-SVRT 完整训练与评估
- [ ] 新算法开发与 SOTA 对比
- [ ] 统一 Benchmark 评测框架
- [ ] TensorBoard/WandB 训练监控

## 📄 License

各方法遵循其原始仓库的 License。本项目代码部分 MIT License。
