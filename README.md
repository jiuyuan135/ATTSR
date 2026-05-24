# High-Resolution Optical Satellite Image Guided DEM Super-Resolution via Topographic-Aware Transformer

<p align="center">
  <a href="https://ieeexplore.ieee.org/Xplore/home.jsp"><img src="https://img.shields.io/badge/Paper-TGRS_2026-blue"></a>
  <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg"></a>
  <a href="https://github.com/yourusername/yourrepo/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%3E%3D1.10-orange"></a>
</p>

This repository is the official PyTorch implementation of the paper **"High-Resolution Optical Satellite Image Guided DEM Super-Resolution via Topographic-Aware Transformer"** (Accepted by *IEEE Transactions on Geoscience and Remote Sensing*, 2026).


## 🛠 Prerequisites

* Operating System: Linux / Windows
* Python 3.8+


**Environment Setup:**
```bash

conda create -n SR python=3.8
conda activate SR
pip install -r requirements.txt
```

**Data Preparation:**
```bash
data/
├── train/
│   ├── img/
│   └── dem/
└── test/
    ├── img/
    └── dem/
```

Usage:
Training
```bash
python train.py --data_root ./data --save_dir ./checkpoints
```

Prediction
```bash
python infer.py --data_root ./data --model_path ./checkpoints/ATTSR.pth --save_dir ./results
```

📝 Citation
```bash
@ARTICLE{11455240,
  author={Tang, Yubin and Yan, Enping and Xiong, Yujiu and Jiang, Jiawei and Sun, Hua and Mo, Dengkui},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={High-Resolution Optical Satellite Image-Guided DEM Super-Resolution via Topographic-Aware Transformer}, 
  year={2026},
  volume={64},
  number={},
  pages={1-17},
  keywords={Transformers;Image reconstruction;Spatial resolution;Satellite images;Superresolution;Optical sensors;Optical imaging;Interpolation;Convolution;Analytical models;Digital elevation model (DEM);super-resolution (SR);Swin Transformer;topographic-aware attention (TAA);transformer-based topography neural network (TTSR)},
  doi={10.1109/TGRS.2026.3677203}}

```
