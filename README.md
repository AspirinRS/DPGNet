# DPGNet: Closed-Loop Remote Sensing Image Change Detection via Active Difference-Prior Guidance

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat)](LICENSE)

</div>

---

PyTorch implementation of **DPGNet** for high-resolution remote sensing image change detection.

DPGNet formulates difference representations as active priors and performs progressive change reasoning through three key components:
- **SPRM**: Statistical Prior-Guided Rectification Module
- **DRP**: Difference Retrospective Probe
- **SPGA**: Semantic Prior-Guided Aggregation Module

<p align="center">
  <img src="images/Overall architecture.jpg" width="90%">
</p>

---

## 🛠️ Installation

### Requirements

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch>=2.0.1 torchvision>=0.15.2 timm>=1.0.23 einops>=0.8.1 numpy>=1.24.4 Pillow>=10.4.0 tqdm>=4.67.1 thop
```

> Tested with Python 3.8, PyTorch 2.0, and CUDA 11.x.

---

## 📁 Dataset Preparation

### Supported Datasets

| Dataset | Download |
|---------|----------|
| LEVIR-CD | [Link](https://justchenhao.github.io/LEVIR/) |
| WHU-CD | [Link](https://study.rsgis.whu.edu.cn/pages/download/) |
| SYSU-CD | [Link](https://github.com/liumency/SYSU-CD) |

### Directory Structure

Please download the datasets from their official sources and organize them as follows:

```text
Data_Directory/
├── LEVIR-CD/
│   ├── train/
│   │   ├── A/
│   │   ├── B/
│   │   └── label/
│   ├── val/
│   └── test/
├── WHU-CD/
└── SYSU-CD/
```

> ⚠️ **Note**: Labels should be binary PNG images with pixel value **255** = change, **0** = no change.

---

## 🚀 Quick Start

### Training

Run training with the default paper setting:

```bash
python trainval.py
```

### Evaluation

Run evaluation on the test set:

```bash
python test.py
```

Please configure the dataset path, checkpoint path, and other options in `option.py` or via command-line arguments according to your setup.

---

## 📂 Project Structure

```text
DPGNet/
├── model/
│   ├── network.py              
│   └── block/
│       ├── sprm.py              
│       ├── sprm_components.py   
│       ├── spga.py            
│       ├── drp.py              
│       └── heads.py            
├── data/
│   ├── cd_dataset.py          
│   └── transform.py            
├── util/
│   └── metric_tool.py         
├── trainval.py                
├── test.py                    
└── option.py                   
```

---

## 📊 Experimental Results

### Visual Comparison

<p align="center">
  <img src="images/Visual comparison results on LEVIR-CD.jpg" width="100%">
  <br>
  <em>Visual comparison results on LEVIR-CD dataset</em>
</p>

<p align="center">
  <img src="images/Visual comparison results on WHU-CD.jpg" width="100%">
  <br>
  <em>Visual comparison results on WHU-CD dataset</em>
</p>

<p align="center">
  <img src="images/Visual comparison results on SYSU-CD.jpg" width="100%">
  <br>
  <em>Visual comparison results on SYSU-CD dataset</em>
</p>

---

## 🙏 Acknowledgments

We thank the open-source remote sensing community for providing valuable benchmark datasets and code resources.

---

## 📄 License

This project is released under the Apache 2.0 License.

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

Made with ❤️ for the Remote Sensing Community

</div>