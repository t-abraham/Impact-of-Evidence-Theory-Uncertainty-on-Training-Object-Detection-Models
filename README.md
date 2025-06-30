# Impact of Evidence Theory Uncertainty on Training Object Detection Models

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A PyTorch-based implementation of an evidence-theoretic ensemble for object detection that quantifies and leverages uncertainty during inference. This repository accompanies the paper **“Impact of Evidence Theory Uncertainty on Training Object Detection Models”**.

---

## 🚀 Features

- **Ensemble Multiple Detectors**  
  Fuse predictions from one or more pretrained detectors (e.g. Faster R-CNN, YOLOv5, Detectron2)  
- **Evidence-Theoretic Fusion**  
  • Confidence-based filtering & IoU grouping  
  • Dempster–Shafer combination to produce a fused label and conflict measure (uncertainty)  
- **Bounding-Box Aggregation**  
  - **Max-Score**: choose the highest-scoring box  
  - **Evidence-Weighted**: average coordinates weighted by normalized evidence  
- **Uncertainty Quantification**  
  Retain per-box conflict score (K) for downstream filtering or analysis  
- **Configurable Thresholds** for confidence and IoU  
