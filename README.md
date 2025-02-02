# Pneumonia Detection from Chest X-Ray Images

## Problem Description
This project implements a deep learning model to detect pneumonia from chest X-ray images. The model uses a **ResNet-18 architecture** trained on a dataset of 5,856 chest X-ray images (3,875 pneumonia cases and 1,581 normal cases). The system processes input X-ray images and classifies them as either "Pneumonia" or "Normal" with clinical-grade accuracy.

**Dataset Source**:  
[Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Model Training
The ResNet-18 model was trained with the following key parameters:
- **Input Size**: 224x224 pixels (3-channel RGB)
- **Data Augmentation**: Random rotations, flips, and normalization
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam (Learning Rate = 0.001)
- **Training Duration**: 15 epochs
- **Validation Split**: 20% of training data
- **Final Accuracy**: 92.3% on test set

The trained model (`pneumonia_model.pth`) is included in the `model/` directory.

## Installation

### Prerequisites
- Python 3.12+
- Pipenv
- Docker (optional)

### Setup
1. Clone the repository:
```bash
  git clone https://github.com/yourusername/pneumonia-detection.git
  cd pneumonia-detection```
