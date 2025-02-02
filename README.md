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
- **Train, Validate, and Test Split**: 70% : 20% : 10% 
- **Final Accuracy**: 92.3% on test set

The trained model (`pneumonia_model.pth`) is included in the `model/` directory.

## Installation

### Prerequisites
- Python 3.12+
- Pipenv
- Docker (optional)

### Setup
1. Clone the repository:
``bash
  git clone https://github.com/yourusername/pneumonia-detection.git
  cd pneumonia-detection``

2. Install dependencies using Pipenv:
  ``bash
  pip install pipenv
  pipenv install``

3. Activate virtual environment:
  ``bash
  pipenv shell``

## Docker Setup
1. Build the Docker image:
   ``bash
    docker build -t pneumonia-detector .``
   
3. Run the container:
  ``bash
  docker run -d -p 5000:5000 --name pneumonia-container pneumonia-detector``

3. Access the web interface at:  
`http://localhost:5000`

## Run directly from app.py
1. Start the Flask application:
  ``bash
  python app/app.py``

## Usage
1. Open web browser and navigate to:
`http://localhost:5000`

3. Upload a chest X-ray image (JPEG/PNG format)

4. Get instant prediction result (Pneumonia/Normal)

## Project Structure
Capstone/
├── app/
│ ├── app.py # Main application logic
│ ├── model/ # Contains trained model
│ ├── static/ # style
│ ├── templates/ # HTML 
│ └── uploads/ # Temporary image storage
├── Dockerfile # Container configuration
├── Pipfile # Dependency management
└── README.md # This documentation

## Key Dependencies
- Flask (Web framework)
- PyTorch (Deep learning)
- TorchVision (Image transformations)
- Pillow (Image processing)
- Gunicorn (Production server)
