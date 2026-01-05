# Waste Classification Model

A deep learning image classification model using Convolutional Neural Networks (CNN) to classify waste into different categories. Built with PyTorch.

## Project Structure

```
WasteClassification/
├── model.ipynb              # Original Jupyter notebook
├── requirements.txt         # Python dependencies
├── setup.sh                # Environment setup script
├── README.md               # This file
├── models/                 # Trained models directory
│   ├── best_model.pth     # Best model based on test accuracy
│   ├── latest_model.pth   # Final model after all epochs
│   └── training_history.pth  # Training metrics and history
└── src/
    ├── processing.py       # Data loading and preprocessing
    ├── train.py           # Model training script
    ├── evaluate.py        # Model evaluation and visualization
    └── predict.py         # Inference script for new images
```

## Setup

### Initial Setup

Run the setup script to create a virtual environment and install all dependencies:

```bash
./setup.sh
```

This will:
- Create a virtual environment in the `venv` folder
- Activate the environment
- Upgrade pip
- Install all required dependencies from `requirements.txt`

### Using the Environment Later

**To activate the environment:**
```bash
source venv/bin/activate
```

**To deactivate when done:**
```bash
deactivate
```

### Manual Installation

If you prefer to set up manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- kagglehub >= 0.2.0
- matplotlib >= 3.7.0
- scikit-learn >= 1.3.0
- Pillow >= 10.0.0
- tqdm >= 4.65.0
- numpy >= 1.24.0

## Usage

### 1. Training the Model

Train the model on the garbage classification dataset:

```bash
cd src
python train.py
```

**With custom parameters:**

```bash
# Basic training with custom parameters
python train.py --epochs 20 --batch-size 32 --lr 0.001

# Advanced options
python train.py --epochs 30 --batch-size 64 --lr 0.0005 --hidden-units 128

# All available options
python train.py --epochs 20 \
                --batch-size 32 \
                --lr 0.001 \
                --hidden-units 64 \
                --model-path my_model.pth \
                --history-path my_history.pth \
                --seed 42
```

**Available arguments:**
- `--epochs`: Number of training epochs (default: 45)
- `--batch-size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--hidden-units`: Number of hidden units in first conv layer (default: 64)
- `--save-dir`: Directory to save models (default: ../models)
- `--history-path`: Path to save training history (default: ../models/training_history.pth)
- `--seed`: Random seed for reproducibility (default: 42)

This will:
- Download the dataset from Kaggle
- Prepare train/test splits with stratification
- Train the CNN model
- **Automatically save the best model** (based on test accuracy) to `models/best_model.pth`
- Save the latest model after all epochs to `models/latest_model.pth`
- Save training history to `models/training_history.pth`

### 2. Evaluating the Model

Evaluate the trained model and generate visualizations:

```bash
cd src
python evaluate.py
```

This will:
- Load the trained model
- Generate accuracy and loss curves
- Create a confusion matrix
- Print classification report
- Visualize CNN filters
- Generate activation maps

**Output files:**
- `accuracy_curve.png` - Training and test accuracy over epochs
- `loss_curve.png` - Training and test loss over epochs
- `confusion_matrix.png` - Confusion matrix on test set
- `conv_filters.png` - Learned filters from first convolutional layer
- `activation_maps_layer_*.png` - Activation maps for each layer

### 3. Making Predictions

Use the trained model to predict on new images:

**Command-line usage:**

```bash
cd src

# Predict on a single image
python predict.py --image path/to/image.jpg

# Predict on multiple images
python predict.py --images img1.jpg img2.jpg img3.jpg

# Show top 5 predictions
python predict.py --image path/to/image.jpg --top-k 5

# Don't display images (for headless environments)
python predict.py --image path/to/image.jpg --no-display

# Use custom model and history files
python predict.py --image path/to/image.jpg \
                  --model my_model.pth \
                  --history my_history.pth
```

**Available arguments:**
- `--image`: Path to a single image file to classify
- `--images`: Paths to multiple images to classify
- `--model`: Path to trained model (default: ../models/best_model.pth)
- `--history`: Path to training history file (default: ../models/training_history.pth)
- `--top-k`: Number of top predictions to show (default: 3)
- `--no-display`: Do not display images (useful for servers)

**Using the prediction API in Python:**

```python
from predict import WasteClassifier
import torch

# Load classes from history
history = torch.load('../models/training_history.pth')
classes = history['classes']

# Initialize classifier (uses best model by default)
classifier = WasteClassifier(
    model_path='../models/best_model.pth',
    classes=classes,
    device='cpu'  # or 'cuda' if available
)

# Predict on a single image
predicted_class, confidence, all_probs = classifier.predict('path/to/image.jpg')

# Predict on multiple images
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = classifier.predict_batch(image_paths, top_k=3)
```

## Model Architecture

The `WasteClassifierModelV1` uses a CNN architecture with:
- 3 convolutional blocks with increasing feature maps (64 → 128 → 256)
- ReLU activations
- Max pooling layers
- Global average pooling
- Fully connected classifier with dropout (0.5)
- Output layer for multi-class classification

## Dataset

The model uses the [Garbage Dataset Classification](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification) from Kaggle, which is automatically downloaded when running the training script.

## Training Details

- **Input size:** 224x224 RGB images
- **Batch size:** 32
- **Epochs:** 45
- **Optimizer:** Adam (lr=0.001)
- **Loss function:** CrossEntropyLoss
- **Learning rate scheduler:** ReduceLROnPlateau (patience=3, factor=0.5)
- **Data augmentation:** Random resized crop, horizontal flip
- **Normalization:** ImageNet statistics

## GPU Support

The code automatically detects and uses GPU (CUDA) if available, otherwise falls back to CPU.

To check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## License

This project is for educational purposes.

## Author

Created as part of a waste classification project using deep learning.
