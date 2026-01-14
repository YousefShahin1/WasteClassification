# Waste Classification Model

A deep learning image classification model using Convolutional Neural Networks (CNN) to classify waste into different categories. Built with PyTorch.

## Project Structure

```
WasteClassification/
├── model.ipynb              # Original Jupyter notebook
├── requirements.txt         # Python dependencies
├── setup.sh                # Environment setup script
├── launch_gui.sh           # Launch GUI application
├── launch_complete.sh      # Launch API + GUI together
├── README.md               # This file
├── GUI_README.md           # GUI-specific documentation
├── models/                 # Trained models directory
│   ├── best_model.pth     # Best model based on test accuracy
│   ├── latest_model.pth   # Final model after all epochs
│   └── training_history.pth  # Training metrics and history
└── src/
    ├── processing.py       # Data loading and preprocessing
    ├── train.py           # Model training script
    ├── evaluate.py        # Model evaluation and visualization
    ├── predict.py         # Inference script for new images
    ├── api.py             # FastAPI REST API server
    └── gui.py             # Modern Tkinter GUI application
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
- fastapi >= 0.104.0 (for API server)
- uvicorn >= 0.24.0 (for API server)
- python-multipart >= 0.0.6 (for file uploads)
- requests >= 2.31.0 (for GUI API client)

## Usage

The project offers three ways to interact with the waste classification model:

1. **GUI Application** - Modern desktop interface for training, evaluation, and predictions
2. **REST API** - FastAPI server for integration with other applications
3. **Command Line** - Traditional CLI scripts for automation and scripting

### GUI Application

The easiest way to use the model is through the graphical interface.

**Launch the GUI:**

```bash
# Launch GUI and API together (recommended)
./launch_complete.sh

# Or launch GUI only
./launch_gui.sh

# Or manually
cd src
python gui.py
```

**Features:**
- 🎯 **Train Tab**: Configure and train models with custom hyperparameters
- 📊 **Evaluate Tab**: View training curves, confusion matrix, and model metrics
- 🔍 **Predict Tab**: Upload images for classification with confidence scores
- Real-time training progress with live charts
- Automatic API server management
- Model performance visualization

See [GUI_README.md](GUI_README.md) for detailed GUI documentation.

### REST API Server

The FastAPI server provides a REST interface for programmatic access.

**Start the API server:**

```bash
cd src
python api.py
```

The API will be available at `http://localhost:8000`

**API Documentation:**
- Interactive docs: http://localhost:8000/docs
- OpenAPI spec: http://localhost:8000/openapi.json

**Available Endpoints:**

**Health Check:**
```bash
GET /health
```
Returns service status, model information, and training status.

**Model Information:**
```bash
GET /model/info
```
Returns model architecture, classes, device, and accuracy.

**Single Image Prediction:**
```bash
POST /predict
Content-Type: multipart/form-data
Body: file=<image_file>

# Example with curl:
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@waste_image.jpg"
```

Response:
```json
{
  "predicted_class": "plastic",
  "confidence": 94.52,
  "all_probabilities": {
    "plastic": 94.52,
    "cardboard": 3.21,
    "paper": 1.15,
    ...
  },
  "processing_time": 0.0234
}
```

**Batch Image Prediction:**
```bash
POST /predict/batch
Content-Type: multipart/form-data
Body: files=<image_file1>, files=<image_file2>, ...

# Example with curl:
curl -X POST "http://localhost:8000/predict/batch" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg" \
     -F "files=@image3.jpg"
```

Response:
```json
{
  "predictions": [
    {
      "filename": "image1.jpg",
      "predicted_class": "glass",
      "confidence": 97.83,
      "top_3_predictions": {
        "glass": 97.83,
        "metal": 1.45,
        "plastic": 0.52
      }
    },
    ...
  ],
  "total_images": 3,
  "processing_time": 0.0678
}
```

**Start Training:**
```bash
POST /train
Content-Type: application/json
Body: {
  "epochs": 45,
  "batch_size": 32,
  "learning_rate": 0.001,
  "hidden_units": 64,
  "resume": false
}

# Example with curl:
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{"epochs": 20, "batch_size": 32, "learning_rate": 0.001}'
```

**Get Training Status:**
```bash
GET /train/status
```

**Stop Training:**
```bash
POST /train/stop
```

**Using the API with Python:**

```python
import requests

# Single prediction
with open('waste_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    result = response.json()
    print(f"Predicted: {result['predicted_class']} ({result['confidence']:.2f}%)")

# Batch prediction
files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb')),
    ('files', open('image3.jpg', 'rb'))
]
response = requests.post('http://localhost:8000/predict/batch', files=files)
results = response.json()

for pred in results['predictions']:
    print(f"{pred['filename']}: {pred['predicted_class']} ({pred['confidence']:.2f}%)")

# Start training
training_config = {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
    "hidden_units": 64
}
response = requests.post('http://localhost:8000/train', json=training_config)
print(response.json())

# Check training status
status = requests.get('http://localhost:8000/train/status').json()
print(f"Status: {status['status']}, Epoch: {status['current_epoch']}/{status['total_epochs']}")
```

### Command Line Interface

#### 1. Training the Model

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

#### 2. Evaluating the Model

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

#### 3. Making Predictions

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

## Integration Examples

### Integrate API into Web Application

```javascript
// JavaScript/Node.js example
async function classifyWaste(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  console.log(`Predicted: ${result.predicted_class}`);
  console.log(`Confidence: ${result.confidence}%`);
  return result;
}
```

### Integrate into Python Application

```python
from src.predict import WasteClassifier
import torch

# Load model
history = torch.load('models/training_history.pth')
classifier = WasteClassifier(
    model_path='models/best_model.pth',
    classes=history['classes']
)

# Use in your application
def process_waste_image(image_path):
    predicted_class, confidence, all_probs = classifier.predict(image_path)
    
    if confidence > 80:
        print(f"High confidence: {predicted_class}")
        return predicted_class
    else:
        print(f"Low confidence: {confidence:.2f}%. Manual review needed.")
        return None
```

### Use as Microservice

Deploy the API in a Docker container:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
WORKDIR /app/src

EXPOSE 8000
CMD ["python", "api.py"]
```

```bash
# Build and run
docker build -t waste-classifier .
docker run -p 8000:8000 waste-classifier
```

## Troubleshooting

### GUI Issues

**API Server Not Starting:**
- Check if port 8000 is already in use
- Manually start the API: `cd src && python api.py`
- Check console for error messages

**Import Errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate virtual environment before running

### API Issues

**Model Not Loaded:**
- Ensure `models/best_model.pth` exists
- Run training first: `cd src && python train.py`
- Check file permissions

**CORS Errors (Web Integration):**
Add CORS middleware to `api.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Timeout Errors:**
- Increase timeout in GUI or client code
- Use CPU if GPU is causing issues
- Reduce batch size for batch predictions

## License

This project is for educational purposes.

## Author

Created as part of a waste classification project using deep learning.
