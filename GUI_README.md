# Waste Classification GUI

Modern desktop application for training, evaluating, and using the waste classification model.

## Architecture

The GUI uses the **FastAPI backend for all operations**:
- **Training**: API endpoint `/train` with background task processing
- **Predictions**: API endpoint `/predict`
- **Model Info**: API endpoint `/model/info`
- **Training Status**: API endpoint `/train/status` for real-time monitoring

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Start the API server** (required for all features):
```bash
cd src
python api.py
```

3. **Launch the GUI** (in a separate terminal):
```bash
cd src
python gui.py
```

Or use the launcher:
```bash
bash launch_complete.sh
```

## Features

### 🎯 Train Model Tab
- Configure training parameters:
  - Epochs (default: 45)
  - Batch size (default: 32)
  - Learning rate (default: 0.001)
  - Hidden units (default: 64)
- Resume training from checkpoint
- Real-time training status monitoring via API
- Progress tracking with status updates
- Stop training mid-process

### 📊 Evaluate Model Tab
- View API/model information
- Plot accuracy curves (from training history)
- Plot loss curves (from training history)
- Generate confusion matrix
- Run full evaluation

### 🔍 Predict Tab
- Select and preview images
- Real-time predictions using API
- Confidence scores
- Class probability distribution

## Usage

1. **First Time Setup:**
   - GUI will check if API server is running
   - Offers to start it automatically
   - **API is required for all features (training, prediction, evaluation)**

2. **Training:**
   - Go to "Train Model" tab
   - Set parameters
   - Click "Start Training"
   - Monitor progress in real-time
   - Training runs on API server (background task)
   - Click "Stop Training" to halt (current epoch completes)

3. **Making Predictions:**
   - Ensure API server is running
   - Go to "Predict" tab
   - Click "Select Image"
   - Click "Predict"
   - View results and probabilities

4. **Evaluation:**
   - Go to "Evaluate Model" tab
   - Click "API Info" to view loaded model details
   - Click any evaluation button
   - View results in the output area

## API Server

The API server provides:
- `/health` - Health check (includes training status)
- `/model/info` - Model information
- `/predict` - Single image prediction
- `/predict/batch` - Batch predictions
- **`/train` - Start training with custom parameters**
- **`/train/status` - Get real-time training status**
- **`/train/stop` - Stop ongoing training**

**API runs on:** http://localhost:8000

**API Docs:** http://localhost:8000/docs

## Training via API

Training is now handled entirely by the API:
- Supports custom epochs, batch size, learning rate, hidden units
- Background task processing (non-blocking)
- Real-time status monitoring
- Resume from checkpoint support
- Graceful stop functionality

**Example Training Request:**
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 45,
    "batch_size": 32,
    "learning_rate": 0.001,
    "hidden_units": 64,
    "resume": false
  }'
```

**Check Training Status:**
```bash
curl "http://localhost:8000/train/status"
```

## Troubleshooting

**API not connecting:**
- Ensure api.py is running in a separate terminal
- Check that port 8000 is not in use
- Verify model files exist in `../models/`

**Training fails:**
- Check API logs for detailed error messages
- Ensure enough disk space
- Verify dataset download completed
- Check training status via `/train/status` endpoint

**Training already in progress error:**
- Wait for current training to complete
- Check status with "API Info" button
- Use `/train/stop` to stop current training

**Predictions fail:**
- Restart API server
- Check model file exists: `../models/best_model.pth`
- Verify image format (JPG, PNG)

## Requirements

- Python 3.8+
- PyTorch with CUDA support (optional, for GPU)
- FastAPI and Uvicorn
- All packages in requirements.txt
