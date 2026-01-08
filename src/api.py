"""
FastAPI service for Waste Classification Model
"""
import os
import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from typing import List
from pydantic import BaseModel
import uvicorn
from datetime import datetime

from train import WasteClassifierModelV1
from torchvision import transforms


class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    predicted_class: str
    confidence: float
    all_probabilities: dict
    processing_time: float


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""
    predictions: List[dict]
    total_images: int
    processing_time: float


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    device: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response model for model info"""
    model_path: str
    num_classes: int
    classes: List[str]
    device: str
    best_accuracy: float
    model_architecture: str


# Initialize FastAPI app
app = FastAPI(
    title="Waste Classification API",
    description="API for classifying waste images using a trained CNN model",
    version="1.0.0"
)

# Global variables
model = None
classes = None
device = None
transform = None
model_path = None
best_accuracy = None


def load_model():
    """Load the trained model and classes"""
    global model, classes, device, transform, model_path, best_accuracy
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model path
    model_path = '../models/best_model.pth'
    history_path = '../models/training_history.pth'
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found at {history_path}")
    
    # Load history and classes
    history = torch.load(history_path, map_location=device)
    classes = history['classes']
    best_accuracy = history.get('best_accuracy', 0.0)
    
    # Load model
    num_classes = len(classes)
    model = WasteClassifierModelV1(
        input_shape=3,
        hidden_units=64,
        output_shape=num_classes
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Setup transform
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    
    print(f"Model loaded successfully on {device}")
    print(f"Classes: {classes}")


def predict_image(image: Image.Image):
    """
    Predict the class of an image
    
    Args:
        image: PIL Image object
        
    Returns:
        predicted_class, confidence, all_probabilities
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Preprocess image
    image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    start_time = datetime.now()
    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    predicted_class = classes[predicted_idx.item()]
    confidence_value = confidence.item() * 100
    
    # Get all class probabilities
    all_probs = {classes[i]: round(probabilities[0][i].item() * 100, 2) 
                 for i in range(len(classes))}
    
    return predicted_class, confidence_value, all_probs, processing_time


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("API will start but predictions will fail until model is loaded")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Waste Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict",
            "batch_predict": "/predict/batch"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Returns the status of the service and model
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=device if device else "unknown",
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_path=model_path,
        num_classes=len(classes),
        classes=classes,
        device=device,
        best_accuracy=round(best_accuracy, 2) if best_accuracy else 0.0,
        model_architecture="WasteClassifierModelV1 (CNN)"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of a single uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        
    Returns:
        Prediction results with class, confidence, and all probabilities
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction
        predicted_class, confidence, all_probs, processing_time = predict_image(image)
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=round(confidence, 2),
            all_probabilities=all_probs,
            processing_time=round(processing_time, 4)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Predict the classes of multiple uploaded images
    
    Args:
        files: List of image files (JPEG, PNG, etc.)
        
    Returns:
        Batch prediction results with predictions for each image
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    start_time = datetime.now()
    predictions = []
    
    for idx, file in enumerate(files):
        # Validate file type
        if not file.content_type.startswith('image/'):
            predictions.append({
                "filename": file.filename,
                "error": "Not an image file"
            })
            continue
        
        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Make prediction
            predicted_class, confidence, all_probs, proc_time = predict_image(image)
            
            predictions.append({
                "filename": file.filename,
                "predicted_class": predicted_class,
                "confidence": round(confidence, 2),
                "top_3_predictions": dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3])
            })
        
        except Exception as e:
            predictions.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_images=len(files),
        processing_time=round(total_time, 4)
    )


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
