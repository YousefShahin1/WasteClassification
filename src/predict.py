"""
Prediction script for Waste Classification Model
"""
import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from train import WasteClassifierModelV1


class WasteClassifier:
    """Wrapper class for easy predictions"""
    
    def __init__(self, model_path, classes, device='cpu'):
        """
        Initialize the classifier
        
        Args:
            model_path: Path to the saved model
            classes: List of class names
            device: Device to run on
        """
        self.device = device
        self.classes = classes
        self.num_classes = len(classes)
        
        # Load model
        self.model = WasteClassifierModelV1(
            input_shape=3,
            hidden_units=64,
            output_shape=self.num_classes
        ).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # Setup transforms
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])
        
        print(f"Model loaded and ready for predictions")
    
    def predict(self, image_path, show_image=True):
        """
        Predict the class of an image
        
        Args:
            image_path: Path to the image file
            show_image: Whether to display the image with prediction
            
        Returns:
            predicted_class, confidence, all_probabilities
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.inference_mode():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            
        predicted_class = self.classes[predicted_idx.item()]
        confidence_value = confidence.item() * 100
        
        # Get all class probabilities
        all_probs = {self.classes[i]: probabilities[0][i].item() * 100 
                     for i in range(self.num_classes)}
        
        if show_image:
            plt.figure(figsize=(8, 6))
            plt.imshow(image)
            plt.title(f"Predicted: {predicted_class} ({confidence_value:.2f}%)")
            plt.axis('off')
            plt.show()
        
        return predicted_class, confidence_value, all_probs
    
    def predict_batch(self, image_paths, top_k=3):
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image paths
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions for each image
        """
        results = []
        
        for image_path in image_paths:
            predicted_class, confidence, all_probs = self.predict(image_path, show_image=False)
            
            # Get top-k predictions
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            top_predictions = sorted_probs[:top_k]
            
            results.append({
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_predictions': top_predictions
            })
            
            print(f"\nImage: {os.path.basename(image_path)}")
            print(f"Prediction: {predicted_class} ({confidence:.2f}%)")
            print(f"Top {top_k} predictions:")
            for i, (cls, prob) in enumerate(top_predictions, 1):
                print(f"  {i}. {cls}: {prob:.2f}%")
        
        return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict waste class for images using trained model')
    parser.add_argument('--image', type=str, help='Path to image file to classify')
    parser.add_argument('--images', nargs='+', help='Paths to multiple images to classify')
    parser.add_argument('--model', type=str, default='../models/best_model.pth', help='Path to trained model (default: ../models/best_model.pth)')
    parser.add_argument('--history', type=str, default='../models/training_history.pth', help='Path to training history file (default: ../models/training_history.pth)')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions to show (default: 3)')
    parser.add_argument('--no-display', action='store_true', help='Do not display images')
    return parser.parse_args()


def main():
    """Main prediction function"""
    # Parse arguments
    args = parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load classes from training history
    if not os.path.exists(args.history):
        print(f"Error: Training history file '{args.history}' not found.")
        print("Please train the model first using train.py")
        return
    
    history = torch.load(args.history, map_location=device)
    classes = history['classes']
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        print("Please train the model first using train.py")
        return
    
    # Create classifier
    classifier = WasteClassifier(
        model_path=args.model,
        classes=classes,
        device=device
    )
    
    # Check if image path(s) provided
    if args.image:
        # Single image prediction
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found.")
            return
        
        print(f"\nPredicting class for: {args.image}")
        predicted_class, confidence, all_probs = classifier.predict(
            args.image, 
            show_image=not args.no_display
        )
        
        print(f"\nPrediction: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"\nTop {args.top_k} predictions:")
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for i, (cls, prob) in enumerate(sorted_probs[:args.top_k], 1):
            print(f"  {i}. {cls}: {prob:.2f}%")
    
    elif args.images:
        # Batch prediction
        # Check if all images exist
        missing_images = [img for img in args.images if not os.path.exists(img)]
        if missing_images:
            print(f"Error: The following image files were not found:")
            for img in missing_images:
                print(f"  - {img}")
            return
        
        print(f"\nPredicting classes for {len(args.images)} images...")
        results = classifier.predict_batch(args.images, top_k=args.top_k)
        
        print("\n" + "="*60)
        print("BATCH PREDICTION SUMMARY")
        print("="*60)
        for result in results:
            print(f"\n{result['image_path']}")
            print(f"  → {result['predicted_class']} ({result['confidence']:.2f}%)")
    
    else:
        # No image provided
        print("\nNo image specified. Please provide an image to classify.")
        print("\nUsage examples:")
        print(f"  python predict.py --image path/to/image.jpg")
        print(f"  python predict.py --images img1.jpg img2.jpg img3.jpg")
        print(f"  python predict.py --image path/to/image.jpg --top-k 5")
        print(f"  python predict.py --image path/to/image.jpg --no-display")


if __name__ == "__main__":
    main()
