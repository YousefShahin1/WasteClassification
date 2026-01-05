"""
Evaluation and visualization script for Waste Classification Model
"""
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torchvision.utils import make_grid
from processing import download_dataset, prepare_data
from train import WasteClassifierModelV1


def plot_accuracy_curve(train_accuracies, test_accuracies, epochs):
    """Plot training and test accuracy curves"""
    plt.figure(figsize=(10, 6))
    
    epoch_range = range(1, epochs + 1)
    
    plt.plot(epoch_range, train_accuracies, label='Train Accuracy', marker='o', color='blue')
    plt.plot(epoch_range, test_accuracies, label='Test Accuracy', marker='s', color='orange')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_curve.png')
    print("Accuracy curve saved to accuracy_curve.png")
    plt.show()


def plot_loss_curve(train_losses, test_losses, epochs):
    """Plot training and test loss curves"""
    plt.figure(figsize=(10, 6))
    
    epoch_range = range(1, epochs + 1)
    
    # Convert tensors to CPU if needed
    train_losses_cpu = [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in train_losses]
    test_losses_cpu = [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in test_losses]
    
    plt.plot(epoch_range, train_losses_cpu, label='Train Loss', marker='o', color='red')
    plt.plot(epoch_range, test_losses_cpu, label='Test Loss', marker='s', color='green')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss Over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    print("Loss curve saved to loss_curve.png")
    plt.show()


def evaluate_model(model, test_loader, classes, device='cpu'):
    """
    Evaluate model and generate confusion matrix and classification report
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        classes: List of class names
        device: Device to run on
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating model on test set...")
    with torch.inference_mode():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
    plt.title('Confusion Matrix on Test Set')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    plt.show()
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=classes))


def get_conv_layers(model, max_layers=3):
    """Helper function to find first few Conv2d layers in the model"""
    convs = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            convs.append(m)
            if len(convs) >= max_layers:
                break
    return convs


def visualize_filters(model):
    """Visualize the learned filters of the first conv layer"""
    conv_layers = get_conv_layers(model, max_layers=1)
    if len(conv_layers) == 0:
        print('No Conv2d layers found in the model.')
        return
    
    conv0 = conv_layers[0]
    # conv0.weight shape: (out_chan, in_chan, kH, kW)
    w = conv0.weight.data.clone().cpu()
    # Normalize weights to 0..1 for visualization
    w_min, w_max = w.min(), w.max()
    w_norm = (w - w_min) / (w_max - w_min + 1e-8)
    # Make a grid (will show filters as RGB images when in_chan==3)
    grid = make_grid(w_norm, nrow=8, normalize=False, pad_value=1)
    plt.figure(figsize=(8, 8))
    plt.title('First Conv Layer Filters')
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('conv_filters.png')
    print("Conv filters saved to conv_filters.png")
    plt.show()


def visualize_activation_maps(model, test_data, classes, device='cpu'):
    """Visualize activation maps for a sample test image"""
    # Get a sample image from test_data
    try:
        sample_img, sample_label = test_data[0]
    except Exception as e:
        print(f'Could not index test_data directly: {e}')
        return
    
    # Prepare image and register hooks to capture activations
    activations = []
    hooks = []
    
    def hook_fn(module, inp, out):
        activations.append(out.detach().cpu())
    
    # Register hooks on the first 3 conv layers
    conv_layers = get_conv_layers(model, max_layers=3)
    for conv in conv_layers:
        hooks.append(conv.register_forward_hook(hook_fn))
    
    # Forward pass the sample image
    model.to(device)
    model.eval()
    with torch.inference_mode():
        img_batch = sample_img.unsqueeze(0).to(device)
        _ = model(img_batch)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Plot activation maps from each hooked layer
    for i, act in enumerate(activations):
        # act shape: (batch, channels, H, W)
        act = act[0]  # take the first (and only) batch item
        n_maps = min(8, act.shape[0])
        # normalize each map for display
        act_min, act_max = act.min(), act.max()
        act_norm = (act - act_min) / (act_max - act_min + 1e-8)
        maps = act_norm[:n_maps].unsqueeze(1)  # (n_maps,1,H,W)
        # make grid and plot
        grid = make_grid(maps, nrow=4, normalize=False, pad_value=1)
        plt.figure(figsize=(10, 4))
        plt.title(f'Activation maps - Conv layer {i+1} (first {n_maps} channels)')
        if grid.shape[0] == 1:
            plt.imshow(grid.squeeze().numpy(), cmap='viridis')
        else:
            plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'activation_maps_layer_{i+1}.png')
        print(f"Activation maps layer {i+1} saved to activation_maps_layer_{i+1}.png")
        plt.show()
    
    print(f'Sample ground-truth label: {classes[sample_label]}')


def load_model(model_path, num_classes, device='cpu'):
    """Load a trained model"""
    model = WasteClassifierModelV1(
        input_shape=3,
        hidden_units=64,
        output_shape=num_classes
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def main():
    """Main evaluation function"""
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Download and prepare data
    print("Downloading dataset...")
    path = download_dataset()
    dataset_path = os.path.join(path, "Garbage_Dataset_Classification", "images")
    
    print("\nPreparing data loaders...")
    train_loader, test_loader, classes, num_classes = prepare_data(dataset_path, batch_size=32)
    
    # Load trained model (use best model by default)
    print("\nLoading trained model...")
    model_path = '../models/best_model.pth'
    if not os.path.exists(model_path):
        print(f"Best model not found at {model_path}, trying latest model...")
        model_path = '../models/latest_model.pth'
    
    model = load_model(model_path, num_classes, device)
    
    # Load training history
    history_path = '../models/training_history.pth'
    history = torch.load(history_path)
    train_accs = history['train_accuracies']
    test_accs = history['test_accuracies']
    train_losses = history['train_losses']
    test_losses = history['test_losses']
    
    epochs = len(train_accs)
    
    # Plot curves
    print("\nPlotting accuracy and loss curves...")
    plot_accuracy_curve(train_accs, test_accs, epochs)
    plot_loss_curve(train_losses, test_losses, epochs)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, test_loader, classes, device)
    
    # Visualize filters
    print("\nVisualizing filters...")
    visualize_filters(model)
    
    # Visualize activation maps
    print("\nVisualizing activation maps...")
    # Get test data subset for visualization
    from torch.utils.data import Subset
    test_indices = list(range(len(test_loader.dataset)))
    test_data = test_loader.dataset
    visualize_activation_maps(model, test_data, classes, device)


if __name__ == "__main__":
    main()
