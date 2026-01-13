"""
Training script for Waste Classification Model
"""
import os
import argparse
import torch
from torch import nn
from tqdm.auto import tqdm
from timeit import default_timer as timer
from processing import download_dataset, prepare_data


class WasteClassifierModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units*2, out_channels=hidden_units*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*2, out_channels=hidden_units*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*4, out_channels=hidden_units*8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*4, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)
        x = self.global_avg_pool(x)
        return self.classifier(x)


def accuracy_fn(y_true, y_pred):
    """Calculate accuracy"""
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start: float, end: float, device: torch.device = None):
    """Print training time"""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
    """Perform a single training step"""
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    
    for batch, (X, y) in enumerate(data_loader):
        # Send data to device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    return train_loss, train_acc


def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device):
    """Perform a single testing step"""
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        # Adjust metrics
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
        return test_loss, test_acc


def train_model(model, train_loader, test_loader, epochs=45, lr=0.001, device='cpu', save_dir='models'):
    """
    Train the model
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        test_loader: Testing data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_dir: Directory to save the best model
        
    Returns:
        model, train_accuracies, test_accuracies, train_losses, test_losses, best_accuracy
    """
    # Setup loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3
    )
    
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []
    best_acc = 0.0
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Measure time
    train_time_start = timer()
    
    # Train and test model
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        
        train_loss, train_acc = train_step(
            data_loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device
        )
        
        test_loss, test_acc = test_step(
            data_loader=test_loader,
            model=model,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device
        )
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"New best accuracy: {best_acc:.4f}% - Model saved to {model_path}")
        
        scheduler.step(test_acc)
    
    train_time_end = timer()
    print_train_time(start=train_time_start, end=train_time_end, device=device)
    
    return model, train_accuracies, test_accuracies, train_losses, test_losses, best_acc


def save_model(model, filepath='waste_classifier_model.pth'):
    """Save the trained model"""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Waste Classification Model')
    parser.add_argument('--epochs', type=int, default=45, help='Number of training epochs (default: 45)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden-units', type=int, default=64, help='Number of hidden units in first conv layer (default: 64)')
    parser.add_argument('--save-dir', type=str, default='../models', help='Directory to save models (default: ../models)')
    parser.add_argument('--history-path', type=str, default='../models/training_history.pth', help='Path to save training history (default: ../models/training_history.pth)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from (default: None)')
    return parser.parse_args()


def main():
    """Main training function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"\nTraining configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden units: {args.hidden_units}")
    print(f"  Random seed: {args.seed}")
    if args.resume:
        print(f"  Resuming from: {args.resume}")
    
    # Download and prepare data
    print("\nDownloading dataset...")
    path = download_dataset()
    dataset_path = os.path.join(path, "Garbage_Dataset_Classification", "images")
    
    print("\nPreparing data loaders...")
    train_loader, test_loader, classes, num_classes = prepare_data(dataset_path, batch_size=args.batch_size)
    
    # Create model
    print("\nCreating model...")
    model = WasteClassifierModelV1(
        input_shape=3,
        hidden_units=args.hidden_units,
        output_shape=num_classes
    ).to(device)
    
    # Load checkpoint if resuming
    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    if args.resume and os.path.exists(args.resume):
        print(f"\nLoading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # New format with full training history and model weights
            model.load_state_dict(checkpoint['model_state_dict'])
            train_accs = checkpoint.get('train_accuracies', [])
            test_accs = checkpoint.get('test_accuracies', [])
            train_losses = checkpoint.get('train_losses', [])
            test_losses = checkpoint.get('test_losses', [])
            print(f"Resumed from epoch {len(train_accs)}")
            print(f"Previous best accuracy: {checkpoint.get('best_accuracy', 0):.4f}%")
        elif 'train_accuracies' in checkpoint:
            # History file only (no model weights) - load from best_model.pth
            print("Training history found, but no model weights in this file.")
            model_path = os.path.join(args.save_dir, 'best_model.pth')
            if os.path.exists(model_path):
                print(f"Loading model weights from {model_path}...")
                model.load_state_dict(torch.load(model_path))
                train_accs = checkpoint.get('train_accuracies', [])
                test_accs = checkpoint.get('test_accuracies', [])
                train_losses = checkpoint.get('train_losses', [])
                test_losses = checkpoint.get('test_losses', [])
                print(f"Resumed from epoch {len(train_accs)}")
                print(f"Previous best accuracy: {checkpoint.get('best_accuracy', 0):.4f}%")
            else:
                print(f"Warning: Could not find model weights at {model_path}")
                print("Starting training from scratch")
        else:
            # Old format - just model weights (state_dict)
            model.load_state_dict(checkpoint)
            print("Loaded model weights from checkpoint")
    
    print(model)
    
    # Train model
    print("\nStarting training...")
    model, train_accs, test_accs, train_losses, test_losses, best_acc = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir
    )
    
    # Save training history with model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_accuracies': train_accs,
        'test_accuracies': test_accs,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'classes': classes,
        'best_accuracy': best_acc
    }, args.history_path)
    print(f"Training history saved to {args.history_path}")


if __name__ == "__main__":
    main()
