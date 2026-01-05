"""
Data processing and loading utilities for Waste Classification
"""
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import kagglehub


class GarbageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # Get all class folders
        self.classes = sorted([d for d in os.listdir(root)
                              if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Build list of (image_path, label) tuples
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def download_dataset():
    """Download the dataset from Kaggle"""
    path = kagglehub.dataset_download("zlatan599/garbage-dataset-classification")
    print("Path to dataset files:", path)
    
    print("\nContents of dataset directory:")
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            print(f"📁 {item}/ - {len(os.listdir(item_path))} items")
        else:
            print(f"📄 {item}")
    
    return path


def get_transforms():
    """Get training and testing transforms"""
    # Define ImageNet normalization stats
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    
    return train_transform, test_transform


def prepare_data(dataset_path, batch_size=32, test_size=0.2, random_state=42):
    """
    Prepare train and test data loaders
    
    Args:
        dataset_path: Path to the dataset images folder
        batch_size: Batch size for data loaders
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        train_loader, test_loader, classes, num_classes
    """
    train_transform, test_transform = get_transforms()
    
    # Load full dataset to get indices
    full_dataset = GarbageDataset(root=dataset_path, transform=None)
    
    # Randomly sample
    num_samples = int(len(full_dataset))
    indices = random.sample(range(len(full_dataset)), num_samples)
    
    # Get corresponding labels
    labels = [full_dataset.samples[i][1] for i in indices]
    
    # Split 80/20 train/test with stratification
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    # Create separate datasets with transforms
    train_dataset = GarbageDataset(root=dataset_path, transform=train_transform)
    test_dataset = GarbageDataset(root=dataset_path, transform=test_transform)
    
    # Create subsets
    train_data = Subset(train_dataset, train_indices)
    test_data = Subset(test_dataset, test_indices)
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Classes: {full_dataset.classes}")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, full_dataset.classes, len(full_dataset.classes)


if __name__ == "__main__":
    # Test the data loading
    path = download_dataset()
    dataset_path = os.path.join(path, "Garbage_Dataset_Classification", "images")
    train_loader, test_loader, classes, num_classes = prepare_data(dataset_path)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {classes}")
