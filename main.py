import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def main():
    # 1. Extract zip
    zip_path = '/content/archive (1).zip'
    extract_path = '/content/extracted_data'
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Zip extracted successfully!")
    else:
        print(f"Warning: {zip_path} not found. Ensure it exists before running.")
        return

    cancer_dir = os.path.join(extract_path, 'train_cancer')
    classes = ['benign', 'malignant']

    # 2. Prepare datasets and train_test_split
    file_paths = []
    labels = []
    for cls in classes:
        cls_path = os.path.join(cancer_dir, cls)
        img_names = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        file_paths.extend(img_names)
        labels.extend([cls] * len(img_names))

    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # 3. Custom PyTorch Dataset
    class CancerDataset(Dataset):
        def __init__(self, file_paths, labels, transform=None):
            self.file_paths = file_paths
            self.labels = labels
            self.transform = transform
            self.class_to_idx = {'benign': 0, 'malignant': 1}
        def __len__(self):
            return len(self.file_paths)
        def __getitem__(self, idx):
            img_path = self.file_paths[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.class_to_idx[self.labels[idx]]
            if self.transform:
                image = self.transform(image)
            return image, label

    # Transforms & Dataloaders
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CancerDataset(train_files, train_labels, transform=train_transforms)
    val_dataset = CancerDataset(val_files, val_labels, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Class Weights
    num_benign = train_labels.count('benign')
    num_malignant = train_labels.count('malignant')
    total = len(train_labels)
    class_weights = torch.tensor([total / (2.0 * num_benign), total / (2.0 * num_malignant)]).to(device)

    # Custom CNN Definition
    class SkinCancerCNN(nn.Module):
        def __init__(self):
            super(SkinCancerCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 28 * 28, 128)
            self.fc2 = nn.Linear(128, 2)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 64 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = SkinCancerCNN().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    model.train()
    print("Starting training...")
    for epoch in range(5):
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        print(f"Epoch {epoch+1}/5 - Loss: {running_loss / len(train_loader.dataset):.4f}")

    # Validation & Evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    print("Accuracy:", accuracy_score(all_targets, all_preds))
    print("
Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=['benign', 'malignant']))

if __name__ == '__main__':
    main()
