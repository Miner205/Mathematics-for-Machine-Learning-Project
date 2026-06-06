import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Note: calculation launched on {device}.\n")


# DATA

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalization
])

train_dataset = datasets.ImageFolder("processed_medical_dataset/train", transform=transform)
test_dataset = datasets.ImageFolder("processed_medical_dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# CLASS IMBALANCE

targets = train_dataset.targets
class_counts = Counter(targets)
total = len(targets)
weights = [total / class_counts[0], total / class_counts[1]]
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
#print("Class weights", class_weights, "\n")


# MODEL

class MedicalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# TRAINING

def train_model(model, train_loader, epochs=10):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"\n--- Training {model.__class__.__name__} ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss = {running_loss:.4f}")


# TESTING

def test_model(model, test_loader):
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    all_labels = []
    all_preds = []
    print(f"\n--- Testing {model.__class__.__name__} ---")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    accuracy = 100 * correct / total
    print(f"\n--- Training and Testing of {model.__class__.__name__} Completed ---")
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    cm = confusion_matrix(all_labels, all_preds)
    return cm


# MAIN

if __name__ == "__main__":

    model = MedicalCNN().to(device)
    train_model(model, train_loader, epochs=10)
    cm = test_model(model, test_loader)

    print("\nConfusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    )
    disp.plot()
    plt.show()
