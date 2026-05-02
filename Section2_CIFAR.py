import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(0)
np.random.seed(0)

# ==========================================
# 1. CUSTOM DATA PREPROCESSORS
# ==========================================
class GrayscaleFlattenTransform:
    def __call__(self, img_tensor):
        r, g, b = img_tensor[0], img_tensor[1], img_tensor[2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.view(-1)

class ColorFlattenTransform:
    def __call__(self, img_tensor):
        return img_tensor.view(-1)

# ==========================================
# 2. ARCHITECTURES
# ==========================================
class Linear_CIFAR_Gray(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1024, 10)
    def forward(self, x):
        return self.layer(x)

class Linear_CIFAR_Color(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3072, 10)
    def forward(self, x):
        return self.layer(x)

class MLP_CIFAR_Gray(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x):
        return self.layers(x)

class MLP_CIFAR_Color(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3072, 1024), nn.ReLU(), nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 10))
    def forward(self, x):
        return self.layers(x)

class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU())
        self.densification = nn.Sequential(nn.Flatten(), nn.Linear(8 * 8 * 64, 10))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.densification(x)
        return x

# ==========================================
# 3. TRAINING & TESTING LOOPS
# ==========================================
def train_model(model, train_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    print(f"\n--- Training {model.__class__.__name__} ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f"  > Processed {batch_idx + 1}/{len(train_loader)} batches...")
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
    return model

def test_model(model, test_loader):
    """THIS IS THE FUNCTION THAT VERIFIES THE MODEL"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"  => 🎯 VERIFICATION: Accuracy on Test Images: {accuracy:.2f}%\n")
    return accuracy

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    transform_gray = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), GrayscaleFlattenTransform()])
    transform_color = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), ColorFlattenTransform()])
    transform_cnn = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    print("Loading datasets...")
    # Make sure your folders are exactly named cifar10_data/train and cifar10_data/test!
    train_gray = DataLoader(datasets.ImageFolder('cifar10_data/train', transform=transform_gray), batch_size=128, shuffle=True)
    test_gray = DataLoader(datasets.ImageFolder('cifar10_data/test', transform=transform_gray), batch_size=128, shuffle=False)
    
    train_color = DataLoader(datasets.ImageFolder('cifar10_data/train', transform=transform_color), batch_size=128, shuffle=True)
    test_color = DataLoader(datasets.ImageFolder('cifar10_data/test', transform=transform_color), batch_size=128, shuffle=False)
    
    train_cnn = DataLoader(datasets.ImageFolder('cifar10_data/train', transform=transform_cnn), batch_size=128, shuffle=True)
    test_cnn = DataLoader(datasets.ImageFolder('cifar10_data/test', transform=transform_cnn), batch_size=128, shuffle=False)

    # 1. Linear Models (Preliminary Test)
    print("\n--- Testing Linear Grayscale ---")
    model_gray_lin = Linear_CIFAR_Gray()
    train_model(model_gray_lin, train_gray)
    test_model(model_gray_lin, test_gray)

    print("\n--- Testing Linear Color ---")
    model_color_lin = Linear_CIFAR_Color()
    train_model(model_color_lin, train_color)
    test_model(model_color_lin, test_color)

    # 2. MLP Models (Preliminary Test with Layers)
    print("\n--- Testing MLP Grayscale ---")
    model_gray_mlp = MLP_CIFAR_Gray()
    train_model(model_gray_mlp, train_gray)
    test_model(model_gray_mlp, test_gray)

    print("\n--- Testing MLP Color ---")
    model_color_mlp = MLP_CIFAR_Color()
    train_model(model_color_mlp, train_color)
    test_model(model_color_mlp, test_color)

    # 3. The Real Model (CNN)
    print("\n--- Testing CNN ---")
    cnn_model = CIFAR_CNN()
    train_model(cnn_model, train_cnn)
    test_model(cnn_model, test_cnn)