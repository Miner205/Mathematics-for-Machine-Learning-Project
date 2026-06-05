import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# to use nvidia gpu instead of cpu ;
# to be able to use it first you need to have an nvidia gpu [apparently it's only recommended to have one, not mandatory],
# then in windows cmd of computer type the command "nvidia-smi" and note the Cuda version printed,
# then in PyCharm cmd/terminal use command "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126" with either 126, 130 or 132 depending of your Cuda version.
# pip command from here (check here for Linux/Mac version too) : https://pytorch.org/
# Note: pour que la bonne version de pytorch/cuda soit installé/utilisé sur mon pc j'ai d'abord dû désintaller l'ancienne version(cpu) avec "pip uninstall torch torchvision"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Note: calculation launched on {device}.\n")

torch.manual_seed(0)
np.random.seed(0)

# CUSTOM DATA PREPROCESSORS

class GrayscaleFlattenTransform:
    def __call__(self, img_tensor):
        r, g, b = img_tensor[0], img_tensor[1], img_tensor[2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.view(-1)

class ColorFlattenTransform:
    def __call__(self, img_tensor):
        return img_tensor.view(-1)


# ARCHITECTURES

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


class CIFAR_CNN_3D(nn.Module):
    def __init__(self):
        super().__init__()
        # On passe en 3D mais on force la profondeur à 1 (kernel 1x3x3) pour ne pas crasher
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)), nn.ReLU(), 
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)), nn.ReLU(), 
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)), nn.ReLU(), 
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)), nn.ReLU()
        )
        self.densification = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(8 * 8 * 64, 10)
        )

    def forward(self, x):
        # Note : PyTorch crash si on ne rajoute pas une dimension de "profondeur"
        # x passe de [batch, 3 canaux, 32, 32] à [batch, 3 canaux, 1 de profondeur, 32, 32]
        x = x.unsqueeze(2) 
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.densification(x)
        return x


# TRAINING & TESTING LOOPS

def train_model(model, train_loader, epochs=5):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    print(f"\n--- Training {model.__class__.__name__} ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f" > Processed {batch_idx + 1}/{len(train_loader)} batches...")
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
    return model


def test_model(model, test_loader):
    """Function to verifies the different models"""
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"\n--- Training of {model.__class__.__name__} Completed ---")
    print(f" => Verification: Accuracy on Test Images: {accuracy:.2f}%\n")
    return accuracy


# MAIN

if __name__ == "__main__":
    
    transform_gray = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), GrayscaleFlattenTransform()])
    transform_color = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), ColorFlattenTransform()])
    transform_cnn = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    print("Loading datasets...")
    train_gray = DataLoader(datasets.ImageFolder('cifar10_data/train', transform=transform_gray), batch_size=128, shuffle=True)
    test_gray = DataLoader(datasets.ImageFolder('cifar10_data/test', transform=transform_gray), batch_size=128, shuffle=False)
    
    train_color = DataLoader(datasets.ImageFolder('cifar10_data/train', transform=transform_color), batch_size=128, shuffle=True)
    test_color = DataLoader(datasets.ImageFolder('cifar10_data/test', transform=transform_color), batch_size=128, shuffle=False)
    
    train_cnn = DataLoader(datasets.ImageFolder('cifar10_data/train', transform=transform_cnn), batch_size=128, shuffle=True)
    test_cnn = DataLoader(datasets.ImageFolder('cifar10_data/test', transform=transform_cnn), batch_size=128, shuffle=False)

    print("\n" + "="*30)
    print("MENU CIFAR TEST")
    print("="*30)
    print("1 - test only CNN 3D")
    print("2 - all tests (Linear models, MLP, CNN 2D and CNN 3D)")
    print("="*30)
    
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        print("\n--- Testing CNN 3D ---")
        cnn_3d_model = CIFAR_CNN_3D()
        train_model(cnn_3d_model, train_cnn, epochs=5)
        test_model(cnn_3d_model, test_cnn)

    elif choice == "2":
        print("\n--- Testing Linear Grayscale ---")
        model_gray_lin = Linear_CIFAR_Gray()
        train_model(model_gray_lin, train_gray)
        test_model(model_gray_lin, test_gray)

        print("\n--- Testing Linear Color ---")
        model_color_lin = Linear_CIFAR_Color()
        train_model(model_color_lin, train_color)
        test_model(model_color_lin, test_color)

        print("\n--- Testing MLP Grayscale ---")
        model_gray_mlp = MLP_CIFAR_Gray()
        train_model(model_gray_mlp, train_gray)
        test_model(model_gray_mlp, test_gray)

        print("\n--- Testing MLP Color ---")
        model_color_mlp = MLP_CIFAR_Color()
        train_model(model_color_mlp, train_color)
        test_model(model_color_mlp, test_color)

        print("\n--- Testing CNN ---")
        cnn_model = CIFAR_CNN()
        train_model(cnn_model, train_cnn)
        test_model(cnn_model, test_cnn)

        print("\n--- Testing CNN 3D ---")
        cnn_3d_model = CIFAR_CNN_3D()
        train_model(cnn_3d_model, train_cnn)
        test_model(cnn_3d_model, test_cnn)
    else:
        print("Invalid choice. Relaunch the script.")
