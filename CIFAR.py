import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)

# --- 1. DATA PREP (Grayscale conversion) ---

def convert_cifar_to_grayscale(images):
    """
    Takes the raw CIFAR color images and turns them into flat grayscale vectors.
    Required by section 2.2 of the project PDF.
    """
    # The exact RGB weights the prof gave us in the instructions
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    
    # np.dot multiplies the RGB values by the weights and adds them up automatically. Much faster than running a massive loop over all 50,000 training images
    grayscale_images = np.dot(images[..., :3], weights)
    
    # CIFAR images are 32x32. Flattening them into 1D arrays of 1024 pixels so they fit perfectly into our linear layers.
    flattened_gray = grayscale_images.reshape(-1, 1024)
    return flattened_gray

def flatten_cifar_color(images):
    """
    Flattens the original color images for the second part of the preliminary tests.
    """
    # 32 pixels * 32 pixels * 3 color channels = 3072 inputs per image
    return images.reshape(-1, 3072)


# --- 2. MODELS (Adapted from our MNIST.py) ---
# I copied the exact structure from our MNIST linear model, but updated the input sizes so it doesn't crash when we feed it the bigger CIFAR images.

class Linear_CIFAR_Gray(nn.Module):
    def __init__(self):
        super().__init__()
        # 1024 inputs because grayscale CIFAR is 32x32. Outputs 10 classes.
        self.layer = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.layer(x)
        return x

class Linear_CIFAR_Color(nn.Module):
    def __init__(self):
        super().__init__()
        # 3072 inputs because color CIFAR is 32x32x3. Outputs 10 classes.
        self.layer = nn.Linear(3072, 10)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.layer(x)
        return x


# --- 3. TRAINING LOOP ---

def train_cifar_model(model, train_loader, epochs=5):
    """
    Runs the gradient descent. Same logic as our MNIST training loop so we stay consistent.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    print(f"\n--- Training {model.__class__.__name__} ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad() 
            outputs = model(images) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
    return model


if __name__ == "__main__":
    print("CIFAR-10 preprocessor ready to go.")
    
    #to run this:
    # 1. Load the CIFAR data (we need to download and read the CIFAR batch files first)
    # 2. Pass the data through convert_cifar_to_grayscale() or flatten_cifar_color()
    # 3. Create the TensorDataset and DataLoader exactly like we did in MNIST.py
    # 4. Call train_cifar_model(Linear_CIFAR_Gray(), your_train_loader)
