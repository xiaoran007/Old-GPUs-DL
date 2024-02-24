import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

# Generate a toy dataset
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 50) # 2 input features, 50 outputs
        self.fc2 = nn.Linear(50, 2) # 50 input features, 2 outputs (classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(device):
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(100): # Using fewer epochs for quick demonstration
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device) # Move data to device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    end_time = time.time()
    print(f"Training completed on {device}. Time taken: {end_time - start_time:.2f} seconds.")


# Check if CUDA is available and select GPU device; otherwise, CPU is used.
if torch.cuda.is_available():
    # Train on GPU
    train_model(torch.device("cuda"))
else:
    print("CUDA is not available. Training on CPU only.")

# Train on CPU for comparison
train_model(torch.device("cpu"))
