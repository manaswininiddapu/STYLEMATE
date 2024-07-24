import os
import sys
import torch
import PIL
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor  # Adjust according to your transforms
from utils.dataset import VirtualTryOnDataset

# Adjust the path to include the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001
data_root = 'dataset.py'  # Adjust this path

# Define transforms (adjust based on your preprocessing needs)
transform = Compose([
    ToTensor(),
    # Add more transforms as needed
])

# Initialize your dataset and dataloader
dataset = VirtualTryOnDataset(root_dir=data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example of using the dataloader
for epoch in range(num_epochs):
    for batch_idx, inputs in enumerate(dataloader):
        # Process your data here (e.g., feed it into your model for training)
        
        print(f'Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Input Shape: {inputs.shape}')