# utils/dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class VirtualTryOnDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Optional: Add transforms if needed
class CustomTransform:
    def __call__(self, image):
        # Implement your transformation logic here
        return image
