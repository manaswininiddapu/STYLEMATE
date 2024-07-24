import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import models
import PIL
import matplotlib.pyplot as plt
#from PIL import Image
from PIL import Image, ImageOps, ImageEnhance
from PIL import __version__ as PILLOW_VERSION

#from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION


from torchvision import transforms
from models.generator import Generator
import numpy as np

# Load the trained generator model
G = Generator()
G.load_state_dict(torch.load('checkpoints/generator.pth'))
G.eval()

# Function to test the virtual try-on
def test_virtual_try_on(person_img_path, clothing_img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    person_img = Image.open(person_img_path).convert('RGB')
    clothing_img = Image.open(clothing_img_path).convert('RGB')
    person_img = transform(person_img).unsqueeze(0)
    clothing_img = transform(clothing_img).unsqueeze(0)

    with torch.no_grad():
        try_on_img = G(person_img, clothing_img)
    
    try_on_img = try_on_img.squeeze().permute(1, 2, 0).numpy()
    try_on_img = (try_on_img * 0.5 + 0.5) * 255
    try_on_img = try_on_img.astype(np.uint8)
    
    plt.imshow(try_on_img)
    plt.show()

# Example usage
test_virtual_try_on('data/person_images/sample_person.jpg', 'data/clothing_images/sample_clothing.jpg')