import torchvision.transforms as transforms

# Define transforms for both the person and clothing images
class PersonTransform:
    def __init__(self, image_size):
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Assuming RGB images
        ])

    def __call__(self, img):
        return self.transforms(img)


class ClothingTransform:
    def __init__(self, image_size):
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Assuming RGB images
        ])

    def __call__(self, img):
        return self.transforms(img)