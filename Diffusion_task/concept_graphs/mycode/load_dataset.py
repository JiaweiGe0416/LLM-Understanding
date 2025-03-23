import os
import glob
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
class MyDataset(Dataset):
    def __init__(self, transform=None, dataset_path="input/dataset_root", training=True, configs=["000"]):
        """
        Custom Dataset class for loading images and metadata.

        Args:
            transform: Transformations to apply to the images.
            dataset_path: Path to the dataset directory.
            training: If True, loads training data, else loads test data.
            configs: List of dataset configurations to load.
        """
        self.training = training
        self.dataset_path = dataset_path
        self.transform = transform

        prefix = "CLEVR"
        ext = ".png"
        self.image_paths = []

        subdir = "train" if training else "test"

        for config in configs:
            path_pattern = os.path.join(dataset_path, subdir, f"{prefix}_{config}_*{ext}")
            new_paths = glob.glob(path_pattern)
            self.image_paths.extend(new_paths)

        self.len_data = len(self.image_paths)
        print(f"Loaded {self.len_data} {'training' if training else 'testing'} images.")

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        json_path = img_path.replace(".png", ".json")

        # Load image and convert to RGB format
        img = Image.open(img_path).convert("RGB")  # Ensure it's an RGB image

        if self.transform is not None:
            img = self.transform(img)  

        # Load metadata
        with open(json_path, "r") as f:
            metadata = json.load(f)

        label = {
            "shape": metadata["shape"],
            "x_position": np.array(metadata["x_position"], dtype=np.float32),
            "y_position": np.array(metadata["y_position"], dtype=np.float32),
            "color": np.array(metadata["color"], dtype=np.float32),
            "size": np.array(metadata["size"], dtype=np.float32),
            "bg_color": np.array(metadata["bg_color"], dtype=np.float32),
            "class": int(metadata["class"])
        }

        return img, label

    def __len__(self):
        return self.len_data


# Function to load dataset
def get_dataloader(batch_size=4, training=True, dataset_path="input/dataset_root", configs=["000"]):
    # transform = transforms.Compose([transforms.ToTensor()])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # Converts to [0,1]
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Rescales to [-1,1]
    # ])
    transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0,1]
    transforms.Lambda(lambda x: x * 2 - 1)  # Maps to [-1,1]
    ])
    dataset = MyDataset(transform=transform, dataset_path=dataset_path, training=training, configs=configs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



if __name__ == '__main__':
    # Example usage
    train_loader = get_dataloader(batch_size=4, training=True)
    
    for img, label in train_loader:
        print("Label:", label)
        print("Image shape:", img.shape)
        break  # Just load one batch for verification
