import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from train import ConditionalUNet  # Import trained DDPM model
from load_dataset import get_dataloader  # Load dataset



# Load trained classifiers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Function to convert a list of label dictionaries into binary class tensors
def convert_labels_2(true_labels, attribute_name):
    # Ensure everything is moved to the correct device
    device = true_labels[0]["size"].device

    # Initialize empty lists for merging
    merged_labels = {
        "shape": [],
        "color": [],
        "size": [],
        "bg_color": []
    }

    # Iterate over the batch dictionaries and append all samples to the merged list
    for batch in true_labels:
        merged_labels["shape"].extend(batch["shape"])  # This is already a list of strings
        merged_labels["color"].append(batch["color"])  # Append tensors
        merged_labels["size"].append(batch["size"])    # Append tensors
        merged_labels["bg_color"].append(batch["bg_color"])  # Append tensors

    # Stack tensors for batch processing
    merged_labels["color"] = torch.cat(merged_labels["color"]).to(device)  # Shape: (total_batch_size, 3)
    merged_labels["size"] = torch.cat(merged_labels["size"]).to(device)  # Shape: (total_batch_size,)
    merged_labels["bg_color"] = torch.cat(merged_labels["bg_color"]).to(device)  # Shape: (total_batch_size, 3)

    # Convert each attribute into a binary tensor with correct shape (batch_size,)
    if attribute_name == "shape":
        return torch.tensor([0 if shape == "circle" else 1 for shape in merged_labels["shape"]], dtype=torch.long, device=device)

    elif attribute_name == "color":
        color_class = (merged_labels["color"][:, 2] > merged_labels["color"][:, 0]).long() # 0 for red, 1 for blue
        return color_class.view(-1)  # Ensure shape is (batch_size,)

    elif attribute_name == "size":
        size_class = (merged_labels["size"] >= 0.5).long()  # 0 for small, 1 for large
        return size_class.view(-1)  # Ensure shape is (batch_size,)

    elif attribute_name == "bg_color":    
        brightness = merged_labels["bg_color"].mean(dim=1)
        bg_class = (brightness >= 0.5).long()  # 0 for dark, 1 for bright
        return bg_class.view(-1)  # Ensure shape is (batch_size,)


    else:
        raise ValueError(f"Unknown attribute name: {attribute_name}")







# Define the classifier structure (must match training)
class AttributeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(AttributeClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Function to load a classifier
def load_classifier(model_path):
    classifier = AttributeClassifier(input_dim=3*32*32, num_classes=2).to(device)  # Flattened image input
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.eval()  # Set to evaluation mode
    return classifier

# Load all classifiers
shape_classifier = load_classifier("models/shape_classifier.pth")
color_classifier = load_classifier("models/color_classifier.pth")
size_classifier = load_classifier("models/size_classifier.pth")
bg_color_classifier = load_classifier("models/bg_color_classifier.pth")

print("All classifiers loaded successfully!")


# Load trained DDPM model
ddpm_model = ConditionalUNet().to(device)
ddpm_model.load_state_dict(torch.load("models/ddpm_conditional_unet.pth", map_location=device))
ddpm_model.eval()
print("DDPM model loaded successfully!")



# Function to generate images using the DDPM model
def generate_images(ddpm_model, dataset_loader, n_T=1000):
    ddpm_model.eval()
    generated_images = []
    true_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataset_loader, desc="Generating Images from DDPM"):
            images = images.to(device)

            # Extract conditioning information
            conditions = torch.stack([
                labels["x_position"],
                labels["y_position"],
                labels["color"][:, 0],
                labels["color"][:, 1],
                labels["color"][:, 2],
                labels["size"],
                labels["bg_color"][:, 0],
                labels["bg_color"][:, 1],
                labels["bg_color"][:, 2],
                labels["class"].float()
            ], dim=1).to(device)

            # Start from pure noise and reverse diffusion
            x_t = torch.randn_like(images).to(device)
            for t in reversed(range(n_T)):
                predicted_noise = ddpm_model(x_t, conditions)
                x_t = x_t - predicted_noise  # Reverse diffusion step

            generated_images.append(x_t.cpu())  # Store generated image
            true_labels.append(labels)  # Store ground truth labels

    return torch.cat(generated_images), true_labels



# Function to classify generated images
def classify_generated_images(generated_images, true_labels):
    correct_shape, correct_color, correct_size, correct_bg = 0, 0, 0, 0
    total_samples = generated_images.shape[0]

    # Flatten images for classification
    generated_images = generated_images.view(generated_images.shape[0], -1).to(device)

    # Predict attributes
    with torch.no_grad():
        pred_shape = torch.argmax(shape_classifier(generated_images), dim=1)
        pred_color = torch.argmax(color_classifier(generated_images), dim=1)
        pred_size = torch.argmax(size_classifier(generated_images), dim=1)
        pred_bg = torch.argmax(bg_color_classifier(generated_images), dim=1)

    # Convert true labels to binary format
    true_shape = convert_labels_2(true_labels, "shape").to(device)
    true_color = convert_labels_2(true_labels, "color").to(device)
    true_size = convert_labels_2(true_labels, "size").to(device)
    true_bg = convert_labels_2(true_labels, "bg_color").to(device)

    # Compute accuracy for each attribute
    correct_shape = (pred_shape == true_shape).sum().item()
    correct_color = (pred_color == true_color).sum().item()
    correct_size = (pred_size == true_size).sum().item()
    correct_bg = (pred_bg == true_bg).sum().item()

    shape_acc = correct_shape / total_samples
    color_acc = correct_color / total_samples
    size_acc = correct_size / total_samples
    bg_acc = correct_bg / total_samples

    # Compute overall correctness (all four attributes must be correct)
    overall_correct = ((pred_shape == true_shape) & 
                       (pred_color == true_color) & 
                       (pred_size == true_size) & 
                       (pred_bg == true_bg)).sum().item()
    
    # Compute overall accuracy
    overall_accuracy = overall_correct / total_samples

    print(f"Shape Accuracy: {shape_acc:.4f}")
    print(f"Color Accuracy: {color_acc:.4f}")
    print(f"Size Accuracy: {size_acc:.4f}")
    print(f"Background Color Accuracy: {bg_acc:.4f}")
    print(f"Overall Multiplicative Accuracy: {overall_accuracy:.4f}")


# Load test dataset
test_loader = get_dataloader(batch_size=8, training=False)

# Generate images
generated_images, true_labels = generate_images(ddpm_model, test_loader)

# Classify generated images
classify_generated_images(generated_images, true_labels)
