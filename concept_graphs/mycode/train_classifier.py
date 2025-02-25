import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from train import ConditionalUNet  # Import trained DDPM model
from load_dataset import get_dataloader  # Load dataset


# Function to convert labels into binary class tensors
def convert_labels(labels, attribute_name):
    if attribute_name == "shape":
        return torch.tensor([1 if shape == "circle" else 0 for shape in labels["shape"]], dtype=torch.long)

    elif attribute_name == "color":
        # Compare blue channel (index 2) with red channel (index 0)
        return (labels["color"][:, 2] > labels["color"][:, 0]).long()  # 0 for red, 1 for blue

    elif attribute_name == "size":
        return (labels["size"] >= 0.5).long()  # 0 for small (<0.5), 1 for large (>=0.5)

    else:
        raise ValueError(f"Unknown attribute name: {attribute_name}")


# Define a simple linear classifier
class AttributeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(AttributeClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Train a classifier for a given attribute
def train_classifier(attribute_name, dataset_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = AttributeClassifier(input_dim=3*32*32, num_classes=2).to(device)  # Change input_dim that matches the image size!!!
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    classifier.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct, total = 0, 0
        progress_bar = tqdm(dataset_loader, desc=f"Training {attribute_name} Classifier - Epoch {epoch+1}/{num_epochs}")

        for images, labels in progress_bar:
            images = images.to(device)
            labels = convert_labels(labels, attribute_name).to(device)  # Convert labels correctly

            images = images.view(images.shape[0], -1)  # Flatten image

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=loss.item())

        accuracy = correct / total
        print(f"{attribute_name} Classifier - Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.4f}")

    return classifier

# Load training data
train_loader = get_dataloader(batch_size=8, training=True)

# Train classifiers
shape_classifier = train_classifier("shape", train_loader)
color_classifier = train_classifier("color", train_loader)
size_classifier = train_classifier("size", train_loader)


# Save classifiers
torch.save(shape_classifier.state_dict(), "models/shape_classifier.pth")
torch.save(color_classifier.state_dict(), "models/color_classifier.pth")
torch.save(size_classifier.state_dict(), "models/size_classifier.pth")
