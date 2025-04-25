import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re
from tqdm import tqdm
from PIL import Image
import argparse
import json


# ------------------------- Define MLP Class -------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dims):
        super().__init__()

        self.output_fc0 = nn.Linear(input_dim, output_dims[0])
        self.output_fc1 = nn.Linear(input_dim, output_dims[1])
        self.output_fc2 = nn.Linear(input_dim, output_dims[2])

    def forward(self, x):
        batch_size = x.shape[0]
        x = x[:,:3,:,:].reshape(batch_size, -1)

        y_pred = {}
        y_pred[0] = self.output_fc0(x)
        y_pred[1] = self.output_fc1(x)
        y_pred[2] = self.output_fc2(x)

        return y_pred 


# ------------------------- Load Trained Model -------------------------
def load_model(model_path, input_dim, output_dims, device):
    model = MLP(input_dim, output_dims)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ------------------------- Process NPZ and Evaluate -------------------------
def evaluate_npz(model, npz_path, device):
    # Extract ground truth from filename (e.g., image_001_ep99.npz -> "001")
    match = re.search(r'image_(\d{3})_ep\d+\.npz', os.path.basename(npz_path))
    if not match:
        raise ValueError(f"Invalid filename format: {npz_path}")
    ground_truth = match.group(1)

    # Load NPZ file
    data = np.load(npz_path)
    total_images = 0
    correct_images = 0
    class_correct = {0: 0, 1: 0, 2: 0}  # For shape, color, size
    class_total = {0: 0, 1: 0, 2: 0}

    for key in data.files:
        images = data[key]  # Shape: (64, 3, 28, 28)
        total_images += images.shape[0]

        #shape_flag = int(ground_truth[0])
        #color_flag = int(ground_truth[1])
        #size_flag = int(ground_truth[2])
        #save_dir = "/data2/amanda/LLM-Understanding/concept_graphs/output/single-body_2d_3classes/seed=300/H32-train1/5000_0.4_256_500_200_0.0001_None_1500_2.0_1"
        #for idx, img in enumerate(images):
        #    img_filename = f"CLEVR_{shape_flag}{color_flag}{size_flag}_{idx:05d}.png"
        #    img = (img * 255).astype(np.uint8)  # Convert to [0,255] range
        #    img_pil = Image.fromarray(np.transpose(img, (1, 2, 0)))  # Convert to HxWxC
        #    img_pil.save(os.path.join(save_dir, img_filename))

        # Convert images to tensor
        images_tensor = torch.tensor(images, dtype=torch.float32).to(device)

        # Run model prediction
        with torch.no_grad():
            outputs = model(images_tensor)

        # Process predictions
        pred_classes = []
        for idx in range(3):  # For shape, color, size
            preds = outputs[idx].argmax(dim=1).cpu().numpy()
            pred_classes.append(preds)

        # Combine predictions and check accuracy
        for i in range(images.shape[0]):
            # Ground truth per class from filename (e.g., "001" -> [0, 0, 1])
            gt_per_class = [int(digit) for digit in ground_truth]

            # Per-class accuracy
            for idx in range(3):
                class_total[idx] += 1
                if pred_classes[idx][i] == gt_per_class[idx]:
                    class_correct[idx] += 1

            # Overall prediction
            predicted = f"{pred_classes[0][i]}{pred_classes[1][i]}{pred_classes[2][i]}"
            if predicted == ground_truth:
                correct_images += 1

    # Calculate overall and per-class error rates
    overall_error_rate = 1 - (correct_images / total_images)
    class_error_rates = {idx: 1 - (class_correct[idx] / class_total[idx]) for idx in range(3)}

    return overall_error_rate, class_error_rates, total_images

# ------------------------- Main Evaluation -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--config', type=str)
    parser.add_argument('--ep', default=199, type=int)
    args = parser.parse_args()
    seed = args.seed
    config = args.config
    ep = args.ep

    with open(f"config_categories/config_category_{config}.json", 'r') as f:
         configs = json.load(f)['H32-train1']
    output_configs = list(set(configs["test"] + configs["train"])) 

    # Paths
    model_path = "working/linear-classifier_single-body_2d_3classes_multi-class.pt"
    npz_folder = f"output/single-body_2d_3classes/seed={seed}/H32-train1/{config}"
    npz_files = [f"image_{out_config}_ep{ep}.npz" for out_config in output_configs]

    # Configs
    pixel_size = 28
    input_dim = pixel_size * pixel_size * 3  # 3 channels
    output_dims = [2, 2, 2]  # Update according to your dataset (e.g., number of shapes, colors, sizes)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model
    model = load_model(model_path, input_dim, output_dims, device)

    # Evaluate all NPZ files and report individual errors
    file_errors = {}

    for npz_file in npz_files:
        npz_path = os.path.join(npz_folder, npz_file)
        overall_error, class_errors, total_images = evaluate_npz(model, npz_path, device)
        file_errors[npz_file] = {
            'overall': overall_error,
            'per_class': class_errors,
            'total_images': total_images
        }

    # Only print final error rates per NPZ file
    f = open(npz_folder + "/evaluation.txt", "w")
    
    print("\n Final Error Rates per NPZ File:")
    for npz_file, errors in file_errors.items():
        print(f"\n {npz_file}:")
        print(f" - Total Images: {errors['total_images']}")
        print(f" - Overall Error Rate: {errors['overall']*100:.2f}%")
        print(f" - Shape Error Rate: {errors['per_class'][0]*100:.2f}%")
        print(f" - Color Error Rate: {errors['per_class'][1]*100:.2f}%")
        print(f" - Size Error Rate: {errors['per_class'][2]*100:.2f}%")

        f.write(f"\n {npz_file}: \n")
        f.write(f" - Total Images: {errors['total_images']} \n")
        f.write(f" - Overall Error Rate: {errors['overall']*100:.2f}% \n")
        f.write(f" - Shape Error Rate: {errors['per_class'][0]*100:.2f}% \n")
        f.write(f" - Color Error Rate: {errors['per_class'][1]*100:.2f}% \n")
        f.write(f" - Size Error Rate: {errors['per_class'][2]*100:.2f}% \n")

    f.close()