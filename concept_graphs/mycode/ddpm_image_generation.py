import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from train import ConditionalUNet, prepare_diffusion_schedule  # Reuse from training code

# Load trained model
def load_trained_model(model_path, input_channels=3, condition_dim=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalUNet(input_channels=input_channels, condition_dim=condition_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# Reverse diffusion process (Sampling)
@torch.no_grad()
def sample_image(model, device, condition, n_T=1000):
    # Prepare diffusion schedule using training function
    betas, alphas, alphas_cumprod = prepare_diffusion_schedule(n_T)
    betas, alphas, alphas_cumprod = betas.to(device), alphas.to(device), alphas_cumprod.to(device)

    img_size = 32  # Fixed to 32x32 as per your setup
    x = torch.randn(1, 3, img_size, img_size).to(device)  # Start from pure noise

    for t in reversed(range(n_T)):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

        # Predict noise using the trained model
        predicted_noise = model(x, condition)

        # Reverse diffusion step
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]

        mean = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)

        if t > 0:
            noise = torch.randn_like(x).to(device)
        else:
            noise = torch.zeros_like(x).to(device)

        x = mean + torch.sqrt(beta_t) * noise  # Sample x_{t-1}

    return x

# Display generated image
def show_image(tensor_image):
    to_pil = ToPILImage()
    tensor_image = (tensor_image + 1) / 2  # Rescale from [-1,1] to [0,1]
    img = tensor_image.squeeze(0).cpu().clamp(0, 1)  # Clamp to [0,1]
    plt.imshow(to_pil(img))
    plt.axis('off')
    plt.show()

# # Example Usage
# if __name__ == "__main__":
#     # Load the trained model
#     model, device = load_trained_model("models/ddpm_conditional_unet.pth")

#     # Define new condition (Example: Circle shape, Blue color, Size 0.7)
#     new_condition = torch.tensor([[1, 0.0, 0.0, 1.0, 0.7]], dtype=torch.float32).to(device)
#     # Format: [shape (1=circle, 0=square), R, G, B, size]

#     # Generate image
#     generated_image = sample_image(model, device, new_condition)

#     # Show generated image
#     show_image(generated_image)
