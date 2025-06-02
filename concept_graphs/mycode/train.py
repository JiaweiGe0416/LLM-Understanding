import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from load_dataset import get_dataloader
import numpy as np
from tqdm import tqdm

# Define Conditional U-Net Model
class ConditionalUNet(nn.Module):
    def __init__(self, input_channels=3, condition_dim=5):
        super(ConditionalUNet, self).__init__()

        self.condition_emb = nn.Linear(condition_dim, 64)

        # Downsampling
        self.down1 = self.conv_block(input_channels, 64)
        self.down2 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Upsampling
        self.up1 = self.conv_block(256 + 128, 128)
        self.up2 = self.conv_block(128 + 64, 64)
        self.final_conv = nn.Conv2d(64, input_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x, condition):
        condition_emb = self.condition_emb(condition).unsqueeze(-1).unsqueeze(-1)
        condition_emb = condition_emb.expand(-1, -1, x.shape[2], x.shape[3])

        # Encoder
        d1 = self.down1(x)
        p1 = self.pool(d1)

        d2 = self.down2(p1)
        p2 = self.pool(d2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        up1 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        up1 = torch.cat([up1, d2], dim=1)
        up1 = self.up1(up1)

        up2 = F.interpolate(up1, scale_factor=2, mode='bilinear', align_corners=False)
        up2 = torch.cat([up2, d1], dim=1)
        up2 = self.up2(up2)

        # Final output
        out = self.final_conv(up2)
        return out


# Define forward diffusion noise schedule
def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

# Compute alphas and betas
def prepare_diffusion_schedule(n_T):
    betas = linear_beta_schedule(n_T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return betas, alphas, alphas_cumprod

# Forward diffusion (adds noise)
def forward_diffusion(x0, t, alphas_cumprod, device):
    noise = torch.randn_like(x0).to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)

    xt = sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise
    return xt, noise  # Return noisy image and actual noise

# Loss function for denoising (MSE loss between predicted and actual noise)
criterion = nn.MSELoss()

# Training function for DDPM
def train_model(n_T=1000, num_epochs=20, batch_size=4, learning_rate=0.0002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare diffusion schedule
    betas, alphas, alphas_cumprod = prepare_diffusion_schedule(n_T)
    betas, alphas, alphas_cumprod = betas.to(device), alphas.to(device), alphas_cumprod.to(device)

    # Load dataset
    train_loader = get_dataloader(batch_size=batch_size, training=True)

    # Initialize model and optimizer
    model = ConditionalUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, labels in progress_bar:
            images = images.to(device)
            
            labels["shape"] = torch.tensor([1 if shape == "circle" else 0 for shape in labels["shape"]], dtype=torch.long)

            # Extract conditioning information
            conditions = torch.stack([
                labels["shape"],
                # labels["x_position"],
                # labels["y_position"],
                labels["color"][:, 0],
                labels["color"][:, 1],
                labels["color"][:, 2],
                labels["size"]
                # labels["bg_color"][:, 0],
                # labels["bg_color"][:, 1],
                # labels["bg_color"][:, 2]
            ], dim=1).to(device)

            # Sample a random diffusion step for each image
            t = torch.randint(0, n_T, (images.shape[0],), device=device)

            # Apply forward diffusion (adding noise)
            xt, noise = forward_diffusion(images, t, alphas_cumprod, device)

            # Predict the noise using Conditional U-Net
            predicted_noise = model(xt, conditions)

            # Compute loss (how close the predicted noise is to the actual noise)
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {epoch_loss / len(train_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "models/ddpm_conditional_unet.pth")
    print("Training complete! Model saved.")



# Ensure script does not execute when imported
if __name__ == '__main__':
    train_model()
