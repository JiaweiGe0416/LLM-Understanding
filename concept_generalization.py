import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def sampling(center_list, var_list, n):
    """
    Generate samples from multiple multivariate Gaussian distributions.

    Parameters:
    center_list: List of d-dimensional vectors representing the mean of Gaussian distributions.
    var_list (list of dxd covariance matrices): List of dxd matrices representing the covariance for Gaussian distributions.
    n (int): Number of samples per Gaussian distribution.

    Returns:
    numpy.ndarray: Array of shape (len(center_list) * n, d) containing sampled points.
    """
    samples = []
    
    for center, cov in zip(center_list, var_list):
        center = np.array(center)
        cov = np.array(cov)
        
        # Sample n points from a multivariate Gaussian distribution
        sampled_points = np.random.multivariate_normal(mean=center, cov=cov, size=n)
        samples.append(sampled_points)
    
    return np.vstack(samples)







class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation_type):
        """
        Parameters:
        - input_dim: Input feature dimension
        - hidden_dims: List of hidden layer sizes (e.g., [10, 20, 10])
        - activation_type: 'ReLU' or 'linear'
        """
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # Add multiple hidden layers (without bias)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=False))  # No bias
            if activation_type == 'ReLU':
                layers.append(nn.ReLU())
            elif activation_type == 'linear':
                layers.append(nn.Identity())  # Linear activation
            prev_dim = hidden_dim  # Update input dim for next layer

        # Output layer (without bias)
        layers.append(nn.Linear(prev_dim, input_dim, bias=False))

        # Combine all layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



    

def train_mlp(samples, hidden_dims, activation_type, learning_rate=0.001, batch_size=128, max_epochs=1000, tol=1e-6):
    """
    Train a multi-layer MLP model.

    Parameters:
    - samples: Training samples (numpy array)
    - hidden_dims: List of hidden layer sizes (e.g., [10, 20, 10])
    - activation_type: 'ReLU' or 'linear'
    - learning_rate: Learning rate for SGD
    - batch_size: Training batch size
    - max_epochs: Maximum number of epochs
    - tol: Convergence tolerance

    Returns:
    - trained model
    - loss history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert data to PyTorch tensors
    samples_tensor = torch.tensor(samples, dtype=torch.float32).to(device)

    # Create DataLoader
    dataset = TensorDataset(samples_tensor, samples_tensor)  # Target is same as input
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define model
    input_dim = samples.shape[1]
    model = MLP(input_dim, hidden_dims, activation_type).to(device)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    loss_history = []
    prev_loss = float('inf')
    
    for epoch in range(max_epochs):
        epoch_loss = 0
        for batch in dataloader:
            x_batch, y_batch = batch
            optimizer.zero_grad()
            output = model(x_batch)
            loss = torch.sum((output - y_batch) ** 2)  # L2 squared loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        loss_history.append(epoch_loss)

        # Check for convergence
        if abs(prev_loss - epoch_loss) < tol:
            print(f"Converged at epoch {epoch}, loss: {epoch_loss:.4f}")
            break

        prev_loss = epoch_loss

        # Print progress every 50 epochs
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

    return model, loss_history





def test_model(model, test_points):
    """
    Given a trained MLP model and a set of test points, return and print predictions.
    
    Parameters:
    - model: Trained PyTorch model.
    - test_points: List or NumPy array of test points (shape: (num_points, input_dim))
    
    Returns:
    - predictions: NumPy array of model outputs.
    """
    # Convert test points to a PyTorch tensor
    test_tensor = torch.tensor(test_points, dtype=torch.float32)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_tensor = test_tensor.to(device)
    
    # Ensure model is in evaluation mode (disables dropout, batch norm updates)
    model.eval()
    
    # Compute model predictions
    with torch.no_grad():  # No need for gradients during inference
        predictions = model(test_tensor).cpu().numpy()  # Convert output to NumPy
    
    # Print results
    for i, (inp, pred) in enumerate(zip(test_points, predictions)):
        print(f"For test point {inp}, the model predicts {pred}")
    
    return predictions  # Return predictions for further analysis if needed
