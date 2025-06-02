import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def sample_data(d, sigma, n, beta):
    """
    Sample n data points (x_i, y_i) from d-dimensional Gaussian with mean 0 and covariance I_d.
    y_i = x_i^T * beta + eps_i, with eps_i ~ N(0, sigma^2).
    """
    X = np.random.randn(n, d)  # n samples of d-dimensional normal with mean 0, variance 1
    eps = np.random.randn(n) * sigma  # noise
    y = X @ beta + eps
    return X, y


class MLP(nn.Module):
    def __init__(self, d, hidden_layers=[64, 64], activation=nn.ReLU):
        """
        d: input dimension
        hidden_layers: list of integers specifying hidden layer sizes
        activation: activation function class (e.g., nn.ReLU, nn.Tanh, nn.LeakyReLU)
        """
        super(MLP, self).__init__()
        layers = []
        input_size = d

        for h in hidden_layers:
            layers.append(nn.Linear(input_size, h))
            layers.append(activation())  # use the provided activation
            input_size = h
        
        layers.append(nn.Linear(input_size, 1))  # output layer
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x).squeeze()


class CosActivation(nn.Module):
    def forward(self, x):
        return torch.cos(x)


def train_flexible_mlp(X_train, y_train, epochs=1000, lr=0.01, hidden_layers=[2048], activation=nn.ReLU):
    d = X_train.shape[1]
    model = MLP(d, hidden_layers, activation)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    loss_history = []
    


    for epoch in range(epochs):
        # full-batch gradient descent
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        loss_value = loss.item()
        loss_history.append(loss_value)
        
    print(f"\n Final training loss: {loss_history[-1]:.6f}")
        
    return model, loss_history



def test_model(model, d, sigma, n_test, beta):
    X_test, y_test = sample_data(d, sigma, n_test, beta)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    with torch.no_grad():
        y_pred = model(X_test_tensor)
    
    mse = nn.MSELoss()(y_pred, y_test_tensor).item()
    return mse

