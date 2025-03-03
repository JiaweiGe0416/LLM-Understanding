import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


import importlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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



    
def train_mlp(samples, hidden_dims, activation_type, targets, learning_rate=0.005, batch_size=128, max_epochs=1000, tol=1e-4):    
    """
    Train a multi-layer MLP model.

    Parameters:
    - samples: Training samples (numpy array)
    - hidden_dims: List of hidden layer sizes (e.g., [10, 20, 10])
    - activation_type: 'ReLU' or 'linear'
    - targets: Training targets (numpy array)
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
    
    targets_tensor = torch.tensor(targets, dtype=torch.float32).to(device)
    dataset = TensorDataset(samples_tensor, targets_tensor) 


    # Create DataLoader
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        loss_history.append(epoch_loss)

        # Check for convergence
        if abs(prev_loss - epoch_loss) < tol:
            # print(f"Converged at epoch {epoch}, loss: {epoch_loss:.4f}")
            break

        prev_loss = epoch_loss

        # Print progress every 50 epochs
        # if epoch % 50 == 0:
        #     print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

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
    # for i, (inp, pred) in enumerate(zip(test_points, predictions)):
    #     print(f"For test point {inp}, the model predicts {pred}")
    
    return predictions  # Return predictions for further analysis if needed





def collect_clusterPredictions(hidden_dim, activation_type, center_list, var_list, test_centers, test_vars, target_fn=None, n_test=5, k=1, lo=10, hi=100, step=1, learning_rate=0.001):
    """
    Collect predictions of model on test set over values of n (number of training samples) between lo and hi. 
    Also collect (averaged over k) training loss history models trained on each of range(lo,hi,step) samples.

    Inputs:
    - hidden_dim: List containing number of hidden neurons per layer
    - activation_type: {'linear', 'ReLU}
    - center_list: List of d-dimensional vectors representing the mean of Gaussian train clusters.
    - var_list (list of dxd covariance matrices): List of dxd matrices representing the covariance for Gaussian train clusters.
    - test_centers: List of d-dimensional centers of Gaussian test clusters
    - test_vars (list of dxd covariance matrices): List of dxd matrices representing the covariance for Gaussian test clusters.
    - target_fn: Function mapping sample points to target points (defaults to identity map)
    - n_test (int): number of test points sampled from each test cluster
    - k (int): number of predictions/models trained for each choice of n
    - lo (int): min n
    - hi (int): max n
    - step (int): intervals of n

    Returns:
    - evals (n_sizes x k x n_testPoints x dim): Array of all model predictions (over lo <= n < hi) for each test point
    - losses: List (ragged in dim 2) of all training loss histories for each n in (lo,hi,step)
    - test_points: List of sampled test points (n_test from each cluster)
    """
    evals = []
    losses = []
    # sample test points
    test_points = sampling(test_centers, test_vars, n_test)

    for i in range(lo,hi,step):
        curr_losses = []    
        curr_tests = []
        for j in range(k):
            # print("on iteration", j, " of k=", k)
            samples = sampling(center_list, var_list, i)    # n_testPoints x 3
            # Train the MLP
            if target_fn is None:
                train_targets = samples # identity map by default
            else:
                train_targets = target_fn(samples)
            trained_model, loss_history= train_mlp(samples, hidden_dim, activation_type, targets=train_targets, max_epochs=10000, learning_rate=learning_rate)
            predictions = test_model(trained_model, test_points) # n_testPoints x 3
            curr_tests.append(predictions)
            curr_losses.append(loss_history)
        
        losses.append(curr_losses)  # curr_losses is k x len(loss_history)
        evals.append(curr_tests)    # curr_tests is k x n_testPoints x 3
        
    return np.array(evals), test_points, losses




def compute_clusterLoss(evals, test_targets, test_centers, n_test):
    """
    Compute test loss averaged over points sampled from each test cluster over num training samples.

    Inputs:
    - evals (np.array): Array (n_sizes x k x n_testPoints x dim) of all model predictions (over lo <= n < hi) for each test point
    - test_targets (List): List of all test targets
    - test_centers (List): List of centers of all test clusters
    - n_test (int): number of test points sampled from each test cluster

    Output: 
    - loss_list (np.array): Array (len(test_centers) x n_sizes) of average test loss in each cluster
    """
    n_sizes = len(evals)
    k = len(evals[0])
    n_testPoints = len(evals[0][0]) # total number of test points = n_test x len(test_centers)
    n_testCenters = len(test_centers)
    
    loss_list = [[] for _ in range(n_testCenters)] # store avg prediction loss per test cluster
    for test_idx in range(n_testPoints):
        cluster_idx = test_idx // n_test    # idx of curr test cluster
        test_target = test_targets[test_idx]
        test_eval = evals[:,:,test_idx] # n_sizes x k x 3
        # print('test_idx =', test_idx)
        # print('test_eval =', test_eval)
        # print('test_eval.shape =', np.array(test_eval).shape)
        # print('test_target =', np.array([np.vstack([test_target] * k)] * n_sizes))
        # print('test_target.shape =', np.array([np.vstack([test_target] * k)] * n_sizes).shape)
        loss = np.sum(np.square(test_eval - np.array([np.vstack([test_target] * k)] * n_sizes)), axis=2) # n_sizes x k
        loss_list[cluster_idx].append(np.mean(loss, axis=1)) #  loss_list after for loop: len(test_centers) x n_test x n_sizes
    loss_list = np.mean(loss_list, axis=1)  # len(test_centers) x n_sizes
    return loss_list


def plot_clusterTestLoss(loss_list, test_centers, hidden_dim, activation_type='ReLU', lo=10, hi=100, step=5, labels=10):
    """
    Plot test loss averaged over points sampled from each test cluster over num training samples.

    Inputs:
    - loss_list (np.array): Array (len(test_centers) x n_sizes) of average test loss in each cluster
    - test_centers (List): List of centers of all test clusters
    - hidden_dim: List containing number of hidden neurons per layer
    - activation_type: {'linear', 'ReLU}
    - lo (int): min n
    - hi (int): max n
    - step (int): intervals of n
    - labels (int): Interval of labels for x axis
    """
    n_testCenters = len(test_centers)
    for cluster_idx in range(n_testCenters):
        plt.plot(loss_list[cluster_idx])
    
    plt.xlabel('n = num_samples')
    plt.ylabel('Test Loss')
    plt.title('Test loss over Num samples (hidden_dim = %s, ' % hidden_dim + activation_type + ')')
    plt.legend([''.join(map(str, test_centers[cluster_idx])) for cluster_idx in range(n_testCenters)])
    plt.xlim(0,(hi-lo)/step)
    plt.xticks(np.arange(0, hi-lo,step)/step, lo + np.arange(0, hi-lo, labels))
    plt.grid(True)
    plt.show()
    
    
def scatter_testPredictions(evals, test_targets, hidden_dim, test_ids, n_test=10, activation_type='ReLU', k=1, lim=True):
    """
    Scatter plot of test predictions for varying num training samples.

    Inputs:
    - evals (np.array): Array (n_sizes x k x n_testPoints x dim) of all model predictions (over lo <= n < hi) for each test point
    - test_targets (List): List of all test targets
    - hidden_dim: List containing number of hidden neurons per layer
    - test_ids: List of test indices to plot
    - n_test (int): number of test points sampled from each test cluster
    - activation_type: {'linear', 'ReLU}
    - k (int): number of predictions/models trained for each choice of n
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    evals = np.array(evals)

    # plot each corner
    test_idList = np.ravel([id * n_test + np.arange(n_test) for id in test_ids])
    #print(test_idList)
    print(evals.shape)
    for test_idx in test_idList:
        # read out predictions for each test point
        test_evals = evals[:, :, test_idx] # n_sizes x k x dim
        #print(test_evals.shape)
        ax.scatter(test_evals[:, :, 0], test_evals[:, :, 1], test_evals[:, :, 2], c=np.arange(len(test_evals)*k, 0, -1), s=10, zorder=1)
        # plot tes point itself
        ax.scatter(test_targets[test_idx, 0], test_targets[test_idx, 1], test_targets[test_idx, 2], c='r', s=20, zorder=2)

    if lim:
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_zlim(-0.1, 1.1)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.title('Test loss over Num samples (hidden_dim = %s, ' % hidden_dim + activation_type + ')')
    plt.show()
    
    
def plot_trainingLoss(losses, loss_n, plot_list, hidden_dim, activation_type='ReLU', k=1, epochs=100):
    """
    Plot a random training loss history from one of k models trained on n train samples for each n in plot_list.

    Inputs:
    - losses (List): List (n_sizes x k x len(loss_history)) of loss histories for all k models trained on n samples for n in loss_n
    - loss_n (List): Sorted list of n's for which we collect loss histories
    - plot_list (List): subset of indices in loss_n for which to actually plot loss histories
    - hidden_dim: List containing number of hidden neurons per layer
    - activation_type: {'linear', 'ReLU}
    - k (int): number of predictions/models trained for each choice of n
    - epochs (int): number of epochs
    """
    n_plots = len(plot_list)
    for loss_idx in plot_list:
        plt.plot(losses[loss_idx][np.random.randint(0, k)])
    
    plt.xlabel('epochs')
    plt.ylabel('Training Loss')
    plt.xlim(0, hi)
    plt.title('Training loss over epochs (hidden_dim = %s, ' % hidden_dim + activation_type + ')')
    plt.legend(['n= % d' % loss_n[loss_idx] for loss_idx in plot_list])
    plt.grid(True)
    plt.show()
    
    
def plot_hiddDim(testLoss_list, test_centers, hidd_list, activation_type='ReLU', lo=100):
    n_testCenters = len(test_centers)
    for t in range(n_testCenters):
        plt.plot(hidd_list, testLoss_list[:, t])

    plt.xlabel('hidden_dim')
    plt.ylabel('test loss')
    plt.title('Test loss over hidden_dim (n = %s, ' % lo + activation_type + ')')
    plt.legend([''.join(map(str, test_centers[cluster_idx])) for cluster_idx in range(n_testCenters)])
    plt.xscale('log')
    plt.grid(True)
    plt.show()
    
    
    
    
# define mapping from samples to targets
def flippedIdentity(samples):
    out = []
    for sample in samples:
        if np.sum(sample) < 1.5: out.append(np.array(sample))
        else: out.append(1-np.array(sample))
    return np.array(out)