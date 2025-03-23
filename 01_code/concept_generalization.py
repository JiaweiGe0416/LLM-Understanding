import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


import importlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    def __init__(self, input_dim, hidden_dims, activation_type='ReLU', use_init=None, m=0):
        """
        Parameters:
        - input_dim: Input feature dimension
        - hidden_dims: List of hidden layer sizes (e.g., [10, 20, 10])
        - activation_type: 'ReLU' or 'linear'
        - use_bias (boolean): include bias in layers if True
        - use_init (string):  use default initialization if None; 
                              initialize with Kaiming uniform (He initialization) if "he"
                              initialize with flipped weights if "flipped"
        - m (int): if initializing with flipped weights, use weights from model m (in range(k))
        """
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # Add multiple hidden layers
        hidden_dim = 0
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim, bias=True)
            if use_init == "flipped": 
                # initialize with saved weights
                pretrained_weights = torch.load("dim=%s_layer=" % hidden_dim + "0_k=%s.pth" % m)
                with torch.no_grad():
                    linear.weight.copy_(pretrained_weights['linear.weight'])
                    linear.bias.copy_(pretrained_weights['linear.bias'])
            elif use_init == "he": 
                # initialize weights around 0 with Kaiming uniform
                nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
                nn.init.uniform_(linear.bias, -(6 / prev_dim) ** 0.5, (6 / prev_dim) ** 0.5)
            layers.append(linear) 

            if activation_type == 'ReLU':
                layers.append(nn.ReLU())
            elif activation_type == 'linear':
                layers.append(nn.Identity())  # Linear activation
            prev_dim = hidden_dim  # Update input dim for next layer

        # Output layer
        linear = nn.Linear(prev_dim, input_dim, bias=True)
        if use_init == "flipped": 
            # initialize with saved weights
            pretrained_weights = torch.load("dim=%s_layer=" % hidden_dim + "1_k=%s.pth" % m)
            with torch.no_grad():
                linear.weight.copy_(pretrained_weights['linear.weight'])
                linear.bias.copy_(pretrained_weights['linear.bias'])
        elif use_init == "he": 
            # initialize weights around 0 with Kaiming uniform
            nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            nn.init.uniform_(linear.bias, -(6 / prev_dim) ** 0.5, (6 / prev_dim) ** 0.5)
        layers.append(linear)

        # Combine all layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



    
def train_mlp(samples, hidden_dims, targets, lr=0.001, opt='SGD', use_init=None, m=0, batch_size=128, max_epochs=10000, tol=1e-7, use_tol=False, debug=False):    
    """
    Train a multi-layer MLP model.

    Parameters:
    - samples: Training samples (numpy array)
    - hidden_dims: List of hidden layer sizes (e.g., [10, 20, 10])
    - targets: Training targets (numpy array)
    - lr: Learning rate for optimizer
    - opt: 'SGD', 'Adam'
    - use_init (string):  use default initialization if None; 
                        initialize with Kaiming uniform (He initialization) if "he"
                        initialize with flipped weights if "flipped"
    - m (int): if initializing with learned weights, use model m of k
    - batch_size: Training batch size
    - max_epochs: Maximum number of epochs
    - tol: Convergence tolerance

    Returns:
    - trained model
    - loss history (num_epochs,)
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
    model = MLP(input_dim, hidden_dims, use_init=use_init, m=m).to(device)

    # Define optimizer
    if opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)


    # Training loop
    loss_history = []
    prev_loss = float('inf')
    
    for epoch in range(max_epochs):
        epoch_loss = 0
        for batch in dataloader:
            x_batch, y_batch = batch
            optimizer.zero_grad()
            output = model(x_batch)

            loss = nn.functional.mse_loss(output, y_batch)
            if (epoch == max_epochs - 1): 
                print("batch loss =", loss)
                print("random sample loss =", nn.functional.mse_loss(output[0], y_batch[0]))

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            epoch_loss += torch.sum(torch.square((output - y_batch)))/len(samples)

        loss_history.append(epoch_loss)
        if (epoch == max_epochs - 1): 
            print("final epoch loss=", epoch_loss)
            
        # Check for convergence
        if use_tol and epoch_loss - prev_loss < tol:
            if debug: print(f"Converged at epoch {epoch}, loss: {epoch_loss:.4f}")
            break

        prev_loss = epoch_loss

        # Print progress every 100 epochs
        if epoch % 1000 == 0:
            if debug: print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

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





def collect_clusterPredictions(hidden_dim, center_list, var_list, test_centers, test_vars, target_fn=None, n_test=20, k=10, lo=100, hi=110, step=10, lr=0.001, opt='SGD', use_tol=False, use_init=None, m=0, debug=True):
    """
    Collect predictions of model on test set over values of n (number of training samples) between lo and hi. 
    Also collect (averaged over k) training loss history models trained on each of range(lo,hi,step) samples.

    Inputs:
    - hidden_dim: List containing number of hidden neurons per layer
    - center_list: List of d-dimensional vectors representing the mean of Gaussian train clusters.
    - var_list (list of dxd covariance matrices): List of dxd matrices representing the covariance for Gaussian train clusters.
    - test_centers: List of d-dimensional centers of Gaussian test clusters
    - test_vars (list of dxd covariance matrices): List of dxd matrices representing the covariance for Gaussian test clusters.
    - target_fn: Function mapping sample points to target points (defaults to identity map)
    - n_test (int): number of test points sampled from each test cluster
    - k (int): number of predictions/models trained for each choice of n
    - lo (int): min n
    - hi (int): max n
    - step (int): intervals of n to collection predictions for
    - lr (int): learning rate for optimizer
    - opt: {'SGD', 'Adam'}
    - use_tol (boolean): terminate when converged or after fixed number of epochs
    - use_init (string): {'he', 'flipped'}

    Returns:
    - evals (n_sizes x k x n_testPoints x dim): Array of all model predictions (over lo <= n < hi) for each test point
    - losses: List (ragged in dim 2) of all training loss histories for each n in (lo,hi,step)
    - test_points: List of sampled test points (  from each cluster)
    - weights: List of learned model weights (n_sizes x 1 x k x n_layers x {size of layer})
    - norms: List of norms of learned model weights (n_sizes x k x n_layers)
    """
    evals = []
    losses = []
    weights = []
    norms = []
    np.set_printoptions(threshold=np.inf)

    # sample test points
    test_samples = sampling(test_centers, test_vars, n_test)

    for i in range(lo,hi,step):
        curr_losses = []    
        curr_tests = []
        curr_weights = []
        curr_norms = []
        for j in range(k):
            if debug: print("on iteration", j, " of k=", k)
            train_samples = sampling(center_list, var_list, i)    # n_testPoints x 3
            # Train the MLP
            if target_fn is None:
                train_targets = train_samples # identity map by default
            else:
                train_targets = target_fn(train_samples)
            trained_model, loss_history= train_mlp(train_samples, hidden_dim, train_targets, lr=lr, opt=opt, use_init=use_init, m=m, use_tol=use_tol, debug=debug)
            predictions = test_model(trained_model, test_samples) # n_testPoints x 3
            curr_tests.append(predictions)
            curr_losses.append(loss_history)

            # Store MLP weights + norms
            w = {}
            for name, param in trained_model.named_parameters():
                w[name] = param.data
            layer_names = list(w.keys())
            curr_weights.append([w[name] for name in layer_names])
            curr_norms.append([])
            for name in layer_names:
                if "weight" in name:
                    curr_norms[j].append(np.linalg.norm(w[name], axis=(0,1)))
                elif "bias" in name:
                    curr_norms[j].append(np.linalg.norm(w[name]))
        
        losses.append(curr_losses)  # curr_losses is k x len(loss_history)
        evals.append(curr_tests)    # curr_tests is k x n_testPoints x 3
        weights.append(curr_weights) # curr_weights is k x 2 x {weight_shape}
        norms.append(curr_norms)    # curr_norms is k x 2
        
    return np.array(evals), test_samples, losses, weights, norms




def compute_clusterLoss(evals, test_targets, test_centers, n_test=20, m=-1):
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
    n_testPoints = len(evals[0][0]) # total number of test points = n_test x len(test_centers)
    n_testCenters = len(test_centers)
    
    loss_list = [[] for _ in range(n_testCenters)] # store avg prediction loss per test cluster
    for test_idx in range(n_testPoints):
        cluster_idx = test_idx // n_test    # idx of curr test cluster
        test_target = test_targets[test_idx]
        test_eval = evals[:,:,test_idx] # n_sizes x k x 3
        loss = np.sum(np.square(test_eval - np.full(test_eval.shape, test_target)), axis=2) # n_sizes x k
        #loss = nn.functional.mse_loss(torch.tensor(test_eval), torch.tensor(test_target))
        if loss.any() > 1:
            print(loss)
            print(test_eval)
            print(test_target)
        if m >= 0:
            loss_list[cluster_idx].append(loss[:, m])
        else:
            loss_list[cluster_idx].append(np.mean(loss, axis=1)) #  loss_list after for loop: len(test_centers) x n_test x n_sizes
        #loss_list[cluster_idx].append(loss)
    loss_list = np.mean(loss_list, axis=1)  # len(test_centers) x n_sizes
    return loss_list


def plot_clusterTestLoss(loss_list, test_centers, hidden_dim, lo=100, hi=110, step=10, labels=10):
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
    
    plt.xlabel('num_samples')
    plt.ylabel('Test Loss')
    plt.title('Test loss (hidden_dim = %s, ' % hidden_dim + 'ReLU)')
    plt.legend([''.join(map(str, test_centers[cluster_idx])) for cluster_idx in range(n_testCenters)])
    plt.xlim(0,(hi-lo)/step)
    plt.xticks(np.arange(0, hi-lo,step)/step, lo + np.arange(0, hi-lo, labels))
    plt.grid(True)
    plt.show()
    
    
def scatter_testPredictions(evals, test_targets, hidden_dim, test_ids, n_test=20, k=10, title=None):
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
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Define a colormap
    cmap = plt.get_cmap("viridis", len(evals)*k)  # discrete colors
    norm = mcolors.Normalize(vmin=0, vmax=len(evals)*k)

    # plot each corner
    test_idList = np.ravel([id * n_test + np.arange(n_test) for id in test_ids])
    #print(test_idList)
    #print(evals.shape)
    for test_idx in test_idList:
        # read out predictions for each test point
        test_evals = evals[:, :, test_idx] # n_sizes x k x dim
        #print(test_evals.shape)
        ax.scatter(test_evals[:, :, 0], test_evals[:, :, 1], test_evals[:, :, 2], c=np.arange(0, len(test_evals)*k, 1), cmap=cmap, norm=norm, s=10, zorder=1)
        # plot test point itself
        ax.scatter(test_targets[test_idx, 0], test_targets[test_idx, 1], test_targets[test_idx, 2], c='r', s=20, zorder=2)

    legend_labels = {}
    for i in range(k):
        legend_labels[i] = "model %s" % i
    legend_labels[k + 1] = "test points"
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(i)), markersize=10) 
            for i in range(k)]
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10))
    ax.legend(handles, legend_labels.values(), loc="upper left", bbox_to_anchor=(1.1, 0.8))
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Leaves space for legend
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_zlim(-0.1, 1.1)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    if title:
        plt.title(title)
    else:
        plt.title('Test predictions (hidden_dim = %s, ' % hidden_dim + 'ReLU)')
    plt.show()
    
    
def plot_trainingLoss(losses, loss_n, plot_list, hidden_dim, trials=[], k=10, epochs=10000):
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
    for loss_idx in plot_list:
        for trial in trials:
            plt.plot(losses[loss_idx][trial])
        if not trials:
            plt.plot(losses[loss_idx])
    
        plt.xlabel('epochs')
        plt.ylabel('Training Loss')
        plt.xlim(0, epochs)
        plt.title('Training loss (hidden_dim = %s, ' % hidden_dim + 'ReLU, n=' + str(loss_n[loss_idx]) + ')')
        if not trials:
            plt.legend(['avg over k models'])
        else:
            plt.legend(['model % d' % j for j in range(k)])
        plt.grid(True)
        plt.show()
    
    
def plot_hiddDim(testLoss_list, test_centers, hidd_list, lo=100):
    n_testCenters = len(test_centers)
    for t in range(n_testCenters):
        plt.plot(hidd_list, testLoss_list[:, t])

    plt.xlabel('hidden_dim')
    plt.ylabel('test loss')
    plt.title('Test loss over hidden_dim (n = %s, ' % lo + 'ReLU)')
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


def plot_norms(norms, hidden_dim, size=0):
    """
    Plot a norm of weights learned by layer for each of k models.

    Inputs:
    - norms (List): List (n_sizes x k x n_layers) of learned weights
    - hidden_dim: List containing number of hidden neurons per layer
    - activation_type: {'linear', 'ReLU}
    """
    k = len(norms[0])
    n_layers = len(norms[0][0])
    for i in range(k):
        plt.plot(norms[size][i])

    plt.ylabel('norm')
    plt.title('Layer norms (hidden_dim = %s, ' % hidden_dim + 'ReLU)')
    if k == 1:
        plt.legend(['avg over k models'])
    else:
        plt.legend(['model % d' % j for j in range(k)])
    plt.xticks(range(n_layers), ['W1', 'b1', 'W2', 'b2'])
    plt.grid(True)
    plt.show()