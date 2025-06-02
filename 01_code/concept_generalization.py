import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

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
    def __init__(self, input_dim, hidden_dims, use_init, var=np.sqrt(6), activation_type='ReLU', folder="", m=0):
        """
        Parameters:
        - input_dim: Input feature dimension
        - hidden_dims: List of hidden layer sizes (e.g., [10, 20, 10])
        - use_init (string):  if None, use default initialization; 
                              if "he", initialize with Kaiming uniform (He initialization);
                              if "pretrained", initialize with pretrained weights;
                              if "perturbed", initialize with perturbed pretrained weights
        - var (float): if initializing with perturbed pretrained weights, scale gaussian noise with var 
                               if initializing with he (around zero), scale bounds by var
        - activation_type: 'ReLU', 'Tanh', or 'linear'
        - folder (string): if initializing with pretrained weights, use weights from folder 
        - m (int): if initializing with pretrained weights, use weights from model m (in range(k))
        """
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # Add multiple hidden layers
        hidden_dim = 0
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim, bias=True)

            if use_init == "pretrained": 
                # initialize with saved weights
                pretrained_weights = torch.load("weights/" + folder + "/dim=%s_layer=" % hidden_dim + "0_k=%s.pth" % m)
                with torch.no_grad():
                    linear.weight.copy_(pretrained_weights['linear.weight'])
                    linear.bias.copy_(pretrained_weights['linear.bias'])
            elif use_init == "perturbed":
                # initialize with perturbed saved weights
                pretrained_weights = torch.load("weights/" + folder + "/dim=%s_layer=" % hidden_dim + "0_k=%s.pth" % m)
                with torch.no_grad():
                    linear.weight.copy_(pretrained_weights['linear.weight'])
                    linear.weight.add_(torch.randn_like(linear.weight) * var * linear.weight.std())
                    linear.bias.copy_(pretrained_weights['linear.bias'])
                    linear.bias.add_(torch.randn_like(linear.bias) * var * linear.bias.std())
            elif use_init == "he" or use_init =="fixedout": 
                # initialize weights around 0 with Kaiming uniform
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(linear.weight, -var * bound, var * bound)
                nn.init.uniform_(linear.bias, -var * bound, var * bound)
    
            layers.append(linear) 

            if activation_type == 'ReLU':
                layers.append(nn.ReLU())
            elif activation_type == 'linear':
                layers.append(nn.Identity())  # Linear activation
            elif activation_type == 'Tanh':
                layers.append(nn.Tanh())
            prev_dim = hidden_dim  # Update input dim for next layer

        # Output layer
        linear = nn.Linear(prev_dim, input_dim, bias=True)
        if use_init == "pretrained": 
            # initialize with saved weights
            pretrained_weights = torch.load("weights/" + folder + "/dim=%s_layer=" % hidden_dim + "1_k=%s.pth" % m)
            with torch.no_grad():
                linear.weight.copy_(pretrained_weights['linear.weight'])
                linear.bias.copy_(pretrained_weights['linear.bias'])
        elif use_init == "perturbed":
            # initialize with perturbed saved weights
            pretrained_weights = torch.load("weights/" + folder + "/dim=%s_layer=" % hidden_dim + "1_k=%s.pth" % m)
            with torch.no_grad():
                linear.weight.copy_(pretrained_weights['linear.weight'])
                linear.weight.add_(torch.randn_like(linear.weight) * var * linear.weight.std())
                linear.bias.copy_(pretrained_weights['linear.bias'])
                linear.bias.add_(torch.randn_like(linear.bias) * var * linear.bias.std())
        elif use_init == "he": 
            # initialize weights around 0 with Kaiming uniform
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(linear.weight, -var * bound, var * bound)
            nn.init.uniform_(linear.bias, -var * bound, var * bound)
        layers.append(linear)

        # Combine all layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



    
def train_mlp(samples, hidden_dims, targets, test_samples=None, test_targets=None, batch_size=128, opt='SGD', lr=0.001, clip=0.0, use_tol=False, tol=1e-7, max_epochs=10000, use_init=None, var=1.0, folder="", m=0, debug=False):    
    """
    Train a multi-layer MLP model.

    Parameters:
    - samples: Training samples (numpy array)
    - hidden_dims: List of hidden layer sizes (e.g., [10, 20, 10])
    - targets: Training targets (numpy array)
    - test_samples: Test samples (numpy array)
    - test_targets: Test targets (numpy array)
    - batch_size: Training batch size
    - opt: 'SGD', 'Adam'
    - lr: Learning rate for optimizer
    - clip (float): clip with maxNorm=clip
    - use_tol (boolean)
    - tol: Convergence tolerance
    - max_epochs (int): Maximum number of epochs
    - use_init (string): if None, use default initialization; 
                         if "he", initialize with Kaiming uniform (He initialization);
                         if "pretrained", initialize with pretrained weights;
                         if "perturbed", initialize with perturbed pretrained weights
    - var (float): if initializing with perturbed pretrained weights, scale gaussian noise with var 
                               if initializing with he (around zero), scale bounds by var
    - folder (string): if initializing with pretrained weights, use weights from folder 
    - m (int): if initializing with learned weights, use model m of k

    Returns:
    - trained model
    - loss history (num_epochs,)]
    - initial weights
    - initial weight norms
    - (optional): test loss history
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
    model = MLP(input_dim, hidden_dims, use_init=use_init, var=var, folder=folder, m=m).to(device)

    # Store initial MLP weights + norms
    w_init = {}
    for name, param in model.named_parameters():
        w_init[name] = copy.deepcopy(param.data)
    layer_names = list(w_init.keys())
    init_weights = [w_init[name] for name in layer_names]
    init_norms = []
    for name in layer_names:
        if "weight" in name:
            init_norms.append(np.linalg.norm(w_init[name], axis=(0,1)))
        elif "bias" in name:
            init_norms.append(np.linalg.norm(w_init[name]))

    # Define optimizer
    if opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)


    # Training loop
    loss_history = []
    prev_loss = float('inf')
    loss_fn = nn.MSELoss()
    testLoss_history = []
    
    for epoch in range(max_epochs):
        epoch_loss = []
        for batch in dataloader:
            x_batch, y_batch = batch
            optimizer.zero_grad()
            output = model(x_batch)

            loss = loss_fn(output, y_batch)
            if (epoch == max_epochs - 1): 
                print("batch loss =", loss.item())
                print("random sample loss =", torch.sum(torch.square((output[0] - y_batch[0]))).item())
                print("random sample loss =", loss_fn(output[0], y_batch[0]).item())

            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()
            epoch_loss.append(loss.item())

        epoch_loss = np.mean(epoch_loss)
        loss_history.append(epoch_loss)
        if test_samples is not None:
            test_centers = [[0,1,1], [1,0,1], [1,1,0], [1,1,1]]
            preds = test_model(model, test_samples)
            testLoss = compute_clusterLoss([[preds]], test_targets, test_centers)
            testLoss_history.append(testLoss)

        if epoch == max_epochs - 1: 
            print("final epoch loss=", epoch_loss)
            
        # Check for convergence
        if use_tol and epoch_loss - prev_loss < tol:
            if debug: print(f"Converged at epoch {epoch}, loss: {epoch_loss:.4f}")
            break

        prev_loss = epoch_loss

        # Print progress every 100 epochs
        if epoch % 1000 == 0:
            if debug: print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

    if testLoss_history:
        return model, loss_history, init_weights, init_norms, testLoss_history
    else: return model, loss_history, init_weights, init_norms





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
    
    return predictions  # Return predictions for further analysis if needed





def collect_clusterPredictions(hidden_dim, center_list, var_list, test_centers=None, test_vars=None, target_fn=None, alpha=0, n_test=20, k=10, lo=100, hi=110, step=10, opt='SGD', lr=0.001, clip=0.0, max_epochs=10000, use_tol=False, use_init=None, var=np.sqrt(6), folder="", debug=True):
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
    - alpha (float): alpha for interpolating flipped & identity maps
    - n_test (int): number of test points sampled from each test cluster
    - k (int): number of predictions/models trained for each choice of n
    - lo (int): min n
    - hi (int): max n
    - step (int): intervals of n to collection predictions for
    - opt: {'SGD', 'Adam'}
    - lr (int): learning rate for optimizer
    - clip (float): clip with maxNorm=clip
    - max_epochs (int): Maximum number of epochs
    - use_tol (boolean): terminate when converged or after fixed number of epochs
    - use_init (string): {'he', 'pretrained'}
    - var (float): if initializing with perturbed pretrained weights, scale gaussian noise with var 
                               if initializing with he (around zero), scale bounds by var
    - folder (string): if initializing with pretrained weights, use weights from folder 

    Returns:
    - (optional) evals (n_sizes x k x n_testPoints x dim): Array of all model predictions (over lo <= n < hi) for each test point
    - (optional) test_samples: List of all points sampled from  test centers used to compute test loss
    - losses: List (ragged in dim 2) of all training loss histories for each n in (lo,hi,step)
    - (optional) test_losses: List (ragged) of all test loss histories 
    - weights: List of learned model weights (n_sizes x 1 x k x n_layers x {size of layer})
    - norms: List of norms of learned model weights (n_sizes x k x n_layers)
    - init_weights: List of initial model weights
    - init_norms: List of iniital model norms
    - r: List of random points used for far corners
    """
    evals = []
    losses = []
    weights = []
    norms = []
    init_weights = []
    init_norms = []
    testLosses = []
    r = []
    np.set_printoptions(threshold=np.inf)

    # sample test points
    test_samples = None
    test_targets = None
    if test_centers is not None:
        test_samples = sampling(test_centers, test_vars, n_test)

    for i in range(lo, hi, step):
        curr_losses = []    
        curr_tests = []
        curr_weights = []
        curr_norms = []
        curr_init_w = []
        curr_init_n = []
        curr_testLosses = []
        for j in range(k):
            if debug: print("on iteration", j, " of k=", k)
            train_samples = sampling(center_list, var_list, i)    # n_testPoints x 3
            # Train the MLP
            if target_fn == "flippedIdentity":
                train_targets = target_fn(train_samples)
            elif target_fn == "randomMap":
                train_targets, r = randomMap(train_samples)
            elif target_fn == "uniformMap":
                train_targets, r = randomMap(train_samples, dist="uniform")
            elif target_fn == "randomBinary":
                train_targets, r = randomBinary(train_samples)
            elif target_fn == "interpolateFlipped":
                train_targets = interpolateFlipped(train_samples, alpha)
                r = alpha
            else: 
                train_targets = train_samples # identity map by default
                test_targets = test_samples
            if test_centers is not None:
                trained_model, loss_history, init_w, init_n, testLoss_history= train_mlp(train_samples, hidden_dim, train_targets, test_samples=test_samples, test_targets=test_targets, opt=opt, lr=lr, clip=clip, max_epochs=max_epochs, use_tol=use_tol, use_init=use_init, var=var, folder=folder, m=j, debug=debug)
                predictions = test_model(trained_model, test_samples) # n_testPoints x 3
                curr_tests.append(predictions)
                curr_testLosses.append(testLoss_history)
            else:
                trained_model, loss_history, init_w, init_n = train_mlp(train_samples, hidden_dim, train_targets, opt=opt, lr=lr, clip=clip, max_epochs=max_epochs, use_tol=use_tol, use_init=use_init, var=var, folder=folder, m=j, debug=debug)
            curr_losses.append(loss_history)
            curr_init_w.append(init_w)
            curr_init_n.append(init_n)
            

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
        init_weights.append(curr_init_w)
        init_norms.append(curr_init_n)
    
    if test_centers is not None:
        testLosses.append(curr_testLosses)
        return np.array(evals), test_samples, losses, testLosses, weights, norms, init_weights, init_norms, r
    else:
        return losses, weights, norms, r




def compute_clusterLoss(evals, test_targets, test_centers, n_test=20, m=-1):
    """
    Compute test loss averaged over points sampled from each test cluster over num training samples.

    Inputs:
    - evals (np.array): Array (n_sizes x k x n_testPoints x dim) of all model predictions (over lo <= n < hi) for each test point
    - test_targets (List): List of all test targets
    - test_centers (List): List of centers of all test clusters
    - n_test (int): number of test points sampled from each test cluster
    - m (int): use model m of k

    Output: 
    - loss_list (np.array): Array (len(test_centers) x n_sizes) of average test loss in each cluster
    """
    n_testPoints = len(evals[0][0]) # total number of test points = n_test x len(test_centers)
    n_testCenters = len(test_centers)
    
    loss_list = [[] for _ in range(n_testCenters)] # store avg prediction loss per test cluster
    for test_idx in range(n_testPoints):
        cluster_idx = test_idx // n_test    # idx of curr test cluster
        test_target = test_targets[test_idx]
        test_eval = np.array(evals)[:,:,test_idx] # n_sizes x k x 3
        loss = np.sum(np.square(test_eval - np.full(test_eval.shape, test_target)), axis=2) # n_sizes x k

        if loss.any() > 1:
            print(loss)
            print(test_eval)
            print(test_target)
        if m >= 0:
            loss_list[cluster_idx].append(loss[:, m])
        else:
            loss_list[cluster_idx].append(np.mean(loss, axis=1)) #  loss_list after for loop: len(test_centers) x n_test x n_sizes

    loss_list = np.mean(loss_list, axis=1)  # len(test_centers) x n_sizes
    return loss_list


def plot_clusterTestLoss(loss_list, test_centers, hidden_dim, lo=100, hi=110, step=10, labels=10, folder=None):
    """
    Plot test loss averaged over points sampled from each test cluster over num training samples.

    Inputs:
    - loss_list (np.array): Array (len(test_centers) x n_sizes) of average test loss in each cluster
    - test_centers (List): List of centers of all test clusters
    - hidden_dim: List containing number of hidden neurons per layer
    - lo (int): min n
    - hi (int): max n
    - step (int): intervals of n
    - labels (int): Interval of labels for x axis
    - folder (str): folder name to save figure in
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

    if folder:
        os.makedirs("outputs/" + folder + "/" + str(hidden_dim), exist_ok=True)
        plt.savefig("outputs/" + folder + "/"+ str(hidden_dim) + "/test.jpg", bbox_inches="tight")
    plt.show()
    
    
def scatter_testPredictions(evals, test_targets, hidden_dim, test_ids, n_test=20, k=10, title=None, folder=None, file=""):
    """
    Scatter plot of test predictions for varying num training samples.

    Inputs:
    - evals (np.array): Array (n_sizes x k x n_testPoints x dim) of all model predictions (over lo <= n < hi) for each test point
    - test_targets (List): List of all test targets
    - hidden_dim: List containing number of hidden neurons per layer
    - test_ids: List of test indices to plot
    - n_test (int): number of test points sampled from each test cluster
    - k (int): number of predictions/models trained for each choice of n
    - title (str): plot title
    - folder (str): folder name to save figure in
    - file (str): file name
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Define a colormap
    cmap = plt.get_cmap("viridis", len(evals)*k)  # discrete colors
    norm = mcolors.Normalize(vmin=0, vmax=len(evals)*k)

    # plot each corner
    test_idList = np.ravel([id * n_test + np.arange(n_test) for id in test_ids])
    for test_idx in test_idList:
        # read out predictions for each test point
        test_evals = evals[:, :, test_idx] # n_sizes x k x dim
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

    if folder:
        os.makedirs("outputs/" + folder + "/" + str(hidden_dim), exist_ok=True)
        plt.savefig("outputs/" + folder + "/"+ str(hidden_dim) + "/scatter" + file + ".jpg", bbox_inches="tight")
    plt.show()
    
    
def plot_trainingLoss(losses, loss_n, plot_list, hidden_dim, trials=[], k=10, epochs=10000, folder=None, file=""):
    """
    Plot a random training loss history from one of k models trained on n train samples for each n in plot_list.

    Inputs:
    - losses (List): List (n_sizes x k x len(loss_history)) of loss histories for all k models trained on n samples for n in loss_n
    - loss_n (List): Sorted list of n's for which we collect loss histories
    - plot_list (List): subset of indices in loss_n for which to actually plot loss histories
    - hidden_dim: List containing number of hidden neurons per layer
    - k (int): number of predictions/models trained for each choice of n
    - epochs (int): number of epochs
    - folder (str): folder name to save figure in
    - file (str): file name to save figure under
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
        if trials:
            plt.legend(['model % d' % j for j in range(k)])
        plt.grid(True)

        if folder:
            os.makedirs("outputs/" + folder + "/", exist_ok=True)
            plt.savefig("outputs/" + folder + "/" + file +".jpg", bbox_inches="tight")
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
    
    
# define mapping from samples to targets in flipped map
def flippedIdentity(samples):
    out = []
    for sample in samples:
        if np.sum(sample) < 1.5: out.append(np.array(sample))
        else: out.append(1-np.array(sample))
    return np.array(out)

# randomly sample targets of far corners from zero-centered gaussian
def randomMap(samples, dist=None, sigma=50, bound=2.0):
    samples = np.array(samples)
    out = []
    mu = [0,0,0]
    cov = [[sigma,0,0],[0,sigma,0],[0,0,sigma]]
    if dist == "uniform":
        r = np.random.uniform(0.0, bound, (4,3))
    else: 
        r = np.random.multivariate_normal(mean=mu, cov=cov, size=4)
    far_points = np.array([[1,1,0],[1,0,1],[0,1,1],[1,1,1]])
    for sample in samples:
        if np.sum(sample) < 1.5: # near corner ex
            out.append(np.array(sample))
        else: # far corner ex
            dist = np.array([np.linalg.norm(sample - pt) for pt in far_points])

            minidx = np.argmin(dist)
            out.append(r[minidx] + sample - far_points[minidx])
    return np.array(out), r

# randomly sample targets of far corners from all eight corners
def randomBinary(samples):
    samples = np.array(samples)
    out = []
    corners = [[0,0,0],[0,0,1],[0,1,0],[1,0,0],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]
    r_idx = np.random.randint(0,8,4)
    r = [corners[i] for i in r_idx]
    far_points = np.array(corners[4:])
    for sample in samples:
        if np.sum(sample) < 1.5: # near corner ex
            out.append(np.array(sample))
        else: # far corner ex
            dist = np.array([np.linalg.norm(sample - pt) for pt in far_points])

            minidx = np.argmin(dist)
            out.append(r[minidx] + sample - far_points[minidx])
    return np.array(out), r

# interpolate targets of far corners between identity and flipped map
def interpolateFlipped(samples, alpha=0):
    samples = np.array(samples)
    out = []
    flippedSamples = flippedIdentity(samples)

    for idx, sample in enumerate(samples):
        if np.sum(sample) < 1.5: # near corner ex
            out.append(np.array(sample))
        else: # far corner ex
            out.append(alpha * flippedSamples[idx] + (1 - alpha) * sample)
    return np.array(out)


def plot_norms(norms, hidden_dim, size=0, param="", folder=None, file=""):
    """
    Plot a norm of weights learned by layer for each of k models.

    Inputs:
    - norms (List): List (n_sizes x k x n_layers) of learned weights
    - hidden_dim: List containing number of hidden neurons per layer
    - param (str): if "weights", plot norms of weights only
                   if "bias", plot norms of biases only
                   else, plot both
    - folder (str): folder name to save figure in
    - file (str): file name to save figure under
    """
    k = len(norms[0])
    n_layers = len(norms[0][0])//2
    if param == "weights":
        idx = 2 * np.arange(n_layers)
        for i in range(k):
            plt.plot([norms[size][i][id] for id in idx])
        plt.xticks(range(n_layers), ['W%s' % n for n in range(n_layers)])
    elif param == "bias":
        idx = 2 * np.arange(n_layers) + 1
        for i in range(k):
            plt.plot([norms[size][i][id] for id in idx])
        plt.xticks(range(n_layers), ['b%s' % n for n in range(n_layers)])
    else:
        for i in range(k):
            plt.plot(norms[size][i])
        plt.xticks(range(2*n_layers), ['W1', 'b1', 'W2', 'b2'])
    plt.ylabel('norm')
    plt.title('Layer norms (hidden_dim = %s, ' % hidden_dim + 'ReLU)')
    if k == 1:
        plt.legend(['avg over k models'])
    else:
        plt.legend(['model % d' % j for j in range(k)])
    plt.grid(True)

    if folder:
        os.makedirs("outputs/" + folder + "/" + str(hidden_dim), exist_ok=True)
        plt.savefig("outputs/" + folder + "/"+ str(hidden_dim) + "/" + file +".jpg", bbox_inches="tight")
    plt.show()


def plot_weightsSVD(n_hiddDim, hidd_list, k, folder=""):
    for i in range(n_hiddDim):
        _, ax = plt.subplots(1, 2, figsize=(10, 4))
        hidden_dim = hidd_list[i]
        for m in range(k):
            W1 = torch.load("weights/" + folder + "/dim=%s_layer=" % hidden_dim + "0_k=%s.pth" % m)
            W2 = torch.load("weights/" + folder + "/dim=%s_layer=" % hidden_dim + "1_k=%s.pth" % m)
            _, S1, _ = np.linalg.svd(W1['linear.weight'], full_matrices=True)
            _, S2, _ = np.linalg.svd(W2['linear.weight'], full_matrices=True)
            ax[0].plot(S1, marker='o', markersize=5)
            ax[1].plot(S2, marker='o', markersize=5)
        ax[0].set_title("Singular Values of W_1 (hidden_dim = %s)" % hidden_dim)
        ax[1].set_title("Singular Values of W_2 (hidden_dim = %s)" % hidden_dim)
        ax[0].set_xlabel("singular values")
        ax[1].set_xlabel("singular values")
        ax[0].grid(True)
        ax[1].grid(True)
        plt.legend(['model % d' % j for j in range(k)], loc="upper left", bbox_to_anchor=(1.1, 0.8))