import jax.numpy as jnp
import numpy as np
from jax import random
import matplotlib.pyplot as plt
    

def shuffle(Y, C):
    """
    Shuffle the last C of pairs in the Y array.
    """
    if C != 0:
        # Determin the number of entries to shuffle
        num_to_shuffle = int(C * Y.shape[0]) 
        
        # Split the array into two parts: the part to keep in order and the part to shuffle
        Y_first = np.copy(Y[:-num_to_shuffle])
        Y_last = np.copy(Y[-num_to_shuffle:])
        
        # Shuffle the last portion
        np.random.shuffle(Y_last)
        
        # Concatenate the two parts back together
        Y_shuffled = np.vstack((Y_first, Y_last))
        
        return Y_shuffled
    else:
        return np.array(Y)


def generate_means(L, d):
    # Create an equidistant point matrix in higher dimensions
    assert d > 1, "d should be greater than 1"
    
    # Set distance
    distance = L
    
    # Initialize points at the origin
    O = np.zeros(d)
    A = np.zeros(d)
    B = np.zeros(d)
    
    # Set first point
    A[0] = distance
    
    # For B, we move along the second dimension in a manner so that OB = OA
    B[1] = (np.sqrt(3) * distance) / 2
    B[0] = distance / 2
    means = np.vstack([O, A, B])

    return means
        

def sampler_gclusters(seed, L, d, theta, N, C):
    """
    Sample an X from any one of the three equidistant two-dimensional Gaussian distributions
    then the Y is sampled from the clockwise adjacent Gaussian.
    The means of these three Gaussian clusters are the vertices an equilateral triangle.
    Assuming one of the means is always (0, 0)
    
    Inputs:
        key    : Random.PRNGKey(seed)
        L      : Scalar (positive)
                 The length of the edge of the equilateral triangle.
                 L is the level of difficulty to tell the dependence between X and Y,
                 the smaller the value of L, the harder it is to tell the independence.                 
        theta  : Scalar (positive)
                 The angle between the edge of the equilateral triangle and the positive x-axis.
        cov    : 2-by-2 covariance matrix (assume to be identity matrix)
        N      : Integer 
                 Number of samples.
        permute: boolean
                 If true: permute Y, otherwise not.
    
    Outputs:
        X: (N, 2)
           Each X[i] is a sample drawn from any one of the three Gaussian distributions.
        Y: (N, 2) 
           Each Y[i] is a sample drawn from the clockwise adjacent Gaussian to the Gaussian where X[i] is drawn.
    """ 
    # means
    means = generate_means(L, d)
    # Covariance matrix
    var = 5.0
    cov = np.eye(d)*var

    np.random.seed(seed)
    idx_Y = np.random.choice(3, size=(N, ), p=np.array([1/3, 1/3, 1/3]))
    idx_X = (idx_Y + 1) % 3
    
    # Sample X, Y
    X = [np.random.multivariate_normal(mean=means[i], cov=cov, size=(N,)) for i in idx_X]
    Y = [np.random.multivariate_normal(mean=means[i], cov=cov, size=(N,)) for i in idx_Y]
    X, Y = np.array(X)[:, 0], np.array(Y)[:, 1]
    Y = shuffle(Y, C)
    
    return X, Y


def plot_sampler_gclusters(L, theta, X, Y):
    plt.figure(figsize=(8, 8))
    # means
    means = [
        jnp.array([0,0]),
        jnp.array([L * jnp.cos(theta), L * jnp.sin(theta)]),
        jnp.array([(L / 2) * jnp.cos(theta) - (L * jnp.sqrt(3) / 2) * jnp.sin(theta), 
                   (L / 2) * jnp.sin(theta) + (L * jnp.sqrt(3) / 2) * jnp.cos(theta)])  
    ]
    mu_1 = [m[0] for m in means]
    mu_2 = [m[1] for m in means]
    plt.scatter(mu_1, mu_2, color='red', marker='x', alpha=1.0, label='Gaussian Centers')

    x_1 = [x[0] for x in X]
    x_2 = [x[1] for x in X]
    plt.scatter(x_1, x_2, color='blue', marker='o', alpha=0.2, label='X Samples')   
    y_1 = [y[0] for y in Y]
    y_2 = [y[1] for y in Y]
    plt.scatter(y_1, y_2, color='orange', marker='o', alpha=0.2, label='Y Samples') 
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Samples from Three Gaussian Clusters')
    plt.legend()
    plt.xticks(range(int(min(x_1))-1, int(max(x_1))+1, 5))
    plt.yticks(range(int(min(y_1))-1, int(max(y_1))+1, 5))
    # plt.grid(True)
    plt.show()

