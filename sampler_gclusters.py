import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt


def sampler_gclusters(key, L, theta, N, permute):
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
    means = [
        jnp.array([0,0]),
        jnp.array([L * jnp.cos(theta), L * jnp.sin(theta)]),
        jnp.array([(L / 2) * jnp.cos(theta) - (L * jnp.sqrt(3) / 2) * jnp.sin(theta), 
                   (L / 2) * jnp.sin(theta) + (L * jnp.sqrt(3) / 2) * jnp.cos(theta)])  
    ]
    # Covariance matrix
    var = 5
    cov = jnp.array([[var, 0], [0, var]])
    
    key, subkey = random.split(key, 2)
    # Select which Gaussian to sample X from
    idx_Y = random.choice(subkey, 3, shape=(N, ), p=jnp.array([1/3, 1/3, 1/3]))
    # Sample Y from the next Gaussian clockwisely (Dependence between X and Y)
    idx_X = (idx_Y + 1) % 3
    
    # Sample X and Y
    key_X = random.split(key, N)
    X = [random.multivariate_normal(key_X, mean=means[i], cov=cov, shape=(N,)) for key_X, i in zip(key_X, idx_X)]
    key_Y = random.split(key, N)
    Y = [random.multivariate_normal(key_Y, mean=means[i], cov=cov, shape=(N,)) for key_Y, i in zip(key_Y, idx_Y)]
    
    if permute == False:
        X, Y = jnp.array(X)[:, 0], jnp.array(Y)[:, 1]
        return X, Y # Dependent
    else:
        idx_Y_permuted = random.permutation(subkey, idx_Y)
        Y_permuted = [random.multivariate_normal(key_Y, mean=means[i], cov=cov, shape=(N,)) 
                      for key_Y, i in zip(key_Y, idx_Y_permuted)]
        X, Y = jnp.array(X)[:, 0], jnp.array(Y_permuted)[:, 1]
        return X, Y # Independent

    
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
    
    
# # Test example
# N = 1000
# L = 1
# theta = 0   
# key = random.PRNGKey(0)
# X, Y = sampler_gclusters(key, L, theta, N, permute = False)
# plot_sampler_gclusters(L, theta, X, Y)

