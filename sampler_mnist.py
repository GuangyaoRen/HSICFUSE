import pickle
import jax.numpy as jnp
from jax import random
import numpy as np

with open('dataset/mnist.data', 'rb') as handle:
    mnist_dic = pickle.load(handle)

def ind_X_Y_corrupt(key, N, corrupt_proportion):
    assert corrupt_proportion >= 0
    assert corrupt_proportion <= 1

    max_idx_X = 6000
    key, split_key = random.split(key)
    idx_Y = random.randint(split_key, (N,), 0, 10)
    
    key, split_key = random.split(key)
    idx_X = random.randint(split_key, (N,), 0, max_idx_X)

    X = jnp.array([mnist_dic[str(int(idx_Y[i]))][int(idx_X[i])] for i in range(N)])

    if corrupt_proportion > 0:
        key, split_key = random.split(key)
        idx_corrupt = random.choice(split_key, jnp.arange(N), shape=(int(N * corrupt_proportion),), replace=False)
        
        key, split_key_for_corrupt = random.split(key)
        idx_Y = jnp.where(jnp.isin(jnp.arange(N), idx_corrupt), 
                          random.randint(split_key_for_corrupt, (N,), 0, 10), 
                          idx_Y)

    # Y = idx_Y.reshape(N, 1).astype(jnp.float64)
    Y = idx_Y.reshape(N, 1).astype(jnp.float32)
    return X, Y

# # Example usage
# key = random.PRNGKey(0)
# X, Y = ind_X_Y_corrupt(key, 1000, 0.1)

def ind_X_Y_corrupt_np(seed, N, corrupt_proportion):
    assert corrupt_proportion >= 0
    assert corrupt_proportion <= 1
    max_idx_X = 6000
    rs = np.random.RandomState(0)
    idx_Y = rs.randint(10, size=N)
    idx_X = rs.randint(max_idx_X, size=N)
    X = np.array([mnist_dic[str(idx_Y[i])][idx_X[i]] for i in range(N)])
    if corrupt_proportion > 0:
        idx_corrupt = rs.choice([i for i in range(N)], size=int(N * corrupt_proportion), replace=False)
        idx_Y[idx_corrupt] = rs.randint(10, size=int(N * corrupt_proportion))
    Y = np.array(idx_Y.reshape(N, 1), dtype="float64")
    return X, Y