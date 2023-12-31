import jax
import jax.numpy as jnp
from jax import random, jit, vmap, lax
from functools import partial
from jax.scipy.special import logsumexp


@partial(jit, static_argnums = (3, 4, 5, 6, 7, 8))
def hsicfuse(
    X, 
    Y,
    key,
    alpha               = 0.05,
    kernels             = ("laplace", "gaussian"),
    lambda_multiplier   = 1,
    number_bandwidths   = 10, # only at this value, the empirical alpha is around 0.05
    number_permutations = 2000,
    return_p_val        = False
):
    """
    Independet HSIC-FUSE test.
    
    Given a date of (X, Y) pairs from an unknown distribution P_xy,
    return 0 if the test fails to reject the null    (i.e., P_xy  = P_xP_y independent), or
    return 1 if the test rejects the null            (i.e., P_xy != P_xP_y dependent)
    
    Parameters
    ----------
    X                  : array
                         The shape of X is (m, d), where n is the number of samples and d is the dimension
    Y                  : array
                         The shape of X is (n, d), where n is the number of samples and d is the dimension 
    key                : scalar
                         Jax random key (can be generated by jax.random.PRNKey(seed) for an integer seed).
    alpha              : scalar
                         The level of significance, must in (0, 1).
    kernels            : str or list
                         The list should contain strings. The value of the strings must be: 
                         "gaussian", "laplace", "rq", "imq", 
                         "matern_0.5_l1", "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1",
                         "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2", "matern_4.5_l2".
    lambda_multiplier  : scalar
                         The value of lamba_multiplier must be position.
                         The regulariser lambda is taken to be jnp.sart(n*(n-1))*lambda_multiplier,
    number_bandwidths  : int
                         The number of bandwidths per kernel to include in the collection.
    number_permutations: int
                         The number of permuted test statistics to approximate the quantiles.
    return_p_val       : boolean
                         If true, the p-value is returned.
                         If false, the test output Indicator{p_val <= alpha} is returned.
                         
    Returns
    -------
    output             : int
                         0 if the HSIC-FUSE test fails to reject the null
                           (i.e., P_xy  = P_xP_y independent)
                         1 if the HSIC-FUSE test rejects the null
                           (i.e., P_xy != P_xP_y dependent)
    """
    # Assertions
    m = X.shape[0]
    n = Y.shape[0]
    # mn = m + n
    assert n >= 2 and m >= 2
    assert m == n
    assert 0 < alpha and alpha < 1
    assert lambda_multiplier > 0
    assert number_bandwidths >= 1 and type(number_bandwidths) == int
    assert number_permutations > 0 and type(number_permutations) == int
    if type(kernels) is str:
        # convert to list
        kernels = (kernels,)
    for kernel in kernels:
        assert kernel in (
            "gaussian",
            "laplace",
            "rq",
            "imq",
            "matern_0.5_l2",
            "matern_1.5_l2",
            "matern_2.5_l2",
            "matern_3.5_l2",
            "matern_4.5_l2",
            
            "matern_0.5_l1",
            "matern_1.5_l1",
            "matern_2.5_l1",
            "matern_3.5_l1",
            "matern_4.5_l1",
        )
    # Lists of kerenel for l1 and l2
    all_kernels_l1 = (
        "laplace",
        "matern_0.5_l1",
        "matern_1.5_l1",
        "matern_2.5_l1",
        "matern_3.5_l1",
        "matern_4.5_l1",
    )
    all_kernels_l2 = (
        "gaussian",
        "rq",
        "imq",
        "matern_0.5_l2",
        "matern_1.5_l2",
        "matern_2.5_l2",
        "matern_3.5_l2",
        "matern_4.5_l2",
    )
    number_kernels = len(kernels)
    kernels_l1 = [k for k in kernels if k in all_kernels_l1]
    kernels_l2 = [k for k in kernels if k in all_kernels_l2]

    # Setup for permutations
    key, subkey = random.split(key)
    B = number_permutations
    # (B+1, m): rows of permuted indices
    idx = random.permutation(subkey, jnp.array([[i for i in range(m)]] * (B + 1)), axis=1, independent=True)  
    # set the last row to be the original indices (identity map)
    idx = idx.at[B].set(jnp.array([i for i in range(m)]))
    
    # Compute all permuted HSIC estimates
    N = number_bandwidths * number_kernels
    M = jnp.zeros((N, B + 1)) # each entry of M stores a statistic
    kernel_count = -1 # to count which kernel we are for all different bandwidths 
    for r in range(2):
        kernels_l = (kernels_l1, kernels_l2)[r]
        l = ("l1", "l2")[r]
        if len(kernels_l) > 0:
            # pairwise distance matrix for X
            pairwise_matrix_X = distances(X, X, l, matrix=True)
            # pairwise distance matrix for Y
            pairwise_matrix_Y = distances(Y, Y, l, matrix=True)
            
            # collection of bandwidths
            def compute_bandwidths(distances, number_bandwidths):
                median = jnp.median(distances)
                distances = distances + (distances == 0) * median
                dd = jnp.sort(distances)
                lambda_min = dd[(jnp.floor(len(dd) * 0.05).astype(int))] / 2
                lambda_max = dd[(jnp.floor(len(dd) * 0.95).astype(int))] * 2
                bandwidths = jnp.linspace(lambda_min, lambda_max, number_bandwidths)
                return bandwidths
            
            # distance and bandwidths for X 
            distance_X = pairwise_matrix_X[jnp.triu_indices(pairwise_matrix_X.shape[0])]
            bandwidths_X = compute_bandwidths(distance_X, number_bandwidths) 
            # distance and bandwidths for Y 
            distance_Y = pairwise_matrix_Y[jnp.triu_indices(pairwise_matrix_Y.shape[0])]
            bandwidths_Y = compute_bandwidths(distance_Y, number_bandwidths)            
            
            # compute all permuted HSIC estimates for either l1 or l2
            for j in range(len(kernels_l)):
                kernel = kernels_l[j]
                kernel_count += 1 # starts from 0
                for i in range(number_bandwidths):
                    # compute kerenel matrix K and set diagonal to zero
                    bandwidth_X = bandwidths_X[i]
                    K = kernel_matrix(pairwise_matrix_X, l, kernel, bandwidth_X)
                    K = K.at[jnp.diag_indices(K.shape[0])].set(0)
                    # compute kerenel matrix L and set diagonal to zero
                    bandwidth_Y = bandwidths_Y[i]
                    L = kernel_matrix(pairwise_matrix_Y, l, kernel, bandwidth_Y)
                    L = L.at[jnp.diag_indices(L.shape[0])].set(0)
                    
                    

                    # # compute nomalizer
                    # nomalizer = jnp.sqrt(jnp.sum(K**4)) * jnp.sqrt(jnp.sum(L**4)) / (n*(n-1))
                    # # a vector of ones with dimension of n
                    # ones = jnp.ones(n) 
                    # # compute HSIC permuted values (only permute L)
                    # # followting the equation (5) from Song et al. 2012 (quadratic time)
                    # compute_hsic = lambda index : (jnp.trace(K @ L[index][:, index]) 
                    #                                + (ones @ K @ ones) * (ones @ L @ ones) / ((n-1)*(n-2))
                    #                                - 2 * (ones @ K @ L[index][:, index] @ ones) / (n-2)
                    #                               ) / (n*(n-3)) 
                    # # vectorise the equation
                    # # hsic_values = vmap(compute_hsic)(idx) # This allows parallel computation # Batch size too large
                    # hsic_values = lax.map(compute_hsic, idx)
                    # # set each row of M to be the HSIC values (the last one is the original statistic(s))
                    # M = M.at[kernel_count * number_bandwidths + i].set(hsic_values / jnp.sqrt(nomalizer))

                    # compute HSIC permuted values (B + 1, )
                    # Song et al., Feature Selection via Dependence Maximization, Equation 5
                    nomalizer = jnp.sqrt(jnp.sum(K**4)) * jnp.sqrt(jnp.sum(L**4)) / (n*(n-1))
                    def compute_hsic(index):
                        L_perm = L[index][:, index]
                        ones = jnp.ones((n, 1))
                        term1 = jnp.trace(K @ L_perm)
                        term2 = jnp.sum(K)*jnp.sum(L_perm) / ((n-1) * (n-1))
                        term3 = (2 / (n-2)) * jnp.dot(jnp.dot(jnp.dot(ones.T, K), L_perm), ones)
                        result = (term1 + term2 - term3) / (n * (n-3))
                        return result[0,0]
                    hsic_values = lax.map(compute_hsic, idx) # (B + 1, )
                    
                    # hsic_term_2 = jnp.sum(K) * jnp.sum(L) / (m - 1) / (m - 2)
                    # def compute_hsic(index): 
                    #     K_perm = K[index][:, index]
                    #     K_perm_L = K_perm @ L
                    #     hsic_term_1 = jnp.trace(K_perm_L)
                    #     hsic_term_3 = jnp.sum(K_perm_L) / (m - 2)
                    #     return (hsic_term_1 + hsic_term_2 - 2 * hsic_term_3) / m / (m - 3)
                    # hsic_values = lax.map(compute_hsic, idx) # (B + 1, )

                    # def compute_hsic(index):
                    #     L_perm = L[index][:, index]
                    #     term = 0
                    #     for i in range(n):
                    #         for j in range(n):
                    #             for r in range(n):
                    #                 for s in range(n):
                    #                     if i != j and i != r and i != s and j != r and j != s and r != s:
                    #                         term += 0.25 * (K[i,j] - K[i,r] - K[j,r] + K[r,s]) * (L_perm[i,j] - L_perm[i,r] - L_perm[j,r] + L_perm[r,s])
                    #     return term/n/(n-1)/(n-2)/(n-3)
                    # hsic_values = lax.map(compute_hsic, idx) # (B + 1, )
                    
                    # def compute_hsic(index): 
                    #     L_perm = L[index][:, index]
                    #     K_L_perm = K @ L_perm
                    #     hsic_term_1 = jnp.trace(K_L_perm)
                    #     hsic_term_3 = jnp.sum(K_L_perm) / (m - 2)
                    #     return (hsic_term_1 + hsic_term_2 - 2 * hsic_term_3) / m / (m - 3)
                    # hsic_values = lax.map(compute_hsic, idx) # (B + 1, )
                    
                    # set each row of M to be the HSIC values (the last one is the original statistic(s))
                    M = M.at[kernel_count * number_bandwidths + i].set(hsic_values / jnp.sqrt(nomalizer))   
    
    # compute permuted and original statistics
    all_statistics = logsumexp(lambda_multiplier * M, axis=0, b = 1 / N) # (B+1,)
    original_statistic = all_statistics[-1] # (1,)
    
    # compute statistics and test output
    p_val = jnp.mean(all_statistics >= original_statistic)
    output = p_val <= alpha # p-value <= alpha, we reject null
    
    # return output
    if return_p_val:
        return output.astype(int), p_val 
    else:
        return output.astype(int) # In Jax: False.astype(int) = 0


def distances(X, Y, l, max_samples=None, matrix=False):
    if l == "l1":
        def dist(x, y):
            z = x - y
            return jnp.sum(jnp.abs(z))
    elif l == "l2":
        def dist(x, y):
            z = x - y
            return jnp.sqrt(jnp.sum(jnp.square(z)))
    else:
        raise ValueError("Value of 'l' must be either 'l1' or 'l2'.")
    vmapped_dist = vmap(dist, in_axes=(0, None))
    pairwise_dist = vmap(vmapped_dist, in_axes=(None, 0))
    output = pairwise_dist(X[:max_samples], Y[:max_samples])
    if matrix:
        return output
    else:
        return output[jnp.triu_indices(output.shape[0])]                       


def kernel_matrix(pairwise_matrix, l, kernel, bandwidth, rq_kernel_exponent=0.5):
    """
    Compute kernel matrix for a given kernel and bandwidth.

    inputs: pairwise_matrix: (2m,2m) matrix of pairwise distances
            l: "l1" or "l2" or "l2sq"
            kernel: string from ("gaussian", "laplace", "imq", "matern_0.5_l1", "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1", "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2", "matern_4.5_l2")
    output: (2m,2m) pairwise distance matrix

    Warning: The pair of variables l and kernel must be valid.
    """
    d = pairwise_matrix / bandwidth
    if kernel == "gaussian" and l == "l2":
        return jnp.exp(-(d**2) / 2)
    elif kernel == "laplace" and l == "l1":
        return jnp.exp(-d * jnp.sqrt(2))
    elif kernel == "rq" and l == "l2":
        return (1 + d**2 / (2 * rq_kernel_exponent)) ** (-rq_kernel_exponent)
    elif kernel == "imq" and l == "l2":
        return (1 + d**2) ** (-0.5)
    elif (kernel == "matern_0.5_l1" and l == "l1") or (
        kernel == "matern_0.5_l2" and l == "l2"
    ):
        return jnp.exp(-d)
    elif (kernel == "matern_1.5_l1" and l == "l1") or (
        kernel == "matern_1.5_l2" and l == "l2"
    ):
        return (1 + jnp.sqrt(3) * d) * jnp.exp(-jnp.sqrt(3) * d)
    elif (kernel == "matern_2.5_l1" and l == "l1") or (
        kernel == "matern_2.5_l2" and l == "l2"
    ):
        return (1 + jnp.sqrt(5) * d + 5 / 3 * d**2) * jnp.exp(-jnp.sqrt(5) * d)
    elif (kernel == "matern_3.5_l1" and l == "l1") or (
        kernel == "matern_3.5_l2" and l == "l2"
    ):
        return (
            1 + jnp.sqrt(7) * d + 2 * 7 / 5 * d**2 + 7 * jnp.sqrt(7) / 3 / 5 * d**3
        ) * jnp.exp(-jnp.sqrt(7) * d)
    elif (kernel == "matern_4.5_l1" and l == "l1") or (
        kernel == "matern_4.5_l2" and l == "l2"
    ):
        return (
            1
            + 3 * d
            + 3 * (6**2) / 28 * d**2
            + (6**3) / 84 * d**3
            + (6**4) / 1680 * d**4
        ) * jnp.exp(-3 * d)
    else:
        raise ValueError('The values of "l" and "kernel" are not valid.')

        