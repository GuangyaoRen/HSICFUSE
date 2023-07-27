import jax
import jax.numpy as jnp
from jax import vmap, random, jit, lax
from jax.flatten_util import ravel_pytree
from functools import partial
import itertools


def G(x):
    """
    Function G defined in Section 5.4 of our paper.
    input: x: real number
    output: G(x): real number
    """
    #if -1 < x and x < -0.5:
    #    return np.exp(-1 / (1 - (4 * x + 3) ** 2))
    #if -0.5 < x and x < 0:
    #    return - np.exp(-1 / ( 1 - (4 * x + 1) ** 2))   
    #return 0  
    output = jax.lax.cond(
        jnp.logical_and(-1 < x, x < 0), 
        lambda: jax.lax.cond(
            jnp.logical_and(-1 < x, x < -0.5), 
            lambda: jnp.exp(-1 / (1 - (4 * x + 3) ** 2)), 
            lambda: - jnp.exp(-1 / ( 1 - (4 * x + 1) ** 2)),
        ),
        lambda: 0.,
    )
    return output  


def f_theta(x, p, s, perturbation_multiplier=1):
    """
    Function f_theta defined in in Section 5.4 (Eq. (17)) of our paper.
    inputs: x: (d,) array (point in R^d)
            p: non-negative integer (number of perturbations)
            s: positive number (smoothness parameter of Sobolev ball (Eq. (1))
            perturbation_multiplier: positive number (c_d in Eq. (17))
            seed: integer random seed (samples theta in Eq. (17))
    output: real number f_theta(x) 
    """
    #x = np.atleast_1d(x)
    d = x.shape[0]
    #assert perturbation_multiplier * p ** (-s) * jnp.exp(-d) - 1e-07 <= 1, "density is negative"
    # set {1,...,p}^d
    I = list(itertools.product([i + 1 for i in range(p)], repeat=d))  
    
    #for i in range(len(I)):
    #    output += jnp.prod(jnp.array([G(x[r] * p - I[i][r]) for r in range(d)]))   
    I = jnp.array(I)
    compute_prod = lambda i : jnp.prod(
        lax.map(lambda r : G(x[r] * p - I[i, r]), jnp.arange(d))
    )
    output = jnp.sum(lax.map(compute_prod, jnp.arange(len(I))))

    output *= p ** (-s) * perturbation_multiplier
    
    #if np.min(x) >= 0 and np.max(x) <= 1:
    #    output += 1
    output += jax.lax.cond(
        jnp.logical_and(jnp.min(x) >= 0, jnp.max(x) <= 1), 
        lambda: 1.,
        lambda: 0.,
    )
    return output


# https://github.com/google/jax/discussions/11219
def sample_fun(key, function, num_samples, d, ymax):
     def rejection_sample(args):
        key, all_x, i = args
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(
            subkey, 
            minval=0, 
            maxval=1, 
            shape=(1,),
        )
        key, subkey = jax.random.split(key)
        y = jax.random.uniform(
            subkey, 
            minval=0, 
            maxval=ymax, 
            shape=(1,),
        )

        passed = (y < function(x)).astype(bool)
        all_x = all_x.at[i].add((passed * x)[0])
        i = i + passed[0]
        return key, all_x, i
     X_init = jnp.zeros((num_samples, d))
     _, X, _ = jax.lax.while_loop(
         lambda args: args[2] < num_samples, 
         rejection_sample, 
         (key, X_init, 0),
     )
     return X

@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def sampler_perturbed_uniform(
    key,
    num_random,
    num_samples,
    d,
    number_perturbations,
    scale=1,
):
    sample_fun_vmap = jit(
        partial(
            vmap(sample_fun, in_axes=(0, None, None, None, None))
        ),
        static_argnums=(1, 2, 3, 4),
    )
    p = number_perturbations
    
    # stretch the perturbation to have amplitude 1 * scale
    s = 1 
    perturbation_multiplier = jnp.exp(d) * p ** s * scale 
    
    function = lambda x : f_theta(
        x, 
        p, 
        s, 
        perturbation_multiplier,
    )
    ymax = 1 + scale
    
    keys = random.split(key, num=num_random)
    return sample_fun_vmap(keys, function, num_samples, d, ymax)


def sampler_perturb_2ST(
    key,
    m,
    n,
    d,
    num_random,
    num_perturbations,
    scale=1,
):
    keys = random.split(key)
    X = sampler_perturbed_uniform(
        keys[0],
        num_random,
        m,
        d,
        num_perturbations,
        scale,
    )
    Y = random.uniform(keys[1], shape=(num_random, n, d))
    return X, Y


def sampler_perturb_IndT(
    key,
    m,
    dX,
    dY,
    num_random,
    num_perturbations,
    scale=1,
):
    d = dX + dY
    Z = sampler_perturbed_uniform(
        key,
        num_random,
        m,
        d,
        num_perturbations,
        scale,
    )
    return Z[:, :, :dX], Z[:, :, dX:]

