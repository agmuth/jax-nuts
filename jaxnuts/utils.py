import jax.numpy as jnp
from typing import NamedTuple


class NutsHyperParams(NamedTuple):
    delta: float=0.5 * (0.95 + 0.25)
    gamma: float=0.05
    kappa: float=0.75
    t_0: int=10
    delta_max: int=1_000
    

class NutsLikelihoods(NamedTuple):
    theta_loglik: callable=None
    theta_loglik_grad: callable=None
    theta_r_loglik: callable=None
    theta_r_lik: callable=None


class BuildTreeWhileLoopArgs(NamedTuple):
    # passed in by call to function
    theta_star: jnp.array = None
    r_star: jnp.array = None
    u: float = None
    v: int = None
    j: int = None
    eps: float = None
    theta_0: jnp.array = None
    r_0: jnp.array = None
    
    
    
    # HMC path vars
    s: int = None
    n: int = None

    # dual averaging vars
    theta_prime: jnp.array = None
    alpha: float = None
    n_alpha: int = None

    left_leaf_nodes: jnp.array = None
    
    # counter
    i: int = None
    
    prng_key: jnp.array = None
        
    

    """Should be subset of `BuildTreeWhileLoopArgs`"""
    # passed in by call to function
    theta_star: jnp.array
    r_star: jnp.array
    u: float
    v: int
    j: int
    eps: float
    theta_0: jnp.array
    r_0: jnp.array
    
    # HMC path vars
    theta_prime = jnp.array
    s: int
    n: int

    # dual averaging vars
    alpha: float
    n_alpha: int

    # counter
    i: int
        
    left_leaf_nodes: jnp.array