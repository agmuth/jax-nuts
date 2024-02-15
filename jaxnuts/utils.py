import jax.numpy as jnp
from typing import NamedTuple


# class NutsHyperParams(NamedTuple):
#     delta: float=0.5 * (0.95 + 0.25)
#     gamma: float=0.05
#     kappa: float=0.75
#     t_0: int=10
#     delta_max: int=1_000


# class NutsLikelihoods(NamedTuple):
#     theta_loglik: callable=None
#     theta_loglik_grad: callable=None
#     theta_r_loglik: callable=None
#     theta_r_lik: callable=None


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


class SampleWhileLoopArgs(NamedTuple):
    prng_key: jnp.array = (None,)
    theta_plus_minus: jnp.array = (None,)
    r_plus_minus: jnp.array = (None,)
    theta_prime: jnp.array = (None,)
    theta_m: jnp.array = (None,)
    n: int = (0,)
    s: int = (0,)
    alpha: float = (0,)
    n_alpha: int = (0,)
    u: float = (0,)
    eps: float = (0,)
    theta_m_minus_one: jnp.array = (None,)
    r_0: jnp.array = (None,)
    j: int = 0


class SampleForLoopArgs(NamedTuple):
    prng_key: jnp.array = (None,)
    theta_samples: jnp.array = (None,)
    eps_bar: float = (None,)
    H_bar: float = (None,)
    mu: float = (None,)
    M_adapt: int = (None,)


class DualAveragingArgs(NamedTuple):
    eps: float = (None,)
