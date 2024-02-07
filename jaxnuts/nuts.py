import jax
import jax.numpy as jnp
from typing import Union, Tuple, Dict
from jax.lax import while_loop, cond
import numpy as np

from jax.tree_util import register_pytree_node_class, Partial

# class PRNGKeySequence:
#     def __init__(self, seed: int) -> None:
#         self.key = jax.random.PRNGKey(seed=seed)

#     def __next__(self):
#         _, self.key = jax.random.split(self.key)
#         return self.key

#     def __iter__(self):
#         return self

#     def __call__(self):
#         return self.__next__()


@register_pytree_node_class
class PRNGKeySequence:
    def __init__(self, seed: int) -> None:
        self.key = jax.random.PRNGKey(seed=seed)

    @jax.jit
    def __call__(self):
        _, self.key = jax.random.split(self.key)
        return self.key

    def tree_flatten(self):
        _, self.key = jax.random.split(self.key)
        children = (self.key[-1],)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class NoUTurnSampler:
    def __init__(self, loglik):
        
        self.theta_loglik = Partial(jax.jit(loglik))
        self.theta_loglik_grad = Partial(jax.jit(jax.grad(loglik)))
        self.theta_r_loglik = Partial(
            jax.jit(lambda theta, r: loglik(theta) - 0.5 * jnp.dot(r, r))
        )
        self.theta_r_lik = Partial(
            jax.jit(lambda theta, r: jnp.exp(self.theta_r_loglik(theta, r)))
        )

        # TODO: move to `__call__`
        self.seed = 1234
        self.png_key_seq = PRNGKeySequence(self.seed)
        self.delta_max = 1_000

    def tree_flatten(self):
        children = (self.theta_loglik,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    
    def __call__(
        self, theta_0, M, M_adapt=None, delta=None, gamma=None, kappa=None, t_0=None
    ):
        M_adapt = M_adapt if M_adapt else M // 2
        delta = delta if delta else 0.5 * (0.95 + 0.25)
        gamma = gamma if gamma else 0.05
        kappa = kappa if kappa else 0.75
        t_0 = t_0 if t_0 else 10

        eps = self._find_reasonable_epsilon(theta=theta_0)
        # eps = 1.
        mu = jnp.log(10 * eps)
        eps_bar = 1.0
        H_bar = 0.0

        dim_theta = theta_0.shape[0]
        theta_samples = jnp.empty((M + 1, dim_theta))
        theta_samples = theta_samples.at[0].set(theta_0)

        for m in range(1, M + 1):
            theta_m = theta_samples[m - 1]
            r_0 = jax.random.multivariate_normal(
                key=self.png_key_seq(),
                mean=jnp.zeros(dim_theta),
                cov=jnp.eye(dim_theta),
            )
            u = jax.random.uniform(
                key=self.png_key_seq(), minval=0, maxval=self.theta_r_lik(theta_m, r_0)
            )

            # initialize vars
            theta_m_minus_one = theta_samples[m - 1]
            theta_minus, theta_plus = (
                theta_samples[m - 1],
                theta_samples[m - 1],
            )
            r_minus, r_plus = r_0, r_0

            j = 0
            s = 1
            n = 1
            

            while s == 1:
                v_j = 2 * jax.random.bernoulli(self.png_key_seq()) - 1

                if v_j == -1:
                    (
                        theta_minus,
                        r_minus,
                        theta_prime,
                        n_prime,
                        s_prime,
                        alpha,
                        n_alpha,
                    ) = self._build_tree_for_loop(theta_minus, r_minus, u, v_j, j, eps, theta_m_minus_one, r_0)
                else:
                    (
                        theta_plus,
                        r_plus,
                        theta_prime,
                        n_prime,
                        s_prime,
                        alpha,
                        n_alpha,
                    ) = self._build_tree_for_loop(theta_plus, r_plus, u, v_j, j, eps, theta_m_minus_one, r_0)

                if s_prime == 1:
                    if (
                        jax.random.uniform(key=self.png_key_seq(), minval=0, maxval=1)
                        < n_prime / n
                    ):  # prob of transistion
                        theta_m = theta_prime
                    
                    
                theta_delta = theta_plus - theta_minus
                s *= (
                    s_prime
                    * (jnp.dot(theta_delta, r_minus) >= 0)
                    * (jnp.dot(theta_delta, r_plus) >= 0)
                )
                n += n_prime
                j += 1

            if m < M_adapt:  # adapt accpetance params
                # split out updates to avoid having to save vectors
                H_bar *= 1 - 1 / (m + t_0)
                H_bar += +(delta - alpha / n_alpha) / (m + t_0)
                # on log scale
                eps = mu - jnp.sqrt(m) / gamma * H_bar
                eps_bar = m**-kappa * eps + (1 - m**-kappa) * jnp.log(eps_bar)
                # exponentiate for next iter
                eps = jnp.exp(eps)
                eps_bar = jnp.exp(eps_bar)
            else:
                eps = eps_bar
            
            theta_samples = theta_samples.at[m].set(theta_m)

        return theta_samples

    def _build_tree_for_loop(self, theta, r, u, v, j, eps, theta_0, r_0):
        joint_loglik_0 = self.theta_r_loglik(theta_0, r_0)
        
        s_prime = 1
        n_prime = 0
        alpha_prime = 0
        n_alpha_prime = 0

        theta_prime = theta

        for _ in range(2**j):
            theta_plus_or_minus, r_plus_or_minus = self._leapfrog(theta, r, eps * v)
            joint_loglik_plus_or_minus = self.theta_r_loglik(theta_plus_or_minus, r_plus_or_minus)
            
            
            n_prime += int(u <= jnp.exp(joint_loglik_plus_or_minus))

            if u <= jnp.exp(joint_loglik_plus_or_minus):
                n_prime += 1
                if (
                    jax.random.uniform(key=self.png_key_seq(), minval=0, maxval=1)
                    < 1 / n_prime
                ):  # prob of transistion
                    theta_prime = theta_plus_or_minus

            s_prime = int(jnp.log(u) < joint_loglik_plus_or_minus + self.delta_max)
        # can move this inside forloop for earyly termination?
        theta_delta = (
            theta_plus_or_minus - theta
        ) * v  # need to reverse order if going backwards in time
        s_prime *= int(
            (jnp.dot(theta_delta, r_plus_or_minus) >= 0) * (jnp.dot(theta_delta, r) >= 0)
        )
        alpha_prime += min(1, jnp.exp(joint_loglik_plus_or_minus - joint_loglik_0))
        n_alpha_prime += 1

        return theta_plus_or_minus, r_plus_or_minus, theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime

    
    @jax.jit
    def _leapfrog(self, theta, r, eps):
        r_tilde = r + 0.5 * eps * self.theta_loglik_grad(theta)
        theta_tilde = theta + eps * r_tilde
        r_tilde = r_tilde + 0.5 * eps * self.theta_loglik_grad(theta_tilde)
        return theta_tilde, r_tilde

    
    def _find_reasonable_epsilon(self, theta):
        ln2 = jnp.log(2)
        dim_theta = theta.shape[0]
        eps = 1.0
        r = jax.random.multivariate_normal(
            key=self.png_key_seq(),
            mean=jnp.zeros(dim_theta),
            cov=jnp.eye(dim_theta),
        )

        theta_prime, r_prime = self._leapfrog(theta, r, eps)
        ln_p = self.theta_r_loglik(theta, r)
        ln_p_prime = self.theta_r_loglik(theta_prime, r_prime)

        alpha = 2 * int(ln_p_prime - ln_p > -ln2) - 1
        while alpha * (ln_p_prime - ln_p) > -alpha * ln2:
            eps *= 2.0**alpha
            theta_prime, r_prime = self._leapfrog(theta, r, eps)
            ln_p_prime = self.theta_r_loglik(theta_prime, r_prime)

        return eps
