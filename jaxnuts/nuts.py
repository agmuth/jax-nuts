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
    def __init__(
        self,
        loglik,
        delta=0.5 * (0.95 + 0.25),
        gamma=0.05,
        kappa=0.75,
        t_0=10,
        delta_max=1_000,
        seed=1234,
    ):
        self.theta_loglik = Partial(jax.jit(loglik))
        self.theta_loglik_grad = Partial(jax.jit(jax.grad(loglik)))
        self.theta_r_loglik = Partial(
            jax.jit(lambda theta, r: loglik(theta) - 0.5 * jnp.dot(r, r))
        )
        self.theta_r_lik = Partial(
            jax.jit(lambda theta, r: jnp.exp(self.theta_r_loglik(theta, r)))
        )

        self.delta = delta
        self.gamma = gamma
        self.kappa = kappa
        self.t_0 = t_0
        self.delta_max = delta_max
        self.seed = seed
        self.png_key_seq = PRNGKeySequence(self.seed)

    def tree_flatten(self):
        children = (
            self.theta_loglik,
            self.delta,
            self.gamma,
            self.kappa,
            self.t_0,
            self.delta_max,
            self.seed,
        )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __call__(
        self,
        theta_0,
        M,
        M_adapt=None,
    ):
        eps = self._find_reasonable_epsilon(theta=theta_0)
        eps_bar = 1.0
        H_bar = 0.0
        mu = jnp.log(10 * eps)

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
                    ) = self._build_tree_while_loop(
                        theta_minus, r_minus, u, v_j, j, eps, theta_m_minus_one, r_0
                    )
                else:
                    (
                        theta_plus,
                        r_plus,
                        theta_prime,
                        n_prime,
                        s_prime,
                        alpha,
                        n_alpha,
                    ) = self._build_tree_while_loop(
                        theta_plus, r_plus, u, v_j, j, eps, theta_m_minus_one, r_0
                    )

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
                eps, eps_bar, H_bar = self._dual_average(eps, eps_bar, H_bar, mu, alpha, n_alpha, m)
            else:
                eps = eps_bar

            theta_samples = theta_samples.at[m].set(theta_m)

        return theta_samples

    @jax.jit
    def _dual_average(self, eps, eps_bar, H_bar, mu, alpha, n_alpha, m):
        # split out updates to avoid having to save vectors
        H_bar *= 1 - 1 / (m + self.t_0)
        H_bar += +(self.delta - alpha / n_alpha) / (m + self.t_0)
        
        # on log scale
        eps = mu - jnp.sqrt(m) / self.gamma * H_bar
        eps_bar = m**-self.kappa * eps + (1 - m**-self.kappa) * jnp.log(eps_bar)
        
        # exponentiate for next iter
        eps = jnp.exp(eps)
        eps_bar = jnp.exp(eps_bar)

        return eps, eps_bar, H_bar

    def _build_tree_single_step(self, theta_star, r_star, u, v, eps, theta_0, r_0):
        ln_u = jnp.log(u)

        theta_double_star, r_double_star = self._leapfrog(
            theta_star, r_star, v * eps
        )  # push edge out one leapfrog step
        joint_loglik_double_star = self.theta_r_loglik(theta_double_star, r_double_star)
        joint_loglik_0 = self.theta_r_loglik(theta_0, r_0)

        n = (
            ln_u <= joint_loglik_double_star
        )  # indicator for if new edge state is eligible
        s = (
            ln_u <= joint_loglik_double_star + self.delta_max
        )  # early termination criteria (equ. 3)

        alpha = jnp.minimum(1.0, jnp.exp(joint_loglik_double_star - joint_loglik_0))
        n_alpha = 1

        return theta_double_star, r_double_star, n, s, alpha, n_alpha

    @staticmethod
    def _check_for_u_turn(theta_plus, r_plus, theta_minus, r_minus, v):
        theta_delta = (
            theta_plus - theta_minus
        ) * v  # need to reverse order if args passed in backwards
        return (jnp.dot(theta_delta, r_plus) >= 0) * (
            jnp.dot(theta_delta, r_minus) >= 0
        )

    def _build_tree_while_loop(self, theta_star, r_star, u, v, j, eps, theta_0, r_0):
        left_leaf_nodes = jnp.array(
            [(theta_star, r_star)] * j
        )  # array for storing leftmost leaf nodes in any subtree currently under consideration

        # HMC path vars
        theta_prime = theta_star
        s = 1
        n = 1

        # dual averaging vars
        alpha = 0.0
        n_alpha = 0

        i = 0  # counter
        while s == 1 and i < 2**j:
            i += 1  # incr here to align with b-tree 1-indexing
            (
                theta_double_star,
                r_double_star,
                n_prime,
                s_prime,
                alpha_prime,
                n_alpha_prime,
            ) = self._build_tree_single_step(
                theta_star, r_star, u, v, eps, theta_0, r_0
            )

            # update HMC path vars
            s *= s_prime
            n += n_prime

            if n_prime == 1:
                if (
                    jax.random.uniform(key=self.png_key_seq(), minval=0, maxval=1)
                    < 1 / n
                ):  # prob of transistion
                    theta_prime = theta_double_star

            # update dual averaging vars
            alpha += alpha_prime
            n_alpha += n_alpha_prime
            if j == 0:
                continue  # no u-turns possible

            # check for u-turn / update reference data used to check for u-turns
            if i % 2 == 1:
                """
                if `i` is odd then it is the left-most leaf node of at least one balanced subtree
                -> overwrite correspoding stale position value in `left_leaf_nodes`

                `i` is the left-most leaf node in a tree of height `k` if i%2**k == 1
                we can range k from 1 to `j` to get if the current state is a left-most leaf node in the
                current subtree of height `k`. Summing these values gives a unique index to store these
                states in `left_leaf_nodes`. Moreover it is safe to overwrite values in `left_leaf_nodes`
                using this indexing schema since we are moving across the b-tree left-to-right (not proven
                here but if a state is a left-most leaf node `l` times by the time we get to the next
                state used `l` times we will have already made `l` u-turn checks against the previous
                `l` times state)
                """

                idx = -1  # python is 0-indexed b-tree is 1-indexed
                for k in range(1, j + 1):
                    idx += i % (2**k) == 1
                left_leaf_nodes = left_leaf_nodes.at[idx].set(
                    (theta_double_star, r_double_star)
                )
            else:
                """
                if `i` is even then it is the right-most leaf node of at least one balanced subtree
                -> check for u-turn

                `i` is the right-most leaf node in a tree of height `k` if i%(2**k) == 0
                we can range from k from 1 to j to get which subtrees the current
                state is a right most leaf node of
                """
                for k in range(1, j + 1):
                    if i % (2**k) != 0:
                        continue
                    s *= self._check_for_u_turn(
                        theta_double_star,
                        r_double_star,
                        left_leaf_nodes[k][0],  # theta
                        left_leaf_nodes[k][1],  # r
                        v,
                    )

        return theta_double_star, r_double_star, theta_prime, n, s, alpha, n_alpha

    def _build_tree_for_loop(self, theta, r, u, v, j, eps, theta_0, r_0):
        joint_loglik_0 = self.theta_r_loglik(theta_0, r_0)

        s_prime = 1
        n_prime = 0
        alpha_prime = 0
        n_alpha_prime = 0

        theta_prime = theta

        for _ in range(2**j):
            theta_plus_or_minus, r_plus_or_minus = self._leapfrog(theta, r, eps * v)
            joint_loglik_plus_or_minus = self.theta_r_loglik(
                theta_plus_or_minus, r_plus_or_minus
            )

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
            (jnp.dot(theta_delta, r_plus_or_minus) >= 0)
            * (jnp.dot(theta_delta, r) >= 0)
        )
        alpha_prime += min(1, jnp.exp(joint_loglik_plus_or_minus - joint_loglik_0))
        n_alpha_prime += 1

        return (
            theta_plus_or_minus,
            r_plus_or_minus,
            theta_prime,
            n_prime,
            s_prime,
            alpha_prime,
            n_alpha_prime,
        )

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
