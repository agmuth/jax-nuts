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
        # loglik = jax.jit(loglik)
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

    # def __call__(
    #     self, theta_0, M, M_adapt=None, delta=None, gamma=None, kappa=None, t_0=None
    # ):
    #     # TODO: accept as args
    #     M_adapt = M_adapt if M_adapt else M // 2
    #     delta = delta if delta else 0.5 * (0.95 + 0.25)
    #     gamma = gamma if gamma else 0.05
    #     kappa = kappa if kappa else 0.75
    #     t_0 = t_0 if t_0 else 10

    #     eps = self._find_reasonable_epsilon(theta=theta_0)
    #     # eps = 1.
    #     mu = jnp.log(10 * eps)
    #     eps_bar = 1.0
    #     H_bar = 0.0

    #     dim_theta = theta_0.shape[0]
    #     theta_samples = jnp.empty((M + 1, dim_theta))
    #     theta_samples = theta_samples.at[0].set(theta_0)

    #     for m in range(1, M + 1):
    #         theta_m = theta_samples[m - 1]
    #         r_0 = jax.random.multivariate_normal(
    #             key=self.png_key_seq(),
    #             mean=jnp.zeros(dim_theta),
    #             cov=jnp.eye(dim_theta),
    #         )
    #         u = jax.random.uniform(
    #             key=self.png_key_seq(), minval=0, maxval=self.theta_r_lik(theta_m, r_0)
    #         )

    #         # initialize vars
    #         theta_m_minus_one = theta_samples[m - 1]
    #         theta_minus, theta_plus = (
    #             theta_samples[m - 1],
    #             theta_samples[m - 1],
    #         )
    #         r_minus, r_plus = r_0, r_0

    #         j = 0
    #         s = 1
    #         n = 1

    #         while s == 1:
    #             v_j = 2 * jax.random.bernoulli(self.png_key_seq()) - 1

    #             if v_j == -1:
    #                 (
    #                     theta_minus,
    #                     r_minus,
    #                     _,
    #                     _,
    #                     theta_prime,
    #                     n_prime,
    #                     s_prime,
    #                     alpha,
    #                     n_alpha,
    #                 ) = self._build_tree(
    #                     theta_minus, r_minus, u, v_j, j, eps, theta_m_minus_one, r_0
    #                 )
    #             else:
    #                 (
    #                     _,
    #                     _,
    #                     theta_plus,
    #                     r_plus,
    #                     theta_prime,
    #                     n_prime,
    #                     s_prime,
    #                     alpha,
    #                     n_alpha,
    #                 ) = self._build_tree(
    #                     theta_plus, r_plus, u, v_j, j, eps, theta_m_minus_one, r_0
    #                 )

    #             if s_prime == 1:
    #                 if jax.random.uniform(self.png_key_seq()) <= jnp.where(
    #                     n_prime == 0, 0, n_prime / n
    #                 ):
    #                     theta_m = theta_prime

    #             n += n_prime
    #             theta_delta = theta_plus - theta_minus
    #             s *= (
    #                 s_prime
    #                 * (jnp.dot(theta_delta, r_minus) >= 0)
    #                 * (jnp.dot(theta_delta, r_plus) >= 0)
    #             )
    #             j += 1

    #         if m < M_adapt:  # adapt accpetance params
    #             # split out updates to avoid having to save vectors
    #             H_bar *= 1 - 1 / (m + t_0)
    #             H_bar += +(delta - alpha / n_alpha) / (m + t_0)
    #             # on log scale
    #             eps = mu - jnp.sqrt(m) / gamma * H_bar
    #             eps_bar = m**-kappa * eps + (1 - m**-kappa) * jnp.log(eps_bar)
    #             # exponentiate for next iter
    #             eps = jnp.exp(eps)
    #             eps_bar = jnp.exp(eps_bar)
    #         else:
    #             eps = eps_bar

    #         theta_samples = theta_samples.at[m].set(theta_m)

    #     return theta_samples

    def __call__(
        self, theta_0, M, M_adapt=None, delta=None, gamma=None, kappa=None, t_0=None
    ):
        # TODO: accept as args
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
            # if m == 5:
            #     print()
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
            C = [(theta_m, r_0)]

            while s == 1:
                v_j = 2 * jax.random.bernoulli(self.png_key_seq()) - 1

                if v_j == -1:
                    (
                        theta_minus,
                        r_minus,
                        C_prime,
                        s_prime,
                    ) = self._build_tree_backwards_in_time(
                        theta_minus, r_minus, u, v_j, j, eps
                    )
                else:
                    (theta_plus, r_plus, C_prime, s_prime) = self._build_tree_forwards_in_time(
                        theta_plus, r_plus, u, v_j, j, eps
                    )

                if s_prime == 1:
                    C += C_prime
                theta_delta = theta_plus - theta_minus
                s *= (
                    s_prime
                    * (jnp.dot(theta_delta, r_minus) >= 0)
                    * (jnp.dot(theta_delta, r_plus) >= 0)
                )
                j += 1

            idx = jax.random.randint(self.png_key_seq(), shape=(1,), minval=0, maxval=len(C))
            theta_m = C[idx[0]][0]
            theta_samples = theta_samples.at[m].set(theta_m)

        return theta_samples

    def _build_tree_forwards_in_time(self, theta, r, u, v, j, eps):
        C_prime = list()
        s_prime = 1
        
        
        for _ in range(2**j):
            theta_prime, r_prime = self._leapfrog(theta, r, eps * v)
            joint_loglik_prime = self.theta_r_loglik(theta_prime, r_prime)
            
            if u <= jnp.exp(joint_loglik_prime):
                C_prime.append((theta_prime, r_prime))
            
            s_prime = int(jnp.log(u) < joint_loglik_prime + self.delta_max)
        
        theta_delta = theta_prime - theta
        s_prime *= int(
            (jnp.dot(theta_delta, r) >= 0)
            * (jnp.dot(theta_delta, r_prime) >= 0)
        )
        return theta_prime, r_prime, C_prime, s_prime
    
    
    def _build_tree_backwards_in_time(self, theta, r, u, v, j, eps):
        C_prime = list()
        s_prime = 1
        
        
        for _ in range(2**j):
            theta_prime, r_prime = self._leapfrog(theta, r, eps * v)
            joint_loglik_prime = self.theta_r_loglik(theta_prime, r_prime)
            
            if u <= jnp.exp(joint_loglik_prime):
                C_prime.append((theta_prime, r_prime))
            
            s_prime = int(jnp.log(u) < joint_loglik_prime + self.delta_max)
        
        theta_delta = theta - theta_prime
        s_prime *= int(
            (jnp.dot(theta_delta, r_prime) >= 0)
            * (jnp.dot(theta_delta, r) >= 0)
        )
        return theta_prime, r_prime, C_prime, s_prime
            

    def _build_tree(self, theta, r, u, v, j, eps):
        if j == 0:
            theta_prime, r_prime = self._leapfrog(theta, r, eps * v)
            joint_loglik_prime = self.theta_r_loglik(theta_prime, r_prime)
            C_prime = list()
            if u <= jnp.exp(joint_loglik_prime):
                C_prime.append((theta_prime, r_prime))
            s_prime = int(jnp.log(u) < joint_loglik_prime + self.delta_max)
            return theta_prime, r_prime, theta_prime, r_prime, C_prime, s_prime
        else:
            (
                theta_minus,
                r_minus,
                theta_plus,
                r_plus,
                C_prime,
                s_prime,
            ) = self._build_tree(theta, r, u, v, j - 1, eps)
            if v == -1:
                (
                    theta_minus,
                    r_minus,
                    _,
                    _,
                    C_double_prime,
                    s_double_prime,
                ) = self._build_tree(theta_minus, r_minus, u, v, j - 1, eps)
            else:
                (
                    _,
                    _,
                    theta_plus,
                    r_plus,
                    C_double_prime,
                    s_double_prime,
                ) = self._build_tree(theta_plus, r_plus, u, v, j - 1, eps)

            theta_delta = theta_plus - theta_minus
            s_prime *= int(
                s_double_prime
                * (jnp.dot(theta_delta, r_minus) >= 0)
                * (jnp.dot(theta_delta, r_plus) >= 0)
            )
            C_prime += C_double_prime
        return (theta_minus, r_minus, theta_plus, r_plus, C_prime, s_prime)

    # def _build_tree(self, theta, r, u, v, j, eps, theta_0, r_0):
    #     if j == 0:
    #         # base case - take one leapfrog step in the direction of v
    #         theta_prime, r_prime = self._leapfrog(theta, r, eps * v)

    #         joint_loglik_prime = self.theta_r_loglik(theta_prime, r_prime)
    #         joint_loglik_0 = self.theta_r_loglik(theta_0, r_0)
    #         delta_loglik = joint_loglik_prime - joint_loglik_0

    #         n_prime = int(u <= jnp.exp(joint_loglik_prime))
    #         s_prime = int(jnp.log(u) < joint_loglik_prime + self.delta_max)

    #         alpha = min(1, jnp.exp(delta_loglik))
    #         n_alpha = 1

    #         return (
    #             theta_prime,
    #             r_prime,
    #             theta_prime,
    #             r_prime,
    #             theta_prime,
    #             n_prime,
    #             s_prime,
    #             alpha,
    #             n_alpha,
    #         )
    #     else:
    #         # recursion - build up left and right subtrees
    #         (
    #             theta_minus,
    #             r_minus,
    #             theta_plus,
    #             r_plus,
    #             theta_prime,
    #             n_prime,
    #             s_prime,
    #             alpha_prime,
    #             n_alpha_prime,
    #         ) = self._build_tree(theta, r, u, v, j - 1, eps, theta_0, r_0)
    #         if s_prime == 1:
    #             if v == -1:
    #                 (
    #                     theta_minus,
    #                     r_minus,
    #                     _,
    #                     _,
    #                     theta_double_prime,
    #                     n_double_prime,
    #                     s_double_prime,
    #                     alpha_double_prime,
    #                     n_alpha_double_prime,
    #                 ) = self._build_tree(
    #                     theta_minus, r_minus, u, v, j - 1, eps, theta_0, r_0
    #                 )
    #             else:
    #                 (
    #                     _,
    #                     _,
    #                     theta_plus,
    #                     r_plus,
    #                     theta_double_prime,
    #                     n_double_prime,
    #                     s_double_prime,
    #                     alpha_double_prime,
    #                     n_alpha_double_prime,
    #                 ) = self._build_tree(
    #                     theta_plus, r_plus, u, v, j - 1, eps, theta_0, r_0
    #                 )

    #             if jax.random.uniform(self.png_key_seq()) <= (
    #                 n_double_prime / (n_prime + n_double_prime) if n_double_prime else 0
    #             ):
    #                 theta_prime = theta_double_prime

    #             theta_delta = theta_plus - theta_minus
    #             s_prime *= (
    #                 s_double_prime
    #                 * (jnp.dot(theta_delta, r_minus) >= 0)
    #                 * (jnp.dot(theta_delta, r_plus) >= 0)
    #             )
    #             n_prime += n_double_prime
    #             alpha_prime += alpha_double_prime
    #             n_alpha_prime += n_alpha_double_prime

    #     return (
    #         theta_minus,
    #         r_minus,
    #         theta_plus,
    #         r_plus,
    #         theta_prime,
    #         n_prime,
    #         s_prime,
    #         alpha_prime,
    #         n_alpha_prime,
    #     )

    @jax.jit
    def _leapfrog(self, theta, r, eps):
        r_tilde = r + 0.5 * eps * self.theta_loglik_grad(theta)
        theta_tilde = theta + eps * r_tilde
        r_tilde = r_tilde + 0.5 * eps * self.theta_loglik_grad(theta_tilde)
        return theta_tilde, r_tilde

    # @jax.jit #jit here messes up some tests
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

        alpha = cond(ln_p_prime - ln_p > -ln2, lambda: 1, lambda: -1)
        # alpha = 2 * int(ln_p_prime - ln_p > -ln2) - 1

        # while loop parmas
        val = {
            "eps": eps,
            "alpha": alpha,
            "theta_prime": theta_prime,
            "r_prime": r_prime,
            "theta": theta,
            "r": r,
            "ln_p": ln_p,
        }
        val = while_loop(
            self._find_reasonable_epsilon_while_loop_cond,
            self._find_reasonable_epsilon_while_loop_body,
            val,
        )
        eps = val["eps"]
        return eps

    @jax.jit
    def _find_reasonable_epsilon_while_loop_body(self, val: Dict):
        eps, alpha, theta, r = val["eps"], val["alpha"], val["theta"], val["r"]
        eps *= 2.0**alpha
        theta_prime, r_prime = self._leapfrog(theta, r, eps)
        val["eps"], val["theta_prime"], val["r_prime"] = eps, theta_prime, r_prime
        return val

    @jax.jit
    def _find_reasonable_epsilon_while_loop_cond(self, val: Dict):
        ln2 = jnp.log(2)
        alpha, theta_prime, r_prime, ln_p = (
            val["alpha"],
            val["theta_prime"],
            val["r_prime"],
            val["ln_p"],
        )
        return alpha * (self.theta_r_loglik(theta_prime, r_prime) - ln_p) > -alpha * ln2
