from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


class PRNGKeySequence:
    def __init__(self, seed: int) -> None:
        self.key = jax.random.PRNGKey(seed=seed)

    def __next__(self):
        _, self.key = jax.random.split(self.key)
        return self.key

    def __iter__(self):
        return self

    def __call__(self):
        return self.__next__()


class NoUTurnSampler:
    def __call__(self, theta_0, eps, loglik, M, *args, **kwargs):
        seed = 123
        self.loglik = loglik
        self.loglik_grad = jax.grad(loglik)
        self.loglik_hess = jax.jacfwd(jax.jacrev(loglik))
        self.delta_max = 1_000
        self.key = PRNGKeySequence(seed)

        dim_theta = theta_0.shape[0]
        theta_sample = [None] * (M + 1)
        theta_sample[0] = theta_0

        for m in range(1, M + 1):
            r_0 = jax.random.multivariate_normal(
                key=self.key(), mean=jnp.zeros(dim_theta), cov=jnp.eye(dim_theta)
            )
            u = jax.random.uniform(
                key=self.key(),
                minval=0,
                maxval=jnp.exp(
                    self.loglik(theta_sample[m - 1]) - 0.5 * jnp.dot(r_0, r_0)
                ),
            )

            theta_minus, theta_plus, theta_m = (
                theta_sample[m - 1],
                theta_sample[m - 1],
                theta_sample[m - 1],
            )
            r_minus, r_plus = r_0, r_0

            j = 0
            s = 1
            n = 1

            while s == 1:
                v_j = jax.random.choice(self.key(), jnp.array([-1, 1]))

                if v_j == -1:
                    (
                        theta_minus,
                        r_minus,
                        _,
                        _,
                        theta_prime,
                        n_prime,
                        s_prime,
                    ) = self._build_tree(theta_minus, r_minus, u, v_j, j, eps)
                else:
                    (
                        _,
                        _,
                        theta_plus,
                        r_plus,
                        theta_prime,
                        n_prime,
                        s_prime,
                    ) = self._build_tree(theta_plus, r_plus, u, v_j, j, eps)

                if s_prime == 1:
                    if jax.random.uniform(self.key()) <= (n_prime / n):
                        theta_m = theta_prime

                n += n_prime
                theta_delta = theta_plus - theta_minus
                s *= (
                    s_prime
                    * (jnp.dot(theta_delta, r_minus) >= 0)
                    * (jnp.dot(theta_delta, r_plus) >= 0)
                )
                j += 1

  

            theta_sample[m] = theta_m

        return theta_sample

    # def _build_tree(self, theta, r, u, v, j, eps):
    #     if j == 0:
    #         # base case - take one leapfrog step in the direction of v
    #         theta_prime, r_prime = self._leapfrog(theta, r, eps*v)
    #         C_prime = list()
    #         if u < jnp.exp(self.loglik(theta_prime) -0.5*jnp.dot(r_prime, r_prime)):
    #             C_prime += [(theta_prime, r_prime)]
    #         s_prime = (self.loglik(theta_prime) -0.5*jnp.dot(r_prime, r_prime) > jnp.log(u) - self.delta_max)
    #         return theta_prime, r_prime, theta_prime, r_prime, C_prime, s_prime
    #     else:
    #         # recursion - build up left and right subtrees
    #         theta_minus, r_minus, theta_plus, r_plus, C_prime, s_prime = self._build_tree(theta, r, u, v, j-1, eps)
    #         if v == -1:
    #             theta_minus, r_minus, _, _, C_double_prime, s_double_prime = self._build_tree(theta_minus, r_minus, u, v, j-1, eps)
    #         else:
    #             _, _, theta_plus, r_plus, C_double_prime, s_double_prime = self._build_tree(theta_plus, r_plus, u, v, j-1, eps)

    #         theta_delta = theta_plus - theta_minus
    #         s_prime *= s_double_prime * (jnp.dot(theta_delta, r_minus) >= 0) * (jnp.dot(theta_delta, r_plus) >= 0)
    #         C_prime += C_double_prime
    #     return theta_minus, r_minus, theta_plus, r_plus, C_prime, s_prime

    def _build_tree(self, theta, r, u, v, j, eps):
        if j == 0:
            # base case - take one leapfrog step in the direction of v
            theta_prime, r_prime = self._leapfrog(theta, r, eps * v)
            n_prime = int(
                u < jnp.exp(self.loglik(theta_prime) - 0.5 * jnp.dot(r_prime, r_prime))
            )
            s_prime = int(
                self.loglik(theta_prime) - 0.5 * jnp.dot(r_prime, r_prime)
                > jnp.log(u) - self.delta_max
            )
            return (
                theta_prime,
                r_prime,
                theta_prime,
                r_prime,
                theta_prime,
                n_prime,
                s_prime,
            )
        else:
            # recursion - build up left and right subtrees
            (
                theta_minus,
                r_minus,
                theta_plus,
                r_plus,
                theta_prime,
                n_prime,
                s_prime,
            ) = self._build_tree(theta, r, u, v, j - 1, eps)
            if s_prime == 1:
                if v == -1:
                    (
                        theta_minus,
                        r_minus,
                        _,
                        _,
                        theta_double_prime,
                        n_double_prime,
                        s_double_prime,
                    ) = self._build_tree(theta_minus, r_minus, u, v, j - 1, eps)
                else:
                    (
                        _,
                        _,
                        theta_plus,
                        r_plus,
                        theta_double_prime,
                        n_double_prime,
                        s_double_prime,
                    ) = self._build_tree(theta_plus, r_plus, u, v, j - 1, eps)


                if n_double_prime > 0 and jax.random.uniform(self.key()) <= (
                    n_double_prime / (n_prime + n_double_prime)
                ):
                    theta_prime = theta_double_prime

                theta_delta = theta_plus - theta_minus
                s_prime *= (
                    s_double_prime
                    * (jnp.dot(theta_delta, r_minus) >= 0)
                    * (jnp.dot(theta_delta, r_plus) >= 0)
                )
                n_prime += n_double_prime

        return theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime

    def _leapfrog(self, theta, r, eps):
        r_tilde = r + 0.5 * eps * self.loglik_grad(theta)
        theta_tilde = theta + eps * r_tilde
        r_tilde = r_tilde + 0.5 * eps * self.loglik_grad(theta_tilde)
        return theta_tilde, r_tilde


if __name__ == "__main__":
    print("NUTS")

    # def normal_loglik(x, theta=jnp.array([0.0, 1.0])):
    #     mu, sigma = theta[0], theta[1]
    #     std_errs = (x-mu)/sigma
    #     loglik = -0.5 * jnp.dot(std_errs, std_errs) - jnp.log(sigma*jnp.sqrt(2*jnp.pi))
    #     return loglik

    class NormalLogLik:
        def __init__(self, x):
            self.x = x

        def __call__(self, theta):
            mu, sigma = theta[0], theta[1]
            std_errs = (x - mu) / sigma
            loglik = -0.5 * jnp.dot(std_errs, std_errs) - jnp.log(
                sigma * jnp.sqrt(2 * jnp.pi)
            )
            return loglik

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10,))

    loglik = NormalLogLik(x)
    nuts = NoUTurnSampler()

    theta_0 = jnp.array([0.0, 10])
    eps = 0.1
    M = 2
    theta_samples = nuts(theta_0, eps, loglik, M)
    print(theta_samples)
