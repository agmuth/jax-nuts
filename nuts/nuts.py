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
    def __init__(self, loglik):
        self.theta_loglik = loglik
        self.theta_loglik_grad = jax.grad(loglik)
        self.theta_loglik_hess = jax.jacfwd(jax.jacrev(loglik))
        self.theta_r_loglik = lambda theta, r: loglik(theta) - 0.5 * jnp.dot(r, r)
        self.theta_r_lik = lambda theta, r: jnp.exp(self.theta_r_loglik(theta, r))

        # TODO: move to `__call__`
        self.seed = 1234
        self.png_key_seq = PRNGKeySequence(self.seed)
        self.delta_max = 1_000

    def __call__(self, theta_0, eps, M, *args, **kwargs):
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
                    if jax.random.uniform(self.png_key_seq()) <= jnp.where(
                        n_prime == 0, 0, n_prime / n
                    ):
                        theta_m = theta_prime

                n += n_prime
                theta_delta = theta_plus - theta_minus
                s *= (
                    s_prime
                    * (jnp.dot(theta_delta, r_minus) >= 0)
                    * (jnp.dot(theta_delta, r_plus) >= 0)
                )
                j += 1

            theta_samples = theta_samples.at[m].set(theta_m)

        return theta_samples

    def _build_tree(self, theta, r, u, v, j, eps):
        if j == 0:
            # base case - take one leapfrog step in the direction of v
            theta_prime, r_prime = self._leapfrog(theta, r, eps * v)
            joint_lik = self.theta_r_lik(theta_prime, r_prime)
            n_prime = int(u < joint_lik)
            s_prime = int(joint_lik > jnp.log(u) - self.delta_max)
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

                if jax.random.uniform(self.png_key_seq()) <= jnp.where(
                    n_double_prime == 0, 0, n_double_prime / (n_prime + n_double_prime)
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
        r_tilde = r + 0.5 * eps * self.theta_loglik_grad(theta)
        theta_tilde = theta + eps * r_tilde
        r_tilde = r_tilde + 0.5 * eps * self.theta_loglik_grad(theta_tilde)
        return theta_tilde, r_tilde
    
    def _find_reasonable_epsilon(self, theta):
        ln2 = jnp.log(2)
        dim_theta = theta_0.shape[0]
        eps = 1.
        r = jax.random.multivariate_normal(
                key=self.png_key_seq(),
                mean=jnp.zeros(dim_theta),
                cov=jnp.eye(dim_theta),
            )
        loglik0 = self.theta_r_loglik(theta, r)
        
        theta_prime, r_prime = self._leapfrog(theta, r, eps)
        delta_loglik = self.theta_r_loglik(theta_prime, r_prime) - loglik0
        alpha = 2 * int(delta_loglik > -1*ln2) - 1
        while alpha*delta_loglik > -1*alpha*ln2:
            eps *= 2.0**alpha
            theta_prime, r_prime = self._leapfrog(theta, r, eps)
            delta_loglik = self.theta_r_loglik(theta_prime, r_prime) - loglik0
            
        return eps
        
        


if __name__ == "__main__":
    print("NUTS")

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
    x = jax.random.normal(key, (100,))

    loglik = NormalLogLik(x)
    nuts = NoUTurnSampler(loglik=loglik)

    theta_0 = jnp.array([0.0, 10])
    eps = 0.1
    M = 10
    theta_samples = nuts(theta_0, eps, M)
    print(theta_samples)
