
import jax
import jax.numpy as jnp


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
        # loglik = jax.jit(loglik)
        self.theta_loglik = loglik
        self.theta_loglik_grad = jax.grad(loglik)
        # self.theta_loglik_hess = jax.jacfwd(jax.jacrev(loglik))
        self.theta_r_loglik = lambda theta, r: loglik(theta) - 0.5 * jnp.dot(r, r)
        self.theta_r_lik = lambda theta, r: jnp.exp(self.theta_r_loglik(theta, r))

        # TODO: move to `__call__`
        self.seed = 1234
        self.png_key_seq = PRNGKeySequence(self.seed)
        self.delta_max = 1_000

    def __call__(self, theta_0, M):
        # TODO: accept as args
        M_adapt = M // 2
        delta = 0.5 * (0.95 + 0.25)
        gamma = 0.05
        kappa = 0.75
        t_0 = 10

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
                        _,
                        _,
                        theta_prime,
                        n_prime,
                        s_prime,
                        alpha,
                        n_alpha,
                    ) = self._build_tree(
                        theta_minus, r_minus, u, v_j, j, eps, theta_m_minus_one, r_0
                    )
                else:
                    (
                        _,
                        _,
                        theta_plus,
                        r_plus,
                        theta_prime,
                        n_prime,
                        s_prime,
                        alpha,
                        n_alpha,
                    ) = self._build_tree(
                        theta_plus, r_plus, u, v_j, j, eps, theta_m_minus_one, r_0
                    )

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

    def _build_tree(self, theta, r, u, v, j, eps, theta_0, r_0):
        if j == 0:
            # base case - take one leapfrog step in the direction of v
            theta_prime, r_prime = self._leapfrog(theta, r, eps * v)

            joint_loglik_prime = self.theta_r_loglik(theta_prime, r_prime)
            joint_loglik_0 = self.theta_r_loglik(theta_0, r_0)
            delta_loglik = joint_loglik_prime - joint_loglik_0

            n_prime = int(u <= jnp.exp(joint_loglik_prime))
            s_prime = int(jnp.log(u) < joint_loglik_prime + self.delta_max)

            alpha = min(1, jnp.exp(delta_loglik))
            n_alpha = 1

            return (
                theta_prime,
                r_prime,
                theta_prime,
                r_prime,
                theta_prime,
                n_prime,
                s_prime,
                alpha,
                n_alpha,
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
                alpha_prime,
                n_alpha_prime,
            ) = self._build_tree(theta, r, u, v, j - 1, eps, theta_0, r_0)
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
                        alpha_double_prime,
                        n_alpha_double_prime,
                    ) = self._build_tree(
                        theta_minus, r_minus, u, v, j - 1, eps, theta_0, r_0
                    )
                else:
                    (
                        _,
                        _,
                        theta_plus,
                        r_plus,
                        theta_double_prime,
                        n_double_prime,
                        s_double_prime,
                        alpha_double_prime,
                        n_alpha_double_prime,
                    ) = self._build_tree(
                        theta_plus, r_plus, u, v, j - 1, eps, theta_0, r_0
                    )

                if jax.random.uniform(self.png_key_seq()) <= (
                    n_double_prime / (n_prime + n_double_prime) if n_double_prime else 0
                ):
                    theta_prime = theta_double_prime

                theta_delta = theta_plus - theta_minus
                s_prime *= (
                    s_double_prime
                    * (jnp.dot(theta_delta, r_minus) >= 0)
                    * (jnp.dot(theta_delta, r_plus) >= 0)
                )
                n_prime += n_double_prime
                alpha_prime += alpha_double_prime
                n_alpha_prime += n_alpha_double_prime

        return (
            theta_minus,
            r_minus,
            theta_plus,
            r_plus,
            theta_prime,
            n_prime,
            s_prime,
            alpha_prime,
            n_alpha_prime,
        )

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
        loglik0 = self.theta_r_loglik(theta, r)

        theta_prime, r_prime = self._leapfrog(theta, r, eps)
        delta_loglik = self.theta_r_loglik(theta_prime, r_prime) - loglik0
        alpha = 2 * int(delta_loglik > -1 * ln2) - 1
        while alpha * delta_loglik > -1 * alpha * ln2:
            eps *= 2.0**alpha
            theta_prime, r_prime = self._leapfrog(theta, r, eps)
            delta_loglik = self.theta_r_loglik(theta_prime, r_prime) - loglik0

        return eps


if __name__ == "__main__":
    print("NUTS")

    class NormalLogLikSigmaKnown:
        def __init__(self, x, sigma):
            self.x = x
            self.sigma = sigma

        # @jax.jit
        def __call__(self, theta):
            mu = theta[0]
            std_errs = (x - mu) / self.sigma
            loglik = -0.5 * jnp.dot(std_errs, std_errs) - 0.5 * jnp.log(sigma)
            return loglik

    class NormalLogLikMuKnown:
        def __init__(self, x, mu):
            self.x = x
            self.mu = mu

        # @jax.jit
        def __call__(self, theta):
            sigma = theta[0]
            std_errs = (x - self.mu) / sigma
            loglik = -0.5 * jnp.dot(std_errs, std_errs) - 0.5 * jnp.log(sigma)
            return loglik

    key = jax.random.PRNGKey(0)
    mu = 0
    sigma = 1
    x = jax.random.normal(key, (100,)) * sigma + mu

    # loglik = NormalLogLikSigmaKnown(x, sigma)
    loglik = NormalLogLikMuKnown(x, sigma)
    # loglik_jit = jax.jit(loglik)
    # loglik_aot = jax.jit( loglik).lower(jnp.empty((2,))).compile()

    nuts = NoUTurnSampler(loglik=loglik)

    theta_0 = jnp.array([2.0])
    eps = 0.1
    M = 200
    theta_samples = nuts(theta_0, M)
    theta_samples = theta_samples[M // 2 :]
    print(theta_samples.min(), theta_samples.max())
    print(theta_samples.mean(), theta_samples.std())
