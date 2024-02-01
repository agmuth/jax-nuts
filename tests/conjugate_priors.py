import jax
import jax.numpy as jnp

# ref: https://www.johndcook.com/CompendiumOfConjugatePriors.pdf


class NormalProcessPrecisonKnown:
    def __init__(self, x_bar, n, rho, m, p):
        self.x_bar = x_bar
        self.n = n
        self.rho = rho  # precision
        # priors for mu
        self.m = m
        self.p = p

        self.m_prime = (self.m * self.p + self.n * self.rho * self.x_bar) / (
            self.p + self.n * self.rho
        )
        self.p_prime = self.p + self.n * self.rho

    def __call__(self, theta):
        # posterior loglik for mu
        mu = theta[0]
        loglik = 0.0
        # loglik += 0.5 * jnp.log(self.p_prime)
        loglik += (
            -0.5 * self.p_prime * jnp.dot((mu - self.m_prime), (mu - self.m_prime))
        )
        return loglik

    @property
    def posterior_mean(self):
        return self.m_prime


class NormalProcessMeanKnown:
    def __init__(self, ss, n, mu, a, b):
        self.ss = ss
        self.n = n
        self.mu = mu  # mean
        # priors for rho
        self.a = a
        self.b = b

        self.a_prime = self.a + self.n / 2
        self.b_prime = self.b + self.ss / 2

    def __call__(self, theta):
        # posterior loglik for rho

        rho = theta[0]

        if rho <= 0:
            return -jnp.inf

        loglik = 0.0
        loglik += (self.a_prime - 1) * jnp.log(rho)
        loglik += -rho / self.b_prime
        # loglik -= jax.scipy.special.gamma(self.a_prime)
        # loglik -= self.a_prime*jnp.log(self.b_prime)
        return loglik

    @property
    def posterior_mean(self):
        return self.a_prime * self.b_prime
