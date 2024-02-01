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
        loglik = 0
        loglik += 0.5 * jnp.log(self.p_prime)
        loglik += (
            -0.5 * self.p_prime * jnp.dot((mu - self.m_prime), (mu - self.m_prime))
        )
        return loglik

    @property
    def posterior_params(self):
        return self.m_prime, self.p_prime
