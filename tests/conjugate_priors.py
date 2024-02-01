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

        self.m_prime = (m * p + n * rho * x_bar) / (p + n * rho)
        self.p_prime = p + n * rho

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

        self.a_prime = a + n / 2
        self.b_prime = b + ss / 2

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


class BernoulliProcess:
    def __init__(self, x_sum, n, a, b):
        self.x_sum = x_sum
        self.n = n
        self.a = a
        self.b = b

        self.a_prime = a + x_sum
        self.b_prime = b + n - x_sum

    def __call__(self, theta):
        p = self.inv_logit(theta[0])
        loglik = 0
        loglik += (self.a_prime - 1) * jnp.log(p)
        loglik += (self.b_prime - 1) * jnp.log(1 - p)
        return loglik

    @property
    def posterior_mean(self):
        return self.a_prime / (self.a_prime + self.b_prime)

    @staticmethod
    def logit(x):
        # (0, 1) -> R
        return jnp.log(x) - jnp.log(1 - x)

    @staticmethod
    def inv_logit(x):
        # R -> (0, 1)
        return (1 + jnp.exp(-x)) ** -1


class PoissonProcess:
    def __init__(self, x_sum, n, a, b):
        self.x_sum = x_sum
        self.n = n
        self.a = a
        self.b = b

        self.a_prime = a + x_sum
        self.b_prime = b / (1 + n)

    def __call__(self, theta):
        lambda_ = self.inv_log(theta[0])

        loglik = 0.0
        loglik += (self.a_prime - 1) * jnp.log(lambda_)
        loglik += -lambda_ / self.b_prime
        # loglik -= jax.scipy.special.gamma(self.a_prime)
        # loglik -= self.a_prime*jnp.log(self.b_prime)
        return loglik

    @property
    def posterior_mean(self):
        return self.a_prime * self.b_prime

    @staticmethod
    def log(x):
        # R+ -> R
        return jnp.log(x)

    @staticmethod
    def inv_log(x):
        # R -> R+
        return jnp.exp(x)
