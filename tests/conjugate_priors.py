

import jax.numpy as jnp
import numpy as np
from jaxnuts.nuts import NoUTurnSampler

#ref: https://www.johndcook.com/CompendiumOfConjugatePriors.pdf

class NormalProcessPrecisonKnown:
    def __init__(self, x_bar, n, rho, m, p):
        self.x_bar = x_bar
        self.n = n
        self.rho = rho # precision
        # priors for mu
        self.m = m
        self.p = p
        
        self.m_prime = (self.m*self.p + self.n*self.rho*self.x_bar) / (self.p + self.n*self.rho)
        self.p_prime = self.p + self.n*self.rho
        
        
    def __call__(self, theta):
        # posterior loglik for mu
        mu = theta[0]
        loglik = 0
        loglik += 0.5*np.log(self.p_prime)
        loglik += -0.5*self.p_prime*jnp.dot((mu-self.m_prime), (mu-self.m_prime))
        return loglik
        

    @property
    def posterior_params(self):
        return self.m_prime, self.p_prime
        
        
        
if __name__ == "__main__":
    m = 2.0
    p = 1.0
    rho = 2.0
    x_bar = 4.0
    n = 10
    
    cp = NormalProcessPrecisonKnown(x_bar, n, rho, m, p)
    
    
    
    # loglik_jit = jax.jit(loglik)
    # loglik_aot = jax.jit( loglik).lower(jnp.empty((2,))).compile()
    
    
    nuts = NoUTurnSampler(loglik=cp)

    theta_0 = jnp.array([3.0])
    eps = 0.1
    M = 1000
    theta_samples = nuts(theta_0, M)
    theta_samples = theta_samples[M//2:]
    print(theta_samples.min(), theta_samples.max())
    print(theta_samples.mean(), theta_samples.std()**-2)
    print(cp.posterior_params)