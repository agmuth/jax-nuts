from dataclasses import asdict, dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from jaxnuts.nuts import NoUTurnSampler
from tests.conjugate_priors import BernoulliProcess
from jaxnuts.nuts_functional import sample_posterior


@dataclass
class ProblemInstance:
    x_sum: float
    n: int
    a: float
    b: float

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


problems = [ProblemInstance(a=3.0, b=3.0, n=5, x_sum=4.0)]


@pytest.mark.parametrize("problem", problems)
def test_bernoulli_process(problem: ProblemInstance):
    cp = BernoulliProcess(**problem.dict())
    # nuts = NoUTurnSampler(loglik=cp, theta_0=jnp.zeros(1))
    theta_0 = jnp.array([cp.prior_mean])
    theta_0 = cp.logit(theta_0)
    M, M_adapt = 2000, 1000
    theta_samples = sample_posterior(loglik=cp, theta_0=theta_0, M=M, M_adapt=M_adapt)
    theta_samples = theta_samples[M_adapt:]
    theta_samples = cp.inv_logit(theta_samples)
    nuts_posterior_mean = theta_samples.mean()
    nuts_posterior_std = theta_samples.std()

    z_val_obvs = abs(cp.posterior_mean - nuts_posterior_mean) / nuts_posterior_std
    assert nuts_posterior_std > 0
    assert z_val_obvs < 0.5


if __name__ == "__main__":
    test_bernoulli_process(problems[0])
