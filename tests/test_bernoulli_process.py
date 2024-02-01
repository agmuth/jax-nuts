from dataclasses import asdict, dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from jaxnuts.nuts import NoUTurnSampler
from tests.conjugate_priors import BernoulliProcess


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
    nuts = NoUTurnSampler(loglik=cp)
    theta_0 = jnp.array([0.5])
    theta_0 = cp.logit(theta_0)
    M, M_adapt = 400, 200
    theta_samples = nuts(theta_0, M, M_adapt)
    theta_samples = theta_samples[M_adapt:]
    theta_samples = cp.inv_logit(theta_samples)
    nuts_posterior_mean = theta_samples.mean()
    assert np.isclose(cp.posterior_mean, nuts_posterior_mean, rtol=0.1) and (
        theta_samples.std() > 0
    )
