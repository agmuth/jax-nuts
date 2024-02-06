from dataclasses import asdict, dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from jaxnuts.nuts import NoUTurnSampler
from tests.conjugate_priors import PoissonProcess


@dataclass
class ProblemInstance:
    x_sum: float
    n: int
    a: float
    b: float

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


problems = [ProblemInstance(a=3.0, b=2.0, n=10, x_sum=4.0)]


@pytest.mark.parametrize("problem", problems)
def test_poisson_process(problem: ProblemInstance):
    cp = PoissonProcess(**problem.dict())
    nuts = NoUTurnSampler(loglik=cp)
    theta_0 = jnp.array([1.0])
    theta_0 = cp.log(theta_0)
    M, M_adapt = 800, 400
    theta_samples = nuts(theta_0, M, M_adapt)
    theta_samples = theta_samples[M_adapt:]
    theta_samples = cp.inv_log(theta_samples)
    nuts_posterior_mean = theta_samples.mean()
    nuts_posterior_std = theta_samples.std()
    
    z_val_obvs = abs(cp.posterior_mean - nuts_posterior_mean)/nuts_posterior_std
    assert nuts_posterior_std > 0
    assert np.isclose(cp.posterior_mean, nuts_posterior_mean, rtol=0.1) and (
        theta_samples.std() > 0
    )


if __name__ == "__main__":
    test_poisson_process(problems[0])
