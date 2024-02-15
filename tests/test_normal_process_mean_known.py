from dataclasses import asdict, dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from jaxnuts.nuts import NoUTurnSampler
from tests.conjugate_priors import NormalProcessMeanKnown
from jaxnuts.nuts_functional import sample_posterior


@dataclass
class ProblemInstance:
    a: float
    b: float
    mu: float
    ss: float
    n: int

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


problems = [
    ProblemInstance(
        a=3.0,
        b=1.5,
        mu=2.0,
        ss=38.0,
        n=20,
    )
]


@pytest.mark.parametrize("problem", problems)
def test_normal_process_mean_known(problem: ProblemInstance):
    cp = NormalProcessMeanKnown(**problem.dict())
    theta_0 = jnp.array([cp.prior_mean])
    theta_0 = cp.log(theta_0)
    M, M_adapt = 2000, 1000
    theta_samples = sample_posterior(loglik=cp, theta_0=theta_0, M=M, M_adapt=M_adapt)
    theta_samples = theta_samples[M_adapt:]
    theta_samples = cp.inv_log(theta_samples)
    nuts_posterior_mean = theta_samples.mean()
    nuts_posterior_std = theta_samples.std()

    z_val_obvs = abs(cp.posterior_mean - nuts_posterior_mean) / nuts_posterior_std
    assert nuts_posterior_std > 0
    assert z_val_obvs < 0.5


if __name__ == "__main__":
    test_normal_process_mean_known(problems[0])
