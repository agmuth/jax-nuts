from dataclasses import asdict, dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from jaxnuts.nuts import NoUTurnSampler
from tests.conjugate_priors import NormalProcessPrecisonKnown


@dataclass
class ProblemInstance:
    m: float
    p: float
    rho: float
    x_bar: float
    n: int

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


problems = [
    ProblemInstance(
        m=2.0,
        p=1.0,
        rho=2.0,
        x_bar=4.0,
        n=10,
    )
]


@pytest.mark.parametrize("problem", problems)
def test_normal_process_precison_known(problem: ProblemInstance):
    cp = NormalProcessPrecisonKnown(**problem.dict())
    nuts = NoUTurnSampler(loglik=cp)
    theta_0 = jnp.array([cp.prior_mean])
    M, M_adapt = 2000, 1000
    theta_samples = nuts(theta_0, M, M_adapt)
    theta_samples = theta_samples[M_adapt:]
    nuts_posterior_mean = theta_samples.mean()
    nuts_posterior_std = theta_samples.std()

    z_val_obvs = abs(cp.posterior_mean - nuts_posterior_mean) / nuts_posterior_std
    assert nuts_posterior_std > 0
    assert z_val_obvs < 0.84


# if __name__ == "__main__":
#     test_normal_process_precison_known(problems[0])
