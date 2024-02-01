from dataclasses import asdict, dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from jaxnuts.nuts import NoUTurnSampler
from tests.conjugate_priors import NormalProcessMeanKnown


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
    nuts = NoUTurnSampler(loglik=cp)
    theta_0 = jnp.array([1.0])
    M = 200
    theta_samples = nuts(theta_0, M)
    theta_samples = theta_samples[M // 2 :]
    nuts_posterior_mean = theta_samples.mean()
    assert np.isclose(cp.posterior_mean, nuts_posterior_mean, rtol=0.1)


# if __name__ == "__main__":
#     test_normal_process_mean_known(problems[0])
