# jax-nuts
Functional programming implementation of the No-U-Turn Sampler ([NUTS](https://arxiv.org/pdf/1111.4246.pdf)) written in JAX.

## Credit Where Credit's Due
JAX does not support recursive functions. This posed an issue as NUTS makes repeated calls to the recursive function ```BuildTree```. While I did figure out how to unravel this recursion independent of any other sources I believe the earliest example of this goes to the authors of NumPyro for what they call [Iterative-NUTS](https://github.com/pyro-ppl/numpyro/wiki/Iterative-NUTS).
