# https://github.com/google/jax/issues/446
from collections import namedtuple
from jax.tree_util import register_pytree_node
from jax import grad, jit
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple
import jax


class Point2d(NamedTuple):
    x: float
    y: float


class Point3d(NamedTuple):
    x: float
    y: float
    z: float


# Point2d = namedtuple("Point2d", ["x", "y"])

# register_pytree_node(
#     Point2d,
#     lambda xs: (tuple(xs), None),  # tell JAX how to unpack to an iterable
#     lambda _, xs: Point2d(*xs)       # tell JAX how to pack back into a Point2d
# )

# pt3d = Point3d(1, 2, 3)
# pt2d = Point2d(**pt3d._asdict())  # doesn't work

# def f(pt):
#     pt = pt._replace(x=pt.y, y=pt.x)
#     return np.sqrt(pt.x**2 + pt.y**2)

# pt = Point2d(1., 2.)

# # print(f(pt))      # 2.236068
# # print(grad(f)(pt))  # Point2d(x=..., y=...)

# # g = jit(f)
# # print(g(pt))  # 2.236068


# def cond_fun(pt):
#     return pt.x < 10

# def body_fun(pt):
#     return pt._replace(x=pt.x+pt.y)

# pt = Point2d(1, 1)
# pt = lax.while_loop(cond_fun, body_fun, pt)
# print(pt)


# class X:

#     def cond_fun(self, pt):
#         return pt.x < 10

#     def body_fun(self, pt):
#         return pt._replace(x=pt.x+pt.y)


#     def while_fun(self, x, y):
#         pt = Point2d(x, y)
#         pt = lax.while_loop(self.cond_fun, self.body_fun, pt)
#         return pt


# x = X()
# print(x.while_fun(1, 1))


def f(carry, *args):
    i = 0
    while i < 1:
        i += 1
    key, pos = carry
    key, subkey = jax.random.split(key)
    coords = jax.random.uniform(subkey, (2,), minval=0, maxval=1)
    pos += coords
    carry = (key, pos)
    return carry, coords


key = jax.random.PRNGKey(1234)
pos = jnp.zeros((2,))

init = (key, pos)
xs = None
length = 10

carry, ys = lax.scan(f, init, xs, length)
print(carry)
print(ys)
print(ys.sum(axis=0) == carry[1])
