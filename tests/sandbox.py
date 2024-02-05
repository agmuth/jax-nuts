import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class, Partial
from dataclasses import dataclass

@

@register_pytree_node_class
class CallFunc:
    def __init__(self, f: callable):
        f = jax.jit(f)
        self.f = Partial(f)
       
    @jax.jit 
    def call_f(self, *args):
        return self.f(*args)
    
    def tree_flatten(self):
        children = (self.f,)
        aux_data = None
        return (children, aux_data)
    
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    
if __name__ == "__main__":
    x = CallFunc(f = lambda x: x+1)
    print(x.call_f(1))
    
    from jax.lax import while_loop, cond
    n = 1
    cond_fun = lambda x: x < 10
    body_fun = lambda x: x+1
    init_val = 1
    
    n = while_loop(cond_fun, body_fun, init_val)
    print(n)
    
    print(cond(True, lambda: 1, lambda: 0))
    
    def recursion(n):
        if n == 0:
            return n
        return cond(False, lambda: recursion(n-1), lambda: None)
    
    print(recursion(10))