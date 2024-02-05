import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class, Partial

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