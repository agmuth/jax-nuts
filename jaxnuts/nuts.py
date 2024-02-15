import jax
import jax.numpy as jnp
from typing import Union, Tuple, Dict
import numpy as np
from jax import lax
from typing import NamedTuple, Tuple


from jax.tree_util import register_pytree_node_class, Partial

# from jaxnuts.utils import *



@register_pytree_node_class
class NoUTurnSampler:
    def __init__(
        self,
        loglik,
        theta_0,
        M=2_000,
        M_adapt=1_000,
        delta=0.5 * (0.95 + 0.25),
        gamma=0.05,
        kappa=0.75,
        t_0=10,
        delta_max=1_000,
        prng_key=jax.random.PRNGKey(1234),
    ):
        self.theta_loglik = Partial(jax.jit(loglik))
        self.theta_loglik_grad = Partial(jax.jit(jax.grad(loglik)))
        self.theta_r_loglik = Partial(
            jax.jit(lambda theta, r: loglik(theta) - 0.5 * jnp.dot(r, r))
        )
        self.theta_r_lik = Partial(
            jax.jit(lambda theta, r: jnp.exp(self.theta_r_loglik(theta, r)))
        )

        self.theta_0 = theta_0
        self.M = M
        self.M_adapt = M_adapt 
        
        self.delta = delta
        self.gamma = gamma
        self.kappa = kappa
        self.t_0 = t_0
        self.delta_max = delta_max
        self.prng_key = prng_key

        

    def tree_flatten(self):
        children = (
            self.theta_loglik,
            self.theta_0,
            self.M,
            self.M_adapt,
            self.delta,
            self.gamma,
            self.kappa,
            self.t_0,
            self.delta_max,
            self.prng_key,
        )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    @jax.jit
    def _leapfrog(self, theta, r, eps):
        r_tilde = r + 0.5 * eps * self.theta_loglik_grad(theta)
        theta_tilde = theta + eps * r_tilde
        r_tilde = r_tilde + 0.5 * eps * self.theta_loglik_grad(theta_tilde)
        return theta_tilde, r_tilde
    
    
    def _find_reasonable_epsilon(self, theta, prng_key):
        ln2 = jnp.log(2)
        dim_theta = theta.shape[0]
        eps = 1.0
        # TODO: cahnge to call get momentum func
        r = jax.random.multivariate_normal(
            key=prng_key,
            mean=jnp.zeros(dim_theta),
            cov=jnp.eye(dim_theta),
        )

        theta_prime, r_prime = self._leapfrog(theta, r, eps)
        ln_p = self.theta_r_loglik(theta, r)
        ln_p_prime = self.theta_r_loglik(theta_prime, r_prime)

        alpha = 2 * int(ln_p_prime - ln_p > -ln2) - 1
        while alpha * (ln_p_prime - ln_p) > -alpha * ln2:
            eps *= 2.0**alpha
            theta_prime, r_prime = self._leapfrog(theta, r, eps)
            ln_p_prime = self.theta_r_loglik(theta_prime, r_prime)

        return eps
    
    @jax.jit
    def _dual_average(self, eps, eps_bar, H_bar, mu, alpha, n_alpha, m):
        # split out updates to avoid having to save vectors
        H_bar *= 1 - 1 / (m + self.t_0)
        H_bar += +(self.delta - alpha / n_alpha) / (m + self.t_0)

        # on log scale
        eps = mu - jnp.sqrt(m) / self.gamma * H_bar
        eps_bar = m**-self.kappa * eps + (1 - m**-self.kappa) * jnp.log(eps_bar)

        # exponentiate for next iter
        eps = jnp.exp(eps)
        eps_bar = jnp.exp(eps_bar)

        return eps, eps_bar, H_bar
    
    @jax.jit
    def _accept_or_reject_proposed_theta(self, cond, prob, theta_prop, theta_curr, prng_key):
        theta_curr = jnp.where(
            (cond>0)
            * (jax.random.uniform(key=prng_key, minval=0, maxval=1)
            < prob),
            theta_prop,
            theta_curr,
        )
        return theta_curr

    @jax.jit
    def _check_for_u_turn(self, theta_plus, r_plus, theta_minus, r_minus, v):
        theta_delta = (
            theta_plus - theta_minus
        ) * v  # need to reverse order if args passed in backwards
        return (jnp.dot(theta_delta, r_plus) >= 0) * (
            jnp.dot(theta_delta, r_minus) >= 0
        )
        
    def _draw_momentum_vector(self, dim, prng_key):
        r = jax.random.multivariate_normal(
            key=prng_key,
            mean=jnp.zeros(dim),
            cov=jnp.eye(dim),
        )
        return r

    def __call__(
        self,
        
    ):
        self.prng_key, subkey = jax.random.split(self.prng_key)
        eps = self._find_reasonable_epsilon(theta=self.theta_0, prng_key=subkey)
        eps_bar = 1.0
        H_bar = 0.0
        mu = jnp.log(10 * eps)

        # dim_theta = theta_0.shape[0]
        dim_theta = self.theta_0.shape[0]
        theta_samples = jnp.empty((self.M + 1, dim_theta))
        theta_samples = theta_samples.at[0].set(self.theta_0)
        
        """
        TODO:
        split for loop into adapt and sample phase 
        -> avoid if statement at end of loop
        """
        
        
        for m in range(1, self.M + 1):  
            self.prng_key, subkey1, subkey2 = jax.random.split(self.prng_key, 3)
            
            theta_m = theta_samples[m - 1]
            r_0 = self._draw_momentum_vector(dim_theta, subkey1)
            
            u = jax.random.uniform(
                key=subkey2, minval=0, maxval=self.theta_r_lik(theta_m, r_0)
            )

            # initialize vars
            theta_m_minus_one = theta_samples[m - 1]
            theta_plus_minus = jnp.array([theta_samples[m - 1], theta_samples[m - 1]])
            r_plus_minus = jnp.array([r_0, r_0])

            j = 0
            s = 1
            n = 1

            while s == 1:
                self.prng_key, subkey = jax.random.split(self.prng_key)
                idx_star = jax.random.bernoulli(subkey).astype(jnp.int32)
                v_j = 2 * idx_star - 1
                theta_star = theta_plus_minus[idx_star]
                r_star = r_plus_minus[idx_star]
                
                self.prng_key, subkey = jax.random.split(self.prng_key)

                (
                    theta_star,
                    r_star,
                    theta_prime,
                    n_prime,
                    s_prime,
                    alpha,
                    n_alpha,
                ) = self._build_tree_while_loop(
                    theta_star, r_star, u, v_j, j, eps, theta_m_minus_one, r_0, subkey
                )

                theta_plus_minus = theta_plus_minus.at[idx_star].set(theta_star)
                r_plus_minus = r_plus_minus.at[idx_star].set(r_star)

                """
                TODO: func to move from curr val to proposed val
                accept_proposed_theta(lax.cond: bool, prob: float, curr_theta, proposed_theta):
                    return jnp.where(lax.cond * unif < prob, p_theta, c_theta)
                """ 
                
                self.prng_key, subkey = jax.random.split(self.prng_key)
                theta_m = self._accept_or_reject_proposed_theta(s_prime==1, n_prime/n, theta_prime, theta_m, subkey)
                
                

                theta_delta = theta_plus_minus[1] - theta_plus_minus[0]
                s *= (
                    s_prime
                    * (jnp.dot(theta_delta, r_plus_minus[0]) >= 0)
                    * (jnp.dot(theta_delta, r_plus_minus[1]) >= 0)
                )
                n += n_prime
                j += 1

            if m < self.M_adapt:  # adapt accpetance params
                eps, eps_bar, H_bar = self._dual_average(
                    eps, eps_bar, H_bar, mu, alpha, n_alpha, m
                )
            else:
                eps = eps_bar

            theta_samples = theta_samples.at[m].set(theta_m)

        return theta_samples

    

    

    
    @jax.jit
    def _build_tree_single_step(self, theta_star, r_star, u, v, eps, theta_0, r_0):
        ln_u = jnp.log(u)

        theta_double_star, r_double_star = self._leapfrog(
            theta_star, r_star, v * eps
        )  # push edge out one leapfrog step
        joint_loglik_double_star = self.theta_r_loglik(theta_double_star, r_double_star)
        joint_loglik_0 = self.theta_r_loglik(theta_0, r_0)

        n = (
            ln_u <= joint_loglik_double_star
        )  # indicator for if new edge state is eligible
        s = (
            ln_u <= joint_loglik_double_star + self.delta_max
        )  # early termination criteria (equ. 3)

        alpha = jnp.minimum(1.0, jnp.exp(joint_loglik_double_star - joint_loglik_0))
        n_alpha = 1

        return theta_double_star, r_double_star, n, s, alpha, n_alpha
    
    
    
    @jax.jit
    def _build_tree_while_loop(self, theta_star, r_star, u, v, j, eps, theta_0, r_0, prng_key):
        left_leaf_nodes = jnp.array(
            [(jnp.zeros(theta_star.shape), jnp.zeros(r_0.shape))] #* max(1, j)  # get around tracing issues when j = 0
        )  # array for storing leftmost leaf nodes in any subtree currently under consideration        
        # HMC path vars
        theta_prime = theta_star
        s = 1
        n = 1

        # dual averaging vars
        alpha = 0.0
        n_alpha = 0

        i = 0  # counter
        
        (
            theta_star,
            r_star,
            u,
            v,
            j,
            eps,
            theta_0,
            r_0,
            s,
            n,
            theta_prime,
            alpha,
            n_alpha,
            left_leaf_nodes,
            i,
            prng_key,
        ) = lax.while_loop(
            self._build_tree_while_loop_cond,
            self._build_tree_while_loop_body,
            (
                theta_star,
                r_star,
                u,
                v,
                j,
                eps,
                theta_0,
                r_0,
                s,
                n,
                theta_prime,
                alpha,
                n_alpha,
                left_leaf_nodes,
                i,
                prng_key,
            )
        )
       
        
        return (theta_star, r_star, theta_prime, n, s, alpha, n_alpha)

        
        

    
    @jax.jit
    def _build_tree_while_loop_cond(self, val: Tuple):
        (
            _,
            _,
            _,
            _,
            j,
            _,
            _,
            _,
            s,
            _,
            _,
            _,
            _,
            _,
            i,
            _,  
        ) = val
        
        return (s==1)*(i < 2**j)
    
    @jax.jit
    def _build_tree_while_loop_body(self, val: Tuple):
        (
            theta_star,
            r_star,
            u,
            v,
            j,
            eps,
            theta_0,
            r_0,
            s,
            n,
            theta_prime,
            alpha,
            n_alpha,
            left_leaf_nodes,
            i,
            prng_key,  
        ) = val
        
        
        i += 1  # incr here to align with b-tree 1-indexing
        (
            theta_double_star,
            r_double_star,
            n_prime,
            s_prime,
            alpha_prime,
            n_alpha_prime,
        ) = self._build_tree_single_step(
            theta_star, r_star, u, v, eps, theta_0, r_0
        )

        # update HMC path vars
        s *= s_prime
        n += n_prime
        
        # update dual averaging vars
        alpha += alpha_prime
        n_alpha += n_alpha_prime

       
        prng_key, subkey = jax.random.split(prng_key)
        theta_prime = self._accept_or_reject_proposed_theta(n_prime, 1/n, theta_double_star, theta_star, subkey)
        
        left_leaf_nodes = lax.cond(
            (j>0)*(i%2==1),
            self._handle_left_leaf_node_case,
            lambda *args: left_leaf_nodes,
            left_leaf_nodes, theta_double_star, r_double_star, i, j
        )
        
        s = lax.cond(
            (j>0)*(i%2==0),
            self._handle_right_leaf_node_case,
            lambda *args: s,
            left_leaf_nodes, theta_double_star, r_double_star, s, v, i, j
        )
        
        theta_star=theta_double_star
        r_star=r_double_star
        
        
        
        return (
            theta_star,
            r_star,
            u,
            v,
            j,
            eps,
            theta_0,
            r_0,
            s,
            n,
            theta_prime,
            alpha,
            n_alpha,
            left_leaf_nodes,
            i,
            prng_key,
        )
                                
        
        
    @jax.jit
    def _handle_left_leaf_node_case(self, left_leaf_nodes: jnp.array, theta_star: jnp.array, r_star: jnp.array, i: int, j: int) -> jnp.array:
        """
        if `i` is odd then it is the left-most leaf node of at least one balanced subtree
        -> overwrite correspoding stale position value in `left_leaf_nodes`

        `i` is the left-most leaf node in a tree of height `k` if i%2**k == 1
        we can range k from 1 to `j` to get if the current state is a left-most leaf node in the
        current subtree of height `k`. Summing these values gives a unique index to store these
        states in `left_leaf_nodes`. Moreover it is safe to overwrite values in `left_leaf_nodes`
        using this indexing schema since we are moving across the b-tree left-to-right (not proven
        here but if a state is a left-most leaf node `l` times by the time we get to the next
        state used `l` times we will have already made `l` u-turn checks against the previous
        `l` times state)
        """
        
        idx = lax.fori_loop(
            1,
            j + 1,
            lambda k, idx: idx + jnp.array(i % (2**k) == 1, jnp.int32),
            -1,
        )
        idx *= (j>0)  # needed for tracer
        
        left_leaf_nodes = (
            left_leaf_nodes
            .at[idx]
            .set((theta_star, r_star))
        )
        return left_leaf_nodes
    
    @jax.jit
    def _handle_right_leaf_node_case(self, left_leaf_nodes: jnp.array, theta_star: jnp.array, r_star: jnp.array, s: int, v: int, i: int, j: int) -> int:
        """
        if `i` is even then it is the right-most leaf node of at least one balanced subtree
        -> check for u-turn

        `i` is the right-most leaf node in a tree of height `k` if i%(2**k) == 0
        we can range from k from 1 to j to get which subtrees the current
        state is a right most leaf node of
        """
        s *= lax.fori_loop(
            1,
            j + 1,
            lambda k, s: s
            * lax.cond(
                i % (2**k) != 0,
                lambda *args: True,
                self._check_for_u_turn,
                theta_star,
                r_star,
                left_leaf_nodes[k][0],  # theta
                left_leaf_nodes[k][1],  # r
                v,
            ),
            s,
        )
        
        return s
    
    