import jax
import jax.numpy as jnp
from typing import Union, Tuple, Dict
import numpy as np
from jax import lax
from typing import NamedTuple, Tuple


from jax.tree_util import register_pytree_node_class, Partial


# def lax.scan(f, init, xs, length=None):
#   if xs is None:
#     xs = [None] * length
#   carry = init
#   ys = []
#   for x in xs:
#     carry, y = f(carry, x)
#     ys.append(y)
#   return carry, np.stack(ys)

# def lax.while_loop(cond_fun, body_fun, init_val):
#   val = init_val
#   while cond_fun(val):
#     val = body_fun(val)
#   return val

# def lax.fori_loop(lower, upper, body_fun, init_val):
#   val = init_val
#   for i in range(lower, upper):
#     val = body_fun(i, val)
#   return val

# def lax.cond(pred, true_fun, false_fun, *operands):
#   if pred:
#     return true_fun(*operands)
#   else:
#     return false_fun(*operands)
# class DoubleTreeState(NamedTuple):
#     # furthest out state
#     theta_star: jnp.array
#     r_star: jnp.array

#     # beginning state
#     theta_0: jnp.array
#     r_0: jnp.array

#     # current proposed sample from target distn
#     theta_prime: jnp.array

#     eps: float  # leapfrog step length
#     u: float # slice variable
#     v: int # time direction
#     j: int # max tree depth
#     i: int # current walk length
#     s: int # early stopping flag
#     n: int # number of acceptable states encounter


class DoubleTreeState(NamedTuple):
    # furthest out states
    theta_plus_minus: jnp.array
    r_plus_minus: jnp.array

    # beginning state
    theta_0: jnp.array
    r_0: jnp.array

    # current proposed sample from target distn
    theta_prime: jnp.array

    eps: float  # leapfrog step length
    u: float  # slice variable
    v: int  # time direction
    j: int  # max tree depth
    i: int  # current walk length
    s: int  # early stopping flag
    n: int  # number of acceptable states encounter


class LogLikelihoodFuncs(NamedTuple):
    theta_loglik: callable
    theta_loglik_grad: callable


class LeapFrogState(NamedTuple):
    theta: jnp.array
    r: jnp.array
    eps: float
    theta_loglik_grad: callable


class DualAverageState(NamedTuple):
    eps: float
    eps_bar: float
    H_bar: float


class DualAveragePrams(NamedTuple):
    mu: float
    delta: float
    t_0: float
    gamma: float
    kappa: float
    M_adapt: int


def sample_posterior(
    loglik,
    theta_0,
    M=200,
    M_adapt=None,
    delta=0.5 * (0.95 + 0.25),
    gamma=0.05,
    kappa=0.75,
    t_0=10,
    delta_max=1_000,
    prng_key=jax.random.PRNGKey(1234),
):
    # algorithm 6

    if M_adapt is None:
        M_adapt = M // 2

    loglik_funcs = LogLikelihoodFuncs(
        theta_loglik=Partial(jax.jit(loglik)),
        theta_loglik_grad=Partial(jax.jit(jax.grad(loglik))),
        # theta_loglik=loglik,
        # theta_loglik_grad=jax.grad(loglik),
    )

    prng_key, subkey1, subkey2 = jax.random.split(prng_key, 3)
    eps = _find_reasonable_epsilon(theta_0, loglik_funcs.theta_loglik, subkey1)

    dual_average_state = DualAverageState(
        eps=eps,
        eps_bar=1.0,
        H_bar=0.0,
    )

    dual_average_params = DualAveragePrams(
        mu=jnp.log(10 * eps),
        delta=delta,
        t_0=t_0,
        gamma=gamma,
        kappa=kappa,
        M_adapt=M_adapt,
    )

    init = (theta_0, loglik_funcs, dual_average_state, dual_average_params, subkey2)

    # for loop (line 3 nuts paper)
    carry, theta_samples = lax.scan(
        _sample_posterior_scan_f,
        init,
        jnp.arange(1, M + 1),
    )

    return theta_samples


def _sample_posterior_scan_f(carry: Tuple, m: int):
    # for loop (line 3 nuts paper)
    (
        theta_m,  # passed in as theta_{m-1} # maybe change name to theta_0
        loglik_funcs,
        dual_average_state,
        dual_average_params,
        prng_key,
    ) = carry

    # split prng key
    prng_key, subkey1, subkey2 = jax.random.split(prng_key, 3)

    r_0 = _draw_momentum_vector(theta_m, subkey1)

    u = jax.random.uniform(
        key=subkey2,
        minval=0,
        maxval=jnp.exp(loglik_funcs.theta_loglik(theta_m) - 0.5 * jnp.dot(r_0, r_0)),
    )

    # initialize vars
    double_tree_state = DoubleTreeState(
        # furthest out states
        theta_plus_minus=jnp.array([theta_m, theta_m]),
        r_plus_minus=jnp.array([r_0, r_0]),
        theta_0=theta_m,
        r_0=r_0,
        theta_prime=theta_m,
        eps=dual_average_state.eps,
        u=u,
        v=0,
        j=0,
        i=0,
        s=1,
        n=0,
    )
    n, alpha, n_alpha = 0, 0, 0  # need better names
    init_val = (
        double_tree_state,
        loglik_funcs,
        theta_m,
        n,
        alpha,
        n_alpha,
        prng_key,
    )
    val = lax.while_loop(
        _sample_posterior_scan_f_while_loop_cond_fun,
        _sample_posterior_scan_f_while_loop_body_fun,
        init_val,
    )
    
    (
        double_tree_state,
        loglik_funcs,
        theta_m,
        n,
        alpha,
        n_alpha,
        prng_key,
    ) = val

    # dual average here
    dual_average_state = lax.cond(
        m < dual_average_params.M_adapt,
        _dual_average,
        lambda *args: dual_average_state,
        dual_average_state, dual_average_params, alpha, n_alpha, m
    )
    
    carry = (
        theta_m,  # passed in as theta_{m-1} # maybe change name to theta_0
        loglik_funcs,
        dual_average_state,
        dual_average_params,
        prng_key,
    )

    return  carry, theta_m


def _sample_posterior_scan_f_while_loop_cond_fun(val: Tuple):
    double_tree_state = val[0]
    return double_tree_state.s == 1


def _sample_posterior_scan_f_while_loop_body_fun(val: Tuple):
    # while loop starting on line 7 of nuts paper
    (double_tree_state, loglik_funcs, theta_m, n, alpha, n_alpha, prng_key) = val

    (
        prng_key, 
        subkey1,
        subkey2,
        subkey3,
    ) = jax.random.split(
        prng_key, 4
    )  

    # choose direction + double tree
    v_j = 2 * jax.random.bernoulli(subkey1).astype(jnp.int32) - 1
    double_tree_state = double_tree_state._replace(v=v_j)

    double_tree_state_prime, alpha, n_alpha = _double_tree(
        double_tree_state, loglik_funcs, subkey2
    )

    theta_m = _accept_or_reject_proposed_theta(
        cond=double_tree_state_prime.s,
        prob=double_tree_state_prime.n / n,
        theta_pro=double_tree_state_prime.theta_prime,
        theta_cur=theta_m,
        prng_key=subkey3,
    )

    u_turn = _check_for_u_turn(
        double_tree_state_prime.theta_plus_minus[1],
        double_tree_state_prime.theta_plus_minus[0],
        double_tree_state_prime.r_plus_minus[1],
        double_tree_state_prime.r_plus_minus[0],
        1,
    )

    double_tree_state_prime = double_tree_state_prime._replace(
        s=double_tree_state_prime.s * u_turn,
        j=double_tree_state_prime.j + 1,
    )

    n += double_tree_state_prime.n

    val = (double_tree_state_prime, loglik_funcs, theta_m, n, alpha, n_alpha, prng_key)

    return val


def _double_tree(
    double_tree_state: DoubleTreeState, loglik_funcs: LogLikelihoodFuncs, prng_key
):
    # function "BuildTree" in nuts paper

    left_leaf_nodes = jnp.array(
        [
            (
                jnp.zeros(double_tree_state.theta_0.shape),
                jnp.zeros(double_tree_state.r_0.shape),
            )
        ]
    )

    alpha = 0
    n_alpha = 0

    init_val = (
        double_tree_state,
        loglik_funcs,
        alpha,
        n_alpha,
        left_leaf_nodes,
        prng_key,
    )

    val = lax.while_loop(
        _double_tree_while_loop_cond, _double_tree_while_loop_body, init_val
    )

    double_tree_state, _, alpha, n_alpha, _, _ = val

    return double_tree_state, alpha, n_alpha


def _double_tree_while_loop_cond(val: Tuple):
    double_tree_state = val[0]
    return (double_tree_state.s == 1) * (double_tree_state.i < 2**double_tree_state.j)


def _double_tree_while_loop_body(val: Tuple):
    (double_tree_state, loglik_funcs, alpha, n_alpha, left_leaf_nodes, prng_key) = val

    # unpack `double_tree_state`

    theta_plus_minus = double_tree_state.theta_plus_minus
    r_plus_minus = double_tree_state.r_plus_minus
    theta_0 = double_tree_state.theta_0
    r_0 = double_tree_state.r_0
    theta_prime = double_tree_state.theta_prime
    eps = double_tree_state.eps
    u = double_tree_state.u
    v = double_tree_state.v
    j = double_tree_state.j
    i = double_tree_state.i
    s = double_tree_state.s
    n = double_tree_state.n

    i += 1  # incr here to align with b-tree 1-indexing
    idx_star = (v + 1) // 2

    theta_star = theta_plus_minus[idx_star]
    r_star = r_plus_minus[idx_star]

    (
        theta_double_star,
        r_double_star,
        n_prime,
        s_prime,
        alpha_prime,
        n_alpha_prime,
    ) = _simulate_hamiltonian_dynamics_single_step(
        theta_star,
        r_star,
        u,
        v,
        eps,
        theta_0,
        r_0,
        loglik_funcs.theta_loglik,
        loglik_funcs.theta_loglik_grad,
    )

    # update HMC path vars

    theta_plus_minus = theta_plus_minus.at[idx_star].set(theta_double_star)
    r_plus_minus = r_plus_minus.at[idx_star].set(r_double_star)

    s *= s_prime
    n += n_prime

    # update dual averaging vars
    alpha += alpha_prime
    n_alpha += n_alpha_prime

    prng_key, subkey = jax.random.split(prng_key)
    theta_prime = _accept_or_reject_proposed_theta(
        n_prime, 1 / n, theta_double_star, theta_prime, subkey
    )

    # pack up `double_tree_state`

    double_tree_state = DoubleTreeState(
        theta_plus_minus=theta_plus_minus,
        r_plus_minus=r_plus_minus,
        theta_0=theta_0,
        r_0=r_0,
        theta_prime=theta_prime,
        eps=eps,
        u=u,
        v=v,
        j=j,
        i=i,
        s=s,
        n=n,
    )

    left_leaf_nodes, double_tree_state = lax.cond(
        (double_tree_state.j > 0) * (double_tree_state.i % 2 == 1),
        _handle_left_leaf_node_case,
        lambda left_leaf_nodes, double_tree_state: (left_leaf_nodes, double_tree_state),
        left_leaf_nodes, double_tree_state,
    )

    left_leaf_nodes, double_tree_state = lax.cond(
        (double_tree_state.j > 0) * (double_tree_state.i % 2 == 0),
        _handle_right_leaf_node_case,
        lambda left_leaf_nodes, double_tree_state: (left_leaf_nodes, double_tree_state),
        left_leaf_nodes, double_tree_state,
    )

    return (double_tree_state, loglik_funcs, alpha, n_alpha, left_leaf_nodes, prng_key)



def _handle_left_leaf_node_case(left_leaf_nodes, double_tree_state) -> jnp.array:
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
    
    idx_star = (double_tree_state.v + 1) // 2
    idx = lax.fori_loop(
        1,
        double_tree_state.j + 1,
        lambda k, idx: idx + jnp.array(double_tree_state.i % (2**k) == 1, jnp.int32),
        -1,
    )
    idx *= double_tree_state.j > 0  # needed for tracer

    left_leaf_nodes = left_leaf_nodes.at[idx].set(
        (
            double_tree_state.theta_plus_minus[idx_star],
            double_tree_state.r_plus_minus[idx_star],
        )
    )
    return left_leaf_nodes, double_tree_state


def _handle_right_leaf_node_case(left_leaf_nodes, double_tree_state):
    """
    if `i` is even then it is the right-most leaf node of at least one balanced subtree
    -> check for u-turn

    `i` is the right-most leaf node in a tree of height `k` if i%(2**k) == 0
    we can range from k from 1 to j to get which subtrees the current
    state is a right most leaf node of
    """
    
    idx_star = (double_tree_state.v + 1) // 2
    s = lax.fori_loop(
        1,
        double_tree_state.j + 1,
        lambda k, s: s
        * lax.cond(
            double_tree_state.i % (2**k) != 0,
            lambda *args: True,
            _check_for_u_turn,
            double_tree_state.theta_plus_minus[idx_star],
            double_tree_state.r_plus_minus[idx_star],
            left_leaf_nodes[k][0],  # theta
            left_leaf_nodes[k][1],  # r
            double_tree_state.v,  # direction in time
        ),
        double_tree_state.s,
    )
    double_tree_state = double_tree_state._replace(s=s)

    return left_leaf_nodes, double_tree_state


def _find_reasonable_epsilon(theta, theta_loglik, prng_key):  # no state needed
    # not worth jitting
    ln2 = jnp.log(2)
    eps = 1.0
    r = _draw_momentum_vector(theta, prng_key)
    theta_prime, r_prime = _leapfrog(theta, r, eps, theta_loglik)
    ln_p = theta_loglik(theta) - 0.5 * jnp.dot(r, r)
    ln_p_prime = theta_loglik(theta_prime) - 0.5 * jnp.dot(r_prime, r_prime)

    alpha = 2 * int(ln_p_prime - ln_p > -ln2) - 1  # lax.cond
    while alpha * (ln_p_prime - ln_p) > -alpha * ln2:
        eps *= 2.0**alpha
        theta_prime, r_prime = _leapfrog(theta, r, eps, theta_loglik)
        ln_p_prime = theta_loglik(theta_prime) - 0.5 * jnp.dot(r_prime, r_prime)

    return eps


def _draw_momentum_vector(theta, prng_key):  # no state needed
    dim = theta.shape[0]
    r = jax.random.multivariate_normal(
        key=prng_key,
        mean=jnp.zeros(dim),
        cov=jnp.eye(dim),
    )
    return r


def _accept_or_reject_proposed_theta(cond, prob, theta_pro, theta_cur, prng_key):
    u = jax.random.uniform(key=prng_key, minval=0, maxval=1)
    theta_cur = jnp.where(
        # u < prob*cond,
        (cond > 0) * (u < prob),
        theta_pro,
        theta_cur,
    )
    return theta_cur


def _check_for_u_turn(theta_plus, r_plus, theta_minus, r_minus, v):  # no state needed
    theta_delta = (
        theta_plus - theta_minus
    ) * v  # need to reverse order if args passed in backwards
    return (jnp.dot(theta_delta, r_plus) >= 0) * (jnp.dot(theta_delta, r_minus) >= 0)



def _dual_average(
    state: DualAverageState, params: DualAveragePrams, alpha, n_alpha, m
) -> DualAverageState:
    (eps, eps_bar, H_bar) = (state.eps, state.eps_bar, state.H_bar)

    # split out updates to avoid having to save vectors

    H_bar += (params.delta - alpha / n_alpha) / (m + params.t_0)

    # on log scale
    eps = params.mu - jnp.sqrt(m) / params.gamma * H_bar
    eps_bar = m**-params.kappa * eps + (1 - m**-params.kappa) * jnp.log(eps_bar)

    # exponentiate for next iter
    eps = jnp.exp(eps)
    eps_bar = jnp.exp(eps_bar)

    return DualAverageState(eps, eps_bar, H_bar)


def _simulate_hamiltonian_dynamics_single_step(
    theta_star, r_star, u, v, eps, theta_0, r_0, theta_loglik, theta_loglik_grad
):
    delta_max = 1_000
    ln_u = jnp.log(u)

    theta_double_star, r_double_star = _leapfrog(
        theta_star, r_star, v * eps, theta_loglik_grad
    )  # push edge out one leapfrog step
    joint_loglik_double_star = theta_loglik(theta_double_star) - 0.5 * jnp.dot(
        r_double_star, r_double_star
    )
    joint_loglik_0 = theta_loglik(theta_0) - 0.5 * jnp.dot(r_0, r_0)

    n = ln_u <= joint_loglik_double_star  # indicator for if new edge state is eligible
    s = (
        ln_u
        <= joint_loglik_double_star
        + delta_max  # change so that this cond is returned by func `delta_max`
    )  # early termination criteria (equ. 3)

    alpha = jnp.minimum(1.0, jnp.exp(joint_loglik_double_star - joint_loglik_0))
    n_alpha = 1

    return (
        theta_double_star,
        r_double_star,
        n,
        s,
        alpha,
        n_alpha,
    )  # TODO: think can upodate + return state here -> NO update state outside of function


def _leapfrog(theta, r, eps, theta_loglik_grad):  # no state needed
    r_tilde = r + 0.5 * eps * theta_loglik_grad(theta)
    theta_tilde = theta + eps * r_tilde
    r_tilde = r_tilde + 0.5 * eps * theta_loglik_grad(theta_tilde)
    return theta_tilde, r_tilde

