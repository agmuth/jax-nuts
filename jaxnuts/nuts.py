from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import config
from jax.lax import cond, fori_loop, scan, while_loop
from jax.tree_util import Partial

config.update("jax_enable_x64", True)


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


def _draw_momentum_vector(theta: jnp.array, prng_key: jax.random.PRNGKey) -> jnp.array:
    """Draw momentum vecotr from N(0, I).

    Parameters
    ----------
    theta : jnp.array
        position vector (used for dimension)
    prng_key : jax.random.PRNGKey
        prng key to pass to jax.random

    Returns
    -------
    jnp.array
        momentum vector corresponding to `theta`
    """
    dim = theta.shape[0]
    r = jax.random.multivariate_normal(
        key=prng_key,
        mean=jnp.zeros(dim),
        cov=jnp.eye(dim),
    )
    return r


def _leapfrog(
    theta: jnp.array, r: jnp.array, eps: float, theta_loglik_grad: callable
) -> Tuple[jnp.array, jnp.array]:
    """Leapfrog integrator.

    Parameters
    ----------
    theta : jnp.array
        position state
    r : jnp.array
        momentum state
    eps : float
        step size parameter
    theta_loglik_grad : callable
        gradient of log likelihood wrt to theta

    Returns
    -------
    Tuple[jnp.array, jnp.array]
        updated position and momentum states
    """
    r_tilde = r + 0.5 * eps * theta_loglik_grad(theta)
    theta_tilde = theta + eps * r_tilde
    r_tilde = r_tilde + 0.5 * eps * theta_loglik_grad(theta_tilde)
    return theta_tilde, r_tilde


def _simulate_hamiltonian_dynamics_single_step(
    theta_star: jnp.array,
    r_star: jnp.array,
    u: float,
    v: int,
    eps: float,
    theta_0: jnp.array,
    r_0: jnp.array,
    theta_loglik: callable,
    theta_loglik_grad: callable,
) -> Tuple[jnp.array, jnp.array, int, int, float, float,]:
    """Base case of `BuildTree` function in NUTS paper.

    eps : float
        step size for leapfrog integrator
    theta_0 : jnp.array
        starting position state
    r_0 : jnp.array
        starting momentum state
    theta_loglik : callable
        log likelihood wrt to theta
    theta_loglik_grad : callable
        gradient of log likelihood wrt to theta

    Returns
    -------
    Tuple[jnp.array, jnp.array, int, int, float, float,]
        see nuts paper for definition/description of return values
    """
    delta_max = 1_000
    ln_u = jnp.log(u)

    theta_double_star, r_double_star = _leapfrog(
        theta_star, r_star, v * eps, theta_loglik_grad
    )  # push edge out one leapfrog step
    joint_loglik_double_star = theta_loglik(theta_double_star) - 0.5 * jnp.dot(
        r_double_star, r_double_star
    )
    joint_loglik_0 = theta_loglik(theta_0) - 0.5 * jnp.dot(r_0, r_0)

    n = (
        ln_u <= joint_loglik_double_star
    )  # indicator for if new edge state is eligible (!) adds to cardinatilty of set C
    s = (
        ln_u <= joint_loglik_double_star + delta_max
    )  # early termination criteria (equ. 3)

    # Metropolis-Hastings statistics for dual averaging
    alpha = jnp.minimum(
        1.0, jnp.exp(joint_loglik_double_star - joint_loglik_0)
    )  # M-H prob
    n_alpha = 1  # count for taking average later

    return (
        theta_double_star,
        r_double_star,
        n,
        s,
        alpha,
        n_alpha,
    )


def _accept_or_reject_proposed_theta(
    cond: int,
    prob: float,
    theta_pro: jnp.array,
    theta_cur: jnp.array,
    prng_key: jax.random.PRNGKey,
) -> jnp.array:
    """Accept or reject new theta basd on condition + probability.

    Parameters
    ----------
    cond : int
        0/1 boolean gate on accepting new theta
    prob : float
        probability of accepting new theta
    theta_pro : jnp.array
        proposed theta
    theta_cur : jnp.array
        current theta
    prng_key : jax.random.PRNGKey
        prng key to pass to jax.random

    Returns
    -------
    jnp.array
        either proposed or current theta
    """
    u = jax.random.uniform(key=prng_key, minval=0, maxval=1)
    theta_cur = jnp.where(
        u < prob * (cond > 0),
        theta_pro,
        theta_cur,
    )
    return theta_cur


def _check_for_u_turn(
    theta_plus: jnp.array,
    r_plus: jnp.array,
    theta_minus: jnp.array,
    r_minus: jnp.array,
    v: int,
) -> int:
    """Check for U-turn between forward-most and backward-most (in time) position-momentum states.

    Parameters
    ----------
    theta_plus : jnp.array
        forward most in time position state
    r_plus : jnp.array
        forward most in time momentum state
    theta_minus : jnp.array
        backward most in time position state
    r_minus : jnp.array
        backward most in time momentum state
    v : int
        +/- 1 `v` from NUTS algorithm (whether running forwards or backwards in time)

    Returns
    -------
    int
        0/1 - whether a U-turn occured
    """
    theta_delta = (
        theta_plus - theta_minus
    ) * v  # need to reverse order if args passed in backwards
    return (jnp.dot(theta_delta, r_plus) >= 0) * (jnp.dot(theta_delta, r_minus) >= 0)


def _find_reasonable_epsilon(
    theta: jnp.array,
    theta_loglik: callable,
    theta_loglik_grad: callable,
    prng_key: jax.random.PRNGKey,
) -> float:
    """Function to find reasonable starting value of epsilon (algorithm 4 in NUTS paper).

    Parameters
    ----------
    theta : jnp.array
        starting position state
    theta_loglik : callable
        log likelihood wrt to theta
    theta_loglik_grad : callable
        gradient of log likelihood wrt to theta
    prng_key : jax.random.PRNGKey
        prng key to pass to jax.random

    Returns
    -------
    float
        starting value for epsilon
    """
    # not worth jitting -> only called once
    ln2 = jnp.log(2)
    eps = 1.0
    r = _draw_momentum_vector(theta, prng_key)
    theta_prime, r_prime = _leapfrog(theta, r, eps, theta_loglik_grad)
    ln_p = theta_loglik(theta) - 0.5 * jnp.dot(r, r)
    ln_p_prime = theta_loglik(theta_prime) - 0.5 * jnp.dot(r_prime, r_prime)

    alpha = 2 * int(ln_p_prime - ln_p > -ln2) - 1
    while alpha * (ln_p_prime - ln_p) > -alpha * ln2:
        eps *= 2.0**alpha
        theta_prime, r_prime = _leapfrog(theta, r, eps, theta_loglik)
        ln_p_prime = theta_loglik(theta_prime) - 0.5 * jnp.dot(r_prime, r_prime)

    return eps


def _dual_average(
    state: DualAverageState,
    params: DualAveragePrams,
    alpha: float,
    n_alpha: int,
    m: int,
) -> DualAverageState:
    """Update step size parameter via dual averaging (dual average step from algorithm 6).

    Parameters
    ----------
    state : DualAverageState
        dual average state vars
    params : DualAveragePrams
        dual average param vars
    alpha : float
        sum of M-H acceptance probs from current tree
    n_alpha : int
        size of current tree
    m : int
        overall iteration

    Returns
    -------
    DualAverageState
        updated dual average state vars
    """
    (eps, eps_bar, H_bar) = (state.eps, state.eps_bar, state.H_bar)
    H_bar = (1 - 1 / (m + params.t_0)) * H_bar + (params.delta - alpha / n_alpha) / (
        m + params.t_0
    )

    # on log scale
    eps = params.mu - jnp.sqrt(m) / params.gamma * H_bar
    eps_bar = m**-params.kappa * eps + (1 - m**-params.kappa) * jnp.log(eps_bar)

    # exponentiate for next iter
    eps = jnp.exp(eps)
    eps_bar = jnp.exp(eps_bar)

    return DualAverageState(eps, eps_bar, H_bar)


def sample_posterior(
    loglik: callable,
    theta_0: jnp.array,
    M: int = 200,
    M_adapt: int = None,
    delta: float = 0.5 * (0.95 + 0.25),
    gamma: float = 0.05,
    kappa: float = 0.75,
    t_0: int = 10,
    prng_key: jax.random.PRNGKey = jax.random.PRNGKey(1234),
) -> jnp.array:
    """algorithm 6 from NUTS paper

    ref: https://arxiv.org/pdf/1111.4246.pdf

    Parameters
    ----------
    loglik : callable
        see NUTS paper
    theta_0 : jnp.array
        see NUTS paper
    M : int, optional
        see NUTS paper, by default 200
    M_adapt : int, optional
        see NUTS paper, by default None
    delta : float, optional
        see NUTS paper, by default 0.5*(0.95 + 0.25)
    gamma : float, optional
        see NUTS paper, by default 0.05
    kappa : float, optional
        see NUTS paper, by default 0.75
    t_0 : int, optional
        see NUTS paper, by default 10
    prng_key : jax.random.PRNGKey, optional
        by default jax.random.PRNGKey(1234)

    Returns
    -------
    jnp.array
        samples from loglik
    """
    if M_adapt is None:
        M_adapt = M // 2

    loglik_funcs = LogLikelihoodFuncs(
        theta_loglik=Partial(jax.jit(loglik)),
        theta_loglik_grad=Partial(jax.jit(jax.grad(loglik))),
    )

    prng_key, subkey1, subkey2 = jax.random.split(prng_key, 3)
    eps = _find_reasonable_epsilon(
        theta_0, loglik_funcs.theta_loglik, loglik_funcs.theta_loglik_grad, subkey1
    )

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
    carry, theta_samples = scan(
        _sample_posterior_scan_f,
        init,
        jnp.arange(1, M + 1),
    )

    return theta_samples


def _sample_posterior_scan_f(carry: Tuple, m: int) -> Tuple[Tuple, jnp.array]:
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
    n, alpha, n_alpha = 1, 0, 0  # need better names
    init_val = (
        double_tree_state,
        loglik_funcs,
        theta_m,
        n,
        alpha,
        n_alpha,
        prng_key,
    )
    val = while_loop(
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
    dual_average_state = cond(
        m < dual_average_params.M_adapt,
        _dual_average,
        lambda *args: dual_average_state,
        dual_average_state,
        dual_average_params,
        alpha,
        n_alpha,
        m,
    )

    carry = (
        theta_m,  # passed in as theta_{m-1} # maybe change name to theta_0
        loglik_funcs,
        dual_average_state,
        dual_average_params,
        prng_key,
    )

    return carry, theta_m


def _sample_posterior_scan_f_while_loop_cond_fun(val: Tuple) -> bool:
    """main for-loop of algorithm 6

    Parameters
    ----------
    val : Tuple
        see invocation

    Returns
    -------
    bool
        see invocation
    """
    double_tree_state = val[0]
    return double_tree_state.s == 1


def _sample_posterior_scan_f_while_loop_body_fun(val: Tuple) -> Tuple:
    """main for-loop of algorithm 6

    Parameters
    ----------
    val : Tuple
        see invocation

    Returns
    -------
    Tuple
        see invocation
    """
    # while loop starting on line 7 of nuts paper
    (double_tree_state, loglik_funcs, theta_m, n, alpha, n_alpha, prng_key) = val

    # split prng at top of function
    (
        prng_key,
        subkey1,
        subkey2,
        subkey3,
    ) = jax.random.split(prng_key, 4)

    # choose direction + double tree
    v_j = 2 * jax.random.bernoulli(subkey1).astype(jnp.int32) - 1
    double_tree_state = double_tree_state._replace(v=v_j)

    double_tree_state_prime, alpha, n_alpha = _double_tree(
        double_tree_state, loglik_funcs, subkey2
    )

    theta_m = _accept_or_reject_proposed_theta(
        cond=double_tree_state_prime.s,
        prob=double_tree_state_prime.n / n,  # |C| / |B|
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

    n += double_tree_state_prime.n  # update cardinality of set B

    val = (double_tree_state_prime, loglik_funcs, theta_m, n, alpha, n_alpha, prng_key)

    return val


def _double_tree(
    double_tree_state: DoubleTreeState,
    loglik_funcs: LogLikelihoodFuncs,
    prng_key: jax.random.PRNGKey,
) -> Tuple[DoubleTreeState, float, int]:
    """while-loop implementation of recursive "BuildTree" function in algorithm 6

    Parameters
    ----------
    double_tree_state : DoubleTreeState
        params passed + returned by "BuildTree"
    loglik_funcs : LogLikelihoodFuncs
        loglik and gradient
    prng_key : jax.random.PRNGKey

    Returns
    -------
    Tuple[DoubleTreeState, float, int]
        return values from "BuildTree" (tree state + MH/dual averaging vars)
    """

    prng_key, subkey = jax.random.split(prng_key)
    left_leaf_nodes = jnp.array(
        [
            (
                jnp.zeros(double_tree_state.theta_0.shape),
                jnp.zeros(double_tree_state.r_0.shape),
            )
        ]
    )

    # M-H/dual averaging vars
    alpha = 0
    n_alpha = 0

    init_val = (
        double_tree_state,
        loglik_funcs,
        alpha,
        n_alpha,
        left_leaf_nodes,
        subkey,
    )

    val = while_loop(
        _double_tree_while_loop_cond, _double_tree_while_loop_body, init_val
    )

    double_tree_state, _, alpha, n_alpha, _, _ = val

    return double_tree_state, alpha, n_alpha


def _double_tree_while_loop_cond(val: Tuple) -> bool:
    """while-loop implementation of recursive `BuildTree` function in algorithm 6

    Parameters
    ----------
    val : Tuple
        see invocation

    Returns
    -------
    bool
        see invocation
    """
    double_tree_state = val[0]
    return (double_tree_state.s == 1) * (double_tree_state.i < 2**double_tree_state.j)


def _double_tree_while_loop_body(val: Tuple) -> Tuple:
    """while-loop implementation of recursive `BuildTree` function in algorithm 6

    Parameters
    ----------
    val : Tuple
        see invocation

    Returns
    -------
    Tuple
        see invocation
    """
    (double_tree_state, loglik_funcs, alpha, n_alpha, left_leaf_nodes, prng_key) = val

    prng_key, subkey = jax.random.split(prng_key)

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
    n += n_prime  # cardinality of set C

    # update dual averaging/M-H vars
    alpha += alpha_prime
    n_alpha += n_alpha_prime

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

    # update array of left-most leaf nodes of current b subtrees if node is left node (i is odd)
    left_leaf_nodes, double_tree_state = cond(
        (double_tree_state.j > 0) * (double_tree_state.i % 2 == 1),
        _handle_left_leaf_node_case,
        lambda left_leaf_nodes, double_tree_state: (left_leaf_nodes, double_tree_state),
        left_leaf_nodes,
        double_tree_state,
    )

    # check for u-turn against array of left-most leaf nodes of current b subtrees if node is right node (i is even)
    left_leaf_nodes, double_tree_state = cond(
        (double_tree_state.j > 0) * (double_tree_state.i % 2 == 0),
        _handle_right_leaf_node_case,
        lambda left_leaf_nodes, double_tree_state: (left_leaf_nodes, double_tree_state),
        left_leaf_nodes,
        double_tree_state,
    )

    return (double_tree_state, loglik_funcs, alpha, n_alpha, left_leaf_nodes, prng_key)


def _handle_left_leaf_node_case(
    left_leaf_nodes: jnp.array, double_tree_state: DoubleTreeState
) -> Tuple[jnp.array, DoubleTreeState]:
    """If newest/i^{th} node adding while building out tree is a left leaf node this function is called.

    NOTE:
    `i` is a is a left-most leaf node in at least one balanced subtree iff `i` is odd.
    moreover if `i` is a left leaf node of the current binary subtree of height `h` then it is a left
    leaf node for the current binary subtree of height `h-1`, `h-2`, ..., `h-h+1`

    in fact `i` is the left-most leaf node in the current binary tree of height `k` iff
    `k` if i%2**k == 1. iterating over `k` and summing these values or taking the maximum we get
    an index corresponging to the tallest binary subtree that `i` is a left node of. since we
    only consider subtrees of any particular hieght one at a time + sequentially this defines a
    unqiue index for node `i` for the life of the particular subtree under consideration.

    that is, it is safe to overwrite values in the array `left_leaf_nodes` using this indexing
    scheme since we are moving across the b-tree left-to-right when we encounter a state that is
    a left-most leaf node `l` times by the time we get to the next state used `l` times we will
    have already made `l` u-turn checks against the previous `l` times state

    Parameters
    ----------
    left_leaf_nodes : jnp.array
        array of length `j` containning the left leaf node of the current subtree of height `k` at the k^th index
    double_tree_state : DoubleTreeState
        current tree state

    Returns
    -------
    Tuple[jnp.array, DoubleTreeState]
        updated left leaf nodes, treet state unchanged
    """

    idx_star = (double_tree_state.v + 1) // 2
    idx = fori_loop(
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


def _handle_right_leaf_node_case(
    left_leaf_nodes: jnp.array, double_tree_state: DoubleTreeState
) -> Tuple[jnp.array, DoubleTreeState]:
    """If newest/i^{th} node adding while building out tree is a left leaf node this function is called.

    NOTE:
    if `i` is even then it is the right-most leaf node in at least one of the current
    balanced subtrees. in fact it is the right-most leaf node of the current balanced subtree
    of height `k` if i%(2**k) == 0. we can range from k from 1 to j to get which of
    the current subtrees `i` is the right-most leaf node of and check for a u-turn against the
    corresponding left-most leaf node.

    Parameters
    ----------
    left_leaf_nodes : jnp.array
        array of length `j` containning the left leaf node of the current subtree of height `k` at the k^th index
    double_tree_state : DoubleTreeState
        current tree state

    Returns
    -------
    Tuple[jnp.array, DoubleTreeState]
        left leaf nodes is unchanges tree state variable `s` is potentially updated if a u-turn was detected
    """

    idx_star = (double_tree_state.v + 1) // 2
    s = fori_loop(
        1,
        double_tree_state.j + 1,
        lambda k, s: s
        * cond(
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
