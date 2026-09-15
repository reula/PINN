"""
line_search_backtracking.py

Personal implementation of backtracking. Compatible with GPU/TPU and Jit compiled.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable, Tuple, NamedTuple, Optional
from functools import partial


class BacktrackingState(NamedTuple):
    """Loop state of Backtracking"""
    alpha0: jax.Array
    alpha1: jax.Array
    phia0: jax.Array
    phia1: jax.Array
    iteration: jax.Array
    converged: jax.Array
    fc: jax.Array  # Function evaluation counter


class BacktrackingResult(NamedTuple):
    """Backtracking result"""
    a_k: jax.Array  # Learning rate
    f_k: jax.Array  # Function value
    g_k: jax.Array  # Gradient value
    old_fval: jax.Array
    converged: jax.Array
    failed: jax.Array
    nfev: jax.Array
    ngev: jax.Array
    status: jax.Array


@partial(jax.jit, static_argnames=("fun",))
def backtracking(
    fun: Callable,
    xk: jax.Array,
    pk: jax.Array,
    gfk: jax.Array,
    old_fval: jax.Array,
    c1: float = 1e-4,
    alpha0: float = 1.0,
    rholo: float = 0.1,
    rhohi: float = 0.5,
    max_iter: int = 20,
    alphamin: float = 1e-12
) -> BacktrackingResult:
    """
    Backtracking line search compatible with JIT.
    
    Finds a step-lenght that satisfies the Armijo (sufficient decrease) condition:
        f(xk + alpha*pk) <= f(xk) + c1*alpha*grad(f)^T*pk
    Using interpolation of the objective function.
    
    Args:
        fun: Objective function
        xk: Current point (also called phi(0))
        pk: Search direction
        gfk: Gradient of loss at xk
        old_fval: Value of f at xk
        c1: Sufficient decrease parameter (1e-4, typically)
        alpha0: First trial of alpha
        rholo: Minimum reduction factor
        rhohi: Maximum reduction factor
        max_iter: Maximum number of iterations
        alphamin: Minimum value of alpha allowed. Using only Armijo may result in very small alphas. We discard this possiblity.
        
    Returns:
        BacktrackingResult with:
            - a_k: Line search that satisfies (if converged=True) Armijo condition
            - f_k: Value of f(xk + a_k*pk)
            - g_k: Gradient of objective at xk + a_k*pk
            - old_fval: Value of f(xk)
            - converged: If true, then a_k satisfies Armijo
            - failed: True if convergence fails.
            - nfev: Function evaluations
            - ngev: Gradient evaluations (always 1, tecnically you can simply get rid of this variable)
            - status: State (0=ok, 1=failed)
    """
    
    phi0 = old_fval
    derphi0 = jnp.dot(gfk, pk)
    
    # Function phi(s) = f(xk + s*pk)
    def phi(s):
        return fun(xk + s * pk)
    
    # Gradient of objective
    grad_fun = jax.grad(fun)
    
    # phi(alpha0)
    phia1 = phi(alpha0)
    
    # Initial state
    init_state = BacktrackingState(
        alpha0=jnp.array(alpha0),
        alpha1=jnp.array(alpha0),
        phia0=phia1,
        phia1=phia1,
        iteration=jnp.array(0),
        converged=jnp.array(False),
        fc=jnp.array(1)
    )
    
    def cond_fun(state: BacktrackingState) -> jax.Array:
        """Condición de continuación del loop"""
        armijo_satisfied = state.phia1 <= phi0 + c1 * state.alpha1 * derphi0
        alpha_too_small = state.alpha1 <= alphamin
        # Continue if: (Armijo is not satisfied) and (alpha not too small) and (max iters not excedeed)
        return (~armijo_satisfied) & (~alpha_too_small) & (state.iteration < max_iter)
    
    def body_fun(state: BacktrackingState) -> BacktrackingState:
        """One Backtracking iteration"""
        first_iter = state.iteration == 0
        
        # First iteration: Quadratic interpolation (with phi(alpha0), phi(0) and phi'(0))
        alphatemp_quad = -(derphi0 * state.alpha1**2) / (
            2 * (state.phia1 - phi0 - state.alpha1 * derphi0)
        )
        
        # Cubic interpolation (with phi(alpha1), phi(alpha0), phi(0) and phi'(0))
        vec_factor = 1.0 / (state.alpha1 - state.alpha0)
        
        mat_row1_col1 = 1.0 / state.alpha1**2
        mat_row1_col2 = -1.0 / state.alpha0**2
        mat_row2_col1 = -state.alpha0 / state.alpha1**2
        mat_row2_col2 = state.alpha1 / state.alpha0**2
        
        vec_elem1 = state.phia1 - phi0 - derphi0 * state.alpha1
        vec_elem2 = state.phia0 - phi0 - derphi0 * state.alpha0
        
        a = vec_factor * (mat_row1_col1 * vec_elem1 + mat_row1_col2 * vec_elem2)
        b = vec_factor * (mat_row2_col1 * vec_elem1 + mat_row2_col2 * vec_elem2)
        
        a_too_small = jnp.abs(a) <= 1e-100
        discriminant = b**2 - 3 * a * derphi0
        discriminant_valid = discriminant >= 0
        
        alphatemp_cubic = (-b + jnp.sqrt(jnp.maximum(discriminant, 0))) / (3 * a)
        alphatemp_cubic_fallback = -(derphi0 * state.alpha1**2) / (
            2 * (state.phia1 - phi0 - state.alpha1 * derphi0)
        )
        
        alphatemp_cubic = jnp.where(discriminant_valid, alphatemp_cubic, alphatemp_cubic_fallback)
        alphatemp_cubic = jnp.where(a_too_small, -derphi0 / (2 * b), alphatemp_cubic)
        alphatemp = jnp.where(first_iter, alphatemp_quad, alphatemp_cubic)
        alphatemp = jnp.minimum(alphatemp, rhohi * state.alpha1)
        
        new_alpha0 = state.alpha1
        new_phia0 = state.phia1
        new_alpha1 = jnp.maximum(rholo * state.alpha1, alphatemp)
        new_phia1 = phi(new_alpha1)
        
        return BacktrackingState(
            alpha0=new_alpha0,
            alpha1=new_alpha1,
            phia0=new_phia0,
            phia1=new_phia1,
            iteration=state.iteration + 1,
            converged=jnp.array(False),
            fc=state.fc + 1
        )
    
    # Loop execution
    final_state = lax.while_loop(cond_fun, body_fun, init_state)
    
    # Verify convergence
    armijo_satisfied = final_state.phia1 <= phi0 + c1 * final_state.alpha1 * derphi0
    alpha_too_small = final_state.alpha1 <= alphamin
    
    # ONLY CONVERGES IF SATISFIES ARMIJO AND ALPHA IS NOT TOO SMALL
    converged = armijo_satisfied & (~alpha_too_small)
    failed = jnp.logical_not(converged)
    
    # 
    alpha_final = final_state.alpha1
    x_new = xk + alpha_final * pk
    f_new = fun(x_new)
    g_new = grad_fun(x_new)
    
    # Function and gradient evaluations
    nfev = final_state.fc + jnp.array(1)  # +1 por f_new
    ngev = jnp.array(1)  # Una evaluación de gradiente
    
    status = jnp.where(converged, jnp.array(0), jnp.array(1))
    
    return BacktrackingResult(
        a_k=alpha_final,
        f_k=f_new,
        g_k=g_new,
        old_fval=phi0,
        converged=converged,
        failed=failed,
        nfev=nfev,
        ngev=ngev,
        status=status
    )


def backtracking_with_defaults(
    fun: Callable,
    xk: jax.Array,
    pk: jax.Array,
    gfk: Optional[jax.Array] = None,
    old_fval: Optional[jax.Array] = None,
    c1: float = 1e-4,
    alpha0: float = 1.0,
    rholo: float = 0.1,
    rhohi: float = 0.5,
    max_iter: int = 20,
    alphamin: float = 1e-12
) -> BacktrackingResult:
    """
    Wrapper of backtracking that handles none values (no JIT-compilable).
    """
    grad_fun = jax.grad(fun)
    
    if gfk is None:
        gfk = grad_fun(xk)
    if old_fval is None:
        old_fval = fun(xk)
    
    return backtracking(fun, xk, pk, gfk, old_fval, c1, alpha0, rholo, rhohi, max_iter, alphamin)
