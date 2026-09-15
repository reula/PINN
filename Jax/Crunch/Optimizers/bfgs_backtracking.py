import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable, Tuple, NamedTuple, Optional
import time
from functools import partial
import sys
import os

# Add project root to path and import custom backtracking
file_path = os.getcwd()
project_root = os.path.dirname(file_path)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the custom backtracking implementation
from line_search_backtracking import backtracking, BacktrackingResult

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)


class _BFGSResults(NamedTuple):
  """Results from BFGS optimization.

  Parameters:
    converged: True if minimization converged.
    failed: True if line search failed.
    k: integer the number of iterations of the BFGS update.
    nfev: integer total number of objective evaluations performed.
    ngev: integer total number of jacobian evaluations
    nhev: integer total number of hessian evaluations
    x_k: array containing the last argument value found during the search. If
      the search converged, then this value is the argmin of the objective
      function.
    f_k: array containing the value of the objective function at `x_k`. If the
      search converged, then this is the (local) minimum of the objective
      function.
    g_k: array containing the gradient of the objective function at `x_k`. If
      the search converged the l2-norm of this tensor should be below the
      tolerance.
    H_k: array containing the inverse of the estimated Hessian.
    status: int describing end state.
    line_search_status: int describing line search end state (only means
      something if line search fails).
  """
  converged: bool | jax.Array
  failed: bool | jax.Array
  k: int | jax.Array
  nfev: int | jax.Array
  ngev: int | jax.Array
  nhev: int | jax.Array
  x_k: jax.Array
  f_k: jax.Array
  g_k: jax.Array
  H_k: jax.Array
  old_old_fval: jax.Array
  status: int | jax.Array
  line_search_status: int | jax.Array


def minimize_bfgs(
    fun: Callable,
    x0: jax.Array,
    initial_H: jax.Array,
    maxiter: int | None = None,
    norm=jnp.inf,
    gtol: float = 1e-5,
    update_method: str = "bfgs",
    initial_scale: bool = False,
    c1: float = 1e-4,
    alpha0: float = 1.0,
    rholo: float = 0.1,
    rhohi: float = 0.5,
    max_iter: int = 20,
    alphamin: float = 1e-12,
) -> _BFGSResults:
  """Minimize a function using BFGS.

  Implements the BFGS algorithm from
    Algorithm 6.1 from Wright and Nocedal, 'Numerical Optimization', 1999, pg.
    136-143.

  Args:
    fun: function of the form f(x) where x is a flat ndarray and returns a real
      scalar. The function should be composed of operations with vjp defined.
    x0: initial guess.
    maxiter: maximum number of iterations.
    norm: order of norm for convergence check. Default inf.
    gtol: terminates minimization when |grad|_norm < g_tol.
    update_method: método de actualización ('bfgs', 'ssbfgs', 'ssbroyden2')
    initial_scale: usar scaling inicial para ssbroyden2
    c1: parámetro de Armijo para backtracking
    alpha0: tamaño de paso inicial
    rholo: factor de reducción mínimo
    rhohi: factor de reducción máximo
    max_iter: máximo de iteraciones de line search

  Returns:
    Optimization result.
  """

  if maxiter is None:
    maxiter = jnp.size(x0) * 200

  d = x0.shape[0]

  f_0, g_0 = jax.value_and_grad(fun)(x0)
  state = _BFGSResults(
      converged=jnp.linalg.norm(g_0, ord=norm) < gtol,
      failed=False,
      k=0,
      nfev=1,
      ngev=1,
      nhev=0,
      x_k=x0,
      f_k=f_0,
      g_k=g_0,
      H_k=initial_H,
      old_old_fval=f_0 + jnp.linalg.norm(g_0) / 2,
      status=0,
      line_search_status=0,
  )

  def cond_fun(state):
    return (jnp.logical_not(state.converged)
            & jnp.logical_not(state.failed)
            & (state.k < maxiter))    

  def body_fun(state):
    p_k = -_dot(state.H_k, state.g_k)
    
    line_search_results = backtracking(
        fun, 
        xk = state.x_k,
        pk = p_k,
        gfk=state.g_k,
        old_fval=state.f_k,
        c1=c1,
        alpha0=alpha0,
        rholo = rholo,
        rhohi = rhohi,
        max_iter = max_iter,
        alphamin = alphamin,
    )
    
    # CRÍTICO: Si line search falla, detener INMEDIATAMENTE
    # No actualizar nada más, solo marcar como failed y retornar
    def handle_ls_failure(ls_res):
        """Si line search falla, retornar estado sin actualizar"""
        return state._replace(
            nfev=state.nfev + ls_res.nfev,
            ngev=state.ngev + ls_res.ngev,
            failed=True,
            line_search_status=ls_res.status,
        )
    
    def handle_ls_success(ls_res):
        """Si line search tiene éxito, continuar con actualización BFGS"""
        
        s_k = ls_res.a_k * p_k
        x_kp1 = state.x_k + s_k
        f_kp1 = ls_res.f_k
        g_kp1 = ls_res.g_k
        y_k = g_kp1 - state.g_k
        
        # --- MODIFIED rho_k CALCULATION ---
        yk_dot_sk = _dot(y_k, s_k)
        small_denominator_threshold = 1e-32 

        pred_yk_dot_sk_is_problematic = jnp.abs(yk_dot_sk) < small_denominator_threshold

        def calculate_rho_k_normally(operand_yds):
            return 1.0 / (operand_yds + jnp.sign(operand_yds) * 1e-20)

        def handle_problematic_rho_k_denominator(operand_yds):
            return jnp.array(1000.0, dtype=operand_yds.dtype)

        rho_k = jax.lax.cond(
            pred_yk_dot_sk_is_problematic,
            handle_problematic_rho_k_denominator,
            calculate_rho_k_normally,
            yk_dot_sk
        )
        # --- END OF MODIFIED rho_k CALCULATION ---

        # Store current state variables for convenience
        Hk = state.H_k
        yk = y_k
        sk = s_k
        rhok = rho_k
        alpha_k_linesearch = ls_res.a_k
        gfk_old = state.g_k
        dim = state.x_k.shape[0]

        # --- Conditional Hessian Update ---
        if update_method.lower() == "ssbroyden2":
            eps_div = 1e-32 

            Hkyk = _dot(Hk, yk)
            ykHkyk = _dot(yk, Hkyk)
            
            hk = ykHkyk * rhok
            bk = -alpha_k_linesearch * rhok * _dot(sk, gfk_old)
            ak = bk * hk - 1.0

            abs_ak = jnp.abs(ak)
            sqrt_term = jnp.sqrt(abs_ak / (1.0 + abs_ak + eps_div)) 
            rhokm_val = hk * (1.0 - sqrt_term)
            rhokm = jnp.minimum(1.0, rhokm_val) 

            thetakm_den = ak + jnp.sign(ak) * eps_div
            thetakm = (rhokm - 1.0) / thetakm_den
            
            thetakp_den = rhokm + eps_div
            thetakp = 1.0 / thetakp_den

            bk_safe_den = bk + jnp.sign(bk) * eps_div
            term_for_thetak = (1.0 - bk) / bk_safe_den
            thetak = jnp.maximum(thetakm, jnp.minimum(thetakp, term_for_thetak))

            is_initial_step_and_scale_cond = initial_scale & (state.k == 0) & \
                                            jnp.allclose(Hk, jnp.eye(dim, dtype=Hk.dtype), atol=1e-6)

            tauk_A_den = (1.0 + ak * thetak + eps_div)
            tauk_A = hk / tauk_A_den

            rhokk_den = bk + jnp.sign(bk) * eps_div
            rhokk = jnp.minimum(1.0, 1.0 / rhokk_den)
            
            sigmak = 1.0 + thetak * ak
            
            sigmaknm1_exp = lax.cond(dim == 1,
                                    lambda _: 0.0,
                                    lambda _: 1.0 / (1.0 - dim),
                                    None)
            sigmaknm1 = jnp.abs(sigmak) ** sigmaknm1_exp

            tauk_B_cond = thetak <= 0.0
            tauk_B_true = jnp.minimum(rhokk * sigmaknm1, sigmak)
            tauk_B_false_den = thetak + eps_div
            tauk_B_false = rhokk * jnp.minimum(sigmaknm1, 1.0 / tauk_B_false_den)
            tauk_B = jnp.where(tauk_B_cond, tauk_B_true, tauk_B_false)

            tauk = jnp.where(is_initial_step_and_scale_cond, tauk_A, tauk_B)
            tauk_safe = tauk + jnp.sign(tauk) * eps_div

            ykHkyk_safe = ykHkyk + jnp.sign(ykHkyk) * eps_div
            vk = sk * rhok - Hkyk / ykHkyk_safe
            
            phik_den = (1.0 + ak * thetak + eps_div)
            phik = (1.0 - thetak) / phik_den

            term1_H_update = Hk
            term2_H_update_num = _einsum('i,j->ij', Hkyk, Hkyk)
            term2_H_update = term2_H_update_num / ykHkyk_safe
            
            term3_H_update_vk_outer = _einsum('i,j->ij', vk, vk)
            term3_H_update = phik * ykHkyk * term3_H_update_vk_outer
            
            H_numerator = term1_H_update - term2_H_update + term3_H_update
            H_scaled = H_numerator / tauk_safe
            
            term4_H_update_sk_outer = _einsum('i,j->ij', sk, sk)
            term4_H_update = term4_H_update_sk_outer * rhok
            
            H_kp1 = H_scaled + term4_H_update

        elif update_method.lower() == "ssbfgs":
            sy_k = sk[:, jnp.newaxis] * yk[jnp.newaxis, :]
            bk = -alpha_k_linesearch * rhok * _dot(sk, gfk_old)
            tauk = jnp.minimum(1.0, 1.0 / bk)
            Hk_scaled = Hk / tauk
            w = jnp.eye(dim, dtype=rhok.dtype) - rhok * sy_k
            H_kp1 = (_einsum('ij,jk,lk', w, Hk_scaled, w)
                    + rho_k * s_k[:, jnp.newaxis] * s_k[jnp.newaxis, :])

        elif update_method.lower() == "bfgs":
            sy_k = sk[:, jnp.newaxis] * yk[jnp.newaxis, :]
            w = jnp.eye(dim, dtype=rhok.dtype) - rhok * sy_k
            H_kp1 = (_einsum('ij,jk,lk', w, state.H_k, w)
                    + rho_k * s_k[:, jnp.newaxis] * s_k[jnp.newaxis, :])

        else:
            raise ValueError(f"Unknown update_method: {update_method}")

        # Safeguard against non-finite values
        H_kp1 = jnp.where(jnp.isfinite(rho_k), H_kp1, state.H_k)
        H_kp1 = jnp.where(jnp.isfinite(rho_k), H_kp1, state.H_k)
        
        converged = jnp.linalg.norm(g_kp1, ord=norm) < gtol

        return state._replace(
            converged=converged,
            nfev=state.nfev + ls_res.nfev,
            ngev=state.ngev + ls_res.ngev,
            failed=False,
            k=state.k + 1,
            x_k=x_kp1,
            f_k=f_kp1,
            g_k=g_kp1,
            H_k=H_kp1,
            old_old_fval=state.f_k,
            line_search_status=0,
        )
    
    # Decidir qué hacer basándose en si line search falló
    return lax.cond(
        line_search_results.failed,
        handle_ls_failure,
        handle_ls_success,
        line_search_results
    )

  state = lax.while_loop(cond_fun, body_fun, state)
  
  status = jnp.where(
      state.converged,
      0,  # converged
      jnp.where(
          state.k == maxiter,
          1,  # max iters reached
          jnp.where(
              state.failed,
              2 + state.line_search_status, # ls failed (+ reason)
              -1,  # undefined
          )
      )
  )
  state = state._replace(status=status)
  return state


# =============================================================================
# WRAPPER CONVENIENTE
# =============================================================================

def bfgs_optimize(
    fun: Callable,
    x0: jax.Array,
    maxiter: int = 200,
    gtol: float = 1e-5,
    update_method: str = "bfgs",
    c1: float = 1e-4,
    alpha0: float = 1.0,
    verbose: bool = False
) -> _BFGSResults:
    """
    Interfaz conveniente para BFGS.
    
    Args:
        fun: Función objetivo
        x0: Punto inicial
        maxiter: Máximo de iteraciones
        gtol: Tolerancia del gradiente
        update_method: 'bfgs', 'ssbfgs', o 'ssbroyden2'
        c1: Parámetro de Armijo
        alpha0: Tamaño de paso inicial
        verbose: Imprimir información
    """
    
    n = x0.shape[0]
    initial_H = jnp.eye(n, dtype=x0.dtype)
    
    if verbose:
        f_init = fun(x0)
        grad_init = jax.grad(fun)(x0)
        gnorm_init = jnp.linalg.norm(grad_init)
        
        print(f"Optimización BFGS iniciada")
        print(f"Método: {update_method}")
        print(f"Punto inicial: {x0}")
        print(f"f(x0): {f_init:.6e}")
        print(f"||grad(x0)||: {gnorm_init:.6e}")
        print("-" * 60)
    
    result = minimize_bfgs(
        fun, x0, initial_H, 
        maxiter=maxiter,
        gtol=gtol,
        update_method=update_method,
        c1=c1,
        alpha0=alpha0
    )
    
    if verbose:
        print("-" * 60)
        status_msgs = {
            0: "Convergió (||grad|| < gtol)",
            1: "MaxIter alcanzado",
            2: "Line search falló (alpha < 1e-18)",
        }
        msg = status_msgs.get(int(result.status), f"Status {int(result.status)}")
        
        print(f"Estado final: {msg}")
        print(f"Iteraciones: {int(result.k)}")
        print(f"Evaluaciones función: {int(result.nfev)}")
        print(f"Evaluaciones gradiente: {int(result.ngev)}")
        print(f"f(x_final): {float(result.f_k):.6e}")
        print(f"||grad(x_final)||: {float(jnp.linalg.norm(result.g_k)):.6e}")
        print(f"x_final: {result.x_k}")
    
    return result


# =============================================================================
# PRUEBAS
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PRUEBAS DE BFGS CON DETENCIÓN EN LINE SEARCH FAILURE")
    print("=" * 80)
    
    # Test 1: Función cuadrática (debería funcionar)
    print("\n" + "=" * 80)
    print("TEST 1: Función Cuadrática (debería converger)")
    print("=" * 80)
    
    @jax.jit
    def f_quad(x):
        return jnp.sum(x**2)
    
    x0 = jnp.array([1.0, 2.0, 3.0])
    
    result1 = bfgs_optimize(f_quad, x0, maxiter=50, verbose=True)
    
    # Test 2: Función difícil (puede fallar line search)
    print("\n" + "=" * 80)
    print("TEST 2: Función Difícil (puede fallar line search)")
    print("=" * 80)
    
    @jax.jit
    def f_difficult(x):
        # Función con valley muy estrecho
        return jnp.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    x0_hard = jnp.array([10.0, 10.0])  # Punto inicial lejos del óptimo
    
    result2 = bfgs_optimize(f_difficult, x0_hard, maxiter=100, verbose=True)
    
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print(f"""
✓ Si line search falla (alpha < 1e-18):
  - BFGS se detiene INMEDIATAMENTE
  - No se realiza actualización BFGS
  - failed=True, status=2
  - Retorna el último punto válido

✓ Esto previene:
  - Actualizaciones con pasos infinitesimales
  - Corrupción de la matriz Hessiana
  - Propagación de errores numéricos
    """)