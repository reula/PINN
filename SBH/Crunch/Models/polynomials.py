"""Chebyshev polynomial basis functions (used as activation/feature functions)."""
import jax.numpy as jnp

def T0(x): return jnp.ones_like(x)
def T1(x): return x
def T2(x): return 2*x**2 - 1
def T3(x): return 4*x**3 - 3*x
def T4(x): return 8*x**4 - 8*x**2 + 1
def T5(x): return 16*x**5 - 20*x**3 + 5*x
def T6(x): return 32*x**6 - 48*x**4 + 18*x**2 - 1
def T7(x): return 64*x**7 - 112*x**5 + 56*x**3 - 7*x
def T8(x): return 128*x**8 - 256*x**6 + 160*x**4 - 32*x**2 + 1
def T9(x): return 256*x**9 - 576*x**7 + 432*x**5 - 120*x**3 + 9*x
