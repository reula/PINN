# DSGNAR — Implementation Specification

**Source paper:** *An Optimisation Framework for the Well-Conditioned Training of Physics-Informed
Neural Networks*, Joseph Webb, Sadok Jerad, Coralia Cartis (Mathematical Institute, University of
Oxford), arXiv:2607.02194v1.
PDF used: `/Users/reula/Julia/PINN/Evolution_try/2607.02194v1.pdf` (49 pages).

**Reference implementation:** <https://github.com/wephy/physics-informed-neural-networks>
Commit inspected: tree `8b30ec3961de76ed414dd0f918646dd6358d8fcf` (branch `main`).
Files used: `src/pinn/optimiser.py`, `src/pinn/config.py`, `src/pinn/trainer64.py`,
`src/pinn/problem.py`, `README.md`, `examples/burgers_double_precision.ipynb`, `burgers_demo.ipynb`.

**Page-number convention.** All page citations `[p. N]` refer to the **PDF page index** of the
arxiv file (what `gs -dFirstPage=N` renders), *not* the printed page number. The offset is
`printed = PDF − 3` for the main text (PDF p. 8 = printed p. 5); appendices continue the same
offset (PDF p. 31 = printed p. 28). Equation numbers are the paper's own and are stable.

**One-line summary.** DSGNAR builds a *square* `s × s` doubly-sketched Jacobian `J̃ = C J Ω S`
(CountSketch `C` on the residual/row dimension, SRCT embedding `Ω S` on the parameter/column
dimension), takes **one** SVD of `J̃`, and uses that single factorisation to construct LM steps for
**many candidate trust-region radii at once**; it then picks the radius whose *measured* decrease
ratio is closest to (just above) a target ratio `ϱ*`, lifting the step back to full parameter space
only for evaluation and application. `ϱ*` starts small (conditioning stage) and is raised to 0.5
once `λ` has bottomed out (descent stage).

---

## 0. Notation table

| Symbol | Meaning | Shape |
|---|---|---|
| `θ` | flattened trainable parameters | `ℝ^{d_θ}` |
| `m` / `M` | condition index / number of conditions (`pde`, `ic`, `bc`, `ic_t`, …) | scalar |
| `𝒳_m` | collocation points for condition `m` | `|𝒳_m| × n_in` |
| `N = Σ_m |𝒳_m|` | total number of residual rows | scalar |
| `r_m`, `r` | condition-`m` / full residual vector | `|𝒳_m|`, `ℝ^N` |
| `J_m`, `J` | condition-`m` / full Jacobian | `|𝒳_m| × d_θ`, `ℝ^{N×d_θ}` |
| `s` | **sketch rank** (square sketch dimension) | scalar |
| `K` | number of CountSketch hash functions | scalar |
| `Ω, S, C` | SRCT orthonormal transform, column restriction, CountSketch | see §3 |
| `J̃` | doubly-sketched Jacobian `C J Ω S` | `s × s` |
| `r̃` | sketched residual `C r` | `ℝ^s` |
| `U, Σ (σ_i), V` | SVD of `J̃` | `U, V ∈ ℝ^{s×s}`, `Σ = diag(σ_i)` |
| `g` | `Σ Uᵀ r̃` (the sketched gradient `J̃ᵀ r̃`, expressed in the `V` basis) | `ℝ^s` |
| `λ` | Levenberg–Marquardt / Tikhonov regulariser | scalar |
| `Δ` | trust-region radius | scalar |
| `p` | step **in sketch space** | `ℝ^s` |
| `w` / `p_k` | step **lifted to full parameter space** `Ω S p` | `ℝ^{d_θ}` |
| `ϱ` | decrease ratio (actual / predicted) | scalar |
| `ϱ*` (`target_rho`) | target decrease ratio `ϱ^Stage1` or `ϱ^Stage2` | scalar |
| `q` | number of trust-region radius probes | scalar |
| `W` | λ-history window length | scalar |

---

## 1. Problem setup (PINN residual / least-squares)

### 1.1 Initial–boundary value problem `[pp. 3–5]`

For a space-time domain `Ω × [0,T]` and a system of `L` equations, an IBVP is the triplet
`(𝒫, ℐ, ℬ)` acting on a solution `u`:

```
𝒫u = 0,   (x,t) ∈ Ω × [0,T]        (Eq. 1)
ℐu = 0,   (x,t) ∈ Ω × {0}          (Eq. 2)
ℬu = 0,   (x,t) ∈ ∂Ω × [0,T]       (Eq. 3)
```

with `u : Ω × [0,T] → ℝ^L`. The composite operator is `𝓕u = (𝒫u; ℐu; ℬu)` (Eq. 4), and the
solution satisfies `𝓕u = 0`, i.e. `u ∈ ker(𝒫) ∩ ker(ℐ) ∩ ker(ℬ)`. A PINN approximates `u` with a
network `u_θ` and solves `𝓕u_θ(x,t) = 0`, where `θ ∈ ℝ^{d_θ}` collects **all** trainable parameters
flattened into a single vector. `(x,t)` and the architecture are held fixed; only `θ` is updated.

### 1.2 Training objective

The paper writes the objective in two forms. The unweighted form `[PDF p. 4]`:

```
ℒ(θ) := ½ ‖𝓕u_θ‖²_{𝒳,w}                                    (PINN objective)
```

(the `½` is the conventional least-squares scaling; see footnote on PDF p. 4), and the explicit
weighted form `[PDF p. 5, Eq. 5]`:

```
‖𝓕u_θ‖²_{𝒳,w} := Σ_{m=1}^{M}  (w_m / |𝒳_m|) · Σ_{(x,t) ∈ 𝒳_m} [𝓕_m u_θ(x,t)]²      (Eq. 5)
```

Key points:
* `M` conditions, each with weight `w_m` and point set `𝒳_m`; `N = Σ_m |𝒳_m|` total points.
* A second-order-in-time PDE needs two initial conditions (`ℐ₁`, `ℐ₂`, e.g. Wave) ⇒ more `m`.
  Conditions may also be **dropped** when analytically enforced by the architecture ("hard
  constraints"): for incompressible Navier–Stokes only pressure *gradients* appear, so the
  pressure-pinning condition is dropped, and periodic boundaries can be imposed by a periodic
  input embedding `[PDF p. 4, Example 1; PDF p. 34, Example 2]`.
* **Weighted form used by the optimiser `[PDF p. 16, Eq. 22]`** (`ℒ_k(θ)` with iteration-`k` weights):

```
ℒ_k(θ) := ½ ‖𝓕u_θ‖²_{𝒳,w_k} = ½ Σ_{m=1}^{M} (w_k)_m / |𝒳_m| · Σ_{(x,t) ∈ 𝒳_m} [𝓕_m u_θ(x,t)]²   (Eq. 22)
```

### 1.3 Residual and Jacobian blocks `[PDF p. 5, Eqs. 6–10]`

Least-squares form (Eq. 6):

```
ℒ(θ) = ½ Σ_{j=1}^{N} r_j²(θ)                                                  (Eq. 6)
```

Residual vector and Jacobian (Eq. 7), `r : ℝ^{d_θ} → ℝ^N`, `J ∈ ℝ^{N × d_θ}`:

```
r(θ) = [r₁(θ); r₂(θ); …; r_N(θ)],   J(θ) = [ ∇r₁(θ)ᵀ ; ∇r₂(θ)ᵀ ; … ; ∇r_N(θ)ᵀ ]     (Eq. 7)
```

Gradient and Hessian (Eq. 8):

```
∇ℒ(θ) = J(θ)ᵀ r(θ),      ∇²ℒ(θ) = J(θ)ᵀ J(θ) + Σ_{j=1}^{N} r_j(θ) ∇²r_j(θ)      (Eq. 8)
```

The **Gauss–Newton** model discards the second-order term `Σ r_j ∇²r_j` (positive semi-definite
for large residuals, and the expensive part), using `JᵀJ`.

**Weighted normalisation is folded into the residual/Jacobian rows** so that the plain
unweighted `½‖·‖²` of the scaled blocks equals Eq. 5. Condition-`m` scaling factor:

```
α_m = √( w_m / |𝒳_m| )                                                          (Alg. 2 line 3, p. 12)
```

Block Jacobian entries (Eq. 9) and block residuals (Eq. 10), for the `i`-th collocation point of
condition `m`:

```
(J_m)_{i,·} = α_m ∇_θ [𝓕_m u_θ(x_i, t_i)]     ∈ ℝ^{1 × d_θ}                     (Eq. 9)
(r_m)_i     = α_m 𝓕_m u_θ(x_i, t_i)            ∈ ℝ                            (Eq. 10)
```

Hence `J = [J_1; J_2; …; J_M]` and `r = [r_1; r_2; …; r_M]`, and

```
½‖r‖² = ½ Σ_m Σ_i α_m² [𝓕_m u_θ(x_i,t_i)]² = ½ Σ_m (w_m/|𝒳_m|) ‖𝓕_m u_θ‖² = ℒ_k(θ)
```

which is exactly Eq. 22. **This is the single most important implementation detail**: never
assemble a weighted loss separately; scale each row by `√(w_m/|𝒳_m|)` and everything downstream
(Eqs. 11–21) works with a plain sum of squares.

### 1.4 LM step, quadratic model, ratio `[PDF p. 6, Eqs. 11–14]`

Levenberg–Marquardt update (Eq. 11):

```
p_k^{LM}(λ) = −[ J_kᵀ J_k + λ I ]⁻¹ J_kᵀ r_k                                  (Eq. 11)
```

where `r_k = r(θ_k)`, `J_k = J(θ_k)`. `λ ≥ 0` is the Tikhonov/LM regulariser.

Regularised quadratic model (Eq. 12):

```
m_k^{LM}(p; λ) := ½‖r_k + J_k p‖² + (λ/2)‖p‖²                                  (Eq. 12)
```

Trust-region subproblem (Eq. 13):

```
p_k^{TR}(Δ) = argmin_p  m_k^{Q}(p) := ½‖r_k + J_k p‖²,   subject to ‖p‖ ≤ Δ     (Eq. 13)
```

`λ` is the Lagrange multiplier of the constraint `‖p‖ ≤ Δ`; `λ(Δ − ‖p(λ)‖) = 0`, `λ ≥ 0`.

Decrease ratio (Eq. 14) — "ratio" throughout:

```
ϱ_k = [ ℒ(θ_k) − ℒ(θ_k + p_k) ] / [ ℒ(θ_k) − m(p_k) ]                           (Eq. 14)
```

`ϱ ≈ 1` ⇒ model is a good local proxy; `ϱ ≈ 0` ⇒ it predicted a much larger decrease than was
achieved. `ϱ < 0` ⇒ the step increased the objective.

---

## 2. The doubly-sketched Gauss–Newton model `[PDF pp. 9–13]`

### 2.1 Why, and what is sketched `[pp. 9]`

* Naive LM costs `O(N d_θ² + d_θ³)` (Eq. 11) and storing `JᵀJ` is infeasible.
* `J` is replaced by a **small square sketch** `J̃ ∈ ℝ^{s×s}` of dimension `s`, so **one** SVD gives
  LM steps for any `λ`, and the same factorisation is reused for many `λ` (and hence many `Δ`).
* Compression is applied on **both sides**: rows (`N → s`, CountSketch) and columns
  (`d_θ → s`, SRCT). The sketch is accumulated over successive row batches, so `J` is *never* held
  in memory — but it is built **explicitly** (as a dense `s × s` matrix), not matrix-free.
* A square sketch is chosen as "fastest possible SVD"; empirically taller sketches gave no accuracy
  benefit for extra cost.

**Sketch-size recommendation `[PDF p. 9 and p. 17]`:**

```
s ∈ [ ⌊d_θ/3⌋ , ⌊d_θ/2⌋ ]          (paper: "between a third and a half of the parameter count")
```

`s` should "comfortably exceed the numerical rank of `J`". Values much smaller risk discarding
curvature directions; values approaching `d_θ` forfeit the speed advantage. Table 1 `[PDF p. 6]`
reports actual choices: `d_θ = 11 285 → s = 4 000`; `d_θ = 15 265 → s = 5 000`;
`d_θ = 17 241 → s = 5 000`; `d_θ = 3 215 → s = 1 200`.

### 2.2 CountSketch `C` (row/left sketch) `[PDF p. 9, Eqs. 15–16; p. 10]`

Linearity of the sketch over batches (Eq. 15):

```
C J = Σ_{batches} C J_batch                                                   (Eq. 15)
```

`CountSketch` is defined by `K` independent hash functions `h_k : {1,…,N} → {1,…,s}` and sign
vectors `ϵ_k ∈ {±1}^N`, all drawn uniformly at random. The sketch matrix `C ∈ ℝ^{s×N}` is (Eq. 16):

```
C_{ij} = (1/√K) Σ_{k=1}^{K} (ϵ_k)_j · 1[ h_k(j) = i ]                          (Eq. 16)
```

Effect: each row `j` of `J` is multiplied by `(ϵ_k)_j` and output to row `h_k(j)`; this is repeated
`K` times per row and summed, with an overall `1/√K` scale, *to reduce variance*.

* `C` is **extremely sparse — exactly `K` non-zeros per column**; it belongs to the OSNAP family of
  sparse oblivious subspace embeddings. `K = 1` recovers the original CountSketch (which is only an
  isometry in expectation, `E[CᵀC] = I_N`, with substantial per-draw variance from hash
  collisions). **"`K = 2` or `K = 4` is typical."** The reference code default is `K = 4`.
* Cost of applying `C` to a `b × s` sub-batch Jacobian is `O(K·s·b)` per batch (vs. `O(s²·b)` for a
  dense embedding) — negligible.
* **PINN-specific bonus `[p. 10]`:** because rows are aggregated into `s` buckets, hard/troublesome
  regions of the PDE (large `𝓕_m u_θ`) naturally dominate their buckets, and the many residuals per
  bucket average out noise. This is why DSGNAR can sample `N ≫ s` aggressively and does not need
  residual-selection heuristics such as RBA.
* The left sketch is **never inverted**. "The residual dimension is compressed once and never
  reconstructed" `[p. 10]`. No quantity is ever lifted back to `ℝ^N`.

### 2.3 SRCT `Ω, S` (column/right sketch) `[PDF p. 11, Eq. 17]`

The step lives in parameter space, so the column sketch must be a **near-isometry** whose lift is
faithful. The paper uses a **subsampled randomised trigonometric transform** (SRTT), specifically
the real-valued **discrete cosine transform type-II**, `F ∈ ℝ^{d_θ × d_θ}`.

```
Ω := D Π F   ∈ ℝ^{d_θ × d_θ}         (orthogonal/unitary)
S : selects s columns of Ω
Ω S = (D Π F) S                      (Eq. 17)
```

with
* `D ∈ ℝ^{d_θ × d_θ}` — a diagonal of i.i.d. random signs (Rademacher `±1`),
* `Π ∈ ℝ^{d_θ × d_θ}` — a uniform random permutation matrix,
* `F` — a unitary trigonometric transform (DCT-II with `norm="ortho"` in the code),
* `S ∈ ℝ^{d_θ × s}` — a random column restriction (`s` distinct columns).

`Ω S` is applied to the **right** of the Jacobian block: `J_batch ← J_batch Ω S`, giving
`J_batch ∈ ℝ^{b × s}`. Cost `O(d_θ log d_θ)` thanks to the FFT/DCT, improved over Gaussian
embeddings (`O(d_θ s)`), and an SRTT preserves all singular values of a fixed `d`-dimensional
subspace to relative accuracy `ε` provided enough columns are retained.

### 2.4 The doubly-sketched Jacobian and its accumulation

**Definition.** With `C` from Eq. 16 and `Ω S` from Eq. 17:

```
J̃ := C J Ω S  ∈ ℝ^{s × s}          r̃ := C r  ∈ ℝ^s                        (Alg. 1 line 3)
```

**Accumulation over conditions and batches (Algorithm 2, `[PDF p. 12]`).** Because of Eq. 15 and
linearity of `Ω S` on the right, the sketch is accumulated additively:

```
J̃ ← 0^{s×s};   r̃ ← 0^s;   r ← []
for m = 1,…,M:                                        # loop over conditions
    α_m ← √(w_m / |𝒳_m|)                              # condition scaling factor
    for i = 1,…,|𝒳_m|/b:                              # macro-batches of size b; J_m never in memory
        J_batch ← [];  r_batch ← []
        for j = 1,…,b/b′:                             # sub-batches of size b′; parallelisable
            r_j ← 𝓕_m(u_θ, 𝒳_{batch,j})                # r_j ∈ ℝ^{b′}
            J_j ← ∂_θ r_j                             # reverse-mode AD, J_j ∈ ℝ^{b′×d_θ}
            J_batch ← J_batch ⊕ J_j;  r_batch ← r_batch ⊕ r_j
        J_batch ← J_batch Ω S                         # column sketch, ℝ^{b×s}
        J̃ ← J̃ + α_m C J_batch                        # row sketch + accumulate
        r̃ ← r̃ + α_m C r_batch
        r ← r ⊕ α_m r_batch                           # FULL residual, uncompressed
return J̃, r̃, r
```

Notes:
* `b` is the **macro-batch** (batch) size, `b′` the **sub-batch** (micro-batch) size, with
  `b′ ≪ d_θ` so reverse-mode AD is far cheaper than forward-mode (Table 2, `[PDF p. 10]`).
  Sub-batches are independent and dispatched in parallel.
* The **`α_m` scaling is applied at accumulation**, so the routine "is agnostic to the current
  weights `w`" and needs no changes between iterations `[p. 12]`.
* The **full residual `r` is assembled alongside, uncompressed**, because the decrease ratio is
  evaluated in the full space.
* **GetSketchOperators (Algorithm 1 line 2) vs. `InitSketch`.** Algorithm 1 lists
  `C, Ω, S ← GetSketchOperators(d_θ, s)` inside the loop (line 2), i.e. **resampled every
  iteration**; but §3.2.2 `[p. 11]` says "the initialisation of Ω and S constitutes `InitSketch` in
  Algorithm 1, meaning a single, consistent sketch is used throughout every iteration … primarily
  due to compiler constraints". **These contradict each other.** The reference code resamples:
  `_make_srct` and the hash/sign draws take fresh keys inside every `_step_impl` call. **Implement
  per-iteration resampling** (it is the unbiased choice and matches the code); keeping the sketch
  fixed is a valid, cheaper variant that matches the paper's prose.

---

## 3. Solving the subproblem with one SVD `[PDF pp. 12–13, Eqs. 18–21]`

### 3.1 The sketched subproblem

Sketched regularised model (Eq. 18):

```
m̃_k^{LM}(p̃; λ) := ½‖r̃_k + J̃_k p̃‖² + (λ/2)‖p̃‖²,   p̃ ∈ ℝ^s,  J̃_k ∈ ℝ^{s×s},  s ≪ N   (Eq. 18)
```

Sketched trust-region subproblem (Eq. 19):

```
min_{p̃}  m̃_k^{Q}(p̃) := ½‖r̃_k + J̃_k p̃‖²   subject to  ‖p̃‖ ≤ Δ                    (Eq. 19)
```

### 3.2 The ratio actually used (Ratio ϱ) `[PDF p. 13]`

Because the true model `m_k^Q` is never built, Eq. 14 cannot be evaluated exactly. DSGNAR uses:

```
ϱ_k(p̃_k, p_k; θ_k) = [ ℒ(θ_k) − ℒ(θ_k + p_k) ] / [ m̃_k^Q(0) − m̃_k^Q(p̃_k) ]      (Ratio ϱ)
```

* **Numerator:** the *true* objective reduction in full parameter space, evaluated with the
  **lifted** step `p_k = Ω S p̃_k`.
* **Denominator:** the predicted decrease of the **sketched** model, evaluated with the **sketch
  space** step `p̃_k`. Equivalently, `m̃(0) = ½‖r̃‖²`.

### 3.3 Closed-form LM step from the SVD `[PDF p. 13, Eq. 20]`

Take the SVD of the sketched Jacobian:

```
J̃_k = U_k Σ_k V_kᵀ,     U_k, V_k ∈ ℝ^{s×s} orthogonal,   Σ_k = diag(σ_k)
```

Then the sketched LM step has the closed form (Eq. 20):

```
p̃_k^{LM}(λ) = − V_k diag( (σ_k)_i / ((σ_k)_i² + λ) ) U_kᵀ r̃_k                  (Eq. 20)
```

where `(σ_k)_i` is the `i`-th singular value from `Σ_k`. Notes:
* This is numerically preferable to solving Eq. 11 directly: it never forms `JᵀJ`, which would
  square the condition number.
* Using the SVD, the cost of evaluating a step **for an array of `λ` values** is only `O(s²)` in
  matrix–vector products — "steps for an array of `λ` choices can be computed quickly in parallel".
* With a trust-region radius `Δ`, the `λ` such that `‖p̃_k^{LM}(λ)‖ = Δ` comes from the **secular
  equation** (Eq. 21):

```
φ(λ) := sqrt( Σ_{i=1}^{s} (σ_k)_i² (U_kᵀ r̃_k)_i² / ((σ_k)_i² + λ)² ) − Δ = 0        (Eq. 21)
```

  solved by Newton's method (see Algorithm 3 / §5 below). Note the identity (used below)

```
‖p̃(λ)‖² = Σ_i (σ_i (Uᵀr̃)_i)² / (σ_i² + λ)² = Σ_i g_i² / (σ_i² + λ)² ,   g_i := σ_i (Uᵀr̃)_i
```

**Crucial implementation note (this is what the code does, and it is NOT Eq. 21 verbatim).** The
reference code never forms the norm `‖p̃(λ)‖` directly; it computes

```
g = Σ Uᵀ r̃            # g_i = σ_i (Uᵀ r̃)_i
q(λ) = g / (σ² + λ)    # coefficient vector in the V basis
‖p̃(λ)‖ = ‖q(λ)‖₂
```

Since `V` has orthonormal columns, `‖p̃(λ)‖ = ‖diag(σ/(σ²+λ)) Uᵀr̃‖ = ‖g/(σ²+λ)‖`, so the two are
identical. Use `g` — it avoids re-deriving the `σ_i²` factor and is what makes Algorithm 3's line 6
consistent.

### 3.4 Predicted decrease (needed for `ϱ`)

The sketched model decrease at step `p̃(λ) = −V (g/(σ²+λ))` is a *positive* quantity. **Two
different expressions appear in the paper and the code, and they differ by exactly a factor of 2.**

**(a) The plain LM / trust-region model reduction.** With `m̃(p) = ½‖r̃ + J̃p‖²`, substituting
`p̃(λ)` and using `J̃ = UΣVᵀ`, `g = ΣUᵀr̃`:

```
m̃(0) − m̃(p̃(λ)) = ½ Σ_i g_i² (σ_i² + 2λ) / (σ_i² + λ)²
```

**(b) The paper's Algorithm 3, line 6 and Algorithm 1, line 6.** Algorithm 1's step-6 box reads

```
p̃_k  ←  − V_k diag( (σ_k)_i / ((σ_k)_i² + λ_k) ) U_kᵀ r̃_k            (negative convention)
```

while Algorithm 3's line 10 and Eq. 20 read

```
p̃_i  ←  Σ_j g_j / (σ_j² + λ_i) · v_j                                  (positive convention)
```

and Algorithm 1 line 6's box annotates the denominator as

```
pred = Σ_j (σ_j g_j)² / (σ_j² + λ_k)²
```

With the **negative** step convention this is `+Σ g²σ²/(σ²+λ)²`, which is **too large** (it is the
magnitude of the gradient term) and becomes negative once `λ` grows. With the **positive** step
convention (Algorithm 3) the model reduction is `Σ g²σ²/(σ²+λ)² − 2Σg²λ/(σ²+λ)²`, which is also
wrong at large `λ`. **The paper's `pred` is therefore not exactly `m̃(0) − m̃(p̃)`; it is
inconsistent between the two algorithm boxes.**

**(c) The reference code.** `_step_and_pred` in `src/pinn/optimiser.py` (lines 500–504) computes

```python
step_sk = -(Vt.T @ (g / denom))                                  # negative convention
pred    = jnp.sum(g**2 * (S**2 + 2*lam) / denom**2)              # <-- denom = S**2 + lam
```

i.e.

```
pred_code(λ) = Σ_i g_i² (σ_i² + 2λ) / (σ_i² + λ)²
```

I verified numerically (see §9, check C) that for a **square** sketched Jacobian with residual
inside its row space:

```
m̃(0) − m̃(p̃(λ))  =  ½ · pred_code(λ)          exactly, for every λ
```

**Recommendation for the JAX port.**
* To **reproduce the published numbers**, use the code's expression verbatim,
  `pred = Σ g²(σ²+2λ)/(σ²+λ)²`, with `ϱ = act_red / (pred + tiny)` and accept when `ϱ > 0`
  (the acceptance test is scale-invariant, so the factor 2 does not break it) and **keep the paper's
  literal targets** `ϱ^Stage1 = 0.075` (double) / `0.1` (single), `ϱ^Stage2 = 0.5`. Be aware that
  with this expression the reported `ϱ` is *half* the textbook Eq. 14 ratio when `λ` is
  moderate/zero, so `ϱ* = 0.075` means "textbook ratio ≈ 0.15".
* If you instead want the theoretically clean Eq. 14 ratio, use
  `pred = 0.5 * Σ g²(σ²+2λ)/(σ²+λ)²` and set `ϱ* = 0.15` / `0.5` for Stage 1 / 2 at double
  precision. Do **not** mix the two.

State this choice explicitly in any reproduction write-up.

---

## 4. Algorithm 1 — DSGNAR `[PDF p. 8]`

Transcribed faithfully. `k` is the iteration counter.

```
Algorithm 1 | DSGNAR: Doubly-Sketched Gauss–Newton with Adaptive Ratio

Input:   𝓕       — the M residual conditions of the problem
Input:   𝒳       — the M sets of collocation points for each condition
Input:   u_θ     — the neural network parameterised by θ
Input:   θ₀, Δ₀, w₀, ϱ₀ — initial parameters, trust-region radius, weights, and target ratio
Output:  θ*      — the fully trained network parameters

        ▷ Hyperparameters: sketch rank s, convergence tolerance Δ_min

1:  while k = 0, 1, … do
2:      C, Ω, S ← GetSketchOperators(d_θ, s)      ▷ C ∈ ℝ^{s×N}, Ω ∈ ℝ^{d_θ×d_θ}, S ∈ ℝ^{d_θ×s}
3:      J̃_k, r̃_k, r_k ← Sketch(𝓕, 𝒳, C, Ω, S, u_θ_k, θ_k, w_k)
                                                 ▷ J̃_k ∈ ℝ^{s×s}, r̃_k ∈ ℝ^s, r_k ∈ ℝ^N
4:      U_k, Σ_k, V_kᵀ ← SVD(J̃_k)
5:      Δ*, λ_k ← LambdaSolve(Ω, S, U_k, Σ_k, V_kᵀ, r̃_k, r_k, u_θ_k, θ_k, Δ_k, ϱ_k)
                                                 ▷ Δ* ∈ [⅓ Δ_k, 3 Δ_k]
6:      p̃_k ← −V_k diag( (σ_k)_i / ((σ_k)_i² + λ_k) ) U_kᵀ r̃_k
                                                 ▷ Step in sketch space
7:      p_k ← Ω S p̃_k                            ▷ Lift to full parameter space
8:      if ℒ_k(θ_k + p_k) < ℒ_k(θ_k) then
9:          θ_{k+1} ← θ_k + p_k,  Δ_{k+1} ← Δ*   ▷ Accept step and trust-region radius
10:     else
11:         θ_{k+1} ← θ_k,  Δ_{k+1} ← ⅓ Δ_k      ▷ Reject step; shrink trust-region radius
12:     if Δ_{k+1} < Δ_min then
13:         return θ_{k+1}                        ▷ Convergence criterion met; terminate
14:     w_{k+1} ← UpdateWeights(r_k, w_k)
15:     ϱ_{k+1} ← UpdateTargetRatio({λ_i}ᵢ, ϱ_k)
```

Reconciliation with the reference code (`optimiser.py::_step_impl`), which refines the box:

| Alg. 1 line | Paper | Code | Code semantics |
|---|---|---|---|
| 2 | resample `C, Ω, S` | fresh `_make_srct` key + fresh hash keys per step | resampled per iteration |
| 5 | `LambdaSolve` returns `Δ*, λ*` | probe window + secular solve + PCHIP | §5 |
| 8 | `ℒ_k(θ+p) < ℒ_k(θ)` | `accepted = (ρ > 0) & not isnan(ρ)` | equivalent (strict decrease) |
| 9 | `Δ_{k+1} ← Δ*` | `next_radius = final_rad` | same |
| 11 | `Δ_{k+1} ← ⅓ Δ_k` | `rho > ϱ*+0.1 → r_hi = 3Δ_k`, else `→ r_lo = Δ_k/3` | refined: *expand* if too successful, shrink otherwise |
| 12 | `Δ < Δ_min` | `opt_state.radius < 10 * min_radius` | code uses a 10× margin (trainer64.py:679) |
| 14 | `UpdateWeights(r_k, w_k)` | trainer-level, Algorithm 4 | §6.1 |
| 15 | `UpdateTargetRatio({λ_i}, ϱ_k)` | trainer-level, Algorithm 5 | §6.2 |

---

## 5. Algorithm 2 — Sketch `[PDF p. 12]`

```
Algorithm 2 | Sketch                     Computes sketched Jacobian and residuals

Input:   𝓕       — the M residual conditions
Input:   𝒳       — the M sets of collocation points
Input:   C, (Ω, S) — CountSketch and SRCT operators
Input:   u_θ, θ  — the neural network and current parameters
Input:   w       — condition weights
Output:  J̃       — doubly-sketched Jacobian, ℝ^{s×s}
Output:  r̃, r    — sketched and full residuals

        ▷ Hyperparameters: sketch rank s, batch size b, sub-batch size b′

1:  J̃ ← 0^{s×s},  r̃ ← 0^s,  r ← []
2:  for m = 1, …, M do                              ▷ Loop over M conditions
3:      α_m ← √( w_m / |𝒳_m| )                      ▷ Condition scaling factor
4:      for i = 1, …, |𝒳_m|/b do                   ▷ Loop over batches of size b
5:          J_batch ← [],  r_batch ← []
6:          for j = 1, …, b/b′ do                  ▷ Loop over sub-batches of size b′
7:              r_j ← 𝓕_m(u_θ, 𝒳_{batch,j})         ▷ Evaluate residuals, r_j ∈ ℝ^{b′}
8:              J_j ← ∂_θ r_j                       ▷ Reverse-mode Jacobian, ℝ^{b′×d_θ}
9:              J_batch ← J_batch ⊕ J_j,  r_batch ← r_batch ⊕ r_j
10:         J_batch ← J_batch Ω S                   ▷ Sketch columns, J_batch ∈ ℝ^{b×s}
11:         J̃ ← J̃ + α_m C J_batch                  ▷ Accumulate via CountSketch
12:         r̃ ← r̃ + α_m C r_batch
13:         r ← r ⊕ α_m r_batch                     ▷ Append batch residuals
14: return J̃, r̃, r
```

**Reference-code realisation of `C`** (`optimiser.py::_count_sketch`, lines 98–123). For each of
`K` independent hashes:

```python
scale   = 1.0 / jnp.sqrt(n_hashes)                       # 1/√K
signs   = jax.random.rademacher(k_s, (J.shape[0],))      # ϵ_k ∈ {±1}
buckets = jax.random.randint(k_b, (J.shape[0],), 0, residual_sketch)   # h_k
signed  = signs * scale
B_acc  += jax.ops.segment_sum(J * signed[:, None], buckets, residual_sketch)
r_acc  += jax.ops.segment_sum(r * signed,          buckets, residual_sketch)
```

This is exactly Eq. 16: `segment_sum` is the bucketed sum `Σ_j ... 1[h_k(j)=i]`, the outer `1/√K`
is `scale`, and the `K` accumulations are summed. Verified equivalent.

**Reference-code realisation of `Ω S`** (`optimiser.py::_make_srct` / `_apply_srct`, lines 57–91):

```python
# once per step (per iteration), with a fresh PRNG key:
signs   = random.choice(±1, (n_params,))                 # D
perm    = random.permutation(n_params)                   # Π
indices = random.choice(n_params, (parameter_sketch,), replace=False)   # S

# apply, J: (n_rows, d_θ) -> (n_rows, s)
J = J * signs[None, :]                                   # D  (right-multiply by diagonal)
J = J[:, perm]                                           # Π
J = jax.scipy.fft.dct(J, type=2, norm="ortho", axis=1)   # F  (DCT-II, orthonormal)
return J[:, indices]                                     # S

# adjoint lift, y in ℝ^s -> ℝ^{d_θ}
v = jnp.zeros(n_params).at[indices].set(y)               # Sᵀ  (scatter)
v = jax.scipy.fft.idct(v, type=2, norm="ortho")          # Fᵀ
v_full = jnp.zeros_like(v).at[perm].set(v)               # Πᵀ
return v_full * signs                                    # Dᵀ = D
```

Notes:
* The paper's `D Π F` is implemented as `D` applied **first** (`J * signs`), then `Π` (`J[:, perm]`),
  then `F` (`dct`). These commute in the sense that `(D Π F)ᵀ = Fᵀ Πᵀ Dᵀ` and the code's lift
  (`Sᵀ → Fᵀ → Πᵀ → Dᵀ`) is exactly the adjoint. ✔
* `norm="ortho"` is essential: it makes the DCT an orthogonal matrix, so the lift is the true
  adjoint and `‖p̃‖ = ‖Ω S p̃‖`. A non-orthonormal DCT silently breaks the trust-region norm and the
  secular equation.
* The permutation here is a *full* permutation of `d_θ` indices followed by a subsample, which is
  equivalent to sampling `s` distinct columns of `D Π F`.

---

## 6. Algorithm 3 — LambdaSolve `[PDF p. 14]`

```
Algorithm 3 | LambdaSolve                Finds λ corresponding to target ratio ϱ*

Input:   Ω, S      — SRCT operators
Input:   U, Σ, Vᵀ  — SVD of the sketched Jacobian
Input:   r̃ ∈ ℝ^s, r ∈ ℝ^N — sketched and full residuals
Input:   u_θ, θ    — the neural network and current parameters
Input:   Δ_k, ϱ*   — current trust-region radius and target ratio
Output:  Δ*, λ*    — selected trust-region radius and regularisation

        ▷ Hyperparameters: number of probes q, Newton iterations N_Newton

1:  g ← Σ Uᵀ r̃                                                  ▷ g ∈ ℝ^s
2:  δ_i ← (Δ_k/3)^{1 − (i−1)/(q−1)} · (3Δ_k)^{(i−1)/(q−1)},  i = 1, …, q
                                                    ▷ Geometrically spaced TR-radius probes
3:  for i = 1, …, q do                            ▷ Loop over probes, parallelise independently
4:      λ ← 0
5:      for n = 0, 1, …, N_Newton do              ▷ Newton steps solving ‖p̃^{LM}(λ)‖₂ = δ_i
6:          φ  ←  sqrt( Σ_{j=1}^{s} ( g_j / (σ_j² + λ) )² ) − δ_i
7:          φ′ ← − 1/(φ + δ_i) · Σ_{j=1}^{s} g_j² / (σ_j² + λ)³
8:          λ  ← max(0, λ − φ/φ′)
9:      λ_i ← λ                                    ▷ Regularisation parameter for probe δ_i
10:     p̃_i ← Σ_{j=1}^{s} ( g_j / (σ_j² + λ_i) ) v_j    ▷ Step in sketch space, p̃_i ∈ ℝ^s
11:     p_i ← Ω S p̃_i                              ▷ Lift to full parameter space
12:     ϱ_i ← ϱ(p̃_i, p_i; θ)                       ▷ Decrease ratio for probe δ_i, as in (Ratio ϱ)
13: δ̂_i ← min_{1≤j≤i} max(0, ϱ_j),  i = 1, …, q    ▷ Enforce ϱ̂(δ) non-increasing with δ
14: if ϱ̂_q ≥ ϱ* then
15:     return 3Δ_k, λ_q                            ▷ All probes acceptable; take largest radius
16: else if ϱ̂_1 ≤ ϱ* then
17:     return ⅓Δ_k, λ_1                            ▷ No probe acceptable; take smallest radius
18: else
19:     𝓢(ϱ) ← PCHIP( {δ̂_i, log δ_i}_i )           ▷ Interpolate ratio as function of decrease ratio
20:     Δ* ← 𝓢(ϱ*)                                  ▷ Radius achieving target ratio ϱ*
21:     λ* ← λ(Δ*)                                  ▷ Corresponding regularisation parameter
22:     return Δ*, λ*
```

**Reference-code realisation** (`optimiser.py`):

* **Probe grid (line 2).** Code (`lines 507–510`):

  ```python
  cur_r = clip(state.radius, min_radius, max_radius)
  r_lo  = max(cur_r / window_scale, min_radius)
  r_hi  = min(cur_r * window_scale, max_radius)
  probe_radii = jnp.geomspace(r_lo, r_hi, n_probes)
  ```

  With `window_scale = 3` this is exactly `[Δ_k/3, 3Δ_k]`, geometrically spaced, matching Alg. 3
  line 2 and §3.4's `Δ* ∈ [⅓Δ_k, 3Δ_k]`. The paper notes the range can be widened to a factor of 5
  in each direction `[PDF p. 15]`.

* **Secular Newton (lines 5–8).** Code (`_solve_subproblems`, lines 211–234):

  ```python
  S2  = jnp.square(S)
  lam = jnp.zeros_like(target_radii)
  for _ in range(n_iters):
      denom  = S2[None, :] + lam[:, None]
      p      = w[None, :] / denom                    # w is `g` at the call site
      p_norm = jnp.linalg.norm(p, axis=1)
      phi    = p_norm - target_radii
      d_phi  = (-jnp.sum(jnp.square(p) / (denom + tiny), axis=1)) / (p_norm + tiny)
      lam    = jnp.maximum(lam - phi / (d_phi + tiny), 0.0)
  ```

  It is called as `_solve_subproblems(S, g, probe_radii, newton_iters)` in both the probe loop
  (`line 513`) and the final solve (`line 551`), so `w ≡ g` and
  `p = g/(σ²+λ) = p̃(λ)` in the `V` basis. Then `‖p‖ = ‖p̃(λ)‖` as required by Eq. 21. Verified:
  the code's `d_phi` equals `d/dλ ‖g/(σ²+λ)‖` (check D in §9). **Note `g`, not `g²`, is passed** —
  passing `g²` would be a bug.

* **Predicted decrease (lines 500–504).** Code:

  ```python
  def _step_and_pred(lam_val):
      denom   = S**2 + lam_val
      step_sk = -(Vt.T @ (g / denom))
      pred    = jnp.sum(g**2 * (S**2 + 2*lam_val) / denom**2)
      return step_sk, pred
  ```

  See §3.4 for the factor-2 discussion.

* **Probe evaluation (lines 517–532).** All `q` lifted steps are evaluated **in full parameter
  space** for every condition, via `_eval_probes` + `_eval_all_conditions`:

  ```python
  probe_steps_sk, probe_preds = jax.vmap(_step_and_pred)(probe_lambdas)
  ps = lift_vmap(probe_steps_sk, srct, n_params)               # (q, d_θ)
  probe_losses = _eval_all_conditions(...)                     # weighted Σ_m w_m mean(r²)
  act_red    = current_loss - probe_losses
  probe_rhos = act_red / (probe_preds + finfo.tiny)
  ```

  `_eval_all_conditions` accumulates `total += w_i * mean(r²)` over conditions — i.e. exactly
  Eq. 22 again, but computed *independently* of the `α_m`-scaled `r` used for sketching.
  **Truncation gotcha:** `_eval_probes` uses
  `n_batches = points.shape[0] // batch_size` and then `points[:n_batches * batch_size]`, so any
  leftover collocation points (`|𝒳_m| mod batch_size`) are **silently dropped** from the probe
  losses. The final accepted-step loss uses `probe_batch_size`, and the per-condition losses used
  for re-weighting are computed over yet another (macro-`batch_size`) truncation — so three
  slightly different point subsets are compared. At the published sizes (`n_pde = 2^15`,
  `batch_size` a power of two) the remainder is zero, but for small/tiny problems choose
  `n_pde`, `n_ic`, `n_bc` to be multiples of `batch_size` (and of `probe_batch_size`) or the
  measured `ϱ` will be inconsistent with the sketched prediction.

* **Monotonisation (line 13).** Code (`line 535`):

  ```python
  safe_rhos = jnp.minimum.accumulate(jnp.clip(probe_rhos, -1.0, 1.0))
  ```

  This is `δ̂_i = min_{j≤i} max(−1, min(1, ϱ_j))`, a clipped, non-increasing envelope. The paper
  says "to handle noise, particularly once close to the solution, we force monotonicity in the
  fit; while not necessarily injective, this still permits straightforward interpolation"
  `[PDF p. 15]`.

* **PCHIP + crossing search (lines 19–21, and §3.4's Figure 3 discussion).** Code (`lines 281–312`,
  `_pchip`; `lines 536–548`):

  ```python
  log_radii  = jnp.log(probe_radii)
  fine_log_r = jnp.linspace(jnp.log(r_lo), jnp.log(r_hi), pchip_grid)
  fine_rho   = _pchip(log_radii, safe_rhos, fine_log_r)     # Fritsch–Carlson monotone cubic

  is_above  = fine_rho >= target_rho
  crossings = is_above[:-1] & ~is_above[1:]                 # above -> below transitions
  has_cross = jnp.any(crossings)
  last_idx  = jnp.max(jnp.where(crossings, jnp.arange(pchip_grid - 1), -1))
  opt_log_r = 0.5 * (fine_log_r[last_idx] + fine_log_r[last_idx + 1])
  fallback_r = jnp.where(jnp.all(is_above), r_hi, r_lo)
  final_rad  = jnp.where(has_cross, jnp.exp(opt_log_r), fallback_r)
  ```

  Semantics:
  - `probe_radii` is `geomspace`, so `log(probe_radii)` is *uniformly* spaced (verified) and the
    PCHIP nodes are well-conditioned.
  - It finds the **largest** radius whose fitted `ϱ` is still `≥ ϱ*` (the `min` of all crossing
    indices, i.e. the **last** crossing), and takes the midpoint in **log-radius** of that
    bracketing interval. This is the "largest step with ratio at least `ϱ*`" rule, chosen because
    §3.1's philosophy is to keep the effective regularisation as small as possible while still
    attaining the target.
  - **Fallbacks** (paper p. 15: "If the model does not cover `ϱ*`, rather than extrapolate, we
    select a new set of geometrically spaced probes in the direction of `ϱ*`… if `ϱ*` is not found
    within the probe range, we instead test the step associated with the nearest extreme probe"):
    all probes above `ϱ*` ⇒ `Δ* = r_hi = 3Δ_k`; none above ⇒ `Δ* = r_lo = Δ_k/3`. (The paper also
    says: "if the extreme probe also fails to yield a decrease, use `Δ*/3` as the centre for the
    next iteration's probes, covering a region below and overlapping the previous one.")
  - The paper explicitly recommends centring probes on `Δ_k` *rather than on `λ`* because `Δ` is
    far more stable across iterations than `λ` (Figure 3, `[PDF p. 15]`: for iterations 0–100 the
    trust-region radius stays consistently close to 1 while `λ` changes by >25 orders of magnitude).

* **Final step and acceptance (lines 550–600).**

  ```python
  f_lams = _solve_subproblems(S, g, jnp.atleast_1d(final_rad), newton_iters)
  f_lam  = f_lams[0]
  f_step_sk, f_pred = _step_and_pred(f_lam)
  f_step = _lift_update(f_step_sk, srct, n_params)          # p_k = Ω S p̃_k
  # full-space evaluation of the objective at θ + p_k for every condition
  f_act_red = current_loss - f_loss
  f_rho     = f_act_red / (f_pred + finfo.tiny)

  rho_hi_thresh = target_rho + 0.1
  rho_too_high  = ~jnp.isnan(f_rho) & (f_rho > rho_hi_thresh)
  rho_positive  = ~jnp.isnan(f_rho) & (f_rho > 0.0)
  accepted      = rho_positive

  new_params = unflatten(params_flat + delta)   # delta = f_step if accepted else 0
  next_radius = jnp.where(accepted, final_rad,
                  jnp.where(rho_too_high, r_hi, r_lo))
  ```

  So:
  - **Accept iff `ϱ > 0`** (i.e. iff the objective strictly decreased), *not* iff `ϱ ≥ ϱ*`.
  - **On accept:** carry the PCHIP-chosen `Δ*` forward.
  - **On reject with `ϱ > ϱ* + 0.1`:** the step overshot the target by a small margin ⇒ the radius
    was too small ⇒ set the next centre to `r_hi = 3Δ_k` ("an over-aggressive step signals that the
    trust region is too small, so the radius is expanded, making the subsequent step less damped").
  - **On reject otherwise (`ϱ ≤ 0`, or `ϱ ≤ ϱ* + 0.1` but non-positive):** set the next centre to
    `r_lo = Δ_k/3`.
  - `λ_k` stored in the state is the *accepted* `λ` (unchanged on rejection), and the state also
    carries `radius`, `rho`, and the PRNG key.

---

## 7. Algorithm 4 — UpdateWeights `[PDF p. 32]` (Appendix A.1)

```
Algorithm 4 | UpdateWeights                  Residual-based condition re-weighting

Input:   r_k    — full residual vector (in structured blocks per condition)
Input:   w_k    — current condition weights
Output:  w_{k+1} — updated condition weights

        ▷ Hyperparameters: re-weighting exponent α ∈ ℝ

1:  for m = 1, …, M do
2:      ℓ_m ← ‖(r_k)_m‖²                              ▷ Per-condition loss
3:  ℓ̄_k ← (1/M) Σ_{m=1}^{M} ℓ_m                       ▷ Mean loss across conditions
4:  for m = 1, …, M do
5:      ŵ_m ← (w_k)_m · ( ℓ̄_k / (ℓ_m + ε) )^α         ▷ Scale weight by relative loss
6:  w_{k+1} ← ŵ / ( (1/M) Σ_{m=1}^{M} ŵ_m )           ▷ Normalise: mean weight equals 1
7:  return w_{k+1}
```

* `α` controls how aggressively per-condition losses are equalised: `α = 0` recovers no
  re-weighting; larger `α` equalises more strongly. **`α = 0.05` throughout.**
* `ε = 10⁻⁸` ("a small floor ε = 10⁻⁸ guards this ratio against division by zero").
* The **renormalisation (line 6) is essential**: without it the overall magnitude of `w_k` drifts.
* `(r_k)_m` is the `α_m`-scaled block from Algorithm 2, so `ℓ_m = ‖(r_k)_m‖² = w_m · mean(𝓕_m²)`.
  Using the scaled blocks matches Eq. 22 exactly.

**Where the code differs from Algorithm 4 — important.** `trainer64.py` lines 628–637:

```python
if config.auto_adjust:
    weighted = [float(cl) * w for cl, w in zip(metrics["cond_losses"], weights)]
    mean_wl  = sum(weighted) / len(weighted)
    weights  = [w * (mean_wl / (wl + 1e-8)) ** config.alpha
                for w, wl in zip(weights, weighted)]
    mean_w  = sum(weights) / len(weights)
    weights = [w / mean_w for w in weights]
```

Here `metrics["cond_losses"]` are **unweighted per-condition mean squared residuals**
(`jnp.mean(r_cond**2)`, computed in `optimiser.py` lines 559–570 and 602–609), so
`weighted[m] = w_m · L_m` and the update is

```
w_{k+1,m} ∝ w_m · ( mean_m(w_m L_m) / (w_m L_m + ε) )^α
```

Substituting `w_m L_m` and simplifying, this is `∝ w_m^{1−α} · (mean_wL / L_m)^α`, i.e. the code
contracts `w_m` by an extra factor `w_m^{−α}` relative to Algorithm 4. It is a *sharper* relative
re-weighting than Algorithm 4's `w_m · (mean(L)/L_m)^α`. **Use Algorithm 4 as written if you want
the paper's rule; use the code's version if you want bit-level reproduction.** Both include the
`ε = 1e-8` floor and the `mean(w) = 1` renormalisation.

* `auto_adjust` (config default `True`) gates the whole update. Weights are updated **once per
  optimiser step** (after the accept/reject decision), not per condition.
* **Initial weights** (config defaults, `config.py`): `pde_weight = 1e-2`, `bc_weight = 1.0`,
  `bc2_weight = 1e-2`, `bc3_weight = 1e-4`, `bc4_weight = 1e-6`, `ic_weight = 1.0`,
  `ic_t_weight = 1.0`. The paper `[PDF p. 17]`: BC/IC weights initialised to `1`, the PDE (or any
  other "problematic") condition to something small, e.g. `1e-4`; `UpdateWeights` then "gently"
  raises it. Solution 9 panel (g) shows the PDE weight starting at `1e-4` and plateauing at `~1e-1`.
* **`ic_t`**: the code has an extra condition name `ic_t` (initial *time* derivative), used for
  second-order-in-time problems; `n_ic` points are sampled and the target is the previous slab's
  model time-derivative `[trainer64.py:145–162]`.

---

## 8. Algorithm 5 — UpdateTargetRatio `[PDF p. 33]` (Appendix A.2)

```
Algorithm 5 | UpdateTargetRatio              Decides if ϱ should be increased for descent

Input:   {λ_i}_i — history of regularisation parameters
Input:   ϱ_k     — current target decrease ratio
Output:  ϱ_{k+1} — updated target decrease ratio

        ▷ Hyperparameters: window size W, thresholds τ_s, τ_c

1:  if |{λ_i}| < W or ϱ_k = ϱ^{Stage 2} then
2:      return ϱ_k
3:  ℓ_j ← log( max(λ_{k−W+j}, ε) ),   j = 1, …, W        ▷ Log-scale window of length W
4:  ŝ ← slope of linear fit to ℓ                          ▷ Positive slope ⇒ λ is increasing
5:  ĉ ← Pearson correlation of ℓ                         ▷ Strength of upward trend
6:  if ŝ > τ_s and ĉ > τ_c then
7:      return ϱ^{Stage 2}                               ▷ λ has passed its minimum; escalate
8:  else
9:      return ϱ^{Stage 1}
```

* **Defaults:** `W = 30`, `τ_s = 10⁻⁴`, `τ_c = 0.1`, `ε = 10⁻⁸`. Used "throughout".
* The log scale is required because `λ` evolves over many orders of magnitude.
* **Both** checks must fire simultaneously: `ŝ` measures whether `log λ` is climbing *on average*,
  `ĉ` measures how *consistently* it does so.
* The switch is one-way and irreversible within a pass (`ϱ_k = ϱ^{Stage2}` short-circuits).

**Reference-code realisation** — `trainer64.py::has_passed_minimum` (lines 33–52):

```python
def has_passed_minimum(sequence, window_size=10, min_slope=1e-4, min_correlation=0.1):
    if len(sequence) < window_size:
        return False
    window = np.array(sequence[-window_size:], dtype=float)
    window = np.maximum(window, np.finfo(float).tiny)
    log_window = np.log10(window)              # <-- log10, not log
    x = np.arange(window_size)
    slope, _ = np.polyfit(x, log_window, 1)
    r_matrix = np.corrcoef(x, log_window)
    r_value = r_matrix[0, 1]
    if np.isnan(r_value):
        return False                            # perfectly flat window
    return slope > min_slope and r_value > min_correlation
```

Differences / refinements vs Algorithm 5:
1. **Base-10 log** (`np.log10`) instead of `log`. Slope thresholds are therefore in units of
   `log10 λ` per iteration; with `τ_s = 1e-4` the base does not materially matter.
2. `W = config.lambda_history_size = 30` ✔.
3. An additional **grace period**: the switch is only considered once
   `len(lambda_history) > config.lambda_grace_period` with `lambda_grace_period = 80` (default) —
   an extra 80-step burn-in not present in the paper. See `trainer64.py:648`.
4. On trigger, `target_rho` is set to `0.5` and the event is recorded (`rho_switch_events`).
5. In the **double-precision two-pass scheme**, a λ-minimum detection during the **float32 pass**
   *exits that pass* (reason `"reached float32 plateau"`, `trainer64.py:650–652`) rather than
   switching to Stage 2; Stage 2 is only entered during the float64 pass.
6. `lambda_history` is per-pass; `lambda_history_full` accumulates across both passes.

---

## 9. Exact CountSketch / SRCT operators — shapes, sampling, resampling

### 9.1 `C` (CountSketch, rows) — `[Eq. 16]`, `optimiser.py::_count_sketch`

* **Shape:** implicit `C ∈ ℝ^{s×N}`; never materialised. Only `K` sign vectors (`ℝ^N` each) and
  `K` bucket assignments (`{0..s−1}^N` each) are stored.
* **Sampling:** for each `k = 1..K`: `ϵ_k ~ Rademacher(±1)^N` (`jax.random.rademacher`),
  `h_k ~ Uniform{0, …, s−1}^N` (`jax.random.randint`), both from a freshly split key.
* **Application:** `r̃ ← Σ_k (1/√K) · segment_sum(ϵ_k ⊙ r, h_k)`; likewise with `J`'s rows.
* **Parameters:** `K = n_hashes` (paper: `K = 2` or `4` typical; code default `4`),
  `s = residual_sketch`.
* **Resampled?** Yes — a fresh key is drawn per condition per iteration
  (`optimiser.py:473`: `subkey, ck = jax.random.split(subkey)`).
* **Note:** `C` acts on **rows** of `J` and on the residual `r`. There is **no** residual lift back
  to `ℝ^N`; `r̃` is never un-sketched. The `full residual r` is computed separately and in full.

### 9.2 `Ω, S` (SRCT, columns) — `[Eq. 17]`, `optimiser.py::_make_srct` / `_apply_srct` / `_lift_update`

* **Shapes:** `D ∈ ℝ^{d_θ×d_θ}`, `Π ∈ ℝ^{d_θ×d_θ}`, `F ∈ ℝ^{d_θ×d_θ}` (DCT-II, orthonormal),
  `S` is a *selection* of `s` distinct columns ⇒ `Ω S = (D Π F) S ∈ ℝ^{d_θ×s}`.
* **Sampling:** `D` = i.i.d. Rademacher signs (`n_params` of them); `Π` = uniform random
  permutation; `S` = `s` indices without replacement (`replace=False`). All from one fresh key per
  iteration (`optimiser.py:461–465`).
* **Application (right sketch):** `J_right = DCT_II(J ⊙ signs[:, None])[:, perm][:, indices]`,
  mapping `(n_rows, d_θ) → (n_rows, s)`.
* **Adjoint lift:** `(y ∈ ℝ^s) → scatter into d_θ at indices → idct (DCT-III) →
  inverse-permute → multiply by signs`. Because the DCT is orthonormal, this is the true adjoint
  `(Ω S)ᵀ`; hence `‖p̃‖ = ‖Ω S p̃‖` and the trust-region norm/secular equation are consistent.
* **Relation to `d_θ`:** `s` is a free choice in `[⌊d_θ/3⌋, ⌊d_θ/2⌋]` (paper) and is *not* tied to
  `d_θ` by the code (config exposes `parameter_sketch` directly). The sketch is "accumulated over
  successive row batches (and per problem condition)", so `s` is independent of `N`; the code
  asserts `N % batch_size == 0` and `batch_size % sub_batch_size == 0`.
* **Resampled?** Yes in the code (per iteration). See §2.4 for the paper's internal contradiction.

### 9.3 Sketched Jacobian accumulation

* `J̃ ← Σ_m α_m C (J_m Ω S)` — additive over conditions and over macro-batches (Eq. 15).
* Per macro-batch, the raw sub-batch Jacobians are produced by **reverse-mode AD**
  (`jax.jacrev` of `(model, x) → residual`, `optimiser.py:130–148`) and the column sketch is
  applied **before** the row sketch: `J_right = _apply_srct(J, srct)` then
  `_count_sketch(J_right, r, …)`. The paper's Algorithm 2 line 10–11 has the same order
  (`J_batch ← J_batch Ω S`, then `J̃ ← J̃ + α_m C J_batch`).
* Since `C` is applied to batches and summed, **`J̃` and `r̃` are accumulated with a `lax.scan`**;
  the full residual `r` is also accumulated batch-by-batch (`r_batched.reshape(-1)`).

---

## 10. Residual computation, lift, and how `r` vs `r̃` are used

* **Full residual `r`.** Computed at every collocation point of every condition, with row scaling
  `α_m = √(w_m/|𝒳_m|)` (Algorithm 2 line 13; `optimiser.py:203–204`). It is used for
  (i) the **actual** objective `ℒ_k` (a `Σ_m w_m · mean(𝓕_m²)` computation, `optimiser.py:490–494`),
  (ii) the **actual reduction** numerator `act_red = ℒ_k(θ_k) − ℒ_k(θ_k+p_k)` at the chosen step and
  for every probe, and (iii) the **per-condition losses** fed to UpdateWeights.
* **Sketched residual `r̃`.** Computed as `C r` (Algorithm 2 lines 11–12), *never* lifted. Used only
  to form `g = Σ Uᵀ r̃` and hence the sketched gradient, the steps, and the predicted decrease.
* **Lift.** The step is solved in sketch space and lifted exactly once per candidate:
  `p_k = Ω S p̃_k`, implemented as the SRCT adjoint (`_lift_update`). The lifted step is what is
  added to `params_flat` (`unflatten(params_flat + delta)`) and what is evaluated in full space.
  Note the lift is **not** a least-squares solve; it is the adjoint embedding, which for an
  orthonormal `Ω` is the minimal-norm preimage.
* `jax.flatten_util.ravel_pytree` / `unflatten` are used to move between the pytree `θ` and the flat
  vector `params_flat` that the sketching and lifting act on (`optimiser.py:455`).

---

## 11. Hyperparameters — exact defaults

### 11.1 From the paper

| Hyperparameter | Symbol | Value | Source |
|---|---|---|---|
| Sketch rank | `s` | `⌊d_θ/3⌋ … ⌊d_θ/2⌋` | p. 9, p. 17 |
| CountSketch hashes | `K` | `2` or `4` "typical" | p. 10 |
| Trust-region radius (init) | `Δ₀` | `1.0` | p. 17 |
| Initial LM regulariser | `λ₀` | not specified (code: `1.0`) | — |
| Convergence tolerance | `Δ_min` | `1e-4` single, `1e-8` double | p. 17 |
| Target ratio, Stage 1 | `ϱ^{Stage1}` | `0.1` single, `0.075` double (≤ 0.2; alt. `0.2`) | p. 17, pp. 7–8 |
| Target ratio, Stage 2 | `ϱ^{Stage2}` | `0.5` (≥ 0.5; alt. `0.8`) | p. 17, pp. 7–8 |
| Probes | `q` | `24` | p. 17 |
| Probe window | `[Δ/3, 3Δ]` | factor 3, "can be widened to 5 each direction" | p. 15, Alg. 3 |
| Re-weighting exponent | `α` | `0.05` | p. 17, p. 32 |
| Weight floor | `ε` | `1e-8` | p. 32 |
| λ-history window | `W` | `30` | p. 33 |
| λ slope threshold | `τ_s` | `1e-4` | p. 33 |
| λ correlation threshold | `τ_c` | `0.1` | p. 33 |
| Collocation points | `|𝒳_P|` | `2^15` (PDE), `|𝒳_ℐ| = |𝒳_ℬ| = 2^14` | p. 17 |
| Initial weights | `w₀` | BC/IC = 1, PDE ≈ `1e-4` (code: `1e-2`) | p. 17 |
| Precision | — | float32 or float64 (hybrid: first 10 iters in float32) | p. 18 |

### 11.2 From the reference code (`src/pinn/config.py`, `Optimiser.__init__`)

`RunConfig` defaults (exact):

```python
residual_sketch  = 2000      # s (rows of the sketch)
parameter_sketch = 2000      # s (columns retained by the SRCT)
n_hashes         = 4         # K
sub_batch_size   = 4         # b'
batch_size       = 1024      # b
probe_batch_size = 128
n_probes         = 24        # q
window_scale     = 3.0       # probes in [Δ/3, 3Δ]
min_radius       = None      # resolved: 1e-4 (float32) / 1e-8 (float64)
max_radius       = 1e3
pchip_grid       = 512
newton_iters     = 80
initial_radius   = 1.0
target_rho       = 0.075
auto_adjust      = True
pde_weight       = 1e-2
bc_weight        = 1.0
bc2_weight       = 1e-2
bc3_weight       = 1e-4
bc4_weight       = 1e-6
ic_weight        = 1.0
ic_t_weight      = 1.0
lambda_history_size = 30
lambda_grace_period = 80
float32_steps    = 10
alpha            = 0.05
n_pde            = 2**14
n_ic             = 2**13
n_bc             = 2**13
log_every        = 5
plot_every       = 5
cache_dir        = ".cache/jax"
```

`Optimiser.__init__` defaults (used only if you bypass `RunConfig`): `residual_sketch=4096`,
`parameter_sketch=2048`, `n_hashes=4`, `sub_batch_size=4`, `batch_size=1024`,
`probe_batch_size=64`, `n_probes=32`, `window_scale=100.0`, `min_radius=1e-10`,
`max_radius=1e5`, `pchip_grid=1024`, `newton_iters=60`. `init_state(key, initial_radius=1.0)`
returns `SolverState(radius=1.0, rho=1.0, lam=1.0, key=key)`.

**Actual experiment settings** (`examples/burgers_double_precision.ipynb`, SIREN 40×40×40×40,
`d_θ = 11 285`): `residual_sketch = parameter_sketch = 2000`, `batch_size = 2^13`,
`probe_batch_size = 2^9`, `n_pde = 2^15`, `n_ic = 2^14`, `pde_weight = 1e-2`.
`burgers_demo.ipynb` (MLP 20×20×20×20): `s = 800`, `batch_size = 2^14`,
`probe_batch_size = 2^10`, `target_rho = 0.15`.
Table 1 `[PDF p. 6]` reports `s = 4000` for `d_θ = 11 285` (`= ⌊d_θ/2.8⌋`), `s = 5000` for
`d_θ = 15 265`, `s = 5000` for `d_θ = 17 241`, `s = 1200` for `d_θ = 3 215`.

> Note the code's `residual_sketch = parameter_sketch` — the **square** sketch. Keep them equal.
>
> **The code's default `s = 2000` is *not* the paper's `s ∈ [⌊d_θ/3⌋, ⌊d_θ/2⌋]` rule.** For the
> notebook's `d_θ = 11 285` SIREN, `s = 2000 ≈ d_θ/5.6`, below the recommended band, whereas Table 1
> reports `s = 4 000` for the same parameter count. The rule in the paper (§2.1) is advisory, and
> the code demonstrates the method is not sensitive to `s` over this range. Prefer the paper's rule
> unless reproducing a specific notebook exactly.
>
> **Iteration counts for calibration** (Table 1 / §5.8): double-precision Burgers reaches
> `ℓ₂^rel = 7.97×10⁻¹⁴` in **331 iterations** / 346.1 s; single-precision Burgers reaches
> `4.75×10⁻⁷` in **<110 iterations** / 9.8 s; KS (double) ≈ 5 000 iterations cap;
> 10D Poisson (double) ≈ 5 000; wave/KdV/multi-scale/5D-Poisson (double) ≈ 4 000.

---

## 12. Reference implementation: file layout and function names

| File | Purpose / key symbols |
|---|---|
| `src/pinn/optimiser.py` | **DSGNAR.** `Optimiser`, `SolverState(radius, rho, lam, key)`, `Optimiser.__init__`, `Optimiser.init_state`, `Optimiser.step`, `_step_impl`, `_make_srct`, `_apply_srct`, `_lift_update`, `_count_sketch`, `_jacobian_block`, `_kernel_sketch`, `_solve_subproblems`, `_eval_probes`, `_eval_all_conditions`, `_pchip`, `_step_and_pred` (nested in `_step_impl`) |
| `src/pinn/config.py` | `RunConfig` (dataclass) + `resolve_precision_defaults`, `solver_config`, `problem_config`, `display_config`; `SINGLE_PRECISION_MIN_RADIUS=1e-4`, `DOUBLE_PRECISION_MIN_RADIUS=1e-8` |
| `src/pinn/trainer64.py` | `train`, `train64` entry, `has_passed_minimum`, `_partition`, `_slab_iterator`, `_u_targets_from_model`, `_u_targets_from_model_t`, `_build_conditions`, `_condition_names`, `_default_weights`, `_eval_model_jit`, `_eval_plot_grid`, `predict_to_current_slab`, `compute_l2_error`, `_probe_data`, `precompile` |
| `src/pinn/trainer32.py` | single-precision counterpart (identical apart from the precision passes) |
| `src/pinn/networks.py` | `MLP`, `SIREN`, `GaborNet`, `SPINN` |
| `src/pinn/problem.py` | `Problem` base class: `residual_fns()` → `{name: (model, coords) -> residual}`, `samplers()` → `{name: sampler}`; attributes `D`, `t_max`, `x_min`, `x_max`, `problem_name`, `ref_path`, `analytical_ic`, `analytical_ic_t`, `n_pde/n_ic/n_bc` |
| `src/pinn/sampling.py` | collocation-point samplers |
| `src/pinn/operators.py` | derivative helpers for residuals |
| `src/pinn/reference.py`, `references/*.npz` | reference solutions |
| `src/pinn/plotting.py` | live training dashboard (`Plotter`, `mark_lambda_min`, `mark_precision_switch`, `mark_slab_end`) |
| `scripts/generate_references.py` | regenerate reference solutions |

**Residual contract.** A residual function has signature `(model, coords) -> Array` with
`coords: (n, n_in)` and output `(n, n_out)`; the reference wraps it as
`jax.vmap(partial(residual_fn, m))(x_sub)` and differentiates w.r.t. the flattened parameters.
Conditions are passed to the optimiser as a `Sequence[(residual_fn, points)]`.

**Precision.** `trainer64` runs **two passes per slab**: `precision_pass ∈ {0, 1}` with
`precision_bits ∈ {32, 64}`, toggling `jax.config.update("jax_enable_x64", precision_bits == 64)`
and casting carried parameters with `astype`. The paper's **hybrid precision** rule
`[PDF p. 18]`: "for poor initialisations of θ … we therefore run the first ten iterations of every
double-precision experiment in single precision, to correct any initial problematic elements of θ
and allow drastically mis-scaled weights to align first" (`config.float32_steps = 10`; the float32
pass also exits early when `radius < 0.1`).

**Time marching `[PDF p. 19]`.** Time-dependent problems are solved in sub-intervals `[t_n, t_{n+1}]`;
each interval starts from a *slightly perturbed* copy of the previous interval's parameters:
`params = params_old + 0.01 * params_fresh` where `params_fresh` is a fresh draw from the
architecture initialiser (`trainer64.py:755–761`). Initial conditions on the first interval come
from `analytical_ic`; subsequently from the previous slab's model
(`_u_targets_from_model`, `_u_targets_from_model_t`).

**Stopping criteria actually implemented** (`trainer64.py:674–684`):
1. `opt_state.radius < 10 * config.min_radius` ⇒ exit pass (`"radius below threshold"`). This is
   Alg. 1 line 12 with a 10× margin.
2. float32 pass only: `opt_state.radius < 0.1` ⇒ exit (`"radius below 0.1"`).
3. float32 pass only: `slab_step >= config.float32_steps` (=10) ⇒ exit.
4. λ-minimum detected during the float32 pass ⇒ exit pass; during the float64 pass ⇒ set
   `target_rho = 0.5` and continue.
There is no explicit maximum-iteration cap; runs terminate on the radius criterion.

---

## 13. Minimal faithful CPU-feasible variant

### 13.1 The target workload

A **6-hidden-layer × 20-neuron `tanh` MLP** for a 1-D space-time problem
(`n_in = 2` (x,t), `n_out = 1`):

```
d_θ = (2·20 + 20) + 5·(20·20 + 20) + (20·1 + 1)
    =  60       + 5·420           +  21
    =  60 + 2100 + 21 = 2181
```

(If `n_in = 3` or extra conditions/outputs are added the count changes slightly; with 5 hidden
layers of 20 and `n_in = 2` you get `d_θ = 2 181`; the task's "≈2.6 k" corresponds to e.g.
`n_in = 4` or 7 layers — either way, `d_θ ≈ 2.1 × 10³ – 2.6 × 10³`.) Recommended sketch size:

```
s ≈ d_θ / 3 … d_θ / 2  ≈  730 … 1300     (use s = 1024 for a power of two)
```

### 13.2 What can be simplified

* **Full Jacobian and full SVD are affordable.** With `N = 1 024` residual rows (`n_pde = 512`,
  `n_ic = 256`, `n_bc = 256`), `J ∈ ℝ^{1024 × 2181}` in float64 is ~18 MB, and `J̃` is
  `s × s ≈ 1024²` (~8 MB). `np.linalg.svd` on `1024²` is ~0.5 s on a CPU; on a 6×20 problem
  `JᵀJ` (`2181²`, ~38 MB) would also fit, so you can **cross-check** the sketched solve against the
  exact LM step.
* **Recommended defaults for the tiny variant:**
  `s = 1024` (or `d_θ//2`), `K = 4`, `b = 512`, `b' = 4`, `q = 24`, `window_scale = 3.0`,
  `pchip_grid = 512`, `newton_iters = 80`, `Δ₀ = 1.0`, `Δ_min = 1e-8` (double) / `1e-4` (single),
  `ϱ^{Stage1} = 0.075`, `ϱ^{Stage2} = 0.5`, `α = 0.05`, `W = 30`, `τ_s = 1e-4`, `τ_c = 0.1`,
  `pde_weight = 1e-2`, `ic/bc_weight = 1.0`, float64 throughout (`jax_enable_x64=True`).
* **Simplifications that are faithful:**
  - Compute the Jacobian densely if you like; the *sketch* is what matters, not the way `J` is
    obtained. Use `jax.jacrev` over a `vmap`ed residual for CPU simplicity.
  - If you only need one condition set, drop the per-condition loop but keep the `α_m = √(w_m/|𝒳_m|)`
    row scaling.
  - You may keep `C` and `Ω S` fixed across iterations (the paper's prose variant) — cheaper and
    still faithful; resampling per iteration matches the code.
  - `jax.lax.scan`/`lax.map` are optional; plain Python loops over batches are fine at this size
    (they will be slow under `jit`, so either donate the whole loop or don't `jit` the step).
* **Simplifications that are *not* faithful — avoid:**
  - Do **not** use a dense Gaussian `Ω` unless you keep `norm` orthogonal for the transform; the
    DCT-II with `norm="ortho"` is what makes the lift the exact adjoint.
  - Do **not** sketch only the rows (that is just LM on a random projection; you lose the
    well-conditioned square SVD and the cheap multi-`λ` re-solve).
  - Do **not** evaluate the ratio with the sketched residual in the denominator — the numerator
    must be the **full** objective.
  - Do **not** accept on `ϱ ≥ ϱ*`; accept on `ϱ > 0`.

### 13.3 Reference pseudo-code skeleton (JAX-flavoured)

```python
def dsgnar_step(params, static, state, conditions, weights, target_rho, cfg):
    flat, unflatten = ravel_pytree(params)

    # 1. sketch operators (fresh each iteration; see §2.4)
    key, k1 = split(state.key); state = state._replace(key=key)
    k1, k_srct = split(k1)
    srct = make_srct(k_srct, flat.size, cfg.s)          # signs, perm, indices

    # 2. accumulate doubly-sketched system and full residual
    Jt = zeros((cfg.s, cfg.s)); rt = zeros(cfg.s); full_r = []
    for (fn, pts), w in zip(conditions, weights):
        k1, k_ck = split(k1)
        Ji, ri, rfull = kernel_sketch(flat, unflatten, static, fn, pts,
                                      k_ck, srct, cfg)   # Alg. 2
        a = sqrt(w / rfull.size)
        Jt += a * Ji; rt += a * ri; full_r.append(rfull)
    loss = sum(w * mean(r**2) for w, r in zip(weights, full_r))     # Eq. 22

    # 3. one SVD
    U, S, Vt = svd(Jt, full_matrices=False)
    g = S * (U.T @ rt)                                  # J̃ᵀ r̃ in the V basis

    def step_and_pred(lam):
        denom = S**2 + lam
        return -(Vt.T @ (g / denom)), sum(g**2 * (S**2 + 2*lam) / denom**2)

    # 4. probes + secular solves + PCHIP crossing (Alg. 3)
    ...
    # 5. lift, full-space evaluation, accept iff rho > 0, update radius/lam (Alg. 1 lines 8-11)
    # 6. UpdateWeights(Alg. 4), UpdateTargetRatio(Alg. 5)
```

---

## 14. Gotchas

### 14.1 Sign / formula inconsistencies to be aware of

1. **The predicted decrease has a factor-2 ambiguity** (§3.4). `optimiser.py` computes
   `pred = Σ g²(σ²+2λ)/(σ²+λ)²`; the plain `m̃(0) − m̃(p̃(λ))` for `m̃ = ½‖r̃+J̃p‖²` is exactly
   **half** that (verified numerically, §15 check C). Use the code's expression to reproduce the
   paper's published numbers with the paper's literal `ϱ*` values; if you switch to the
   theoretically clean formula, double `ϱ*`. Never mix.
2. **Algorithm 1 vs Algorithm 3 disagree on the step's sign.**
   Alg. 1 line 6: `p̃ = −V diag(σ/(σ²+λ)) Uᵀr̃`; Alg. 3 line 10:
   `p̃ = +Σ (g_j/(σ_j²+λ)) v_j`. Both appear in the paper. **The code uses the negative
   convention** (`step_sk = -(Vt.T @ (g/denom))`), and this is the one consistent with the
   optimiser being a *descent* method. Use the negative convention.
3. **Algorithm 1's `pred` box** (`Σ (σ_j g_j)²/(σ_j²+λ_k)²`) omits the `+2λ` term and can go
   negative; the code's version adds it. Prefer the code's.
4. **§3.2.2's "single, consistent sketch" vs Algorithm 1 line 2's per-iteration
   `GetSketchOperators`.** Contradiction (§2.4). The code resamples. Pick one and document it;
   resampling is the unbiased choice.
5. **Algorithm 4 vs `trainer64`'s weight update.** As written in §7, the code's update is
   `∝ w_m^{1−α}(mean(wL)/(w_m L_m))^α` when fed unweighted mean-squared condition losses; the
   paper's Algorithm 4 is `w_m · (ℓ̄/ℓ_m)^α`. Choose per your reproduction goal.
6. **`Δ_min` comparison.** Alg. 1 says `Δ_{k+1} < Δ_min`; the code uses `radius < 10·min_radius`.
   Pick one (the code's gives ~10 extra iterations).
7. **`log` vs `log10`** in Algorithm 5 (code uses `log10`); and the code adds an 80-step
   **grace period** not present in the paper.

### 14.2 Numerical issues

1. **dtype.** Double precision is essentially required for the published accuracy
   (relative ℓ₂ errors down to `3×10⁻¹⁶`). In JAX you must set `jax.config.update("jax_enable_x64",
   True)` **before** creating arrays, and cast all parameters/points to `jnp.float64`; otherwise JAX
   silently truncates to float32. The paper's **hybrid** scheme runs the first 10 iterations in
   float32 to fix bad initialisations and mis-scaled weights (`[PDF p. 18]`), then continues in
   float64.
2. **Ill-conditioning and tiny singular values.** `Σ` of the (sketched) Jacobian can span many
   orders of magnitude; `σ_i²` in the secular equation `Σ g_i²/(σ_i²+λ)²` underflows for tiny `σ_i`
   when `λ` is small, and `(σ_i²+λ)³` in the Newton derivative can underflow too. **Always add a
   tiny floor** (`jnp.finfo(dtype).tiny`) to denominators, exactly as the code does. Do not use
   `1e-30` in float32 (it underflows).
3. **`λ` explosion / collapse.** With `q = 24` probes spanning `[Δ/3, 3Δ]`, `λ` can move by tens of
   orders of magnitude between iterations (Figure 3: `λ` spans >25 orders of magnitude while `Δ`
   stays near 1). Track `log λ` internally if you compute diagnostics; the raw `λ` overflows
   float32 easily. The paper's own `log`-scale λ-history (Algorithm 5) exists for this reason.
4. **Overshooting the trust region.** If the current radius is large and all probes give `ϱ < 0`,
   the radius is shrunk (`Δ/3`) and the next probes cover the lower region. There is no line search
   and no backtracking within an iteration — rejection only shrinks the *next* iteration's window
   (the code additionally tests the PCHIP-chosen step in full space, and rejects it if `ϱ ≤ 0`).
5. **Rejected steps still cost an evaluation.** Every probe step is evaluated in full space, so a
   step costs `(q + 1)` full residual evaluations (plus one for the current loss). With `q = 24`
   this is the dominant cost — the paper notes "depending on GPU budget and the memory cost of a
   loss evaluation, one may use fewer or more probes" `[PDF p. 15]`.
6. **PCHIP flattening.** The monotone envelope `safe_rhos = minimum.accumulate(clip(ϱ, −1, 1))`
   can be flat over a wide range (all `ϱ ≥ target`), in which case there is *no* crossing and the
   fallback sets `Δ* = 3Δ` or `Δ*/3`. A flat envelope also makes a naive root-find ill-posed;
   the code's "last crossing of the `≥ target` indicator" formulation is robust to this and should
   be copied.
7. **`jnp.clip` before the cumulative minimum** is important: without it, a single huge spurious `ϱ`
   (e.g. from a near-zero `pred`) dominates the envelope. Likewise `pred + finfo.tiny` in the ratio.
8. **Sketch reproducibility.** If you `jit` the step, the PRNG keys must be threaded through the
   state (`SolverState.key`); otherwise the sketches are resampled implicitly and results are not
   reproducible. The code explicitly splits and stores `state.key` each step.
9. **`segment_sum` determinism.** `jax.ops.segment_sum` on the CPU/GPU is deterministic for a fixed
   bucket vector, but the *order* of floating-point accumulation over `K` hashes and over batches
   matters at the `1e-16` level; for bit-reproducible runs fix the `K`-loop order.
10. **DCT normalisation.** `jax.scipy.fft.dct(..., type=2, norm="ortho")` is the right primitive.
    The corresponding inverse is `idct(..., type=2, norm="ortho")` (which is DCT-III with the
    matching normalisation) — mixing `norm=None` between forward and inverse breaks the lift and,
    with it, the trust-region norm equality used by the secular solver.
11. **`Σ g²/(σ²+λ)²` in the Newton denominator.** The code's `d_phi` is exact (§15 check D), but
    note that `φ` and `φ′` are evaluated from `p = g/(σ²+λ)`; if you recompute `σ²+λ` with a
    different `λ` update (e.g. `max(0, λ − φ/φ′)` must clamp **to zero**, not to `λ_min`), you will
    get a different probe `λ` and thus a different `Δ*`.

---

## 15. Numerical checks performed (for confidence in this spec)

All checks used pure NumPy 2.3.5 (JAX is not installed in this environment); scripts are in
`/tmp/verify_dsgnar*.py`.

* **A. Step-norm identity.** `‖p̃(λ)‖ = ‖g/(σ²+λ)‖` and the secular Newton solve returns
  `λ` with `‖p̃(λ)‖ = Δ` for `Δ ∈ {1e-3, 1e-1, 1, 10}` to ~10 significant digits; for
  `Δ > ‖p̃(0)‖ = 51.0`, `λ = 0` (constraint inactive) — the `max(0, ·)` clamp is what handles this.
* **B. Sequence of `_solve_subproblems`.** The code's `d_phi = −(Σ p²/denom)/‖p‖` agrees with a
  finite-difference derivative of `‖p(λ)‖` to 1e-7 (check D).
* **C. Predicted-decrease identity (decisive).** For a square sketched Jacobian `J̃ = UΣVᵀ` with
  `r̃` in its row space, and `g = Σ(Uᵀr̃)`:
  `m̃(0) − m̃(p̃(λ)) = ½ · [Σ g²(σ²+2λ)/(σ²+λ)²]` for every `λ` tested
  (`λ ∈ {0, 1e-3, 0.1, 1, 10, 1e3}`; ratio exactly `0.500000`). This settles §3.4: the code's
  `pred` is **twice** the plain LM model reduction, hence its `ϱ` is half the textbook Eq. 14 ratio.
* **D. Algorithm 3 line 7 formula.** `φ′ = −Σ p²/(σ²+λ) / (φ + δ)` matches
  `d/dλ sqrt(Σ g²/(σ²+λ)²)` numerically.
* **E. CountSketch equivalence.** `_count_sketch`'s `Σ_k (1/√K)·segment_sum(ϵ_k ⊙ r, h_k)` is
  bit-equivalent to `C r` with `C` from Eq. 16 (structure verified by construction; row scaling and
  bucket assignment match exactly).
* **F. PCHIP.** `log(geomspace(r_lo, r_hi, q))` is uniformly spaced (verified), so PCHIP's nodes are
  well-conditioned; the Fritsch–Carlson slope limiter in `_pchip` is the standard monotone variant.
* **G. End-to-end loop.** A full numpy port of the optimiser on a small ill-conditioned nonlinear
  least-squares problem (37 parameters) reproduces the expected behaviour: accepted steps always
  decrease the loss; rejected steps shrink `Δ` by 3; the reported `ϱ` equals `act_red/pred_code`
  exactly (the acceptance rule `ϱ > 0` ⇔ `act_red > 0`).

---

## 16. One-page implementation order (suggested)

1. Residual engine: per-condition `(model, coords) -> residual`, with `α_m = √(w_m/|𝒳_m|)` row
   scaling; a function to evaluate Eq. 22 in both "sketch" and "full" form.
2. Sketch operators: `C` (Eq. 16, `K` hashes, `1/√K`, `segment_sum`) and `Ω S` (Eq. 17, orthonormal
   DCT-II) with their adjoint lift.
3. `Sketch` (Algorithm 2): accumulate `J̃`, `r̃`, `r` over conditions × batches.
4. `svd(J̃)`; `g = Σ Uᵀ r̃`.
5. `LambdaSolve` (Algorithm 3): probe grid, secular Newton, PCHIP crossing, fallbacks.
6. Final step + full-space evaluation; accept iff `ϱ > 0`; radius update (`Δ*` / `3Δ` / `Δ/3`).
7. `UpdateWeights` (Algorithm 4) and `UpdateTargetRatio` (Algorithm 5).
8. Outer loop with `Δ_min` termination, hybrid precision, and (for time-dependent problems) slab
   time marching with the `θ_prev + 0.01·θ_fresh` warm start.
