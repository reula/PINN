# Multipole moments: Geroch–Hansen and Thorne, specialised to static spacetimes

A working note for `Stationary/`.  It says what the two definitions are, what they become when
the spacetime is static, which closed forms can actually be evaluated for the solutions this repo
uses, and what (if anything) the current `stationary/multipoles.py` measures.  Every formula is
quoted from, or checked against, a source listed in section 7; the checks are in section 5.

### Notation and conventions

* Signature $(-,+,+,+)$, geometrized units $G=c=1$; $a,b,\dots$ are three-dimensional indices,
  $\mu,\nu,\dots$ four-dimensional.
* The code's static vacuum completion is $g = -\lambda\,dt^2 + \lambda^{-1}h_{ij}dx^idx^j$ with
  $\lambda = e^{2U}$ (`geometry_invariants.py`, `reading="vacuum"`), so
  $\lambda = -\xi^a\xi_a$ for $\xi = \partial_t$ and $U = \tfrac12\log\lambda$.
* Spatial Cartesian coordinates are $x^i = (\rho\cos\varphi,\rho\sin\varphi,z)$; the exact Weyl
  assets are built in canonical Weyl coordinates in `weyl.py`, the spherical asset in `exact.py`.
  Lengths are the *chart* ones unless `physical_factor(cfg)` is applied.
* $\langle\,\cdot\,\rangle$ / superscripts $\langle a_1\cdots a_l\rangle$ and
  $\mathcal C[\cdot]$ denote the symmetric trace-free (STF) part; $r=\lvert\mathbf x\rvert$,
  $n^i=x^i/r$, $\hat n$ a unit vector.
* Three different families of numbers are called "multipoles" in the literature and are **not**
  the same: the Newtonian moments $I_l$, the Weyl coefficients $a_l$ (equivalently the axis
  coefficients $U^{(l)}$), and the Geroch–Hansen/Thorne moments $M_l$.  Section 1.5 tabulates
  them for the solutions used here.
* Two normalisations of the GH/Thorne moments circulate, differing by $(2l-1)!!$; they are
  defined operationally in section 1.4, and **N1** (weak-field limit $=I_l$, metric coefficient
  $2$, Kerr $M_l+\mathrm iS_l=M(\mathrm ia)^l$) is used throughout this note.

### Contents

0. [Bottom line](#0-bottom-line)
1. [Why the naive coefficient is not a moment](#1-why-the-naive-coefficient-is-not-a-moment) —
   chart dependence, non-linearity, origin gauge, and the objects that get confused
2. [Geroch–Hansen](#2-gerochhansen-geroch-1970-static-hansen-1974-stationary)
3. [Thorne's ACMC definition and Gürsel's equivalence](#3-thornes-acmc-definition-and-its-equivalence-to-gh)
4. [Usable formulae in the static case](#4-what-static-buys-you-usable-formulae) — Weyl class,
   FHPS/Gürlebeck, Ernst coefficients, finite-radius and source integrals, uniqueness
5. [Calibration table](#5-calibration-table-with-the-arithmetic)
6. [What this means for this repo](#6-what-this-means-for-this-repo)
7. [Sources](#7-sources)

---

## 0. Bottom line

1. **$S_{lm}$ in `multipoles.py` is not a multipole moment.**  It is the $Y_{lm}$ coefficient of
   $\lambda-\lambda_\infty$ on a coordinate sphere, kept at fixed power $\rho^{-(l+1)}$.  It is a
   useful *tail diagnostic* (it detects a wrong decay power) but it is chart dependent and it is
   not what Geroch–Hansen (GH) or Thorne define.  Even for the exact assets it is not the moment:
   for the spherical asset its asymptotic $l=0$ value is $-2\sqrt{4\pi}\,kR_0$, not $M_0=R_0$
   (section 5.3).
2. **Static $\Rightarrow$ the current moments vanish identically**: the twist
   $\omega_\alpha = \epsilon_{\alpha\beta\gamma\delta}\xi^\beta\nabla^\gamma\xi^\delta$ is zero
   when $\xi$ is hypersurface orthogonal, so $\Phi_J=0$ and all $S_l=0$.  A static spacetime is
   described by mass moments alone.  (Conversely, Xanthopoulos: a stationary spacetime with all
   angular moments zero is static.)
3. **The moment is a functional of a potential, not of $\lambda$ itself, and it is non-linear.**
   In the static axisymmetric Weyl class, with
   $U(\rho=0,z)=\sum_{r\ge0}U^{(r)}/\lvert z\rvert^{r+1}$ on the axis, the FHPS/Gürlebeck
   relations are $M_0=-U^{(0)}$, $M_1=-U^{(1)}$, $M_2=\tfrac13(U^{(0)})^3-U^{(2)}$ and
   $M_3=(U^{(0)})^2U^{(1)}-U^{(3)}$.
   The cubic term is real: the Curzon solution has $U=-M/R$ (no quadrupole at all in $U$) yet
   $M_2=-M^3/3$.  This is why "read the coefficient off $\lambda$" cannot work beyond the linear
   regime.
4. **Static axisymmetric gives a cheap, exact route.**  For `weyl.py`'s rods the axis expansion is
   closed form (section 5.4), so the moments are closed form; no conformal completion, no
   asymptotic fit, is needed for the exact assets.  For a *network* solution the cheapest honest
   route is a flux/source integral (section 4.4) or an ACMC extraction, not the sphere
   coefficients.
5. **Normalisation is a swamp in this literature.**  Fix it once, by tests, against
   Schwarzschild ($M_l=0$, $l\ge1$), Kerr ($M_l+\mathrm iS_l = M(\mathrm ia)^l$), the Newtonian
   limit ($M_l\to I_l$) and Curzon ($M_2=-M^3/3$).  Section 3.3 says exactly what to pin.

---

## 1. Why the naive coefficient is not a moment

### 1.1 The Newtonian definition, for reference

For a Newtonian source of density $\mu$ with potential $\Phi\to0$ at infinity,

$$\Phi(\mathbf x) = -\frac{M}{r} - \sum_{n\ge1}\frac{(2n-1)!!}{n!}\frac{1}{r^{n+1}}M^{i_1\dots i_n}n_{i_1\dots i_n}, \qquad M^{i_1\dots i_n}=\int_V y^{i_1}\cdots y^{i_n}\,\mu\,d^3y,$$

with $\langle\cdot\rangle$ the symmetric trace-free part and $M_n$ the axisymmetric scalar
$M_n=\int\mu\,r^nP_n(\cos\theta)\,d^3x$ so that $\Phi=-\sum_n M_nP_n(\cos\theta)/r^{n+1}$
([HP16, eqs. (11), (12), (21)]).  So in the *flat* case the coefficient of $P_n/r^{n+1}$ **is**
the moment, because the field is linear in the source and the coordinates are fixed by the flat
metric.  General relativity removes both of those supports.

### 1.2 Failure 1: chart dependence

Take Kerr in Boyer–Lindquist coordinates:

$$g_{tt} = -\Big(1-\frac{2Mr}{r^2+a^2\cos^2\theta}\Big) = -1+\frac{2M}{r}-\frac{2Ma^2}{3r^3}-\frac{4Ma^2}{3r^3}P_2(\cos\theta)+O(r^{-4}).$$

The $P_2/r^3$ coefficient is $-\tfrac43Ma^2$.  The GH/Thorne quadrupole of Kerr is
$M_2=-Ma^2$ (N1, section 1.4; e.g. [May22, §1]).  The two differ, and the reason is not that
one of them is wrong: Boyer–Lindquist coordinates are not an ACMC chart — the ACMC conditions
constrain the whole metric, not one component — and the coordinate transformation that brings
them to ACMC form shifts the $P_2$ coefficient at order $r^{-3}$.  *Any* extraction that reads a
coefficient off a written-down metric must first fix the coordinates, which is exactly what
Thorne's ACMC conditions do and what GH do by construction with a conformal completion.  The
size of the shift for a numerical metric is estimated in [PA12] (section 4.4).

There is a second, purely scaling, effect the repo already handles: a change of length unit
$\rho\to c\rho$ multiplies the coefficient of $Y_{lm}/\rho^{l+1}$ by $c^{\,l+1}$.  That is why
`multipoles.py` reports $S_{lm}$ "in the physical chart" via `physical_factor`.  But scaling is
only the trivial part of the chart dependence; a general coordinate change mixes $l$.

### 1.3 Failure 2: non-linearity.  The Curzon solution

The Curzon solution is the static axisymmetric Weyl solution with

$$U = -\frac{M}{R},\qquad R=\sqrt{\rho^2+z^2},\qquad e^{2k} = \exp\!\Big(-\frac{M^2\rho^2}{2R^4}\Big) .$$

Its Newtonian potential has *only* a monopole: $U=-M/R$ has no $P_2$ at all, so a naive
$l=2$ extraction from $U$ returns zero.  Its GH/Thorne quadrupole is nevertheless non-zero,

$$M_0=M,\qquad M_2=-\frac{M^3}{3},\qquad (\text{odd moments }0),$$

which follows from the FHPS relation above, and is confirmed independently three times by direct
asymptotic metric expansions of the Curzon–Chazy metric, all giving the $P_2/r^3$ coefficient
$\pm\frac23M^3$:

* [MC15, eq. (16)] (slowly rotating Curzon–Chazy): in $(+{-}{-}{-})$,
  $g_{tt}=1-\frac{2M}{r}+\frac23\frac{M^3}{r^3}P_2+\dots$, i.e. in $(-,+,+,+)$
  $g_{00}=-1+\frac{2M}{r}-\frac23\frac{M^3}{r^3}P_2+\dots$;
* Aguirregabiria–Bel–Martín–Molina–Ruiz [ABMMR01, eq. (115)]:
  $h_{00}=2M/R-\frac23\frac{M^3}{R^3}P_2+\frac{38}{105}\frac{M^5}{R^5}P_4+\dots$ (identical in
  harmonic and quo-harmonic coordinates);
* Gu [Gu10, eq. (4.6)]: $U=1-R_s/R+\frac1{12}(R_s/R)^3P_2+\dots$ with $R_s=2M$.

Comparing with $g_{00}=-1+2M/r+2M_2P_2/r^3+\dots$ gives $M_2=-\tfrac13M^3$.  (The
Hartle–Thorne comparison in [MC15], $Q=M^3/3$, is the same statement in HT's sign convention.)
The same expansions give a non-zero $P_4$ coefficient, reported as $M_4=+19M^5/105$ in the
coefficient-2 convention [ABMMR01, eq. (115)] — noted here as reported, not re-derived, but it
is a reminder that "no Weyl multipoles" does not mean "no GH moments" at any order.

The lesson is not "Curzon is weird"; it is that the moment is built from a *potential that is a
non-linear function of the metric function*, and the Einstein equations couple that non-linearity
to the angular structure.

### 1.4 Failure 3: gauge/origin, and the two normalisations in use

The GH moments are STF tensors at the point at infinity $\Lambda$; they change in a prescribed,
finite-dimensional way under a change of conformal factor, which is a change of origin (mass
dipole) [Ger70, Han74].  A dipole moment $M_1$ is therefore gauge: the centre-of-mass condition
sets it to zero.  Any single scalar coefficient extracted on a sphere is not.

There are (at least) two normalisations in common use, related by $(2l-1)!!$:

* **N1 (used below, and in most no-hair phenomenology).**  In ACMC coordinates
  $g_{00} = -1+\frac{2M}{r}+2\sum_{l\ge2}M_lP_l(\cos\theta)/r^{l+1}+(\text{lower harmonics})$,
  current moments in $g_{0j}$ likewise; weak field $M_l\to I_l$ (the Newtonian moment); Kerr
  obeys $M_l+\mathrm iS_l=M(\mathrm ia)^l$, i.e. $M_0=M$, $M_2=-Ma^2$, $M_4=Ma^4,\dots$ and
  $S_1=Ma$, $S_3=-Ma^3,\dots$, all odd-order mass and even-order current moments zero
  [CGu16, §5.1; May22, §1].
* **N2 (STF-tensor normalisation).**  The other convention puts the $(2l-1)!!$ on the STF tensors
  instead of on the scalars: for an axisymmetric configuration
  $M^{\rm STF}_{a_1\dots a_l}=(2l-1)!!\,M_l\,k_{a_1}\cdots k_{a_l}$, i.e. N2 $=(2l-1)!!\times$N1.

The labels collide across the literature — [CGu16] calls N1 "the normalisation of Geroch–Hansen"
while [May22] calls N1 "the Thorne normalisation" and writes the tensor-to-scalar conversion with
the prefactor in its eq. (16) — so do not trust the names.  Pin the numbers, as follows.

Where the $(2l-1)!!$ and the $1/l!$ come from: the STF tensors differ from the scalar moments by
$(2l-1)!!$, because the contraction identity is
$M^{\langle a_1\cdots a_l\rangle}n_{\langle a_1}\cdots n_{a_l\rangle}=l!\,M_lP_l(\cos\theta)$ [CGu16, eq. (6.14)].  Hence
a paper that writes $M^{\langle\cdots\rangle}=(2l-1)!!\,I^{\langle\cdots\rangle}$ and one that writes
$M_l=\int\rho r^lP_l\,d^3x$ may be describing the same scalar moments; whereas a paper whose
$M_l$ misses $I_l$ at leading order is using a different scalar normalisation.  Decide by the
weak-field limit (check 3 of section 3.3), never by the symbol.

### 1.5 The three objects that get confused

For the solutions this repo uses, the three families of numbers are:

| | Newtonian moments $I_l$ | Weyl coefficients $a_l\equiv U^{(l)}$ | GH/Thorne moments $M_l$ (N1) |
|---|---|---|---|
| what they are | coefficients of $\Phi_N=-\sum_l I_lP_l/r^{l+1}$; $I_l=\int\mu\,r^lP_l\,d^3x$ | axis coefficients of the Weyl potential, $U(\rho{=}0,z)=\sum_l U^{(l)}/\lvert z\rvert^{l+1}$ ("Weyl multipoles") | §§2–3; non-linear functional of $U$ (FHPS, §4.2) |
| Curzon, $U=-M/R$ | $M,0,0,0,\dots$ | $-M,0,0,0,\dots$ | $M,\ 0,\ -M^3/3,\ 0,\dots$ |
| single rod $[-m,m]$ | $m,\ 0,\ m^3/3,\ 0,\ m^5/5,\dots$ | $-m,\ 0,\ -m^3/3,\ 0,\ -m^5/5,\dots$ | $m,\ 0,\ 0,\ 0,\ 0,\dots$ |
| default two-rod pair | $2,\ 0,\ 31/6,\ 0,\dots$ | $-2,\ 0,\ -31/6,\ 0,\dots$ | $2,\ 0,\ 5/2,\ 0,\dots$ |

Read across the rod row: the Newtonian quadrupole $m^3/3$ and the Weyl coefficient $-m^3/3$ are
both non-zero, yet the GH moment vanishes — the rod is Schwarzschild.  Read the Curzon row: both
$I_2$ and $a_2$ vanish and the GH moment does not.  Read the two-rod row: $M_2=I_2-\frac13M_0^3$
(§4.2), so the answer is neither of its neighbours.  Statements in the literature that "the
Curzon metric has no multipole moments" refer to the middle column $a_l$, not the last
[MS21; BH05; CCLD21]; [HP16, §5.3] writes the same distinction as $\bar M_n^{K}=-a_n$.

---

## 2. Geroch–Hansen (Geroch 1970 static; Hansen 1974 stationary)

The construction, in the form given by [CGu16, App.] and [May22, §2.1]:

Let $\xi^a$ be the asymptotically timelike Killing field, and define

$$\lambda = -\xi^a\xi_a (>0), \qquad \omega_\alpha = \epsilon_{\alpha\beta\gamma\delta}\xi^\beta\nabla^\gamma\xi^\delta \equiv \omega_{,\alpha}\quad(\text{vacuum}) .$$

Split the four-metric along $\xi$ (Kaluza–Klein form, written in this repo's signature)

$$g_{\mu\nu}dx^\mu dx^\nu = -\lambda\,(dt+\mathcal A)^2 + \lambda^{-1}h_{ij}dx^idx^j, \qquad h_{ab} = -\xi^2 g_{ab}+\xi_a\xi_b ,$$

so that $h$ is the Riemannian metric on the space of orbits.  Introduce the mass and twist
potentials

$$\Phi_M = \frac{\lambda^2+\omega^2-1}{4\lambda},\qquad \Phi_J = \frac{\omega}{2\lambda},$$

conformally rescale them, $\tilde\Phi_{M,J}=\Omega^{-1/2}\Phi_{M,J}$, where
$\tilde h_{ab}=\Omega^2h_{ab}$ brings spatial infinity to a single regular point $\Lambda$
($\Omega(\Lambda)=D_a\Omega(\Lambda)=0$, $D_aD_b\Omega(\Lambda)=2\tilde h_{ab}(\Lambda)$), and
define the STF tensors recursively,

$$P^{\,M,J}_{a_1\cdots a_{l+1}} = \mathcal C\Big[\tilde D_{a_{l+1}}P^{\,M,J}_{a_1\cdots a_l} - \tfrac12 l(2l-1)\tilde R_{a_la_{l+1}}P^{\,M,J}_{a_1\cdots a_{l-1}}\Big],$$

with $\mathcal C$ the symmetric trace-free part and $\tilde R$ the Ricci tensor of $\tilde h$.
The moments are the values at $\Lambda$,

$$M_{a_1\cdots a_l} = \frac{P^M_{a_1\cdots a_l}(\Lambda)}{(2l-1)!!},\qquad S_{a_1\cdots a_l} = \frac{l+1}{2l\,(2l-1)!!}P^J_{a_1\cdots a_l}(\Lambda)$$

[N1 normalisation; in N2 the prefactors are absent].

Three features matter more than the formulae:

* the **Ricci term is not optional**: it is what makes the moments transform correctly (by a
  shift of origin) when $\Omega$ is changed; dropping it gives a different, origin-dependent
  answer [Ger70; May22, §2.1];
* the moments are **geometric invariants** of the asymptotically flat vacuum metric, up to the
  origin choice;
* smoothness of $\tilde\Phi_{M,J}$ at $\Lambda$ is a theorem for stationary vacuum solutions
  (Hansen; see also the analyticity results of Beig–Simon and Kundu), and it can fail for
  metrics that fall off too slowly or have logarithmic behaviour at infinity — in which case the
  moments are not defined at all.  (The asymptotic-flatness requirement itself can be relaxed to
  include a NUT parameter [May22, footnote 1].)  This is a real restriction on what a
  finite-radius numerical solve can claim: the moment is defined at $\Lambda$, not at
  $\rho_{\rm out}$.

**Static specialisation.**  $\xi$ is hypersurface orthogonal, so $\omega_\alpha=0$ and

$$\Phi_J=0 \;\Longrightarrow\; S_{a_1\cdots a_l}=0 \ \ \forall l, \qquad \Phi_M = \frac{\lambda^2-1}{4\lambda} = \frac14\big(\lambda-\lambda^{-1}\big).$$

Note that Geroch's 1970 static formulation is usually written with the potential
$\tilde\psi = \Omega^{-1/2}\big(1-(-\xi^a\xi_a)^{1/2}\big) = \Omega^{-1/2}(1-e^{U})$ [Gür14,
eq. (4)], which is *not* the same function of $\lambda$ as $\Phi_M$ above (they agree only to
linear order in $U$).  Different papers use different (non-linearly related) static seed
potentials; the closed forms of section 4 are the safe, checkable statement of the outcome.

**Weak-field limit.**  With the STF conventions of [CGu16, App.] the tensors satisfy
$M^{\langle a_1\cdots a_l\rangle}=(2l-1)!!\,I^{\langle a_1\cdots a_l\rangle}$; equivalently the *scalar* moments obey
$M_l\to I_l$ in N1 (section 1.4).  This is the anchor for the normalisation, and it is what fixes
the factor in front of the metric expansion in section 3.1.

---

## 3. Thorne's ACMC definition, and its equivalence to GH

### 3.1 The definition

Thorne (1980) defines the moments as the coefficients of the metric in *asymptotically Cartesian
and mass-centred* (ACMC) coordinates: coordinates that are asymptotically Cartesian, whose origin
is the centre of mass, and in which at order $r^{-(l+1)}$ only harmonics with $l'\le l$ appear.
In such coordinates [Tho80; CGu16, eqs. (6.13)–(6.15)]

$$g_{00} = -1+\frac{2M}{r} + \sum_{l\ge2}\frac{1}{r^{l+1}}\Big(\frac{2}{l!}M^{\langle a_1\cdots a_l\rangle}n_{a_1}\cdots n_{a_l} +(\text{lower harmonics})\Big),$$

$$g_{0j} = -2\sum_{l\ge1}\frac{1}{r^{l+1}}\Big(\frac{1}{l!}\epsilon_{jka_l}S^{\langle ka_1\cdots a_{l-1}\rangle} n^{\langle a_1}\cdots n^{a_l\rangle}+(\text{lower harmonics})\Big),$$

which for an axisymmetric configuration with axis unit vector $\hat k$ reduces to

$$g_{00} = -1+\frac{2M}{r}+2\sum_{l\ge2}\frac{M_lP_l(\cos\theta)}{r^{l+1}} +(\text{lower harmonics}),\qquad g_{0j}\ \text{built from } S_l .$$

In de Donder coordinates the moments are source integrals,

$$M^{\langle a_1\cdots a_l\rangle}=(2l-1)!!\!\int\tau^{00}x^{\langle a_1}\cdots x^{a_l\rangle}d^3x,\qquad M_l=\int\tau^{00}r^lP_l(\cos\theta)\,d^3x$$

[CGu16, eqs. (6.16)–(6.17)], i.e. precisely the Newtonian expressions with the effective
stress–energy $\tau^{\mu\nu}$ — the operational statement of the normalisation.

**Caveat, and a typo in the review.**  In [CGu16] eq. (6.13) carries the factor $2$ and
eq. (6.15) drops it.  The Newtonian limit resolves it: $g_{00}=-1-2\Phi_N$ and
$\Phi_N=-\sum_l I_lP_l/r^{l+1}$ give $g_{00}=-1+2M/r+2\sum_{l\ge2}I_lP_l/r^{l+1}$, so the
coefficient is $2$.  Used consistently, this is what made the Curzon check of section 1.3 close.

### 3.2 Gürsel's equivalence

Gürsel (1983) proved that, whenever both are defined, the GH and Thorne moments agree [Gür83;
May22, §2.3].  Mayerson (2022) extended both formalisms and the equivalence to stationary,
*non-vacuum* spacetimes, under a mild topological condition and a "gauge fixing" of an improved
twist vector; he also notes that Thorne's ACMC formalism may be marginally more general
(existence of ACMC coordinates vs. smoothness at $\Lambda$).

For this repo this means: for a static vacuum solution, whichever route is implemented, the
target is one set of numbers — the $M_l$ of section 1.4, N1.

### 3.3 Pin the normalisation with four checks

1. **Schwarzschild**: $M_0=M$, $M_l=0$ for $l\ge1$ (all normalisations).
2. **Kerr** (normalisation only): $M_l+\mathrm i\,S_l=M(\mathrm ia)^l$, so $M_0=M$, $M_1=0$,
   $S_1=Ma$, $M_2=-Ma^2$, $S_2=0$, $M_3=0$, $S_3=-Ma^3,\dots$ — already planned as test #8 in
   `ROTATING_PLAN.md`.
3. **Newtonian limit**: for a weakly gravitating, slowly varying source, $M_l\to I_l$, the
   ordinary $M_l=\int\mu r^lP_l\,d^3x$.
4. **Curzon**: $M_2=-M^3/3$ (N1) — the cheapest non-trivial non-linear check, and the one that
   distinguishes N1 from N2 and catches a missing cubic term.

---

## 4. What static buys you: usable formulae

### 4.1 The static Weyl class in this repo's variables

Static axisymmetric vacuum, in canonical Weyl coordinates,

$$ds^2 = -e^{2U}dt^2+e^{-2U}\big[e^{2k}(d\rho^2+dz^2)+\rho^2d\varphi^2\big],$$

with $U$ axisymmetric and flat-harmonic, $\Delta_{\rm flat}U=0$, and $k$ fixed by
$k_{,\rho}=\rho(U_{,\rho}^2-U_{,z}^2)$, $k_{,z}=2\rho U_{,\rho}U_{,z}$, $k\to0$ at infinity
(`weyl.py`; [BG24, eqs. (10)–(14)]).  In these coordinates $\lambda=e^{2U}$, the orbit 3-metric
is $h=e^{2k}(d\rho^2+dz^2)+\rho^2d\varphi^2$, and the only non-trivial field equation is
Laplace's for $U$.  This is the class for which the moments have closed forms.

Write the axis values as

$$U(\rho=0,z)=\sum_{r\ge0}\frac{U^{(r)}}{\lvert z\rvert^{r+1}} .$$

### 4.2 FHPS/Gürlebeck: moments from the axis coefficients

The result quoted by Gürlebeck [Gür14, eq. (8)], attributing it to Fodor–Hoenselaers–Perjés
[FHP89] (who obtained the moments from the Ernst potential on the axis, with explicit expressions
up to $l=10$ in that paper), is

$$M_0=-U^{(0)},\quad M_1=-U^{(1)},\quad M_2=\tfrac13(U^{(0)})^3-U^{(2)},\quad M_3=(U^{(0)})^2U^{(1)}-U^{(3)}$$

in the N1 normalisation (section 3.3 fixes it).  (The published axis expansion is printed with
the sum starting at $r=1$; that is a typo in the source — $r=0$ is required for $U^{(0)}$ to
exist and is the reading used in Gürlebeck's own source integrals.)  Structure worth remembering:

* $M_l = -U^{(l)}$ for $l\le1$, and the relation is **already non-linear at $l=2$**;
* writing $I_2:=-U^{(2)}$ (the quadrupole coefficient of $U$ itself) and $U^{(0)}=-M_0$,
  $M_2 = I_2-\tfrac13M_0^3$: the moment is the Newtonian-type coefficient *minus* a monopole
  cubed term, which is the whole of the Curzon effect;
* the moments are *not* the $U^{(r)}$, and they are *not* the $S_{lm}$ of $\lambda$; even for a
  weak source the normalisation difference is a factor (section 3.3);
* the map is **linear only for the mass and the mass dipole**: [Gür14, §3] stresses that from
  $l=2$ on the passage $U^{(l)}\to M_l$ is non-linear, so the moments of individual sources do
  not superpose.  A two-rod solution is not the sum of two Schwarzschild multipole series, and a
  distorted black hole plus an external field is not the sum of their separate series either;
* the map is known explicitly only up to a finite order ($l=10$ in [FHP89]); beyond that one
  needs the recurrence itself.  For a solver that only wants $l\le4$–$6$, the quoted relations
  plus the recursion are enough.

### 4.3 The same statement from the Ernst potential (and where the first correction is)

Sotiriou–Apostolatos [SA04] compute the first five moments of a stationary axisymmetric vacuum
from the power-series coefficients of the Ernst potential on the axis and correct errors in the
earlier literature.  In their notation the static Ernst potential is
$\xi=(1-F)/(1+F)$ with $F=e^{2U}=\lambda$, i.e. $\xi=-\tanh U$ in this repo's variables, and with
$m_i$ the coefficients of $\tilde\xi=\Omega^{-1/2}\xi=\sum a_{ij}\bar\rho^i\bar z^j$ on the axis
($m_i\equiv a_{0i}$), their eq. (24) gives, with no electromagnetic field,

$$P_0=m_0,\quad P_1=m_1,\quad P_2=m_2,\quad P_3=m_3,\qquad P_4 = m_4-\tfrac17 m_0\big(m_0m_2-m_1^2\big),$$

$$P_5 = m_5-\tfrac13 m_0\big(m_0m_3-m_1m_2\big) -\tfrac1{21}m_1\big(m_0m_2-m_1^2\big),\ \dots$$

Two facts follow, both useful as diagnostics:

* the axis coefficients of the **Ernst** potential equal the moments for $l\le3$, and the first
  correction is at $l=4$; whereas the axis coefficients of the **Weyl** potential $U$ already
  differ at $l=2$ (the cubic term).  Which potential you expand determines where the
  non-linearity bites;
* the recursive algorithm is explicit and testable (it is what a `gh_moments` implementation
  should reproduce); [SA04] also stresses dimensional consistency — $m_i$ has dimension
  $[\text{length}]^{i+1}$ — which is a cheap bug-catcher.

### 4.4 Finite-radius routes (for numerical solutions)

Three tools exist for obtaining true moments from a metric that is only known numerically in a
bounded region.

#### 4.4.1 Gürlebeck's source integrals and quasi-local line integral (Gürlebeck 2014)

This is the most directly implementable of the three.  With the scalars $W=\rho$ (Weyl radius)
and $Z=\zeta$ (axis potential, the potential conjugate to $W$), continued into the interior, and
$U=\tfrac12\log\lambda$, the *Weyl* coefficients are line integrals over a meridian curve
$\gamma_{\mathcal B}$ running from the north to the south pole of any surface $\mathcal S$
enclosing all sources:

$$U^{(r)}=\frac14\int_{\gamma_{\mathcal B}}\big(N_+^{(r)}U_{,A}\hat s^A+N_-^{(r)}U_{,A}\hat n^A\big)\,d\gamma,$$

with $\hat s^A,\hat n^A$ the unit tangent and normal to $\gamma_{\mathcal B}$ and

$$N_-^{(r)}=\sum_{k=0}^{\lfloor r/2\rfloor}\frac{2(-1)^{k+1}r!\,\rho^{2k+1}\zeta^{\,r-2k}}{4^k(k!)^2(r-2k)!},\qquad N_+^{(r)}=\sum_{k=0}^{\lfloor (r-1)/2\rfloor}\frac{2(-1)^{k+1}r!\,\rho^{2k+2}\zeta^{\,r-2k-1}}{4^k(k!)^2(r-2k-1)!(2k+2)}$$

[Gür14, eqs. (16)–(17)].  There are equivalent surface and volume forms,

$$U^{(r)}=\frac{1}{8\pi}\int_{\mathcal S}\frac{e^U}{W}\Big(N_-^{(r)}U_{,\hat n}-N^{(r)}_{+,W}Z_{,\hat n}U+N^{(r)}_{+,Z}W_{,\hat n}U\Big)\,d\mathcal S=\sum_i\big(U^{(r)}_i+U^{(r)H}_i\big),$$

[Gür14, eqs. (21), (23)], the volume form localising the contribution of each source (matter in
$\mathcal V_i$, horizons through $\mathcal S_i^{\mathcal H}$), with the black-hole part exactly a
Schwarzschild series [Gür15].  The $U^{(r)}$ are then converted to $M_l$ through the non-linear
relations of section 4.2 — contributions superpose in the $U^{(r)}$ but **not** in the $M_l$.
For this repo these integrals are the natural bridge from the chart the network works in to
moment values, and checking that they are radius-independent once all sources are enclosed is a
direct error bar on the result.

#### 4.4.2 Relativistic generalised Gauss theorem (Hernández-Pastora et al. 2016)

For static spacetimes the moment can be written as a flux integral over a sphere.  The Newtonian
version is

$$M_n = \frac{1}{4\pi}\oint\Big\{r^nP_n(\cos\theta)\,\partial_k\Phi - \Phi\,\partial_k\big[r^nP_n(\cos\theta)\big]\Big\}d\sigma^k$$

[HP16, eqs. (16), (22)]; the relativistic version uses the conformal (quotient) metric and the
potential $\log\xi$ (their eqs. (75)–(81)), and reproduces the Thorne moments.  Crucially, this
holds **only in asymptotically Cartesian harmonic coordinates**, and this is not a technicality:
in canonical Weyl spherical coordinates the same integral returns only the fraction
$\frac{n+1}{2n+1}a_n$ of the Weyl coefficient, while using their conformal metric returns $-a_n$
[HP16, §5.3(B), eqs. (83)–(84)].  The paper's message is precisely that a "moment integral" is
meaningful only in the right coordinates.

#### 4.4.3 ACMC extraction

Fit the metric to the ACMC form of section 3.1, solving order by order for the coordinate
transformation that removes the non-ACMC pieces (this is what numerical-relativity codes do to
measure $M_l$).  It requires an asymptotic expansion, so it is limited by the outer boundary
condition; it does not need axisymmetry.  For the practical size of the coordinate correction on
a numerical metric see [PA12]: in quasi-isotropic coordinates the naive quadrupole is shifted,
$M_2^{\rm GH}=M_2^{\rm naive}-\frac43(\frac14+b)M^3$, where $b$ is a metric coefficient that
vanishes for $b=-\frac14$ — a useful estimate of how large a chart-induced error a moment
extraction can carry.

For *this* repo the ranking is: 4.4.1/4.4.2 for a solver that wants invariants from a bounded
region; the FHPS axis formula for anything axisymmetric that is already in canonical Weyl
coordinates; 4.4.3 only when a full non-axisymmetric extraction is really needed.

### 4.5 Uniqueness and rigidity (why the moments are worth computing)

For static vacuum spacetimes the moments are not just diagnostics, they are coordinates on the
solution space:

* **Beig–Simon** proved Geroch's conjecture in the static case: two static solutions with the same
  multipole moments are identical (at least in a neighbourhood of infinity) [BS80; Liu, §4.5].
  Kundu extended the local uniqueness to stationary vacuum metrics.
* **Xanthopoulos** proved that a static spacetime is flat **iff** all its multipole moments
  vanish; and that a stationary spacetime is static iff all its angular moments vanish
  [Xan79].
* **Gürsel**: a stationary spacetime is axisymmetric iff all its moments are axisymmetric.
* Combined with Israel (and Bunting–Masood / Robinson) — the only static vacuum black hole is
  Schwarzschild — a static vacuum black hole has $M_l=0$ for all $l\ge1$: the "no-hair" statement
  in multipole language.
* **Gürlebeck 2015** sharpens this for a *distorted* static axisymmetric black hole in an external
  field: the hole contributes exactly a Schwarzschild multipole series to the asymptotic field;
  external matter contributes additively [Gür14, Gür15; BG24, §I and §IV].  This is directly
  relevant to the repo's two-rod (Israel–Khan) assets: the strut between the rods is a
  distributional source, so the moments are not those of two Schwarzschild holes.

---

## 5. Calibration table (with the arithmetic)

Throughout, N1 normalisation and $\lambda=e^{2U}$.

### 5.1 Flat space

$U\equiv0$: all $M_l=0$.  Xanthopoulos's theorem is the converse (section 4.5).

### 5.2 A single rod = Schwarzschild

A rod spanning $z\in[a,b]$, mass $m=(b-a)/2$, has [Gür14, §2.1; `weyl.py`'s `U_of`]

$$U(0,z>\max b) = \tfrac12\ln\frac{z-b}{z-a} \;\Longrightarrow\; U^{(r)} = -\frac{1}{2(r+1)}\big(b^{\,r+1}-a^{\,r+1}\big),$$

and for the symmetric rod $[-m,m]$ this gives $U^{(0)}=-m$, $U^{(2)}=-m^3/3$,
$U^{(4)}=-m^5/5$, all odd-index $U^{(r)}=0$.  Hence

$$M_0=-U^{(0)}=m,\qquad M_2=\tfrac13(-m)^3-\big(-\tfrac{m^3}{3}\big)=0,$$

and likewise $M_l=0$ for $l\ge1$: the rod is Schwarzschild.  This is the check that catches
sign and factor errors in $M_2$.

### 5.3 The repo's spherical asset

`exact.exact_fields(R0,k)` is, in the vacuum reading, Schwarzschild of mass $M=R_0$ in harmonic
coordinates (`geometry_invariants.py` docstring, verified there against the Kretschmann scalar
and the areal radius).  Therefore, exactly,

$$M_0 = R_0,\qquad M_l = 0\ \ (l\ge1).$$

Note what the existing diagnostic does on the same field.  Here the areal radius is
$r_a=\rho+R_0$ (vacuum reading), while the chart radius is $\rho$, and
$\lambda = k(1-2R_0/r_a)$; hence in the chart

$$\lambda-k = -2kR_0\frac{\rho}{\rho+R_0}\cdot\frac1{\rho} = -\frac{2kR_0}{\rho}+\frac{2kR_0^2}{\rho^2}-\dots$$

The $l=0$ coefficient that `multipoles.py` reports is therefore
$S_{00} = \sqrt{4\pi}\,\rho\,(\lambda-k) = -2\sqrt{4\pi}\,kR_0\,\dfrac{\rho}{\rho+R_0}$,
which tends to $-2\sqrt{4\pi}\,kR_0$ only as $\rho\to\infty$: asymptotically
$S_{00}/M_0=-2\sqrt{4\pi}\,k$, not $1$, and at finite radius the ratio also drifts like
$\rho/(\rho+R_0)$ (the $1/\rho^2$ term leaking into a $1/\rho$ coefficient, which is the drift
recorded in `NEXT.md`).  This is the cleanest single illustration that $S_{lm}\ne M_l$ — the
mismatch here is not non-linearity but the chart radius versus the areal radius, i.e. exactly the
ambiguity ACMC coordinates are designed to remove.

### 5.4 Israel–Khan: two rods, closed form

For rods $[a_i,b_i]$ (each of mass $m_i=(b_i-a_i)/2$), the same expansion gives

$$U^{(r)} = -\frac{1}{2(r+1)}\sum_i\big(b_i^{\,r+1}-a_i^{\,r+1}\big),$$

so with the FHPS relations

$$M_0=\sum_i m_i\ \ (\text{total mass}),\qquad M_1=\tfrac14\sum_i\big(b_i^2-a_i^2\big)\ \ (\text{centre of mass}),$$

$$M_2 = -\tfrac13M_0^3+\tfrac16\sum_i\big(b_i^3-a_i^3\big),\qquad M_3 = -M_0^2\sum_i\tfrac14\big(b_i^2-a_i^2\big) +\tfrac18\sum_i\big(b_i^4-a_i^4\big).$$

For the default symmetric pair `Rods.symmetric(half_length=1.0, half_gap=0.5)`, i.e. rods
$[-2.5,-0.5]$ and $[0.5,2.5]$ (masses $1,1$): $\sum(b^3-a^3)=31$, so

$$M_0=2,\quad M_1=0,\quad M_2=-\tfrac83+\tfrac{31}{6}=\tfrac{15}{6}=2.5,\quad M_3=0 .$$

The Newtonian rod quadrupole is $I_2=\sum_i\frac16(b_i^3-a_i^3)=\frac{31}{6}\approx5.167$, and
$M_2=I_2-\frac13M_0^3=5.167-2.667=2.5$ — the same identity as in section 4.2.  (If the code's
asset uses different masses, only the numbers change; the formula does not.)

The numbers above, and the claim of section 4.3 that the **Ernst** axis coefficients equal the
moments for $l\le3$, were re-derived here independently: expanding
$\xi=(1-\lambda)/(1+\lambda)$ with $\lambda=\prod_i(z-b_i)/(z-a_i)$ in $u=1/z$ and multiplying by
$z$ (which is what $\Omega^{-1/2}$ does on the axis, since $R=z$ and $\bar z=1/z$) gives
$m_0,m_1,m_2,m_3$, and they agree with the FHPS values for the single rod, the default symmetric
pair and an unequal pair (e.g. rods $[-4,-2],[1,3]$: both routes give $M=(2,-1,11,-16,\dots)$).
That is a useful cross-check to build into the tests: two algebraically different routes, same
numbers.

### 5.5 Curzon

$U=-M/R$ on the axis: $U^{(0)}=-M$, $U^{(r)}=0$ for $r\ge1$.  Hence

$$M_0=M,\qquad M_2=-\tfrac13M^3,$$

in agreement with the three asymptotic expansions cited in section 1.3.  This is the non-linear
check that a "read the coefficient" implementation fails.

**The confusion to expect.**  Curzon has *no Weyl multipoles* — its Weyl coefficients are
$a_0=-M$, $a_n=0$ ($n\ge1$) — while it *does* have GH moments.  Statements in the literature
that "the Curzon metric has no multipole moments", or "$M_2=0$", are about the Weyl/source
coefficients $a_n$, not the GH moments [MS21; BH05; CCLD21]; the Weyl monopole is the Curzon
solution, which is not Schwarzschild.  Whenever a claimed moment disagrees, check which of the
two objects is meant and in which normalisation (section 1.4).

### 5.6 Kerr (normalisation anchor only; not static)

$$M_l+\mathrm i\,S_l = M(\mathrm ia)^l$$

equivalently $M_l=M(-a^2)^{l/2}$ for even $l$ and $M_l=0$ for odd $l$, while $S_l=0$ for even
$l$ and $S_l=\mathrm{Im}\,[M(\mathrm ia)^l]$ for odd $l$.  Hence $M_0=M$, $M_2=-Ma^2$,
$M_4=Ma^4$, and $S_1=Ma$, $S_3=-Ma^3$, $S_5=Ma^5$.  The case $a=0$
is the static Schwarzschild of section 5.2 (only $M_0$).  (Mayerson's introduction states this
identity but its inline component formula for $S_l$ has a shifted index; use the complex
identity.)

---

## 6. What this means for this repo

### 6.1 The current estimator

`stationary/multipoles.py` computes, for a field on a coordinate sphere of chart radius $\rho$,
the coefficients $a_{lm}(\rho)=\int(\lambda-\lambda_\infty)Y_{lm}\,d\Omega$ and reports
$S_{lm}=\rho_{\rm phys}^{l+1}a_{lm}$.  As the tests pin it, this is exactly the constant in front
of $Y_{lm}/\rho_{\rm phys}^{l+1}$ for a *pure tail*, and the drift of $S_{lm}$ over a range of
radii is a genuine detector of a wrong decay power.  That is a legitimate purpose, and the module
should keep doing it.

It should not be labelled "multipoles", and it cannot be compared with inner data as if $S_1$
resp. $S_2$ were the dipole/quadrupole moments.  Concretely, all four failure modes of section 1
apply:

* chart dependence (the harmonic chart's radial coordinate is not an ACMC radius; the
  $1/\rho^2$ term in the spherical asset leaks into $S_{00}$);
* non-linearity (the same field's $M_2$ contains $-\frac13M_0^3$, absent from any linear reading
  of $\lambda$);
* origin gauge (an imposed $S_1$ is a centre-of-mass shift, which GH absorb into the definition
  of the origin);
* normalisation: even in the weak-field limit, and even in coordinates where $\lambda$'s tail
  is a pure multipole series, the two objects differ by an $l$-dependent factor.  With
  $\lambda-1 = 2\Phi_N$ and $\Phi_N=-\sum_l M_lP_l/r^{l+1}$ one has
  $S_{l0} = -4M_l/\sqrt{4\pi(2l+1)}$ for the leading $m=0$ component, so e.g. $M_0$ is not
  $S_{00}$ (and section 5.3 shows the mismatch is much worse when the chart radius is not the
  areal radius).  In the non-linear regime there is no fixed factor at all.

Suggested minimal rename/documentation change: call them *harmonic tail coefficients*
($h_{lm}$, say), keep the tests, and state in the report that they are not GH moments.  The
existing `NEXT.md` table (network vs exact vs inner data) remains exactly as useful under the new
name.

### 6.2 A GH-moment module: what to implement, in order

1. **Exact axisymmetric assets (cheap, closed form).**  For any `Rods` configuration, compute
   $U^{(r)}$ from $U^{(r)}=-\frac{1}{2(r+1)}\sum_i(b_i^{r+1}-a_i^{r+1})$ and apply the FHPS
   relations (extend as needed with the FHP89 recursion for $l\ge4$).  Tests: Schwarzschild rod
   ($M_l=0$, $l\ge1$), the default two-rod numbers of section 5.4, Curzon (constructed by
   setting `U_of` to $-M/R$), and flat space.
2. **Spherical asset.**  In the **vacuum** reading, $M_0=R_0$, $M_l=0$ for $l\ge1$.  In the
   **scalar** reading ($g=-dt^2+h$) the four-metric is not a stationary vacuum metric, so GH
   moments are not defined for it; the geometry layer must always state which reading a reported
   multipole belongs to, as `geometry_invariants.py` already does with `reading=`.
3. **Network solutions.**  Two honest options:
   * for axisymmetric runs, extract $U$ on the axis in canonical Weyl coordinates — which
     requires constructing those coordinates — and apply FHPS.  This is the most accurate and
     the cheapest at high $l$, but it is a real piece of geometry work (the network lives in a
     harmonic chart, not in Weyl coordinates);
   * otherwise use the finite-radius integrals of section 4.4 — Gürlebeck's quasi-local line
     integral in the scalars $(W,Z)$, which is the one that does not require constructing
     canonical Weyl coordinates globally, or the Hernández-Pastora flux integral in asymptotically
     Cartesian harmonic coordinates — and *measure* the radius dependence: for a true solution of
     the static vacuum equations in the right coordinates it should be radius independent once the
     source is enclosed; the residual dependence is a direct, catchable error bar on the moment.
     This fits the existing `geometry_invariants.py` style (surface integrals over spheres).
4. **Rotating runs (P3 and later).**  Keep the Kerr identity of section 3.3 as the acceptance
   test; the twist potential and $\Phi_J$ from `ROTATING_PLAN.md` §2 are exactly the input the
   GH current moments need.
5. **Report provenance, not just numbers.**  Expose one entry point per route and return, besides
   $M_l$, the route used (closed-form axis, flux integral, or ACMC fit), the curve/radius/surface
   at which an integral was taken, and the intermediate $U^{(r)}$ (or $m_i$) values.  Because the
   $U^{(r)}\to M_l$ map is non-linear from $l=2$ on (§4.2), a wrong axis coefficient can hide
   inside an apparently reasonable $M_l$; the intermediates are what make the number auditable.

### 6.3 Caveats specific to this codebase

* **Finite outer boundary.**  GH moments live at $\Lambda$.  The repo imposes Robin conditions at
  $\rho_{\rm out}=100$–$200$; the $l=0$ "drift" already documented is $\sim40\%$ for the spherical
  asset and is an expansion effect, not an error in the estimator.  A moment extracted by an
  asymptotic fit inherits that; a flux integral over an inner surface does not (at the price of
  needing the interior field to be trustworthy).
* **Rods are horizons, not matter.**  The Weyl rods are degenerate surfaces at $\rho=0$; the
  string/strut between rods is a distributional source.  A moment computed by integrating over a
  surface that does not enclose all of the axis singularity will be wrong (Gürlebeck's source
  integrals assume a surface enclosing *all* sources, matter and singularities).  The
  Israel–Khan moments of section 5.4 are the target for the two-rod asset, not "two times
  Schwarzschild".
* **Rotated configurations.**  `weyl.rotated_fields` is the same solution about a rotated axis.
  Its GH moment *tensors* are those of the unrotated solution carried by the rotation, so the
  scalar moments about the symmetry axis are unchanged; the axis/FHPS route still applies once
  that axis is identified from the axial Killing vector, whereas a coordinate $z$-axis-based
  extraction would silently mix components.
* **Chart factor.**  Convert with $M_l^{\rm phys}=(\text{factor})^{l+1}M_l^{\rm chart}$, exactly
  as `multipole_constants` already does for $S_{lm}$ — but note that this is only the trivial
  rescaling part of the chart dependence (section 1.2).
* **Non-axisymmetric static runs.**  The FHPS axis route and Gürlebeck's source integrals assume
  axisymmetry.  For a general static solution use ACMC extraction, or the STF form of the flux
  integral [HP16, eq. (57)] in asymptotically Cartesian harmonic coordinates.  Do not fall back on
  sphere coefficients.

---

## 7. Sources

Labels in brackets are the ones used in the text.

Primary definitions and reviews:

* **[Ger70]** R. Geroch, *Multipole moments. II. Curved space*, J. Math. Phys. **11**, 2580 (1970) —
  static construction.  <https://pubs.aip.org/aip/jmp/article-abstract/11/8/2580/388057>
* **[Han74]** R. O. Hansen, *Multipole moments of stationary space-times*, J. Math. Phys. **15**,
  46 (1974).
* **[Tho80]** K. S. Thorne, *Multipole expansions of gravitational radiation*, Rev. Mod. Phys.
  **52**, 299 (1980) — ACMC definition.  <https://inspirehep.net/literature/158081>
* **[Gür83]** Y. Gürsel, *Multipole moments for stationary systems: the equivalence of the
  Geroch–Hansen formulation and the Thorne formulation*, Gen. Rel. Grav. **15**, 737 (1983).
  <https://ui.adsabs.harvard.edu/abs/1983GReGr..15..737G/abstract>
* **[CGu16]** V. Cardoso, L. Gualtieri, *Testing the black hole "no-hair" hypothesis*, Class.
  Quantum Grav. **33**, 174001 (2016), arXiv:1607.03133 — appendix with both constructions in one
  notation (note the dropped factor $2$ in its eq. (6.15)).
  <https://ar5iv.labs.arxiv.org/html/1607.03133>
* **[May22]** D. R. Mayerson, *Gravitational multipoles in general stationary spacetimes*,
  arXiv:2210.05687 — careful statement of the GH recursion, normalisations, and the GH/Thorne
  equivalence.  <https://ar5iv.labs.arxiv.org/html/2210.05687>

Static case, explicit formulae and source integrals:

* **[FHP89]** G. Fodor, C. Hoenselaers, Z. Perjés, *Multipole moments of axisymmetric systems in
  relativity*, J. Math. Phys. **30**, 2252 (1989), DOI 10.1063/1.528551 — moments from the Ernst
  potential on the axis, explicit to $l=10$.
* **[Gür14]** N. Gürlebeck, *Source integrals for multipole moments in static and axially
  symmetric spacetimes*, Phys. Rev. D **90**, 024041 (2014), arXiv:1207.4500 — the
  $\sum_r U^{(r)}$ relations, eq. (8) (the sum is printed from $r=1$; $r=0$ is intended),
  quasi-local line integrals, and finite-region source integrals.
  <https://ar5iv.labs.arxiv.org/html/1207.4500>
* **[Gür15]** N. Gürlebeck, Phys. Rev. Lett. **114**, 151102 (2015), arXiv:1503.03240 — black
  holes in external fields; the hole contributes a Schwarzschild series.
* **[SA04]** T. P. Sotiriou, T. A. Apostolatos, *Corrections and comments on the multipole moments
  of axisymmetric electrovacuum spacetimes*, arXiv:gr-qc/0407064 — the explicit $P_4$, $P_5$
  corrections and the static reduction (the static Ernst potential is
  $\xi=(1-\lambda)/(1+\lambda)$).  <https://ar5iv.labs.arxiv.org/html/gr-qc/0407064>
* **[HP16]** J. L. Hernández-Pastora, J. Martín-Martín, E. Ruiz, *Source integrals of multipole
  moments for static space-times*, arXiv:1604.07192 — generalised Gauss theorem, flux integrals,
  and the coordinate-dependence demonstration.
  <https://ar5iv.labs.arxiv.org/html/1604.07192>
* **[PA12]** G. Pappas, T. A. Apostolatos, *Revising the multipole moments of numerical
  spacetimes, and its consequences*, Phys. Rev. Lett. **108**, 231104 (2012), arXiv:1201.6067 —
  the coordinate-reading correction to a naive $M_2$.
* **[ABMMR01]** J. M. Aguirregabiria, L. Bel, J. Martín, A. Molina, E. Ruiz, *Comparing metrics at
  large: harmonic vs quo-harmonic coordinates*, Gen. Rel. Grav. **33**, 1809 (2001),
  arXiv:gr-qc/0104019 — Curzon expansion to $P_4$.
* **[Gu10]** Y.-Q. Gu, *The series solution to the metric of stationary vacuum with axisymmetry*,
  Chinese Phys. B **19**, 030402 (2010), arXiv:0811.0449 — another Curzon expansion.
* **[MC15]** P. Montero-Camacho, F. Frutos-Alfaro, C. Gutiérrez-Chaves, I. Cordero-García, *Slowly
  rotating Curzon–Chazy metric*, Rev. Mat. Teor. Apl. **22**(2), 265 (2015), arXiv:1405.2899 — the
  asymptotic expansion used above.  <https://www.redalyc.org/pdf/453/45341139004.pdf>
* **[BG24]** C. Barceló, R. Carballo-Rubio, L. J. Garay, G. García-Moreno, *No-hair and
  almost-no-hair results for static axisymmetric black holes...*, Class. Quantum Grav. **42**,
  075020 (2025), arXiv:2410.08128 — Weyl form, multipolar bookkeeping, Curzon appendix.
  <https://ar5iv.labs.arxiv.org/html/2410.08128>
* **[MS21]** D. Malafarina, S. Sagynbayeva, *What a difference a quadrupole makes?*, Gen. Rel.
  Grav. **53**, 112 (2021), arXiv:2009.12839 — Weyl coefficients vs gravitational mass multipoles.
* **[BH05]** T. Bäckdahl, M. Herberthson, *Static axisymmetric space-times with prescribed
  multipole moments*, Class. Quantum Grav. **22**, 1607 (2005), arXiv:gr-qc/0502012.
* **[CCLD21]** Y.-Z. Chen, Y.-J. Chen, S.-L. Li, W.-S. Dai, arXiv:2110.09725 — source multipole
  integrals vs field multipole moments for Curzon.

Uniqueness/injectivity:

* **[BS80]** R. Beig, W. Simon, *Proof of a multipole conjecture due to Geroch*, Commun. Math.
  Phys. **78**, 75 (1980), DOI 10.1007/BF01941970 — uniqueness of static asymptotically flat
  vacuum solutions from their GH multipole moments.
  <https://ui.adsabs.harvard.edu/abs/1980CMaPh..78...75B/abstract>
* **[Xan79]** B. C. Xanthopoulos, *Multipole moments in general relativity*, J. Phys. A **12**,
  1025 (1979), DOI 10.1088/0305-4470/12/7/018 — "a stationary space-time for which all the angular
  momentum multipole moments vanish is static and ... a static space-time for which all the mass
  multipole moments vanish is flat".
* **[Isr67]** W. Israel, *Event horizons in static vacuum space-times*, Phys. Rev. **164**, 1776
  (1967), DOI 10.1103/PhysRev.164.1776 — Schwarzschild uniqueness; the vanishing of the higher
  moments of a static vacuum black hole is a corollary of it, not a statement in that paper.
* **[Liu]** Further statements and references: the introduction and §4.5 of
  <https://liu.diva-portal.org/smash/get/diva2:17945/FULLTEXT01.pdf>.

---

### Appendix: summary of the checks to add

| test | expected |
|---|---|
| spherical asset (vacuum reading) | $M_0=R_0$, $M_l=0\ (l\ge1)$ |
| single rod $[-m,m]$ | $M_0=m$, $M_l=0\ (l\ge1)$ |
| default symmetric two-rod pair | $M_0=2$, $M_1=0$, $M_2=2.5$, $M_3=0$ |
| Curzon | $M_0=M$, $M_2=-M^3/3$ (reported: $M_4=19M^5/105$) |
| two routes agree, $l\le3$ | FHPS from $U^{(r)}$ $=$ Ernst axis coefficients $m_i$ |
| Weyl coefficients are not moments | Curzon $U^{(l)}=(-M,0,0,\dots)$ while $M_l=(M,0,-M^3/3,\dots)$ |
| static currents | $S_l=0$ identically for every static solution |
| Kerr (normalisation) | $M_l+\mathrm iS_l=M(\mathrm ia)^l$ |
| flat space | all $M_l=0$ (and conversely: Xanthopoulos) |
| origin gauge | a shift of origin changes $M_1$ by $M_0\times$shift and mixes higher moments; mass-centring sets $M_1=0$ |
| units | $M_l$ scales as $[\text{length}]^{l+1}$ |

The first five rows are the ones that catch essentially every implementation error; the two-route
row is the cheapest independent check (the closed-form FHPS map and the Ernst axis expansion are
algebraically unrelated), and it was used to produce the numbers in section 5.4.
