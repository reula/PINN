# smoke_qn

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `fourier` with 6 modes
* network: 6 layers x 20 neurons, tanh, 2401 parameters
* collocation: 1024 points, sampler `uniform`
* optimiser: `ssbroyden`

final training loss: **1.457319e-03**

space-time relative L2 error: **5.187887e+00**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 1.416364e+00 | 1.541689e+00 |
| 1 | 3.107932e+00 | 3.401667e+00 |
| 1.5 | 5.934210e+00 | 5.165116e+00 |
| 2 | 9.377341e+00 | 7.116206e+00 |

wall time: 20.8 s

* phase 1: `ssbroyden` in 200 iterations, 16.4 s, stopped: budget
