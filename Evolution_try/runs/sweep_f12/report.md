# sweep_f12

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `fourier` with 12 modes
* network: 6 layers x 20 neurons, tanh, 2641 parameters
* collocation: 2048 points, sampler `random`
* optimiser: `ssbroyden`

final training loss: **1.620056e-04**

space-time relative L2 error: **1.130837e+00**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 3.364978e-01 | 3.667324e-01 |
| 1 | 1.013043e+00 | 9.053136e-01 |
| 1.5 | 1.371095e+00 | 1.117771e+00 |
| 2 | 1.837974e+00 | 1.369165e+00 |

wall time: 270.1 s

* phase 1: `ssbroyden` in 2500 iterations, 259.1 s, stopped: budget
