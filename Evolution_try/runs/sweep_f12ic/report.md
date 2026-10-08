# sweep_f12ic

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `fourier_ic` with 12 modes
* network: 6 layers x 20 neurons, tanh, 2681 parameters
* collocation: 2048 points, sampler `random`
* optimiser: `ssbroyden`

final training loss: **1.039019e-05**

space-time relative L2 error: **7.996934e-01**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 8.225818e-01 | 9.051667e-01 |
| 1 | 9.344713e-01 | 9.489294e-01 |
| 1.5 | 9.071701e-01 | 9.256994e-01 |
| 2 | 9.063169e-01 | 9.203165e-01 |

wall time: 337.0 s

* phase 1: `ssbroyden` in 2500 iterations, 329.4 s, stopped: budget
