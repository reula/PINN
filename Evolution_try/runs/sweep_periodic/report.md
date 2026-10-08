# sweep_periodic

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 2048 points, sampler `random`
* optimiser: `ssbroyden`

final training loss: **1.552126e-08**

space-time relative L2 error: **9.958220e-04**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 8.300213e-04 | 6.901101e-04 |
| 1 | 9.474741e-04 | 9.256037e-04 |
| 1.5 | 1.344369e-03 | 1.080458e-03 |
| 2 | 1.251259e-03 | 1.321877e-03 |

wall time: 424.7 s

* phase 1: `ssbroyden` in 2253 iterations, 410.0 s, stopped: budget
