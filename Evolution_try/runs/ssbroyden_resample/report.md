# ssbroyden_resample

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 2201 points, sampler `random`
* optimiser: `ssbroyden`

final training loss: **3.745856e-10**

space-time relative L2 error: **2.268752e-04**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 8.760042e-05 | 8.621995e-05 |
| 1 | 1.526613e-04 | 1.328890e-04 |
| 1.5 | 2.502725e-04 | 1.814484e-04 |
| 2 | 4.051472e-04 | 2.400075e-04 |

wall time: 1359.5 s

* phase 1: `ssbroyden` in 8000 iterations, 1355.8 s, stopped: budget
