# dsgnar_fresh

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 2201 points, sampler `random`
* optimiser: `dsgnar`

final training loss: **1.143737e-15**

space-time relative L2 error: **9.378705e-07**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 2.936756e-07 | 1.901071e-07 |
| 1 | 6.605411e-07 | 4.161580e-07 |
| 1.5 | 1.136084e-06 | 6.431821e-07 |
| 2 | 1.609665e-06 | 8.194894e-07 |

wall time: 1964.4 s

* phase 1: `dsgnar` in 206 iterations, 1959.5 s, stopped: radius below threshold
