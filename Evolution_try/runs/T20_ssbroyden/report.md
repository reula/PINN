# T20_ssbroyden

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 20]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 2201 points, sampler `random`
* optimiser: `ssbroyden`

final training loss: **6.976302e-12**

space-time relative L2 error: **9.072394e-01**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 2 | 9.917520e-01 | 9.941178e-01 |
| 4 | 9.825419e-01 | 9.874404e-01 |
| 6 | 9.738881e-01 | 9.810537e-01 |
| 8 | 9.618717e-01 | 9.719765e-01 |
| 10 | 9.511522e-01 | 9.636617e-01 |
| 12 | 9.430450e-01 | 9.572138e-01 |
| 14 | 9.367637e-01 | 9.521094e-01 |
| 16 | 9.306377e-01 | 9.470436e-01 |
| 18 | 9.235972e-01 | 9.411038e-01 |
| 20 | 9.168604e-01 | 9.352613e-01 |

wall time: 2510.7 s

* phase 1: `ssbroyden` in 19763 iterations, 2504.6 s, stopped: budget
