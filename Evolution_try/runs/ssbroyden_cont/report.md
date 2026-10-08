# ssbroyden_cont

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 8192 points, sampler `random`
* optimiser: `ssbroyden`

final training loss: **3.957579e-10**

space-time relative L2 error: **7.805982e-05**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 6.142417e-05 | 5.034069e-05 |
| 1 | 7.984973e-05 | 6.124288e-05 |
| 1.5 | 8.828227e-05 | 7.639774e-05 |
| 2 | 1.118930e-04 | 1.005057e-04 |

wall time: 671.3 s

* phase 1: `ssbroyden` in 3000 iterations, 666.2 s, stopped: budget
