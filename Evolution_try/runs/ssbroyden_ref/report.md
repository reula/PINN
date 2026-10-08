# ssbroyden_ref

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 4096 points, sampler `random`
* optimiser: `ssbroyden`

final training loss: **4.701286e-10**

space-time relative L2 error: **9.037811e-05**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 8.256721e-05 | 6.797041e-05 |
| 1 | 9.058054e-05 | 7.231606e-05 |
| 1.5 | 1.103772e-04 | 9.850387e-05 |
| 2 | 1.167695e-04 | 8.584348e-05 |

wall time: 4279.4 s

* phase 1: `ssbroyden` in 9753 iterations, 4268.1 s, stopped: budget
