# T20_dsgnar

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 20]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 2201 points, sampler `random`
* optimiser: `dsgnar`

final training loss: **1.670181e-06**

space-time relative L2 error: **2.386464e+01**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 2 | 2.796062e+00 | 1.409840e+00 |
| 4 | 6.449588e+00 | 2.943014e+00 |
| 6 | 1.044022e+01 | 4.629618e+00 |
| 8 | 1.467480e+01 | 6.413064e+00 |
| 10 | 1.905587e+01 | 8.257350e+00 |
| 12 | 2.348527e+01 | 1.012426e+01 |
| 14 | 2.794048e+01 | 1.199807e+01 |
| 16 | 3.246272e+01 | 1.389657e+01 |
| 18 | 3.712302e+01 | 1.585581e+01 |
| 20 | 4.199604e+01 | 1.792393e+01 |

wall time: 3515.6 s

* phase 1: `dsgnar` in 97 iterations, 3507.1 s, stopped: radius below threshold
