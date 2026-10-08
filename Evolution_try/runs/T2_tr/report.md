# T2_tr

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 512 points, sampler `random`
* optimiser: `trustregion`

final training loss: **1.836403e-03**

space-time relative L2 error: **9.334090e-01**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 7.690787e-01 | 7.524432e-01 |
| 1 | 8.120554e-01 | 6.464840e-01 |
| 1.5 | 1.186356e+00 | 6.943184e-01 |
| 2 | 1.304180e+00 | 7.582235e-01 |

wall time: 1840.4 s

* phase 1: `trustregion` in 40 iterations, 1825.2 s, stopped: budget
