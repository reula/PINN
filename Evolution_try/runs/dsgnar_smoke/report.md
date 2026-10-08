# dsgnar_smoke

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 512 points, sampler `uniform`
* optimiser: `dsgnar`

final training loss: **1.624422e-02**

space-time relative L2 error: **7.494755e-01**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 5.980675e-01 | 5.812308e-01 |
| 1 | 1.030485e+00 | 7.179824e-01 |
| 1.5 | 9.683919e-01 | 9.521942e-01 |
| 2 | 6.664389e-01 | 5.298715e-01 |

wall time: 16.1 s

* phase 1: `dsgnar` in 20 iterations, 14.3 s, stopped: budget
