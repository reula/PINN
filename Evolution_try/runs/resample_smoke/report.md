# resample_smoke

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 2201 points, sampler `uniform`
* optimiser: `ssbroyden`

final training loss: **7.380955e-05**

space-time relative L2 error: **8.721240e-02**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 3.960032e-02 | 3.451225e-02 |
| 1 | 7.300110e-02 | 7.397474e-02 |
| 1.5 | 1.002967e-01 | 9.701483e-02 |
| 2 | 1.452772e-01 | 1.316121e-01 |

wall time: 29.5 s

* phase 1: `ssbroyden` in 378 iterations, 26.5 s, stopped: budget
