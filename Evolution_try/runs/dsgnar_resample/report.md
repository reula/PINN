# dsgnar_resample

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 2201 points, sampler `random`
* optimiser: `dsgnar`

final training loss: **4.772735e-14**

space-time relative L2 error: **2.125405e-06**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 3.200521e-07 | 3.368535e-07 |
| 1 | 7.328563e-07 | 7.544963e-07 |
| 1.5 | 2.044172e-06 | 1.332397e-06 |
| 2 | 4.221947e-06 | 2.762478e-06 |

wall time: 1597.3 s

* phase 1: `dsgnar` in 69 iterations, 1591.8 s, stopped: radius below threshold
