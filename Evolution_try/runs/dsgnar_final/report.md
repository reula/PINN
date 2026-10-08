# dsgnar_final

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 2201 points, sampler `random`
* optimiser: `dsgnar`

final training loss: **3.497441e-14**

space-time relative L2 error: **7.246673e-07**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 7.402269e-07 | 6.106312e-07 |
| 1 | 6.747807e-07 | 6.219995e-07 |
| 1.5 | 8.763953e-07 | 7.513555e-07 |
| 2 | 9.248555e-07 | 8.256893e-07 |

wall time: 644.8 s

* phase 1: `dsgnar` in 86 iterations, 640.9 s, stopped: radius below threshold
