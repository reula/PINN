# sched_smoke

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 512 points, sampler `random`
* optimiser: `dsgnar`

final training loss: **6.285738e-11**

space-time relative L2 error: **4.017553e-05**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 2.636597e-05 | 1.940903e-05 |
| 1 | 2.867269e-05 | 2.042705e-05 |
| 1.5 | 3.237582e-05 | 2.376712e-05 |
| 2 | 7.427042e-05 | 5.279638e-05 |

wall time: 113.7 s

* phase 1: `dsgnar` in 92 iterations, 107.1 s, stopped: radius below threshold
