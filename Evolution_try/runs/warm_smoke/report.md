# warm_smoke

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 512 points, sampler `uniform`
* optimiser: `ssbroyden`

final training loss: **5.635475e-10**

space-time relative L2 error: **2.280630e-04**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 1.373584e-04 | 1.337295e-04 |
| 1 | 1.804802e-04 | 1.416071e-04 |
| 1.5 | 2.808314e-04 | 2.009792e-04 |
| 2 | 3.605984e-04 | 2.325376e-04 |

wall time: 8.0 s

* phase 1: `ssbroyden` in 250 iterations, 5.3 s, stopped: budget
