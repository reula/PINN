# T20sat_ssbroyden

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 20]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 2201 points, sampler `random`
* optimiser: `ssbroyden`

final training loss: **1.509861e-07**

space-time relative L2 error: **3.474525e+00**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 2 | 1.737444e+00 | 9.031486e-01 |
| 4 | 3.568456e+00 | 1.717474e+00 |
| 6 | 4.567072e+00 | 2.137766e+00 |
| 8 | 4.880504e+00 | 2.271110e+00 |
| 10 | 4.737313e+00 | 2.213338e+00 |
| 12 | 4.319448e+00 | 2.039183e+00 |
| 14 | 3.746578e+00 | 1.791904e+00 |
| 16 | 3.078192e+00 | 1.496216e+00 |
| 18 | 2.329520e+00 | 1.166410e+00 |
| 20 | 1.525432e+00 | 8.101360e-01 |

wall time: 368.0 s

* phase 1: `ssbroyden` in 7255 iterations, 365.7 s, stopped: budget
