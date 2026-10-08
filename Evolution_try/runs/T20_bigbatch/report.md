# T20_bigbatch

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 20]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 8192 points, sampler `random`
* optimiser: `ssbroyden`

final training loss: **1.300341e-07**

space-time relative L2 error: **1.097219e+01**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 2 | 2.103752e+00 | 1.063052e+00 |
| 4 | 3.861506e+00 | 1.843829e+00 |
| 6 | 5.623577e+00 | 2.589355e+00 |
| 8 | 7.400563e+00 | 3.342085e+00 |
| 10 | 9.179030e+00 | 4.097534e+00 |
| 12 | 1.100177e+01 | 4.866035e+00 |
| 14 | 1.287284e+01 | 5.650868e+00 |
| 16 | 1.477941e+01 | 6.456044e+00 |
| 18 | 1.671790e+01 | 7.272862e+00 |
| 20 | 1.869828e+01 | 8.109237e+00 |

wall time: 1211.1 s

* phase 1: `ssbroyden` in 4000 iterations, 1195.3 s, stopped: budget
