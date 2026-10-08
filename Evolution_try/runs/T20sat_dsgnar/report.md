# T20sat_dsgnar

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 20]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 2201 points, sampler `random`
* optimiser: `dsgnar`

final training loss: **5.938236e-06**

space-time relative L2 error: **4.993792e+00**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 2 | 1.470834e+00 | 1.271744e+00 |
| 4 | 2.036271e+00 | 1.534215e+00 |
| 6 | 2.647171e+00 | 1.812121e+00 |
| 8 | 3.187007e+00 | 2.046455e+00 |
| 10 | 3.741111e+00 | 2.285675e+00 |
| 12 | 4.407228e+00 | 2.572555e+00 |
| 14 | 5.257119e+00 | 2.935274e+00 |
| 16 | 6.351366e+00 | 3.398718e+00 |
| 18 | 7.739667e+00 | 3.986594e+00 |
| 20 | 9.462777e+00 | 4.723669e+00 |

wall time: 529.9 s

* phase 1: `dsgnar` in 150 iterations, 527.4 s, stopped: budget
