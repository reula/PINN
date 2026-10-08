# sweep_f6ic

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `fourier_ic` with 6 modes
* network: 6 layers x 20 neurons, tanh, 2441 parameters
* collocation: 2048 points, sampler `random`
* optimiser: `ssbroyden`

final training loss: **7.396228e-06**

space-time relative L2 error: **1.321523e-01**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 5.861290e-02 | 6.187506e-02 |
| 1 | 1.031391e-01 | 8.292711e-02 |
| 1.5 | 1.644980e-01 | 1.207523e-01 |
| 2 | 2.151390e-01 | 1.396783e-01 |

wall time: 339.0 s

* phase 1: `ssbroyden` in 2500 iterations, 321.0 s, stopped: budget
