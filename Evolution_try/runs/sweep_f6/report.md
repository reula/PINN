# sweep_f6

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `fourier` with 6 modes
* network: 6 layers x 20 neurons, tanh, 2401 parameters
* collocation: 2048 points, sampler `random`
* optimiser: `ssbroyden`

final training loss: **1.538071e-06**

space-time relative L2 error: **3.899707e-02**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 1.845484e-02 | 2.061266e-02 |
| 1 | 3.389131e-02 | 2.757082e-02 |
| 1.5 | 4.486180e-02 | 3.492863e-02 |
| 2 | 6.408855e-02 | 4.371131e-02 |

wall time: 293.9 s

* phase 1: `ssbroyden` in 2500 iterations, 291.3 s, stopped: budget
