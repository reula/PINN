# dsgnar_ref

* equation: `wave2` with `c = 1.0`, domain `x in [-1.0, 1.0]`, `t in [0, 2.0]`, periodic in x
* initial data: `u0 = gaussian(sigma=0.2)`, `v0 = -c u0'`, hard-coded as `u = u0 + t v0 + t^2 N`
* features: `periodic` with 1 modes
* network: 6 layers x 20 neurons, tanh, 2201 parameters
* collocation: 2048 points, sampler `random`
* optimiser: `dsgnar`

final training loss: **2.981662e-11**

space-time relative L2 error: **8.370332e-05**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 0.5 | 1.500417e-05 | 9.687981e-06 |
| 1 | 5.778678e-05 | 4.011439e-05 |
| 1.5 | 1.158226e-04 | 9.561806e-05 |
| 2 | 1.345498e-04 | 1.169233e-04 |

wall time: 918.4 s

* phase 1: `dsgnar` in 150 iterations, 916.0 s, stopped: budget
