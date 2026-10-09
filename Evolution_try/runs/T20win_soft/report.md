# T20win_soft (windowed)

* equation: `wave2`, `c = 1.0`, `x in [-1.0, 1.0]`, `t in [0, 20]`, periodic in x
* ansatz: window 1 is `t2` on the exact initial data; later windows are a plain network with a penalty (`w_ic = 100`) pulling `u` and `u_t` onto the previous window's solution at the shared edge, so an inherited error can be corrected
* windows: 10 of `dt = 2`, hand-over `soft` (w_ic = 100)
* features: `periodic`; network 6 x 20 per window
* optimiser: `dsgnar`

space-time relative L2 error: **1.769723e-05**

| t | rel L2 | max abs err |
|---|--------|-------------|
| 0 | 0.000000e+00 | 0.000000e+00 |
| 2 | 1.905519e-06 | 9.696904e-07 |
| 4 | 1.578930e-06 | 8.994569e-07 |
| 6 | 1.556091e-06 | 1.017151e-06 |
| 8 | 1.026698e-06 | 1.040920e-06 |
| 10 | 4.267021e-06 | 2.468932e-06 |
| 12 | 1.428419e-05 | 1.388808e-05 |
| 14 | 1.664441e-05 | 1.853399e-05 |
| 16 | 2.116720e-05 | 2.163014e-05 |
| 18 | 2.985949e-05 | 2.596448e-05 |
| 20 | 3.995736e-05 | 3.061663e-05 |

| window | t range | final loss | inherited rel L2 | window rel L2 | amplification | wall (s) |
|---|---|---|---|---|---|---|
| 1 | [0, 2] | 7.784e-15 | 0.000e+00 | 1.132e-06 | xnan | 188 |
| 2 | [2, 4] | 2.824e-15 | 1.906e-06 | 1.773e-06 | x0.83 | 144 |
| 3 | [4, 6] | 1.066e-14 | 1.579e-06 | 1.120e-06 | x0.99 | 130 |
| 4 | [6, 8] | 5.461e-15 | 1.556e-06 | 1.382e-06 | x0.66 | 136 |
| 5 | [8, 10] | 7.237e-15 | 1.027e-06 | 2.852e-06 | x4.16 | 130 |
| 6 | [10, 12] | 3.124e-12 | 4.267e-06 | 1.228e-05 | x3.35 | 104 |
| 7 | [12, 14] | 3.165e-13 | 1.428e-05 | 1.484e-05 | x1.17 | 116 |
| 8 | [14, 16] | 4.744e-15 | 1.664e-05 | 1.853e-05 | x1.27 | 134 |
| 9 | [16, 18] | 8.523e-14 | 2.117e-05 | 2.516e-05 | x1.41 | 120 |
| 10 | [18, 20] | 1.019e-15 | 2.986e-05 | 3.506e-05 | x1.34 | 150 |

wall time: 1383.9 s
