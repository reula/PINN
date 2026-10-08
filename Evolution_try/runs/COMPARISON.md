# Run comparison

Sorted by space-time relative L2 error.

| run | optimiser | features | modes | n_coll | final loss | rel L2 (space-time) | rel L2 at T | wall (s) | iterations |
|---|---|---|---|---|---|---|---|---|---|
| dsgnar_final | dsgnar | periodic | 1 | 2201 | 3.497e-14 | 7.247e-07 | 9.249e-07 | 644.8 | 86 |
| dsgnar_fresh | dsgnar | periodic | 1 | 2201 | 1.144e-15 | 9.379e-07 | 1.610e-06 | 1964 | 206 |
| dsgnar_resample | dsgnar | periodic | 1 | 2201 | 4.773e-14 | 2.125e-06 | 4.222e-06 | 1597 | 69 |
| sched_smoke | dsgnar | periodic | 1 | 512 | 6.286e-11 | 4.018e-05 | 7.427e-05 | 113.7 | 92 |
| ssbroyden_cont | ssbroyden | periodic | 1 | 8192 | 3.958e-10 | 7.806e-05 | 1.119e-04 | 671.3 | 3000 |
| dsgnar_ref | dsgnar | periodic | 1 | 2048 | 2.982e-11 | 8.370e-05 | 1.345e-04 | 918.4 | 150 |
| ssbroyden_ref | ssbroyden | periodic | 1 | 4096 | 4.701e-10 | 9.038e-05 | 1.168e-04 | 4279 | 9753 |
| ssbroyden_resample | ssbroyden | periodic | 1 | 2201 | 3.746e-10 | 2.269e-04 | 4.051e-04 | 1360 | 8000 |
| warm_smoke | ssbroyden | periodic | 1 | 512 | 5.635e-10 | 2.281e-04 | 3.606e-04 | 8.025 | 250 |
| sweep_periodic | ssbroyden | periodic | 1 | 2048 | 1.552e-08 | 9.958e-04 | 0.001251 | 424.7 | 2253 |
| sweep_f6 | ssbroyden | fourier | 6 | 2048 | 1.538e-06 | 0.039 | 0.06409 | 293.9 | 2500 |
| resample_smoke | ssbroyden | periodic | 1 | 2201 | 7.381e-05 | 0.08721 | 0.1453 | 29.47 | 378 |
| sweep_f6ic | ssbroyden | fourier_ic | 6 | 2048 | 7.396e-06 | 0.1322 | 0.2151 | 339 | 2500 |
| dsgnar_smoke | dsgnar | periodic | 1 | 512 | 0.01624 | 0.7495 | 0.6664 | 16.07 | 20 |
| sweep_f12ic | ssbroyden | fourier_ic | 12 | 2048 | 1.039e-05 | 0.7997 | 0.9063 | 337 | 2500 |
| T20_ssbroyden | ssbroyden | periodic | 1 | 2201 | 6.976e-12 | 0.9072 | 0.9169 | 2511 | 19763 |
| T2_tr | trustregion | periodic | 1 | 512 | 0.001836 | 0.9334 | 1.304 | 1840 | 40 |
| sweep_f12 | ssbroyden | fourier | 12 | 2048 | 1.620e-04 | 1.131 | 1.838 | 270.1 | 2500 |
| smoke_qn | ssbroyden | fourier | 6 | 1024 | 0.001457 | 5.188 | 9.377 | 20.82 | 200 |
| T20_bigbatch | ssbroyden | periodic | 1 | 8192 | 1.300e-07 | 10.97 | 18.7 | 1211 | 4000 |
| T20_dsgnar | dsgnar | periodic | 1 | 2201 | 1.670e-06 | 23.86 | 42 | 3516 | 97 |
| T20_ssb_scout | ssbroyden | periodic | 1 | 2201 | 4.352e-07 | 28.57 | 12.79 | 289.7 | 2000 |
