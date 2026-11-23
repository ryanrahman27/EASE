# E34: frozen-controller robustness comparison

All 1,024 E33 development tasks are retained in each condition, with three unchanged neural checkpoints and the fixed-probe analytical baseline. No training or condition-specific tuning. Every method has 150 task interactions and no separate calibration. Clean success/contact/step counts reproduce E33 exactly. Final test remains reserved.

Noise is independent Gaussian joint-position noise in radians and end-effector noise in meters (sigma 0.001 or 0.005), with an exact target. One deterministic realization per task is shared across methods at equal time. Delay is 40 or 120 ms for both measured positions, with timestamp-aligned commands. History conditions rebuild statistics and tracking bias from only the last 4 or 8 transitions; they test memory restriction, not isolated physical excitation. Success and contact are evaluated on true simulator state.

| Condition | Controller | Success /1024 | Six /512 | Seven /512 | Contacts | Mean steps | Success by 50 | Neural-only / analytical-only |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clean | analytical | 1003 | 491 | 512 | 19 | 19.76 | 992 | 0 / 0 |
| clean | neural_0 | 985 | 477 | 508 | 19 | 27.27 | 930 | 4 / 22 |
| clean | neural_1 | 1000 | 489 | 511 | 18 | 21.29 | 977 | 5 / 8 |
| clean | neural_2 | 995 | 486 | 509 | 16 | 24.90 | 950 | 4 / 12 |
| noise_1mm | analytical | 1002 | 490 | 512 | 18 | 22.05 | 985 | 0 / 0 |
| noise_1mm | neural_0 | 987 | 475 | 512 | 21 | 26.99 | 940 | 5 / 20 |
| noise_1mm | neural_1 | 997 | 486 | 511 | 20 | 22.12 | 970 | 5 / 10 |
| noise_1mm | neural_2 | 999 | 488 | 511 | 18 | 25.28 | 943 | 4 / 7 |
| noise_5mm | analytical | 979 | 473 | 506 | 19 | 36.89 | 858 | 0 / 0 |
| noise_5mm | neural_0 | 972 | 464 | 508 | 23 | 34.93 | 880 | 18 / 25 |
| noise_5mm | neural_1 | 974 | 467 | 507 | 20 | 32.60 | 885 | 18 / 23 |
| noise_5mm | neural_2 | 974 | 466 | 508 | 19 | 35.87 | 853 | 22 / 27 |
| delay_1 | analytical | 1004 | 492 | 512 | 17 | 25.19 | 968 | 0 / 0 |
| delay_1 | neural_0 | 989 | 478 | 511 | 20 | 33.65 | 902 | 3 / 18 |
| delay_1 | neural_1 | 993 | 482 | 511 | 19 | 28.28 | 944 | 3 / 14 |
| delay_1 | neural_2 | 986 | 482 | 504 | 18 | 33.49 | 897 | 4 / 22 |
| delay_3 | analytical | 725 | 360 | 365 | 17 | 75.77 | 510 | 0 / 0 |
| delay_3 | neural_0 | 678 | 341 | 337 | 30 | 93.16 | 279 | 64 / 111 |
| delay_3 | neural_1 | 723 | 346 | 377 | 29 | 81.85 | 395 | 74 / 76 |
| delay_3 | neural_2 | 683 | 335 | 348 | 19 | 90.66 | 320 | 60 / 102 |
| history_4 | analytical | 971 | 467 | 504 | 39 | 24.44 | 951 | 0 / 0 |
| history_4 | neural_0 | 877 | 401 | 476 | 91 | 45.24 | 781 | 7 / 101 |
| history_4 | neural_1 | 949 | 445 | 504 | 57 | 30.39 | 901 | 14 / 36 |
| history_4 | neural_2 | 861 | 376 | 485 | 87 | 47.80 | 752 | 10 / 120 |
| history_8 | analytical | 1008 | 496 | 512 | 16 | 18.66 | 998 | 0 / 0 |
| history_8 | neural_0 | 980 | 471 | 509 | 35 | 26.80 | 955 | 0 / 28 |
| history_8 | neural_1 | 1003 | 491 | 512 | 19 | 20.51 | 989 | 0 / 5 |
| history_8 | neural_2 | 982 | 470 | 512 | 38 | 26.13 | 949 | 0 / 26 |

All 28,672 records pass case-key, finite-error, success/contact and horizon checks. Neural runs block simulator Jacobians and analytical joint-model inverses; the shared 3×3 Cartesian control solve remains intentional. Per-family results are in `summary.json`.

These are correlated tasks on 16 procedural families, using one disturbance realization per task and three learned model seeds. No significance or universal robustness claim follows from these counts. The analytical baseline has fixed clean-condition hyperparameters; this is not a comparison against a noise-tuned estimator.

![Robustness success](robustness.png)
