# E35: noisy/short-history training and matched delay handling

Three original models each receive 5,000 additional AdamW updates (learning rate 0.0001, batch 128). Synthetic histories use training-only motion profiles mixed with Gaussian excitation, lengths 1–32, and correlated measurement differences from noisy positions. The target is the clean weighted ridge solution. Architecture and parameter count remain unchanged. This is a multi-change fine-tuning experiment with additional compute, not an isolated noise ablation.

All 1,024 previously observed E33/E34 development tasks are reused. No final test, new-family confirmation or post-result tuning. Delay-aware control uses gain 0.5/(1+delay steps) for the analytical baseline, old neural models and new neural models. This separates training effects from the shared controller change. Non-delay baseline rows are inherited from E34.

| Condition | Method | Success /1024 | Six /512 | Contacts | Mean steps | Success by 50 | Method-only / analytical-only |
|---|---|---:|---:|---:|---:|---:|---:|
| clean | analytical | 1003 | 491 | 19 | 19.76 | 992 | 0 / 0 |
| clean | old_0 | 985 | 477 | 19 | 27.27 | 930 | 4 / 22 |
| clean | old_1 | 1000 | 489 | 18 | 21.29 | 977 | 5 / 8 |
| clean | old_2 | 995 | 486 | 16 | 24.90 | 950 | 4 / 12 |
| clean | new_0 | 1004 | 492 | 20 | 20.48 | 993 | 3 / 2 |
| clean | new_1 | 1004 | 493 | 16 | 22.08 | 978 | 4 / 3 |
| clean | new_2 | 997 | 486 | 19 | 22.02 | 966 | 3 / 9 |
| noise_1mm | analytical | 1002 | 490 | 18 | 22.05 | 985 | 0 / 0 |
| noise_1mm | old_0 | 987 | 475 | 21 | 26.99 | 940 | 5 / 20 |
| noise_1mm | old_1 | 997 | 486 | 20 | 22.12 | 970 | 5 / 10 |
| noise_1mm | old_2 | 999 | 488 | 18 | 25.28 | 943 | 4 / 7 |
| noise_1mm | new_0 | 1003 | 491 | 20 | 21.01 | 991 | 3 / 2 |
| noise_1mm | new_1 | 1000 | 492 | 18 | 22.74 | 973 | 5 / 7 |
| noise_1mm | new_2 | 1001 | 490 | 19 | 20.55 | 983 | 3 / 4 |
| noise_5mm | analytical | 979 | 473 | 19 | 36.89 | 858 | 0 / 0 |
| noise_5mm | old_0 | 972 | 464 | 23 | 34.93 | 880 | 18 / 25 |
| noise_5mm | old_1 | 974 | 467 | 20 | 32.60 | 885 | 18 / 23 |
| noise_5mm | old_2 | 974 | 466 | 19 | 35.87 | 853 | 22 / 27 |
| noise_5mm | new_0 | 988 | 481 | 19 | 30.68 | 903 | 22 / 13 |
| noise_5mm | new_1 | 983 | 483 | 17 | 30.99 | 901 | 21 / 17 |
| noise_5mm | new_2 | 989 | 478 | 20 | 28.88 | 909 | 18 / 8 |
| delay_1 | analytical | 998 | 487 | 19 | 26.41 | 952 | 0 / 0 |
| delay_1 | old_0 | 991 | 480 | 18 | 34.77 | 886 | 5 / 12 |
| delay_1 | old_1 | 986 | 474 | 23 | 29.28 | 940 | 3 / 15 |
| delay_1 | old_2 | 993 | 483 | 17 | 32.55 | 907 | 9 / 14 |
| delay_1 | new_0 | 999 | 489 | 20 | 29.29 | 947 | 5 / 4 |
| delay_1 | new_1 | 1002 | 490 | 17 | 30.57 | 935 | 5 / 1 |
| delay_1 | new_2 | 993 | 482 | 22 | 28.17 | 949 | 4 / 9 |
| delay_3 | analytical | 986 | 479 | 18 | 42.78 | 792 | 0 / 0 |
| delay_3 | old_0 | 956 | 456 | 24 | 61.05 | 462 | 4 / 34 |
| delay_3 | old_1 | 962 | 451 | 22 | 51.37 | 652 | 10 / 34 |
| delay_3 | old_2 | 955 | 455 | 16 | 57.12 | 535 | 14 / 45 |
| delay_3 | new_0 | 954 | 455 | 19 | 58.80 | 547 | 10 / 42 |
| delay_3 | new_1 | 940 | 445 | 24 | 61.40 | 508 | 9 / 55 |
| delay_3 | new_2 | 955 | 458 | 23 | 53.68 | 653 | 6 / 37 |
| history_4 | analytical | 971 | 467 | 39 | 24.44 | 951 | 0 / 0 |
| history_4 | old_0 | 877 | 401 | 91 | 45.24 | 781 | 7 / 101 |
| history_4 | old_1 | 949 | 445 | 57 | 30.39 | 901 | 14 / 36 |
| history_4 | old_2 | 861 | 376 | 87 | 47.80 | 752 | 10 / 120 |
| history_4 | new_0 | 986 | 482 | 25 | 26.80 | 932 | 27 / 12 |
| history_4 | new_1 | 937 | 450 | 55 | 34.19 | 873 | 21 / 55 |
| history_4 | new_2 | 976 | 468 | 41 | 24.99 | 948 | 19 / 14 |
| history_8 | analytical | 1008 | 496 | 16 | 18.66 | 998 | 0 / 0 |
| history_8 | old_0 | 980 | 471 | 35 | 26.80 | 955 | 0 / 28 |
| history_8 | old_1 | 1003 | 491 | 19 | 20.51 | 989 | 0 / 5 |
| history_8 | old_2 | 982 | 470 | 38 | 26.13 | 949 | 0 / 26 |
| history_8 | new_0 | 1006 | 494 | 18 | 20.27 | 994 | 0 / 2 |
| history_8 | new_1 | 1006 | 494 | 18 | 21.78 | 986 | 0 / 2 |
| history_8 | new_2 | 1006 | 494 | 17 | 19.87 | 986 | 0 / 2 |

29,696 new rollouts and 20,480 inherited rows comprise this comparison. Dataset and parent/new checkpoint hashes and all case keys are verified. Neural evaluation blocks simulator Jacobians and analytical joint-model solves; the intentional shared 3×3 control solve remains.

![E35 comparison](comparison.png)
