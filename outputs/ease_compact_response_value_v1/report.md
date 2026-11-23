# E135: compact regularized task value

32 structured measured features, same four fitting-family splits. Linear signed paired-advantage regression; fitting-only normalization and fixed regularization grid. No privileged object/contact signals. Training targets are final success differences. This is exploratory development after E133.

| Ridge penalty | OOF advantage MSE | Nonzero sign accuracy |
|---|---:|---:|
| 0.01 | 0.11062 | 52.1% |
| 0.1 | 0.10978 | 52.9% |
| 1.0 | 0.10785 | 52.1% |
| 10.0 | 0.10739 | 50.4% |

Zero-advantage reference MSE: 0.10740. Strong regularization nearly returns that reference; better MSE than older neural heads does not establish useful ranking.

Primary lambda 1, margin 0.05: 102/144 reconstructed successes vs E120 97; 9 rescues and 4 lost successes, 80 switches. Zero-loss grid selection still fully abstains. E120 unchanged; no held-placement or unseen-body test. All numerical predictions/weights/normalization and task outcomes independently reproduced. Representative paired successes/failures preserved.
