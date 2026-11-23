# E23 paired-arm pilot

Status: complete

Privileged training control: 256/256; gate passed: True.

Single seed; local reaching goals; no final-test policy evaluation.

| Model | Split | Context | Success | Training-style gate | Mean error at step 10 (n) |
|---|---|---|---:|---|---:|
| history | train | correct | 256/256 | True | 0.03635 (205) |
| history | train | empty | 29/256 | False | 0.09136 (252) |
| history | train | twin | 38/256 | False | 0.14251 (245) |
| history | validation | correct | 62/96 | False | 0.06279 (88) |
| history | validation | empty | 13/96 | False | 0.09072 (91) |
| history | validation | twin | 29/96 | False | 0.08369 (90) |
| pose | train | correct | 214/256 | False | 0.07191 (227) |
| pose | train | empty | 30/256 | False | 0.09306 (252) |
| pose | train | twin | 214/256 | False | 0.07191 (227) |
| pose | validation | correct | 47/96 | False | 0.07937 (92) |
| pose | validation | empty | 13/96 | False | 0.08917 (92) |
| pose | validation | twin | 47/96 | False | 0.07937 (92) |

Frozen controllers and hashes verified. Pose-only correct/twin outcomes match exactly.
Step-10 means exclude already-ended episodes; differing sample counts can confound comparisons.
Validation gates above are descriptive, not training gates or confirmation of adaptation.
