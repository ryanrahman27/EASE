# E145: measured calibration prior

A single fixed analytical control: regularize the local32-transition Jacobian toward the initial measured calibration instead of zero. Same.98 recency,1e-6 ridge, calibration, frozen bias and task controller as E144 uniform32. No learned weights, privileged Jacobian or physics selector.

Success: **67/96**, with 23 collision episodes. Six joints: 35/48; seven joints: 32/48. Uniform32 reference: 74/96. Paired rescues/losses: **1/8**.

Although matched next-motion prediction improved, this analytical prior did not establish a general task-success improvement. Preserving stale calibration directions can trade off local adaptation. All96 fitting outcomes remain saved; fresh and old held placements remain closed. This is known-body development, not unseen-embodiment evidence. E120 unchanged.
