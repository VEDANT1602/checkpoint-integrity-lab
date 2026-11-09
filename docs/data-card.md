# Dataset and runtime boundary

NumPy is not required. A dedicated Torch generator seeded 1602 creates 256 six-dimensional Gaussian examples and nonlinear binary labels. The dataset generator does not depend on the training RNG. Training starts from seed 91; the actual CPU RNG state captures random minibatch selection and dropout. The dataset is fictional and contains no customer, security or employer records.

The task is training-trajectory recovery. There is no held-out accuracy claim, model selection claim or claim that this toy classifier is useful in production. All branches have the same 40-step training protocol. Loss curves and parameter differences compare recovery fidelity, not generalization.

The checkpoint writer calls file fsync before atomic replacement, then publishes a manifest-bound completion marker. It does not fsync a directory or establish power-loss durability. Directory existence is not completion. Loading a candidate may consume RNG while validating model compatibility; callers resuming training must call restore(), which explicitly sets the saved RNG afterward.

Saved artifacts are trusted local data generated during this run. A checksum can detect a mismatch with its manifest but cannot detect an attacker who replaces both. Checkpoints do not have signatures or permissions enforcement. CPU PyTorch version and deterministic settings are recorded; exact cross-version/hardware parity is not promised.
