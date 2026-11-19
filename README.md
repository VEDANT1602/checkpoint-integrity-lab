# Checkpoint Integrity Lab

Three checkpoint directories claim to contain training steps 12, 20 and 28. The step-20 payload was truncated after publication. The step-28 writer stopped before publishing its manifest and completion marker. The reader rejects both and resumes from step 12. [Candidate decisions](results/metrics.json)

The [reader and writer](lab.py) bind the completion marker to the manifest, check payload size and hash, and validate architecture and state before loading it. Training uses a 6–16–2 network with dropout, Adam and random minibatches on 256 synthetic examples. At step 40, each recovery path can be compared with uninterrupted training on the same CPU.

| State restored at step 12 | Maximum absolute parameter difference at step 40 |
| --- | ---: |
| Weights, Adam state and CPU Torch RNG | 0.000000 |
| Weights only | 0.405736 |
| Weights and Adam state, RNG omitted | 0.261498 |

The full restore also reproduces the loss sequence in the recorded PyTorch 2.9.1 CPU environment. The [loss curves](results/loss-curves.csv) and [parity and corruption tests](tests/test_checkpoint.py#L26) show exactly what is compared. The complete [step-12 artifact](models/complete-step12.pt) and [resumed weights](models/resumed-model.pt) are included.

These faults are simulated by orderly local writes. They leave process termination, power loss, multi-host storage and GPU or distributed replay unmeasured. The loader trusts its local checkpoint source and uses CPU mapping with `weights_only=True`. Byte hashes verify the payload against the manifest supplied alongside it.

## Reproduce the recovery

Use a Python 3.11 environment with the pinned CPU build of PyTorch. The runner creates the damaged candidates in a temporary directory.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --output runs/local
```

[Data card](docs/data-card.md) · [Run manifest](results/manifest.json) · [Source and contribution notes](docs/provenance.md)
