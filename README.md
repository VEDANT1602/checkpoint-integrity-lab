# Checkpoint Integrity Lab

The run resumes at step 12 even though there are directories for steps 20 and 28. The reader finds an unfinished write at step 28, then a payload at step 20 whose length or digest disagrees with its manifest. Step 12 is the first candidate that passes all checks. The saved result includes the rejection reason for each directory. [Candidate decisions](results/metrics.json)

Finding an intact file only answers half the recovery question. At step 12, which state does Adam need to follow the uninterrupted run for 28 more updates? The experiment restores weights alone, then adds optimizer state, then restores the CPU random state used by dropout and minibatch sampling. Each run uses the same 6–16–2 network on 256 synthetic examples. [Checkpoint reader and writer](lab.py)

| State restored at step 12 | Maximum absolute parameter difference at step 40 |
| --- | ---: |
| Weights, Adam state and CPU Torch RNG | 0.000000 |
| Weights only | 0.405736 |
| Weights and Adam state, RNG omitted | 0.261498 |

The full restore also reproduces the loss sequence in the recorded PyTorch 2.9.1 CPU environment. The [loss curves](results/loss-curves.csv) and [parity and corruption tests](tests/test_checkpoint.py#L26) show exactly what is compared. The complete [step-12 artifact](models/complete-step12.pt) and [resumed weights](models/resumed-model.pt) are included.

These faults are simulated by orderly local writes. They leave process termination, power loss, multi-host storage and GPU or distributed replay unmeasured. The loader trusts its local checkpoint source and uses CPU mapping with `weights_only=True`. Byte hashes verify the payload against the manifest supplied alongside it.

A useful next experiment would terminate the writer mid-save on the target filesystem and inspect which bytes survive a fresh process. The current fault fixtures deliberately create known partial states.

## Reproduce the recovery

Use a Python 3.11 environment with the pinned CPU build of PyTorch. The runner creates the damaged candidates in a temporary directory.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --output runs/local
```

[Data card](docs/data-card.md) · [Run manifest](results/manifest.json) · [Source and contribution notes](docs/provenance.md)
