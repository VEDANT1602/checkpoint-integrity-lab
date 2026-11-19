# Checkpoint Integrity Lab

Think of restoring a backup. First you check which copy opens. Then you find out whether you can actually pick up where you left off. For a training run I can test that second part by comparing each update after the restart with an uninterrupted run.

I gave the loader three candidates. Step 28 was missing its completion record. A file at step 20 had been truncated after the checkpoint was saved. The loader checked the candidates newest first, logged why it rejected both, and opened step 12. That's the checkpoint I used for the next 28 updates. [Candidate decisions](results/metrics.json) · [Reader and writer](lab.py)

## What I got back at step 40

I trained the same 6–16–2 network on 256 synthetic examples, then changed what I restored from step 12. With weights alone, the largest absolute difference between a resumed parameter and its uninterrupted counterpart reached **0.405736** at step 40. Restoring Adam's state brought the difference down to **0.261498**. Both runs loaded the weights successfully.

Dropout and minibatch sampling also draw from the CPU Torch random generator. When I restored its state alongside the weights and Adam state, every subsequent loss matched the uninterrupted run and the maximum parameter difference at step 40 was **zero**. The comparison used PyTorch 2.9.1 on CPU. [Loss curves](results/loss-curves.csv) · [Parity and corruption tests](tests/test_checkpoint.py#L26) · [Step-12 checkpoint](models/complete-step12.pt) · [Resumed weights](models/resumed-model.pt)

## Reproduce the recovery

Use Python 3.11 and the pinned CPU build of PyTorch. The runner creates the two damaged candidates in a temporary directory.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --output runs/local
```

## Notes to self

On the next run I'd kill the writer during a save on the filesystem I'm targeting, start a fresh process and inspect what survived. Here I deliberately created two known failure states with orderly writes. A power loss or interrupted write on shared storage could leave a different set of files.

The loader uses CPU mapping and `weights_only=True`. It checks the payload against the manifest stored beside it, which assumes the checkpoint source is trusted. GPU and distributed runs would also need their own replay tests.

[Data card](docs/data-card.md) · [Run manifest](results/manifest.json) · [Source and contribution notes](docs/provenance.md)
