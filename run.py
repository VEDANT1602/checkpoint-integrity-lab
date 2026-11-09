"""Train actual CPU models; compare full-state recovery with incomplete restores."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import torch
from lab import dataset, fresh, train, payload, save, load_latest, restore, difference

ROOT = Path(__file__).resolve().parent


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def experiment():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x, y = dataset()
    continuous, continuous_opt = fresh()
    continuous_losses = train(continuous, continuous_opt, x, y, 0, 40)
    interrupted, interrupted_opt = fresh()
    prefix = train(interrupted, interrupted_opt, x, y, 0, 12)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = save(root, payload(interrupted, interrupted_opt, 12))
        checkpoint_bytes = (good / "state.pt").read_bytes()
        train(interrupted, interrupted_opt, x, y, 12, 20)
        corrupt = save(root, payload(interrupted, interrupted_opt, 20))
        (corrupt / "state.pt").write_bytes((corrupt / "state.pt").read_bytes()[:64])
        train(interrupted, interrupted_opt, x, y, 20, 28)
        save(root, payload(interrupted, interrupted_opt, 28), stop_after_payload=True)
        state, rejected = load_latest(root)
    resumed, resumed_opt = restore(state)
    resumed_losses = train(resumed, resumed_opt, x, y, 12, 40)
    weights_only, weights_opt = restore(state, optimizer_state=False, rng_state=False)
    weights_losses = train(weights_only, weights_opt, x, y, 12, 40)
    no_rng, no_rng_opt = restore(state, rng_state=False)
    no_rng_losses = train(no_rng, no_rng_opt, x, y, 12, 40)
    metrics = {"device": "CPU", "torch_version": torch.__version__, "examples": len(x),
               "training_steps": 40, "batch_size": 64, "resume_step": state["step"],
               "checkpoint_candidates": 3, "rejected_candidates": rejected,
               "full_resume_max_parameter_difference": difference(continuous, resumed),
               "full_resume_max_loss_difference": max(abs(a - b) for a, b in zip(continuous_losses, prefix + resumed_losses)),
               "weights_only_max_parameter_difference": difference(continuous, weights_only),
               "optimizer_without_rng_max_parameter_difference": difference(continuous, no_rng),
               "scope": "same-version CPU deterministic dropout/minibatch experiment; no Ray/FSDP/GPU or killed-process/power-loss claim"}
    return metrics, x, y, continuous, resumed, checkpoint_bytes, continuous_losses, prefix + resumed_losses, prefix + weights_losses, prefix + no_rng_losses


def main(output=None):
    output = Path(output) if output else ROOT
    metrics, x, y, reference, resumed, checkpoint, *curves = experiment()
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(parents=True, exist_ok=True)
    with (output / "data/training.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([f"x{i}" for i in range(6)] + ["label"])
        writer.writerows([*row.tolist(), int(target)] for row, target in zip(x, y))
    (output / "models/complete-step12.pt").write_bytes(checkpoint)
    torch.save(resumed.state_dict(), output / "models/resumed-model.pt")
    dump(output / "results/metrics.json", metrics)
    with (output / "results/loss-curves.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["step", "continuous", "full_resume", "weights_only", "optimizer_without_rng"])
        writer.writerows([i + 1, *[curve[i] for curve in curves]] for i in range(40))
    report = ("# Checkpoint recovery result\n\n"
              f"Three candidate checkpoints were inspected; step {metrics['resume_step']} was selected after rejecting the two newer candidates.\n\n"
              f"Full-state resume maximum parameter difference: {metrics['full_resume_max_parameter_difference']:.8g}.\n"
              f"Weights-only restore difference: {metrics['weights_only_max_parameter_difference']:.8g}.\n"
              f"Optimizer restored without RNG difference: {metrics['optimizer_without_rng_max_parameter_difference']:.8g}.\n\n"
              "These are same-version deterministic CPU trajectories on a small authored model. They establish neither distributed recovery nor task accuracy.\n")
    (output / "results/report.md").write_text(report, encoding="utf-8", newline="\n")
    artifacts = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for folder in ("data", "models", "results") for p in (output / folder).glob("*")
                 if p.is_file() and p.name != "manifest.json"}
    dump(output / "results/manifest.json", {"artifacts": artifacts, "source": {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ("lab.py", "run.py")}})
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    main(parser.parse_args().output)
