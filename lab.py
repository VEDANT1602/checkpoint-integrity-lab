"""Trusted-local CPU training checkpoints with explicit completeness and integrity."""
import hashlib
import io
import json
import math
import os
import pickle
from pathlib import Path
import re
import torch

ARCH = {"input": 6, "hidden": 16, "classes": 2, "dropout": 0.2}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def dataset():
    g = torch.Generator().manual_seed(1602)
    x = torch.randn(256, 6, generator=g)
    y = (x[:, 0] + 0.7 * x[:, 1] * x[:, 2] - 0.4 * x[:, 3] > 0).long()
    return x, y


def fresh(seed=91):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.ReLU(),
                                torch.nn.Dropout(0.2), torch.nn.Linear(16, 2))
    return model, torch.optim.Adam(model.parameters(), lr=0.025)


def train(model, optimizer, x, y, start, end):
    for value in (start, end):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("nonnegative integer steps required")
    if end < start:
        raise ValueError("end precedes start")
    losses = []
    model.train()
    for step in range(start, end):
        batch = torch.randperm(len(x))[:64]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x[batch]), y[batch])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def payload(model, optimizer, step):
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("invalid checkpoint step")
    # torch.save immediately captures these state tensors before any more training.
    return {"schema": "training-state.v1", "architecture": ARCH, "step": step,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state()}


def atomic(path, content):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save(root, state, *, stop_after_payload=False):
    step = state["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("invalid checkpoint step")
    directory = Path(root) / f"step-{step:04d}"
    directory.mkdir(parents=True, exist_ok=False)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    content = buffer.getvalue()
    atomic(directory / "state.pt", content)
    if stop_after_payload:
        return directory
    manifest = canonical({"schema": "checkpoint.v1", "step": step, "architecture": ARCH,
                          "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    atomic(directory / "manifest.json", manifest)
    atomic(directory / "COMPLETE", hashlib.sha256(manifest).hexdigest().encode())
    return directory


def load_latest(root):
    root = Path(root).resolve()
    candidates = sorted(((int(p.name[5:]), p) for p in root.iterdir()
                         if p.is_dir() and re.fullmatch(r"step-\d+", p.name)), reverse=True)
    rejected = []
    for expected_step, directory in candidates:
        try:
            if not directory.resolve().is_relative_to(root):
                raise ValueError("checkpoint path escapes root")
            files = [directory / name for name in ("state.pt", "manifest.json", "COMPLETE")]
            if any(not f.is_file() or not f.resolve().is_relative_to(root) for f in files):
                raise ValueError("incomplete checkpoint")
            raw = files[1].read_bytes()
            if files[2].read_bytes() != hashlib.sha256(raw).hexdigest().encode():
                raise ValueError("completion marker does not bind manifest")
            manifest = json.loads(raw)
            if manifest.get("schema") != "checkpoint.v1" or manifest.get("architecture") != ARCH or type(manifest.get("step")) is not int or manifest["step"] != expected_step:
                raise ValueError("manifest schema, architecture or step differs")
            content = files[0].read_bytes()
            if len(content) != manifest["bytes"] or hashlib.sha256(content).hexdigest() != manifest["sha256"]:
                raise ValueError("payload digest or length differs")
            state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
            if state["schema"] != "training-state.v1" or state["architecture"] != ARCH or type(state["step"]) is not int or state["step"] != expected_step:
                raise ValueError("state contract differs")
            for value in state["model"].values():
                if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                    raise ValueError("invalid model tensor")
            def finite_tree(value):
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise ValueError("invalid optimizer tensor")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("invalid optimizer scalar")
                if isinstance(value, dict):
                    for item in value.values():
                        finite_tree(item)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        finite_tree(item)
            finite_tree(state["optimizer"])
            model, optimizer = fresh()
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            for group in optimizer.param_groups:
                for key in ("lr", "eps", "weight_decay"):
                    value = group[key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError("invalid Adam hyperparameter")
                if len(group["betas"]) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v < 1 for v in group["betas"]):
                    raise ValueError("invalid Adam beta")
            probe = torch.Generator()
            probe.set_state(state["torch_rng"])
            return state, rejected
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            rejected.append({"directory": directory.name, "reason": str(exc)})
    raise ValueError("no complete valid checkpoint; rejected=" + json.dumps(rejected))


def restore(state, *, optimizer_state=True, rng_state=True):
    model, optimizer = fresh(999)
    model.load_state_dict(state["model"])
    if optimizer_state:
        optimizer.load_state_dict(state["optimizer"])
    if rng_state:
        torch.set_rng_state(state["torch_rng"])
    return model, optimizer


def difference(left, right):
    return max(float(torch.max(torch.abs(a.detach() - b.detach()))) for a, b in zip(left.parameters(), right.parameters()))
