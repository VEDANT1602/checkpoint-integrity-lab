import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab import dataset, fresh, train, payload, save, load_latest, restore, difference
from run import experiment


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.x, self.y = dataset()
        self.model, self.optimizer = fresh()
        train(self.model, self.optimizer, self.x, self.y, 0, 3)

    def test_full_resume_parity_and_incomplete_restore_divergence(self):
        metrics, *_ = experiment()
        self.assertEqual(metrics["full_resume_max_parameter_difference"], 0)
        self.assertEqual(metrics["full_resume_max_loss_difference"], 0)
        self.assertGreater(metrics["weights_only_max_parameter_difference"], 0.001)
        self.assertGreater(metrics["optimizer_without_rng_max_parameter_difference"], 0.001)
        self.assertEqual(metrics["resume_step"], 12)
        self.assertEqual(len(metrics["rejected_candidates"]), 2)

    def test_incomplete_newest_candidate_falls_back(self):
        save(self.root, payload(self.model, self.optimizer, 3))
        save(self.root, payload(self.model, self.optimizer, 4), stop_after_payload=True)
        state, rejected = load_latest(self.root)
        self.assertEqual(state["step"], 3)
        self.assertEqual(len(rejected), 1)

    def test_corrupt_newest_candidate_falls_back_before_deserialization(self):
        save(self.root, payload(self.model, self.optimizer, 3))
        bad = save(self.root, payload(self.model, self.optimizer, 4))
        (bad / "state.pt").write_bytes(b"broken")
        state, rejected = load_latest(self.root)
        self.assertEqual(state["step"], 3)
        self.assertIn("digest", rejected[0]["reason"])

    def test_marker_binds_manifest(self):
        directory = save(self.root, payload(self.model, self.optimizer, 3))
        (directory / "manifest.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "no complete"):
            load_latest(self.root)

    def test_nothing_valid_does_not_start_from_scratch_silently(self):
        save(self.root, payload(self.model, self.optimizer, 4), stop_after_payload=True)
        with self.assertRaisesRegex(ValueError, "no complete"):
            load_latest(self.root)

    def test_bad_optimizer_state_is_not_accepted(self):
        state = copy.deepcopy(payload(self.model, self.optimizer, 3))
        first = next(iter(state["optimizer"]["state"].values()))
        first["exp_avg"].fill_(float("nan"))
        save(self.root, state)
        with self.assertRaisesRegex(ValueError, "invalid optimizer"):
            load_latest(self.root)

    def test_save_is_create_only(self):
        state = payload(self.model, self.optimizer, 3)
        save(self.root, state)
        with self.assertRaises(FileExistsError):
            save(self.root, state)

    def test_nonfinite_optimizer_hyperparameter_is_rejected(self):
        state = copy.deepcopy(payload(self.model, self.optimizer, 3))
        state["optimizer"]["param_groups"][0]["lr"] = float("nan")
        save(self.root, state)
        with self.assertRaisesRegex(ValueError, "invalid optimizer scalar"):
            load_latest(self.root)

    def test_deterministic_dataset_independent_of_training_rng(self):
        torch.manual_seed(1000)
        other_x, other_y = dataset()
        self.assertTrue(torch.equal(self.x, other_x))
        self.assertTrue(torch.equal(self.y, other_y))

    def test_source_and_artifact_commitments(self):
        manifest = json.loads((ROOT / "results/manifest.json").read_text())
        for section in manifest.values():
            for name, digest in section.items():
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)


if __name__ == "__main__":
    unittest.main()
