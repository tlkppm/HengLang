from __future__ import annotations

import json
import subprocess
import sys
import unittest

from benchmarks.train_text_model import ROOT, PROMPTS, corpus, decode_generation, ExperimentError, parse_arguments


class TextModelTests(unittest.TestCase):
    def test_training_and_held_out_sentence_combinations_are_disjoint(self):
        train, validation = corpus()
        self.assertEqual((len(train), len(validation)), (12, 4))
        self.assertTrue(set(train).isdisjoint(validation))
        self.assertLessEqual(set("".join(validation)), set("".join(train)))

    def test_generation_requires_a_stop_marker_or_full_length(self):
        self.assertEqual(decode_generation("a", [2, 0], "\nab", 4)["text"], "ab")
        for invalid in ([], [1], [0, 2], [3, 0], [True, 0]):
            with self.subTest(invalid=invalid), self.assertRaises(ExperimentError):
                decode_generation("a", invalid, "\nab", 4)

    def test_training_speed_requirement_needs_a_timed_numpy_comparison(self):
        for invalid in (["--require-win"], ["--require-win", "--numpy", "--smoke"]):
            with self.subTest(arguments=invalid), self.assertRaises(ExperimentError):
                parse_arguments(invalid)
        self.assertTrue(parse_arguments(["--require-win", "--numpy"]).require_win)
        self.assertEqual(parse_arguments(["--require-win", "--numpy"]).minimum_speedup, 1.10)
        with self.assertRaises(ExperimentError):
            parse_arguments(["--minimum-speedup", "0"])

    def test_native_step_smoke_trains_and_reports_unseen_examples(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "benchmarks" / "train_text_model.py"), "--smoke", "--native-step"],
            cwd=ROOT, capture_output=True, encoding="utf-8", timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        report = json.loads(result.stdout)
        self.assertTrue(report["configuration"]["native_step"])
        self.assertEqual(report["privacy"], {"machine_information": "omitted"})
        self.assertTrue(report["results"]["heng"]["single_thread_verified"])
        self.assertLess(report["results"]["heng"]["final"]["validation"]["loss"], 3.1)

    def test_native_smoke_trains_and_reports_unseen_examples(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "benchmarks" / "train_text_model.py"), "--smoke"],
            cwd=ROOT, capture_output=True, encoding="utf-8", timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["privacy"], {"machine_information": "omitted"})
        self.assertNotIn(str(ROOT), result.stdout)
        self.assertEqual(report["configuration"]["epochs"], 1)
        self.assertEqual(report["data"]["validation_targets_with_unseen_context_target_pair"], 14)
        heng = report["results"]["heng"]
        self.assertIsNone(heng["training_seconds"])
        self.assertTrue(heng["single_thread_verified"])
        for split in ("train", "validation", "validation_unseen"):
            self.assertLess(heng["final"][split]["loss"], heng["initial"][split]["loss"])
        self.assertEqual([item["prompt"] for item in heng["generations"]], list(PROMPTS))


if __name__ == "__main__":
    unittest.main()
