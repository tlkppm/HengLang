from __future__ import annotations

import ast
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN = ROOT / "tools" / "toolchain.py"
BUILD = ROOT / ".build"


def checksum(data: bytes) -> int:
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def state_bytes(payload: bytes) -> bytes:
    return b"HENGST01" + struct.pack("<IQQ", 1, len(payload), checksum(payload)) + payload


class NativeRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="heng-native-")
        cls.directory = Path(cls.temporary.name)
        cls.addClassCleanup(cls.temporary.cleanup)
        BUILD.mkdir(exist_ok=True)
        cls.program = BUILD / "runtime-unit-test.exe"
        command = [sys.executable, "-B", str(TOOLCHAIN), "cc", "-std=c17", "-O2", "-Wall", "-Wextra", "-Werror"]
        command.extend(str(path) for path in sorted((ROOT / "runtime").glob("*.c")))
        command.extend([str(ROOT / "tests" / "native_runtime.c"), "-o", str(cls.program)])
        compiled = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=240)
        if compiled.returncode:
            raise AssertionError(compiled.stderr.decode("utf-8", "replace"))
        cls.state = cls.directory / "state.hstate"
        cls.success = subprocess.run([str(cls.program), str(cls.state)], cwd=ROOT, capture_output=True, timeout=120)
        if cls.success.returncode:
            raise AssertionError(cls.success.stderr.decode("utf-8", "replace"))

    def run_failure(self, mode: str, path: Path | None = None):
        command = [str(self.program), "fail", mode]
        if path is not None:
            command.append(str(path))
        result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        error = result.stderr.decode("utf-8", "strict")
        self.assertIn("native-case.heng:17:9:", error)
        self.assertNotIn("预期失败分支未触发", error)
        return error

    def test_native_success_and_utf8(self):
        self.assertEqual(self.success.returncode, 0)
        output = self.success.stdout.decode("utf-8", "strict")
        self.assertIn("中文打印", output)
        self.assertIn("PASS native runtime", output)

    def test_exact_binary_state_roundtrip(self):
        self.assertEqual(self.state.read_bytes(), Path(str(self.state) + ".copy").read_bytes())
        self.assertEqual(self.state.read_bytes(), Path(str(self.state) + ".中文").read_bytes())
        content = self.state.read_bytes()
        self.assertEqual(content[:8], b"HENGST01")
        version, size, digest = struct.unpack_from("<IQQ", content, 8)
        self.assertEqual(version, 1)
        self.assertEqual(size, len(content) - 28)
        self.assertEqual(digest, checksum(content[28:]))

    def test_failure_boundaries_and_source_locations(self):
        modes = ["overflow", "multiply", "subtract", "division", "zero", "truth", "index", "pop", "shape", "size", "axis", "slice", "step", "axes", "broadcast", "reshape", "gather", "matmul", "coo_index", "csr_offsets", "csr_duplicate", "max_empty", "mean_empty", "cast_overflow", "result", "arity", "unknown", "threads"]
        for mode in modes:
            with self.subTest(mode=mode):
                self.run_failure(mode)

    def test_missing_state_file(self):
        self.assertIn("无法读取状态文件", self.run_failure("load", self.directory / "missing.hstate"))

    def test_corrupt_state_rejected(self):
        original = self.state.read_bytes()
        mutations = {
            "magic": b"BADMAGIC" + original[8:],
            "version": original[:8] + struct.pack("<I", 2) + original[12:],
            "checksum": original[:-1] + bytes([original[-1] ^ 1]),
            "truncated": original[:-7],
            "trailing": original + b"extra",
        }
        for name, content in mutations.items():
            with self.subTest(name=name):
                path = self.directory / f"corrupt-{name}.hstate"
                path.write_bytes(content)
                self.run_failure("load", path)

    def test_checked_metadata_rejected_with_valid_checksum(self):
        tensor_header = struct.pack("<QQQ", 1, 0, 0)
        mutations = {
            "overflow": tensor_header + struct.pack("<IIQQ", 101, 2, (1 << 63) - 1, 2),
            "negative_dimension": tensor_header + struct.pack("<IIQ", 101, 1, 1 << 63),
            "rank": tensor_header + struct.pack("<II", 101, (1 << 32) - 1),
            "dtype": tensor_header + struct.pack("<II", 0, 0),
            "truncated_scalar": tensor_header + struct.pack("<II", 101, 0),
            "bad_bool": tensor_header + struct.pack("<IIQ", 104, 1, 1) + b"\x02",
            "counts": struct.pack("<QQQ", (1 << 64) - 1, 0, 0),
            "extra_payload": struct.pack("<QQQ", 0, 0, 0) + b"extra",
        }
        for name, payload in mutations.items():
            with self.subTest(name=name):
                path = self.directory / f"metadata-{name}.hstate"
                path.write_bytes(state_bytes(payload))
                self.run_failure("load", path)

    def test_undefined_behavior_instrumentation(self):
        program = BUILD / "runtime-ubsan-test.exe"
        command = [sys.executable, "-B", str(TOOLCHAIN), "cc", "-std=c17", "-O1", "-g", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined", "-fno-sanitize-recover=all"]
        command.extend(str(path) for path in sorted((ROOT / "runtime").glob("*.c")))
        command.extend([str(ROOT / "tests" / "native_runtime.c"), "-o", str(program)])
        compiled = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=240)
        self.assertEqual(compiled.returncode, 0, compiled.stderr.decode("utf-8", "replace"))
        result = subprocess.run([str(program), str(self.directory / "ubsan.hstate")], cwd=ROOT, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertIn(b"PASS native runtime", result.stdout)
        self.assertEqual(result.stderr, b"")

    def test_all_abi_builtins_have_exact_native_signatures(self):
        tree = ast.parse((ROOT / "henglang" / "abi.py").read_text(encoding="utf-8"))
        assignment = next(node for node in tree.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "BUILTINS")
        builtins = ast.literal_eval(assignment.value)
        runtime = (ROOT / "runtime" / "heng_runtime.c").read_text(encoding="utf-8")
        section = runtime.split("static const BuiltinSignature signatures[]={", 1)[1].split("};", 1)[0]
        signatures = {name: int(count) for name, count in re.findall(r'\{"([a-z0-9_]+)",(\d+)\}', section)}
        self.assertEqual(signatures, {name: len(signature[0]) for name, signature in builtins.items()})


if __name__ == "__main__":
    unittest.main()
