from __future__ import annotations

import contextlib
from hashlib import sha256
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch
import unittest

from henglang.__main__ import ROOT, build, configuration, execute, main
from henglang.emitter import emit
from henglang.formatter import format_source
from henglang.frontend import HengError, check, load_program, parse
from henglang.ir import optimize
from tools.toolchain import environment


class CompilerTests(unittest.TestCase):
    def test_cli_version_matches_package_configuration(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            main(["--version"])
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), f"HengLang {configuration()['package']['version']}")

    def test_exclusive_source_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "program.txt"
            path.write_text("fn main() -> void {}", encoding="utf-8")
            with self.assertRaises(HengError):
                load_program(path)

    def test_deterministic_codegen(self):
        program = load_program(ROOT / "tests" / "fixtures" / "language.heng")
        first = emit(optimize(program), ROOT)
        second = emit(optimize(program), ROOT)
        self.assertEqual(first, second)
        self.assertNotIn("eval(", first)
        self.assertNotIn("exec(", first)

    def test_formatter_preserves_semantics_and_is_idempotent(self):
        source = 'fn main()->i64{let x:List<i64>=[1,2];if len(x)>0{return x[0];}else{return 0;}}'
        formatted = format_source(source)
        self.assertEqual(formatted, format_source(formatted))
        check(parse(formatted))

    def test_native_language_program(self):
        result = execute(ROOT / "tests" / "fixtures" / "language.heng")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS: HengLang", result.stdout)
        self.assertIn("衡言语言", result.stdout)

    def test_executable_is_native_and_runs_without_python_on_path(self):
        executable = build(ROOT / "tests" / "fixtures" / "language.heng")
        self.assertEqual(executable.read_bytes()[:2], b"MZ")
        env = environment()
        env["PATH"] = ""
        result = subprocess.run([str(executable)], cwd=ROOT, env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_build_cache_reuses_unchanged_executable(self):
        source = ROOT / "tests" / "fixtures" / "language.heng"
        executable = build(source)
        before = (executable.stat().st_mtime_ns, sha256(executable.read_bytes()).digest())
        self.assertEqual(build(source), executable)
        self.assertEqual(before, (executable.stat().st_mtime_ns, sha256(executable.read_bytes()).digest()))

    def test_complete_framework_build_from_clean_output_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".build") as directory:
            project = Path(directory)
            shutil.copytree(ROOT / "runtime", project / "runtime")
            shutil.copytree(ROOT / "heng", project / "heng")
            shutil.copy2(ROOT / "heng.toml", project / "heng.toml")
            source = project / "main.heng"
            source.write_text('import "heng.heng";fn main(){let tape=ag_tape();let x=ag_variable(tape,tensor([2.0,3.0],[2]),true);let output=ag_sum_all(tape,ag_pow(tape,x,2.0));ag_backward(tape,output,ones([]),false);assert_close(ag_grad(tape,x),tensor([4.0,6.0],[2]),1e-12,1e-12);}', encoding="utf-8")
            self.assertFalse((project / ".build").exists())
            with patch("henglang.__main__.ROOT", project):
                executable = build(source)
            self.assertTrue(executable.is_relative_to(project / ".build"))
            env = environment()
            env["PATH"] = ""
            result = subprocess.run([str(executable)], cwd=project, env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_forced_native_build_is_byte_deterministic(self):
        source = ROOT / "tests" / "fixtures" / "language.heng"
        executable = build(source, force=True)
        first = executable.read_bytes()
        executable = build(source, force=True)
        second = executable.read_bytes()
        offsets = [index for index, pair in enumerate(zip(first, second)) if pair[0] != pair[1]]
        self.assertTrue(first == second, f"原生构建字节不一致，长度 {len(first)}/{len(second)}，前 40 个差异偏移 {offsets[:40]}")
        result = subprocess.run([sys.executable, "-B", "-m", "henglang", "build", str(source), "--force"], cwd=ROOT, env=environment(), capture_output=True, encoding="utf-8", errors="replace", timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(first, executable.read_bytes(), "跨进程强制构建字节不一致")

    def test_native_runtime_error_maps_to_heng_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "failure.heng"
            source.write_text('fn main() -> void { let xs: List<i64> = [1]; print(xs[9]); }\n', encoding="utf-8")
            result = execute(source)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("failure.heng", result.stderr)

    def test_short_circuit_and_evaluation_order(self):
        source = 'struct Box { n: i64 } fn change(b:Box)->i64{b.n+=1;return b.n;} fn main()->void{let b=Box{n:0};assert(change(b)*10+change(b)==12,"求值顺序错误");assert(true||change(b)==9,"短路错误");assert(b.n==2,"短路执行了右侧");}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "order.heng"
            path.write_text(source, encoding="utf-8")
            result = execute(path)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_local_variable_can_shadow_enum_name(self):
        source = 'enum Op{Add}struct Entry{value:i64}fn main(){let Op=Entry{value:1};Op.value=2;assert(Op.value==2,"局部变量遮蔽枚举名称失败");}'
        with tempfile.TemporaryDirectory(dir=ROOT / ".build") as directory:
            path = Path(directory) / "shadow.heng"
            path.write_text(source, encoding="utf-8")
            result = execute(path)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_fusion_preserves_result_and_reduces_allocations(self):
        source = 'fn main()->void{let x=full([8],2.0);let before=allocation_count();let y=relu(add(scale(x,3.0),full([],1.0)));let used=allocation_count()-before;assert_close(y,full([8],7.0),0.0,0.0);print(used);}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fusion.heng"
            path.write_text(source, encoding="utf-8")
            plain = execute(path, optimized=False)
            optimized = execute(path, optimized=True)
            self.assertEqual(plain.returncode, 0, plain.stderr)
            self.assertEqual(optimized.returncode, 0, optimized.stderr)
            self.assertLess(int(optimized.stdout.strip()), int(plain.stdout.strip()))

    def test_fusion_preserves_f32_promotion_rounding_and_side_effect_order(self):
        source = 'fn mutate(x:Tensor)->f64{set(x,[0],100.0);return 0.0;}fn main(){let x=cast(tensor([0.1,1.0000001,-0.3],[3]),"f32");let scaled=scale(x,3.7);let shifted=add(scaled,full([],0.13));let expected=relu(shifted);let actual=relu(add(scale(x,3.7),full([],0.13)));assert(dtype(actual)==dtype(expected),"融合类型提升不一致");assert_close(actual,expected,0.0,0.0);let value=ones([1]);let effected=relu(add(scale(value,2.0),full([],mutate(value))));assert_close(effected,full([1],2.0),0.0,0.0);}'
        with tempfile.TemporaryDirectory(dir=ROOT / ".build") as directory:
            path = Path(directory) / "fusion-order.heng"
            path.write_text(source, encoding="utf-8")
            for enabled in [False, True]:
                with self.subTest(optimized=enabled):
                    result = execute(path, optimized=enabled)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_recursion_limit_reports_a_language_error(self):
        source = 'fn recurse(n:i64)->i64{if n==0{return 0;}return recurse(n-1);}fn main()->void{print(recurse(6000));}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recursion.heng"
            path.write_text(source, encoding="utf-8")
            result = execute(path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("函数调用深度超过安全上限", result.stderr)
            self.assertIn("recursion.heng", result.stderr)

    def test_explicit_output_is_not_reused_for_a_different_program(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".build") as directory:
            root = Path(directory)
            first, second = root / "first.heng", root / "second.heng"
            first.write_text('fn main()->i64{return 11;}', encoding="utf-8")
            second.write_text('fn main()->i64{return 22;}', encoding="utf-8")
            build(second)
            destination = root / "custom.exe"
            build(first, output=destination)
            self.assertEqual(subprocess.run([str(destination)], timeout=30).returncode, 11)
            build(second, output=destination)
            self.assertEqual(subprocess.run([str(destination)], timeout=30).returncode, 22)

    def test_cli_formatter_preserves_line_endings_and_final_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "format.heng"
            for ending, final in [(b"\r\n", True), (b"\n", True), (b"\n", False)]:
                with self.subTest(ending=ending, final=final):
                    original = ending.join([b"fn main()->void{", b"print(1);", b"}"])
                    path.write_bytes(original + (ending if final else b""))
                    self.assertEqual(main(["fmt", str(path), "--write"]), 0)
                    formatted = path.read_bytes()
                    self.assertEqual(formatted.endswith(ending), final)
                    if ending == b"\r\n":
                        self.assertFalse(b"\n" in formatted.replace(b"\r\n", b""))
                    else:
                        self.assertFalse(b"\r" in formatted)
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(main(["fmt", str(path), "--check"]), 0)

    def test_cli_rejects_invalid_source_and_nonpositive_timeout(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertNotEqual(main(["check", str(ROOT / "missing.heng")]), 0)
            self.assertNotEqual(main(["run", str(ROOT / "tests" / "fixtures" / "language.heng"), "--timeout", "0"]), 0)


if __name__ == "__main__":
    unittest.main()
