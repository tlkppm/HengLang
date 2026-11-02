from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tokenize
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("comparison_benchmark", ROOT / "benchmarks" / "compare_frameworks.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def protocol_output(size=2, repeats=20, threads="1"):
    inputs = [benchmark.input_value(size, index, side) for side in ("left", "right") for index in benchmark.sample_positions(size)]
    return "\n".join([
        "HENG_THREADS_V1", threads,
        "HENG_INPUT_V1", *(str(value) for value in inputs),
        "HENG_OUTPUT_V1", *(str(value) for value in benchmark.reference_samples(size)),
        "HENG_TIMINGS_V1", *(str((index + 1) / 1000) for index in range(repeats)), "HENG_END_V1",
    ]) + "\n"


class ComparisonBenchmarkTests(unittest.TestCase):
    def test_defaults_are_requested_median_configuration(self):
        args = benchmark.parse_arguments([])
        self.assertEqual(args.sizes, [128, 256])
        self.assertEqual((args.warmup, args.repeats, args.frameworks), (3, 20, ["numpy"]))

    def test_configuration_rejects_invalid_sizes_and_sample_counts(self):
        cases = [([], 3, 20), ([1], 3, 20), ([2049], 3, 20), ([True], 3, 20),
                 ([2, 2], 3, 20), ([2], 2, 20), ([2], 3, 19), ([2], 3.0, 20)]
        for sizes, warmup, repeats in cases:
            with self.subTest(sizes=sizes, warmup=warmup, repeats=repeats):
                with self.assertRaises(benchmark.BenchmarkError):
                    benchmark.validate_configuration(sizes, warmup, repeats)

    def test_custom_configuration_and_timeout_validation(self):
        args = benchmark.parse_arguments(["--sizes", "3", "5", "--warmup", "4", "--repeats", "22", "--frameworks", "numpy", "torch"])
        self.assertEqual((args.sizes, args.warmup, args.repeats), ([3, 5], 4, 22))
        for arguments in (["--timeout", "nan"], ["--timeout", "0"], ["--build-timeout", "1"],
                          ["--frameworks", "numpy", "numpy"], ["--sizes", "bad"], ["--unknown"]):
            with self.subTest(arguments=arguments):
                with self.assertRaises(benchmark.BenchmarkError):
                    benchmark.parse_arguments(arguments)

    def test_deterministic_inputs_are_distinct_nonconstant_f64_values(self):
        self.assertEqual([benchmark.input_value(2, index, "left") for index in range(4)], [-0.5, -0.25, 0.0, 0.25])
        self.assertEqual([benchmark.input_value(2, index, "right") for index in range(4)], [0.25, 0.0625, -0.125, -0.3125])
        for side, index in (("unknown", 0), ("left", -1), ("right", 4)):
            with self.assertRaises(benchmark.BenchmarkError):
                benchmark.input_value(2, index, side)

    def test_independent_reference_has_known_matrix_product(self):
        self.assertEqual(benchmark.sample_positions(2), [0, 1, 2, 3])
        self.assertEqual(benchmark.reference_samples(2), [-0.09375, 0.046875, -0.03125, -0.078125])

    def test_reference_really_uses_math_fsum(self):
        with mock.patch.object(benchmark.math, "fsum", wraps=math.fsum) as summation:
            result = benchmark.reference_samples(3)
        self.assertEqual(summation.call_count, len(result))
        self.assertEqual(len(result), len(benchmark.sample_positions(3)))

    def test_median_uses_both_middle_samples(self):
        result = benchmark.summarize_timings([float(value) for value in range(20, 0, -1)], 20)
        self.assertEqual(result["median_seconds"], 10.5)
        self.assertEqual(len(result["samples_seconds"]), 20)

    def test_bad_timings_are_rejected(self):
        for values in ([1.0] * 19, [0.0] * 20, [-1.0] * 20, [math.nan] * 20, [math.inf] * 20):
            with self.subTest(values=values[:1]):
                with self.assertRaises(benchmark.BenchmarkError):
                    benchmark.summarize_timings(values, 20)

    def test_output_protocol_parses_and_validates_inputs_and_results(self):
        result = benchmark.parse_heng_output(protocol_output(), 2, 20)
        self.assertEqual(result["validation"]["status"], "PASS")
        self.assertEqual(result["input_validation"]["max_absolute_error"], 0.0)
        self.assertAlmostEqual(result["median_seconds"], 0.0105)
        self.assertTrue(result["threads"]["verified"])
        self.assertEqual(result["threads"]["num_threads"], 1)
        self.assertEqual(result["threads"]["query_symbol"], "get_threads")

    def test_output_protocol_rejects_invalid_or_mismatched_thread_configurations(self):
        for threads in ("0", "2", "256", "-1", "1.0", "nan", "inf", "true", "", "01", "+1", " 1"):
            with self.subTest(threads=threads):
                with self.assertRaisesRegex(benchmark.BenchmarkError, "get_threads"):
                    benchmark.parse_heng_output(protocol_output(threads=threads), 2, 20)

    def test_output_protocol_requires_actual_getter_marker_and_value(self):
        text = protocol_output()
        for invalid in (text.replace("HENG_THREADS_V1", "HENG_THREADS_UNKNOWN"),
                        "\n".join(text.splitlines()[2:]),
                        text.replace("HENG_THREADS_V1\n1\n", "HENG_THREADS_V1\n")):
            with self.assertRaises(benchmark.BenchmarkError):
                benchmark.parse_heng_output(invalid, 2, 20)

    def test_output_protocol_rejects_bad_marker_truncation_and_extra_lines(self):
        text = protocol_output()
        for value in (text.replace("HENG_INPUT_V1", "wrong"), text.rsplit("\n", 2)[0], text + "unexpected\n"):
            with self.assertRaises(benchmark.BenchmarkError):
                benchmark.parse_heng_output(value, 2, 20)

    def test_output_protocol_rejects_bad_input_output_and_timing(self):
        for position, value in ((3, "999"), (12, "999"), (12, "nan"), (17, "bad"), (17, "0")):
            lines = protocol_output().splitlines()
            lines[position] = value
            with self.subTest(position=position, value=value):
                with self.assertRaises(benchmark.BenchmarkError):
                    benchmark.parse_heng_output("\n".join(lines), 2, 20)

    def test_validation_rejects_nonfinite_and_wrong_samples(self):
        for values in ([math.nan], [math.inf], [2.0], []):
            with self.assertRaises(benchmark.BenchmarkError):
                benchmark.validate_samples(values, [1.0], "test")

    def test_heng_sources_pass_frontend_without_compiling(self):
        from henglang.frontend import check, parse
        for size in (2, 3, 128, 256):
            with self.subTest(size=size):
                source = benchmark.heng_source(size, 3, 20)
                program = check(parse(source, "comparison.heng"))
                self.assertIn("main", program.functions)
                self.assertLess(len(source), 5000)
                self.assertIn(f"arange({size * size})", source)
                self.assertIn("set_threads(1)", source)
                self.assertIn('print("HENG_THREADS_V1");', source)
                self.assertIn("print(get_threads());", source)

    def test_help_needs_no_optional_dependencies_or_project_imports(self):
        result = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "benchmarks" / "compare_frameworks.py"), "--help"],
                                cwd=ROOT, capture_output=True, encoding="utf-8", timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--sizes", result.stdout)
        self.assertIn("--frameworks", result.stdout)

    def test_missing_dependency_returns_json_without_installing_or_building(self):
        output = io.StringIO()
        with mock.patch.object(benchmark.importlib, "import_module", side_effect=ImportError), mock.patch.object(benchmark, "run_heng") as native, contextlib.redirect_stdout(output):
            status = benchmark.main([])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "FAIL")
        self.assertIn("不会安装依赖", json.loads(output.getvalue())["error"])
        native.assert_not_called()

    def test_invalid_arguments_return_json_error(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(benchmark.main(["--sizes", "not-a-number"]), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "FAIL")

    def test_preimported_framework_cannot_claim_import_time_thread_settings(self):
        with mock.patch.dict(sys.modules, {"numpy": types.SimpleNamespace()}):
            with self.assertRaisesRegex(benchmark.BenchmarkError, "已在设置线程环境前导入"):
                benchmark.load_framework("numpy")

    def test_thread_environment_is_set_without_exposing_other_variables(self):
        with mock.patch.dict(benchmark.os.environ, {}, clear=True):
            benchmark.configure_threads()
            self.assertTrue(all(benchmark.os.environ[name] == "1" for name in benchmark.THREAD_VARIABLES))
            self.assertEqual(benchmark.os.environ["OMP_DYNAMIC"], "FALSE")

    def test_threadpool_metadata_is_sanitized_and_checks_actual_counts(self):
        pool = {"user_api": "blas", "internal_api": "openblas", "num_threads": 1, "version": "test", "filepath": "private-path", "architecture": "private-architecture"}
        controller = types.SimpleNamespace(threadpool_info=lambda: [pool])
        with mock.patch.object(benchmark.importlib, "import_module", return_value=controller):
            result = benchmark.threadpool_information()
            self.assertTrue(result["verified"])
            self.assertNotIn("private-path", json.dumps(result))
            self.assertNotIn("private-architecture", json.dumps(result))
            pool["num_threads"] = 2
            self.assertFalse(benchmark.threadpool_information()["verified"])
        with mock.patch.object(benchmark.importlib, "import_module", side_effect=ImportError):
            self.assertFalse(benchmark.threadpool_information()["verified"])

    def test_openblas_queries_supported_exports_without_loading_new_dlls(self):
        symbols = ("openblas_get_num_threads64_", "openblas_get_num_threads",
                   "scipy_openblas_get_num_threads64_", "scipy_openblas_get_num_threads")
        for symbol in symbols:
            count = mock.Mock(return_value=1)
            prefix = "scipy_openblas" if symbol.startswith("scipy_") else "openblas"
            version = mock.Mock(return_value=b"OpenBLAS 0.3.31 USE64BITINT DYNAMIC_ARCH")
            library = types.SimpleNamespace(**{symbol: count, prefix + "_get_config64_": version})
            with self.subTest(symbol=symbol), mock.patch.object(benchmark, "loaded_blas_libraries", return_value=[("libopenblas.dll", library)]):
                result = benchmark.openblas_information()
            self.assertTrue(result["verified"])
            self.assertEqual(result["pools"][0]["query_symbol"], symbol)
            self.assertEqual(result["pools"][0]["version"], "0.3.31")
            self.assertEqual(count.argtypes, [])
            self.assertIs(count.restype, benchmark.ctypes.c_int)
            count.assert_called_once_with()

    def test_openblas_missing_unknown_or_multiple_threads_remain_unverified(self):
        for count in (0, -1, 2, 65537):
            library = types.SimpleNamespace(openblas_get_num_threads=mock.Mock(return_value=count))
            with self.subTest(count=count), mock.patch.object(benchmark, "loaded_blas_libraries", return_value=[("openblas.dll", library)]):
                self.assertFalse(benchmark.openblas_information()["verified"])
        known = types.SimpleNamespace(openblas_get_num_threads=mock.Mock(return_value=1))
        for libraries in ([], [("unknownblas.dll", types.SimpleNamespace())],
                          [("openblas.dll", known), ("mkl_rt.dll", types.SimpleNamespace())]):
            with mock.patch.object(benchmark, "loaded_blas_libraries", return_value=libraries):
                self.assertFalse(benchmark.openblas_information()["verified"])

    def test_openblas_query_failures_are_sanitized_and_not_claimed_verified(self):
        library = types.SimpleNamespace(openblas_get_num_threads=mock.Mock(side_effect=OSError("private-path secret-value")))
        with mock.patch.object(benchmark, "loaded_blas_libraries", return_value=[("openblas.dll", library)]):
            result = benchmark.openblas_information()
        self.assertFalse(result["verified"])
        self.assertNotIn("private-path", json.dumps(result))
        with mock.patch.object(benchmark, "loaded_blas_libraries", side_effect=OSError("private-path secret-value")):
            result = benchmark.openblas_information()
        self.assertFalse(result["verified"])
        self.assertNotIn("secret-value", json.dumps(result))

    def test_openblas_version_query_does_not_expose_full_configuration(self):
        library = types.SimpleNamespace(openblas_get_num_threads=mock.Mock(return_value=1),
                                        openblas_get_config=mock.Mock(return_value=b"OpenBLAS /private-path secret-value"))
        with mock.patch.object(benchmark, "loaded_blas_libraries", return_value=[("openblas.dll", library)]):
            result = benchmark.openblas_information()
        self.assertTrue(result["verified"])
        self.assertIsNone(result["pools"][0]["version"])
        self.assertNotIn("private-path", json.dumps(result))
        self.assertNotIn("secret-value", json.dumps(result))

    def test_threadpoolctl_absence_uses_openblas_actual_query(self):
        fallback = {"verified": True, "reason": "queried", "pools": [{"num_threads": 1}]}
        with mock.patch.object(benchmark.importlib, "import_module", side_effect=ImportError), mock.patch.object(benchmark, "openblas_information", return_value=fallback) as query:
            result = benchmark.threadpool_information()
        self.assertEqual(result, fallback)
        query.assert_called_once_with()

    def test_heng_execution_only_invokes_existing_cli_with_bounded_timeout(self):
        args = benchmark.parse_arguments([])
        completed = subprocess.CompletedProcess([], 0, protocol_output(), "")
        with mock.patch.object(benchmark.Path, "write_text") as write, mock.patch.object(benchmark.subprocess, "run", return_value=completed) as execute:
            result = benchmark.run_heng(ROOT / ".build" / "unused-test-directory", 2, args)
        self.assertTrue(result["threads"]["verified"])
        self.assertEqual(result["threads"]["num_threads"], 1)
        self.assertIn("不是全部操作系统线程活动采样", result["threads"]["reason"])
        self.assertEqual(execute.call_args.kwargs["timeout"], 300)
        self.assertEqual(execute.call_args.args[0][1:5], ["-B", "-m", "henglang", "run"])
        self.assertEqual(write.call_args.kwargs["newline"], "\n")

    def test_heng_errors_never_echo_private_diagnostics(self):
        args = benchmark.parse_arguments([])
        completed = subprocess.CompletedProcess([], 1, "", "private-path secret-value")
        with mock.patch.object(benchmark.Path, "write_text"), mock.patch.object(benchmark.subprocess, "run", return_value=completed):
            with self.assertRaises(benchmark.BenchmarkError) as caught:
                benchmark.run_heng(ROOT / ".build" / "unused-test-directory", 2, args)
        self.assertNotIn("secret-value", str(caught.exception))
        with mock.patch.object(benchmark.Path, "write_text"), mock.patch.object(benchmark.subprocess, "run", side_effect=subprocess.TimeoutExpired("private-path", 300)):
            with self.assertRaisesRegex(benchmark.BenchmarkError, "超过总超时"):
                benchmark.run_heng(ROOT / ".build" / "unused-test-directory", 2, args)

    def test_blas_metadata_includes_versions_but_excludes_paths(self):
        configuration = {"Build Dependencies": {"blas": {"name": "openblas", "version": "test-version", "lib directory": "private-path"}}}
        numpy = types.SimpleNamespace(__config__=types.SimpleNamespace(CONFIG=configuration))
        with mock.patch.object(benchmark, "threadpool_information", return_value={"pools": [], "verified": False}):
            result = benchmark.blas_metadata({"numpy": numpy})
        self.assertEqual(result["numpy_build"], {"name": "openblas", "version": "test-version"})
        self.assertNotIn("private-path", json.dumps(result))
        self.assertIn("未核实", result["version_status"])

    def test_single_thread_aggregation_requires_all_shapes_and_libraries_verified(self):
        def row(heng=True, numpy=True, torch=True):
            return {"heng": {"threads": {"verified": heng}}, "frameworks": {
                "numpy": {"threads": {"verified": numpy}}, "torch": {"threads": {"verified": torch}},
            }}
        self.assertTrue(benchmark.single_thread_configuration_verified([row(), row()]))
        for results in ([], [row(False)], [row(numpy=False)], [row(torch=False)],
                        [row(), row(torch=False)], [row(heng="true")], [row(numpy=None)],
                        [{"heng": {"threads": {"verified": True}}, "frameworks": {}}],
                        [{"heng": {}, "frameworks": {"numpy": {"threads": {"verified": True}}}}]):
            with self.subTest(results=results):
                self.assertFalse(benchmark.single_thread_configuration_verified(results))

    def test_run_returns_sanitized_report_with_dynamic_thread_aggregation(self):
        args = benchmark.parse_arguments(["--sizes", "2"])
        module = types.SimpleNamespace(__version__="test-version")
        for heng_verified, framework_verified in ((True, True), (True, False), (False, True), (False, False)):
            with self.subTest(heng=heng_verified, framework=framework_verified):
                heng_timing = {"median_seconds": 1.0, "threads": {"verified": heng_verified}}
                framework_timing = {"median_seconds": 1.0, "threads": {"verified": framework_verified}}
                temporary = mock.MagicMock()
                temporary.__enter__.return_value = ROOT / ".build" / "unused-test-directory"
                with mock.patch.object(benchmark, "load_framework", return_value=module), mock.patch.object(benchmark, "runtime_identity", return_value="test-digest"), mock.patch.object(benchmark, "run_heng", return_value=heng_timing), mock.patch.object(benchmark, "framework_benchmark", return_value=framework_timing), mock.patch.object(benchmark, "blas_metadata", return_value={"runtime_libraries": []}), mock.patch.object(benchmark.tempfile, "TemporaryDirectory", return_value=temporary):
                    with mock.patch.object(benchmark.os, "cpu_count", side_effect=AssertionError("禁止采集本机信息")), mock.patch("platform.system", side_effect=AssertionError("禁止采集本机信息")), mock.patch("platform.release", side_effect=AssertionError("禁止采集本机信息")), mock.patch("platform.machine", side_effect=AssertionError("禁止采集本机信息")), mock.patch("platform.processor", side_effect=AssertionError("禁止采集本机信息")), mock.patch("platform.python_version", side_effect=AssertionError("禁止采集本机信息")), mock.patch("platform.python_implementation", side_effect=AssertionError("禁止采集本机信息")):
                        report = benchmark.run(args)
                self.assertNotIn("hardware", report)
                self.assertNotIn("python", report["versions"])
                self.assertNotIn("python_implementation", report["versions"])
                self.assertEqual(report["privacy"]["machine_information"], "omitted")
                self.assertEqual(report["fair_single_thread"], heng_verified and framework_verified)
                self.assertIn("不保证 CPU 调度隔离", report["fair_single_thread_meaning"])
                self.assertEqual(report["results"][0]["frameworks"]["numpy"]["ratio_framework_over_heng"], 1.0)
                self.assertIn("blas", report["versions"])
                self.assertNotIn(str(ROOT), json.dumps(report))
                temporary.__exit__.assert_called_once()

    def test_files_are_utf8_with_consistent_line_endings_without_comments_or_docstrings(self):
        import ast
        for path in (ROOT / "benchmarks" / "compare_frameworks.py", Path(__file__)):
            with self.subTest(file=path.name):
                content = path.read_bytes()
                self.assertFalse(content.startswith(b"\xef\xbb\xbf"))
                normalized = content.replace(b"\r\n", b"\n")
                self.assertFalse(b"\r" in normalized, "存在独立 CR 换行")
                self.assertFalse(b"\r\n" in content and b"\n" in content.replace(b"\r\n", b""), "存在混合换行")
                source = content.decode("utf-8")
                self.assertFalse(any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(source).readline)))
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.assertIsNone(ast.get_docstring(node))


if __name__ == "__main__":
    unittest.main()
