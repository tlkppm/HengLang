from __future__ import annotations

import math
from pathlib import Path
import random
import re
import struct
import subprocess
import sys
import tempfile
import unittest

from henglang.__main__ import ROOT, build, execute


def rounded(value, dtype):
    if dtype == "f32":
        try:
            return struct.unpack("<f", struct.pack("<f", value))[0]
        except OverflowError:
            return math.copysign(math.inf, value)
    return value


def tensor(values, shape, dtype="f64"):
    literals = [f'f64("{value}")' if not math.isfinite(value) else repr(float(value)) for value in values]
    result = "tensor([" + ",".join(literals) + "],[" + ",".join(map(str, shape)) + "])"
    return result if dtype == "f64" else f'cast({result},"{dtype}")'


def compensated(values):
    total = correction = 0.0
    for value in values:
        adjusted = value - correction
        following = total + adjusted
        correction = (following - total) - adjusted if math.isfinite(following) else 0.0
        total = following
    return total


def matrix_reference(left, right, rows, inner, columns, dtype="f64"):
    return [rounded(compensated(left[row * inner + index] * right[index * columns + column] for index in range(inner)), dtype) for row in range(rows) for column in range(columns)]


class RuntimeFastpathTests(unittest.TestCase):
    def run_heng(self, source, *, success=True):
        directory = ROOT / ".build" / "test-sources"
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            path = Path(temporary) / "fastpaths.heng"
            path.write_text("fn main() -> void {\n" + source + "\n}\n", encoding="utf-8")
            result = execute(path)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def assert_printed(self, output, expected):
        lines = output.splitlines()
        self.assertEqual(len(lines), len(expected), output)
        for line, values in zip(lines, expected):
            match = re.fullmatch(r"tensor\(\[(.*?)\], shape=\[.*?\], dtype=\w+\)", line)
            self.assertIsNotNone(match, line)
            actual = [] if not match.group(1) else [math.nan if "nan" in value.lower() else float(value) for value in match.group(1).split(", ")]
            self.assertEqual(len(actual), len(values), line)
            for position, (number, reference) in enumerate(zip(actual, values)):
                with self.subTest(position=position, reference=reference):
                    if math.isnan(reference):
                        self.assertTrue(math.isnan(number))
                    else:
                        self.assertEqual(number, reference)
                        if reference == 0:
                            self.assertEqual(math.copysign(1, number), math.copysign(1, reference))

    def test_thread_configuration_can_be_verified(self):
        self.run_heng('assert(get_threads()==1,"默认线程配置错误");for workers in [1,2,4,256]{set_threads(workers);assert(get_threads()==workers,"线程配置查询错误");}set_threads(1);assert_close(matmul(ones([2,3]),ones([3,4])),full([2,4],3.0),0.0,0.0);')

    def test_matmul_dtype_specialization_and_block_tails(self):
        generator = random.Random(8103)
        rows, inner, columns = 3, 35, 37
        first = [generator.uniform(-3, 3) for _ in range(rows * inner)]
        second = [generator.uniform(-3, 3) for _ in range(inner * columns)]
        sources = []
        expected = []
        for ld, rd in [("f64", "f64"), ("f32", "f32"), ("f64", "f32"), ("f32", "f64")]:
            left = [rounded(value, ld) for value in first]
            right = [rounded(value, rd) for value in second]
            dtype = "f32" if ld == rd == "f32" else "f64"
            reference = matrix_reference(left, right, rows, inner, columns, dtype)
            sources.append(f'let a={tensor(left,[rows,inner],ld)};let b={tensor(right,[inner,columns],rd)};for optimized in [false,true]{{set_optimized(optimized);for threads in [1,2,4]{{set_threads(threads);let c=matmul(a,b);assert(dtype(c)=="{dtype}","矩阵类型错误");print(c);}}}}')
            expected.extend([reference] * 6)
        self.assert_printed(self.run_heng("".join("{" + source + "}" for source in sources)).stdout, expected)

    def test_matmul_strided_transposed_and_offset_views(self):
        first = [float(index % 17 - 8) / 8 for index in range(6 * 35)]
        second = [float(index % 23 - 11) / 16 for index in range(37 * 35)]
        left = [first[row * 35 + index] for row in [5, 3, 1] for index in range(34, -1, -1)]
        right = [second[column * 35 + index] for index in range(34, -1, -1) for column in range(36, -1, -1)]
        reference = matrix_reference(left, right, 3, 35, 37)
        source = f'let a=slice(slice({tensor(first,[6,35])},0,5,0,-2),1,34,-1,-1);let b=slice(slice(transpose({tensor(second,[37,35])},[1,0]),0,34,-1,-1),1,36,-1,-1);for optimized in [false,true]{{set_optimized(optimized);for threads in [1,4]{{set_threads(threads);print(matmul(a,b));}}}}'
        self.assert_printed(self.run_heng(source).stdout, [reference] * 4)

    def test_matmul_batched_broadcast_and_zero_strides(self):
        left = [float(index - 5) / 4 for index in range(12)]
        right = [float(index % 7 - 3) / 2 for index in range(4 * 3 * 5)]
        reference = []
        for batch in range(2):
            for other in range(4):
                reference.extend(matrix_reference(left[batch * 6:(batch + 1) * 6], right[other * 15:(other + 1) * 15], 2, 3, 5))
        source = f'let a={tensor(left,[2,1,2,3])};let b={tensor(right,[4,3,5])};for optimized in [false,true]{{set_optimized(optimized);for threads in [1,4]{{set_threads(threads);print(matmul(a,b));print(matmul(broadcast_to(ones([1,1]),[2,3]),broadcast_to(full([1,1],2.0),[3,5])));}}}}'
        expected = []
        for _ in range(4):
            expected.extend([reference, [6.0] * 10])
        self.assert_printed(self.run_heng(source).stdout, expected)

    def test_matmul_compensation_nonfinite_and_signed_zero(self):
        left = [1e16] + [1.0] * 33 + [-1e16]
        sources = []
        expected = []
        cases = [(left, [1.0] * (35 * 33), 1, 35, 33), ([math.inf, 1.0, -math.inf, math.nan], [1.0] * 4, 4, 1, 4), ([math.inf, -math.inf, 1.0], [1.0] * 3, 1, 3, 1), ([-0.0], [1.0, -1.0], 1, 1, 2)]
        for dtype in ["f64", "f32"]:
            for first, second, rows, inner, columns in cases:
                a = [rounded(value, dtype) for value in first]
                b = [rounded(value, dtype) for value in second]
                sources.append(f'{{let a={tensor(a,[rows,inner],dtype)};let b={tensor(b,[inner,columns],dtype)};for optimized in [false,true]{{set_optimized(optimized);for threads in [1,4]{{set_threads(threads);print(matmul(a,b));}}}}}}')
                expected.extend([matrix_reference(a, b, rows, inner, columns, dtype)] * 4)
        self.assert_printed(self.run_heng("".join(sources)).stdout, expected)

    def test_contiguous_binary_float_dtypes_and_scalar_directions(self):
        first = [-2.25, -0.0, 1.0000001192092896, 16777217.0, -1e-30, 3.5]
        second = [0.5, -2.0, 1.000000238418579, 3.0, 1e-10, -0.25]
        operations = [("add", lambda a, b: a + b), ("sub", lambda a, b: a - b), ("mul", lambda a, b: a * b), ("div", lambda a, b: a / b)]
        sources = []
        expected = []
        for ld, rd in [("f64", "f64"), ("f32", "f32"), ("f64", "f32"), ("f32", "f64")]:
            left = [rounded(value, ld) for value in first]
            right = [rounded(value, rd) for value in second]
            dtype = "f32" if ld == rd == "f32" else "f64"
            statements = [f'let a={tensor(left,[2,3],ld)};let b={tensor(right,[2,3],rd)};let s={tensor([0.5],[],rd)};']
            for name, operation in operations:
                statements.append(f'print({name}(a,b));print({name}(a,s));')
                expected.extend([[rounded(operation(a, b), dtype) for a, b in zip(left, right)], [rounded(operation(a, 0.5), dtype) for a in left]])
                if name != "div":
                    statements.append(f'print({name}(s,a));')
                    expected.append([rounded(operation(0.5, a), dtype) for a in left])
            sources.append("{" + "".join(statements) + "}")
        self.assert_printed(self.run_heng("".join(sources)).stdout, expected)

    def test_binary_nonfinite_subnormal_and_scalar_left_division(self):
        first = [math.inf, -math.inf, math.nan, 0.0, -0.0, 1e-40, 3e38]
        second = [1.0, -1.0, 2.0, -2.0, -2.0, 0.125, 4.0]
        operations = [("add", lambda a, b: a + b), ("sub", lambda a, b: a - b), ("mul", lambda a, b: a * b), ("div", lambda a, b: a / b)]
        sources = []
        expected = []
        for dtype in ["f64", "f32"]:
            left = [rounded(value, dtype) for value in first]
            right = [rounded(value, dtype) for value in second]
            statements = [f'let a={tensor(left,[7],dtype)};let b={tensor(right,[7],dtype)};let scalar={tensor([0.5],[],dtype)};']
            for name, operation in operations:
                statements.append(f'{{let result={name}(a,b);assert(dtype(result)=="{dtype}","二元结果类型错误");print(result);}}')
                expected.append([rounded(operation(a, b), dtype) for a, b in zip(left, right)])
            statements.append('print(div(scalar,b));')
            expected.append([rounded(0.5 / value, dtype) for value in right])
            sources.append("{" + "".join(statements) + "}")
        self.assert_printed(self.run_heng("".join(sources)).stdout, expected)

    def test_contiguous_offset_singletons_and_strided_fallbacks(self):
        source = 'let base=reshape(arange(24),[4,6]);let offset=slice(base,0,1,3,1);print(add(offset,full([],0.5)));print(scale(offset,0.5));let reversed=slice(base,1,5,-1,-2);print(add(reversed,ones([4,3])));print(scale(reversed,-0.5));print(relu(-reversed));let transposed=transpose(base,[1,0]);print(add(transposed,full([1,4],1.0)));print(fused_affine_relu(transposed,-0.5,2.0));let singleton=transpose(reshape(tensor([2.0,3.0,4.0],[3]),[1,3,1]),[2,1,0]);print(mul(singleton,full([],2.0)));print(sub(full([1,1,1],8.0),singleton));'
        reverse = [float(row * 6 + column) for row in range(4) for column in [5, 3, 1]]
        transposed = [float(row * 6 + column) for column in range(6) for row in range(4)]
        expected = [[value + 0.5 for value in range(6, 18)], [value * 0.5 for value in range(6, 18)], [value + 1 for value in reverse], [value * -0.5 for value in reverse], [0.0] * 12, [value + 1 for value in transposed], [max(value * -0.5 + 2, 0.0) for value in transposed], [4.0, 6.0, 8.0], [6.0, 5.0, 4.0]]
        self.assert_printed(self.run_heng(source).stdout, expected)

    def test_unary_f32_rounding_nonfinite_and_fallbacks(self):
        values = [-math.inf, -3.25, -0.0, 0.0, 1.0, 1.0000001192092896, math.inf, math.nan]
        parameter, bias = 1.0000000596046448, -1.0
        sources = []
        expected = []
        for dtype in ["f64", "f32"]:
            first = [rounded(value, dtype) for value in values]
            scaled = [rounded(value * parameter, dtype) for value in first]
            relu = [value if math.isnan(value) or value > 0 else 0.0 for value in first]
            fused = [value + bias if math.isnan(value + bias) or value + bias > 0 else 0.0 for value in scaled]
            sources.append(f'{{let a={tensor(first,[8],dtype)};print(scale(a,{parameter}));print(relu(a));print(fused_affine_relu(a,{parameter},{bias}));let b=slice(a,0,7,-1,-1);print(scale(b,{parameter}));print(relu(b));print(fused_affine_relu(b,{parameter},{bias}));}}')
            expected.extend([scaled, relu, fused, scaled[::-1], relu[::-1], fused[::-1]])
        self.assert_printed(self.run_heng("".join(sources)).stdout, expected)

    def test_empty_shapes_and_empty_inner_dimension(self):
        source = 'for optimized in [false,true]{set_optimized(optimized);for threads in [1,4]{set_threads(threads);assert_close(matmul(zeros([3,0]),zeros([0,35])),zeros([3,35]),0.0,0.0);assert(numel(matmul(zeros([0,3]),ones([3,35])))==0,"空行错误");assert(numel(matmul(ones([3,2]),zeros([2,0])))==0,"空列错误");assert(numel(matmul(zeros([0,2,3]),ones([1,3,4])))==0,"空批次错误");}}let a=cast(zeros([2,0,3]),"f32");assert(numel(add(a,full([],1.0)))==0,"空广播错误");assert(numel(scale(a,2.0))==0,"空缩放错误");assert(numel(relu(a))==0,"空激活错误");assert(numel(fused_affine_relu(a,2.0,1.0))==0,"空融合错误");assert(numel(add(ones([1,3]),zeros([0,3])))==0,"空广播后备错误");'
        self.run_heng(source)

    def test_mixed_integer_fallback_and_overflow_guards(self):
        self.run_heng('let a=cast(tensor([1.0,2.0,3.0,4.0],[2,2]),"i64");let b=cast(tensor([0.5,1.5,2.5,3.5],[2,2]),"f32");for optimized in [false,true]{set_optimized(optimized);assert_close(matmul(a,b),tensor([5.5,8.5,11.5,18.5],[2,2]),0.0,0.0);}assert_close(add(a,b),tensor([1.5,3.5,5.5,7.5],[2,2]),0.0,0.0);assert(dtype(equal(a,b))=="bool","比较类型错误");assert_close(cast(equal(b,b),"f64"),ones([2,2]),0.0,0.0);')
        for optimized in ["false", "true"]:
            for body, error in [('print(matmul(cast(full([1,2],4611686018427387904.0),"i64"),cast(ones([2,1]),"i64")));', "i64 加法溢出"), ('print(matmul(cast(full([1,1],4611686018427387904.0),"i64"),cast(full([1,1],4.0),"i64")));', "i64 乘法溢出")]:
                result = self.run_heng(f'set_optimized({optimized});set_threads(2);{body}', success=False)
                self.assertIn(error, result.stderr)
        for body, error in [('print(add(cast(full([2],4611686018427387904.0),"i64"),cast(full([2],4611686018427387904.0),"i64")));', "i64 加法溢出"), ('print(div(ones([2]),zeros([])));', "张量除零"), ('print(div(cast(ones([2]),"f32"),cast(zeros([2]),"f32")));', "张量除零")]:
            self.assertIn(error, self.run_heng(body, success=False).stderr)

    def test_undefined_behavior_fastpaths(self):
        source = 'fn main() -> void {for dtype_name in ["f64","f32"]{let a=cast(reshape(arange(210),[6,35]),dtype_name);let b=cast(reshape(arange(1295),[35,37]),dtype_name);let left=slice(a,0,5,0,-2);let right=slice(b,1,36,-1,-1);set_optimized(false);let expected=matmul(left,right);set_optimized(true);set_threads(4);assert_close(matmul(left,right),expected,0.0,0.0);let offset=slice(a,0,1,4,1);let one=cast(full([],1.0),dtype_name);assert_close(add(offset,one),add(clone(offset),one),0.0,0.0);assert_close(relu(scale(offset,-0.5)),relu(scale(clone(offset),-0.5)),0.0,0.0);assert_close(fused_affine_relu(offset,1.0000000596046448,-1.0),relu(add(scale(offset,1.0000000596046448),full([],-1.0))),0.0,0.0);assert(numel(mul(cast(zeros([0,3]),dtype_name),one))==0,"空路径错误");}print("PASS fastpath UBSan");}'
        directory = ROOT / ".build" / "test-sources"
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            path = Path(temporary) / "instrumented.heng"
            path.write_text(source, encoding="utf-8")
            generated = build(path).parent / "program.c"
            executable = Path(temporary) / "instrumented.exe"
            command = [sys.executable, "-B", str(ROOT / "tools" / "toolchain.py"), "cc", "-std=c17", "-O1", "-g", "-fno-fast-math", "-fsanitize=undefined", "-fno-sanitize-recover=all", "-I", str(ROOT / "runtime"), str(generated), str(ROOT / "runtime" / "heng_runtime.c"), "-o", str(executable)]
            compiled = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=240)
            self.assertEqual(compiled.returncode, 0, compiled.stderr.decode("utf-8", "replace"))
            result = subprocess.run([str(executable)], cwd=ROOT, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
            self.assertEqual(result.stderr, b"")
            self.assertIn(b"PASS fastpath UBSan", result.stdout)


if __name__ == "__main__":
    unittest.main()
