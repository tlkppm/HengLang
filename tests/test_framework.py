from __future__ import annotations

import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from henglang.__main__ import ROOT, build, execute
from henglang.frontend import load_program
from test_native_runtime import state_bytes
from test_tensor_reference import tensor_literal
from tools.toolchain import environment


class FrameworkTests(unittest.TestCase):
    def test_all_framework_fixtures_compile_and_meet_their_contract(self):
        fixtures = sorted((ROOT / "tests" / "fixtures").glob("framework_*.heng"))
        expected = {"framework_autograd", "framework_operations", "framework_layers", "framework_optimizers", "framework_data", "framework_state"}
        self.assertTrue(expected.issubset({path.stem for path in fixtures}))
        for source in fixtures:
            with self.subTest(program=source.name):
                load_program(source)
                result = execute(source)
                if source.stem.startswith("framework_error_"):
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn(".heng", result.stderr)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("PASS:", result.stdout)


    def test_initializers_against_independent_random_reference(self):
        def random_values(normal):
            state = 412
            mask = (1 << 64) - 1
            def unit():
                nonlocal state
                state ^= state >> 12
                state ^= (state << 25) & mask
                state ^= state >> 27
                bits = (state * 2685821657736338717) & mask
                return (bits >> 11) / (1 << 53)
            return [math.sqrt(-2 * math.log(1 - unit())) * math.cos(2 * math.pi * unit()) if normal else unit() for _ in range(6)]
        uniform, normal = random_values(False), random_values(True)
        bound = math.sqrt(6 / 8)
        cases = [
            ("nn_init_normal(tape,parameter,-0.25,0.7);", [-0.25 + 0.7 * value for value in normal]),
            ("nn_init_uniform(tape,parameter,-0.8,0.4);", [-0.8 + 1.2 * value for value in uniform]),
            ("nn_init_xavier(tape,parameter,3,5);", [-bound + 2 * bound * value for value in uniform]),
            ("nn_init_kaiming(tape,parameter,3);", [math.sqrt(2 / 3) * value for value in normal]),
        ]
        body = []
        for dtype in ["f64", "f32"]:
            for operation, values in cases:
                expected = [struct.unpack("<f", struct.pack("<f", value))[0] for value in values] if dtype == "f32" else values
                tolerance = 1e-6 if dtype == "f32" else 1e-12
                body.append('{' + f'let tape=ag_tape();let module=nn_identity();let parameter=nn_parameter(tape,module,cast(zeros([2,3]),"{dtype}"));seed(412);{operation}assert(dtype(ag_value(tape,parameter))=="{dtype}","初始化必须保持参数类型");assert_close(ag_value(tape,parameter),{tensor_literal(expected,[2,3])},{tolerance},{tolerance});' + '}')
        with tempfile.TemporaryDirectory(dir=ROOT / ".build") as directory:
            source = Path(directory) / "initializers.heng"
            source.write_text('import "heng.heng";fn main(){' + "".join(body) + '}', encoding="utf-8")
            result = execute(source)
            self.assertEqual(result.returncode, 0, result.stderr)


class StateValidationTests(unittest.TestCase):
    cases = [
        ("bundle.ints[1]=2;", "标识或版本不匹配"),
        ("bundle.tensors[0]=zeros([4]);", "参数形状或类型不匹配"),
        ('bundle.tensors[0]=cast(bundle.tensors[0],"f32");', "参数形状或类型不匹配"),
        ('bundle.tensors[parameter_count]=cast(bundle.tensors[parameter_count],"f32");', "模块缓冲形状或类型不匹配"),
        ("bundle.tensors[parameter_count]=zeros([1,2]);", "模块缓冲形状或类型不匹配"),
        ('bundle.tensors[parameter_count+2]=cast(bundle.tensors[parameter_count+2],"f32");', "优化器张量形状或类型不匹配"),
        ("bundle.tensors[parameter_count+2]=zeros([4]);", "优化器张量形状或类型不匹配"),
        ("bundle.ints[11]=1;", "优化器参数映射不匹配"),
        ("bundle.ints[second_map]=1;", "共享参数映射不匹配"),
        ("let removed=pop(bundle.ints);", "字段不完整"),
        ("let removed=pop(bundle.floats);", "字段不完整"),
        ("push(bundle.ints,0);", "含多余字段"),
        ("push(bundle.floats,0.0);", "含多余字段"),
        ("let removed=pop(bundle.tensors);", "张量数量不匹配"),
        ("bundle.ints[6]=-1;", "优化器类型或步数无效"),
        ("bundle.ints[5]=99;", "优化器类型或步数无效"),
        ("bundle.ints[8]=2;", "优化器布尔字段无效"),
        ("bundle.floats[0]=-1.0;", "优化器超参数无效"),
        ("bundle.floats[0]=item(log(full([],-1.0)),0);", "非有限浮点值"),
        ("bundle.ints[7]=0;", "随机数状态不能为零"),
        ("bundle.ints[11+len(optimizer.parameters)+1]=2;", "模块类型或模式无效"),
        ("bundle.ints[11+len(optimizer.parameters)+14]=3;", "模块配置不匹配"),
        ("push(bundle.tensors,zeros([0]));", "张量数量不匹配"),
    ]

    @classmethod
    def setUpClass(cls):
        (ROOT / ".build").mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".build")
        cls.addClassCleanup(cls.temporary.cleanup)
        directory = Path(cls.temporary.name)
        cls.selector = directory / "case.hstate"
        source = directory / "state-validation.heng"
        selection = cls.selector.as_posix()
        branches = "".join(f"if selected=={index}{{{mutation}}}" for index, (mutation, _) in enumerate(cls.cases))
        body = f'import "heng.heng";fn main(){{let selected=state_ints("{selection}")[0];let tape=ag_tape();let model=nn_sequential();let linear=nn_linear(tape,2,2,true);nn_add(model,linear);nn_add(model,linear);nn_add(model,nn_batch_norm(tape,2,0.00001,0.1));let optimizer=opt_adam(tape,nn_parameters(model),0.01,0.9,0.999,0.00000001,0.0);let bundle=state_pack(tape,model,optimizer);let parameter_count=len(nn_parameters(model));let second_map=11+len(optimizer.parameters)+7+7+len(linear.config)+len(linear.parameters)+7+len(linear.config);{branches}state_validate(tape,model,optimizer,bundle);print("PASS: 状态预验证接受");}}'
        source.write_text(body, encoding="utf-8")
        cls.executable = build(source)

    def run_case(self, selected):
        self.selector.write_bytes(state_bytes(struct.pack("<QQQq", 0, 1, 0, selected)))
        return subprocess.run([str(self.executable)], cwd=ROOT, env=environment(), capture_output=True, encoding="utf-8", errors="replace", timeout=30)

    def test_valid_shared_model_state_is_accepted(self):
        result = self.run_case(-1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS:", result.stdout)

    def test_zero_rng_is_rejected_during_prevalidation(self):
        index = next(index for index, (mutation, _) in enumerate(self.cases) if mutation == "bundle.ints[7]=0;")
        result = self.run_case(index)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("随机数状态不能为零", result.stderr)
        self.assertNotIn("PASS:", result.stdout)

    def test_invalid_state_metadata_is_rejected_before_restore(self):
        for index, (mutation, message) in enumerate(self.cases):
            with self.subTest(mutation=mutation):
                result = self.run_case(index)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(message, result.stderr)
                self.assertIn("heng/state.heng:", result.stderr)
                self.assertNotIn("PASS:", result.stdout)


if __name__ == "__main__":
    unittest.main()
