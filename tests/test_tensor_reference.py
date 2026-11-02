from __future__ import annotations

import math
from pathlib import Path
import random
import tempfile
import unittest

from henglang.__main__ import ROOT, execute


def tensor_literal(values, shape):
    return "tensor([" + ",".join(repr(float(value)) for value in values) + "],[" + ",".join(map(str, shape)) + "])"


class TensorReferenceTests(unittest.TestCase):
    def run_heng(self, source: str, *, success: bool = True):
        directory = ROOT / ".build" / "test-sources"
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            path = Path(temporary) / "reference.heng"
            path.write_text("fn main() -> void {\n" + source + "\n}\n", encoding="utf-8")
            result = execute(path)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def test_random_matrix_multiply_against_independent_scalar_reference(self):
        rng = random.Random(731)
        rows, inner, columns = 5, 35, 41
        left = [rng.uniform(-2, 2) for _ in range(rows * inner)]
        right = [rng.uniform(-2, 2) for _ in range(inner * columns)]
        expected = [math.fsum(left[row * inner + index] * right[index * columns + column] for index in range(inner)) for row in range(rows) for column in range(columns)]
        self.run_heng(f'let a={tensor_literal(left,[rows,inner])};let b={tensor_literal(right,[inner,columns])};let expected={tensor_literal(expected,[rows,columns])};for optimized in [false,true]{{set_optimized(optimized);for threads in [1,2]{{set_threads(threads);assert_close(matmul(a,b),expected,1e-12,1e-12);}}}}')

    def test_batched_matrix_multiply_with_broadcast(self):
        self.run_heng('let a=tensor([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0],[2,2,2]);let b=tensor([1.0,0.0,0.0,2.0],[1,2,2]);assert_close(matmul(a,b),tensor([1.0,4.0,3.0,8.0,5.0,12.0,7.0,16.0],[2,2,2]),0.0,0.0);')

    def test_broadcast_and_empty_dimension(self):
        self.run_heng('assert_close(add(ones([2,1]),tensor([1.0,2.0,3.0],[1,3])),tensor([2.0,3.0,4.0,2.0,3.0,4.0],[2,3]),0.0,0.0);let empty=add(zeros([0,3]),ones([1,3]));assert(numel(empty)==0,"空维度广播错误");assert(shape(empty)[0]==0,"空维度形状错误");assert_close(sum_all(empty),tensor([0.0],[]),0.0,0.0);')

    def test_views_alias_and_shared_version(self):
        self.run_heng('let a=arange(6);let b=reshape(a,[2,3]);let c=transpose(b,[1,0]);let version=tensor_version(a);set(c,[2,1],19.0);assert(item(a,5)==19.0,"视图未共享存储");assert(tensor_version(a)>version,"共享版本未更新");assert_close(reshape(c,[6]),tensor([0.0,3.0,1.0,4.0,2.0,19.0],[6]),0.0,0.0);')

    def test_arbitrary_rank_transpose_and_reduction(self):
        self.run_heng('let a=reshape(arange(24),[2,3,2,2]);let b=transpose(a,[3,0,2,1]);assert_close(sum(b,3,false),tensor([12.0,18.0,48.0,54.0,15.0,21.0,51.0,57.0],[2,2,2]),0.0,0.0);')

    def test_gather_scatter_and_concat(self):
        self.run_heng('let a=tensor([1.0,2.0,3.0,4.0,5.0,6.0],[2,3]);assert_close(gather(a,[2,0],1),tensor([3.0,1.0,6.0,4.0],[2,2]),0.0,0.0);let s=scatter_add(zeros([3]),[1,1],tensor([2.0,3.0],[2]),0);assert_close(s,tensor([0.0,5.0,0.0],[3]),0.0,0.0);assert_close(concat(ones([2,1]),zeros([2,2]),1),tensor([1.0,0.0,0.0,1.0,0.0,0.0],[2,3]),0.0,0.0);')

    def test_stable_softmax_against_python_math(self):
        values = [1000.0, 1001.0, 999.0]
        denominator = math.fsum(math.exp(value - max(values)) for value in values)
        expected = [math.exp(value - max(values)) / denominator for value in values]
        self.run_heng(f'assert_close(softmax({tensor_literal(values,[1,3])},1),{tensor_literal(expected,[1,3])},1e-13,1e-13);')

    def test_sparse_duplicates_and_csr(self):
        self.run_heng('let a=coo([0,0,1],[0,0,1],[2.0,3.0,4.0],2,2);let expected=tensor([5.0,0.0,0.0,4.0],[2,2]);assert_close(sparse_dense(a),expected,0.0,0.0);let b=csr([0,1,2],[0,1],[5.0,4.0],2,2);assert_close(sparse_matmul(b,ones([2,1])),tensor([5.0,4.0],[2,1]),0.0,0.0);assert_close(sparse_dense(sparse_transpose(a)),expected,0.0,0.0);')

    def test_csr_empty_rows_and_transpose_against_dense_reference(self):
        self.run_heng('let a=csr([0,0,2,2,3],[2,0,1],[3.0,2.0,4.0],4,3);let expected=tensor([0.0,0.0,0.0,2.0,0.0,3.0,0.0,0.0,0.0,0.0,4.0,0.0],[4,3]);assert_close(sparse_dense(a),expected,0.0,0.0);let transposed=sparse_transpose(a);assert_close(sparse_dense(transposed),transpose(expected,[1,0]),0.0,0.0);assert_close(sparse_matmul(a,ones([3,2])),tensor([0.0,0.0,5.0,5.0,0.0,0.0,4.0,4.0],[4,2]),0.0,0.0);assert_close(sparse_matmul(transposed,ones([4,1])),tensor([2.0,4.0,3.0],[3,1]),0.0,0.0);let empty=csr([0,0,0],[],[],2,3);assert_close(sparse_dense(empty),zeros([2,3]),0.0,0.0);assert_close(sparse_dense(sparse_transpose(empty)),zeros([3,2]),0.0,0.0);')

    def test_unfold_fold_reference(self):
        self.run_heng('let a=reshape(arange(9),[1,1,3,3]);let u=unfold2d(a,2,2,1,0);assert_close(u,tensor([0.0,1.0,3.0,4.0,1.0,2.0,4.0,5.0,3.0,4.0,6.0,7.0,4.0,5.0,7.0,8.0],[1,4,4]),0.0,0.0);assert_close(fold2d(u,[1,1,3,3],2,2,1,0),tensor([0.0,2.0,2.0,6.0,16.0,10.0,6.0,14.0,8.0],[1,1,3,3]),0.0,0.0);')

    def test_dtypes_and_deterministic_rng(self):
        self.run_heng('let a=cast(tensor([1.75,-2.5],[2]),"f32");assert(dtype(a)=="f32","f32类型错误");assert_close(cast(a,"i64"),tensor([1.0,-2.0],[2]),0.0,0.0);seed(1234);let first=randn([7]);seed(1234);assert_close(first,randn([7]),0.0,0.0);let s=rng_state();let next=rand([9]);rng_restore(s);assert_close(next,rand([9]),0.0,0.0);')

    def test_state_round_trip_and_checksum_failure(self):
        directory = ROOT / ".build" / "test-sources"
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            state = Path(temporary) / "state.bin"
            encoded = str(state).replace("\\", "/")
            self.run_heng(f'state_save("{encoded}",[cast(arange(6),"f32")],[7,9],[0.25]);assert_close(state_tensors("{encoded}")[0],arange(6),0.0,0.0);assert(dtype(state_tensors("{encoded}")[0])=="f32","存储类型未保留");assert(state_ints("{encoded}")[1]==9,"整数状态未保存");assert(state_floats("{encoded}")[0]==0.25,"浮点状态未保存");')
            content = bytearray(state.read_bytes())
            content[-1] ^= 1
            state.write_bytes(content)
            self.run_heng(f'let state=state_tensors("{encoded}");print(len(state));', success=False)

    def test_runtime_rejects_invalid_inputs(self):
        invalid = ['let s=coo([2],[0],[1.0],2,2);print(sparse_dense(s));', 'let s=csr([0,2,1],[0],[1.0],2,2);print(sparse_dense(s));', 'let a=zeros([2]);print(item(a,5));', 'let a=ones([2]);print(slice(a,0,0,2,0));']
        for source in invalid:
            with self.subTest(source=source):
                self.run_heng(source, success=False)


if __name__ == "__main__":
    unittest.main()
