from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from henglang.__main__ import ROOT, execute
from test_tensor_reference import tensor_literal


def finite_difference(function, values, epsilon=1e-6):
    gradients = []
    for index in range(len(values)):
        plus, minus = list(values), list(values)
        plus[index] += epsilon
        minus[index] -= epsilon
        gradients.append((function(plus) - function(minus)) / (2 * epsilon))
    return gradients


class AutogradReferenceTests(unittest.TestCase):
    def run_heng(self, body: str):
        directory = ROOT / ".build" / "gradient-sources"
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            source = Path(temporary) / "gradients.heng"
            source.write_text('import "ops.heng";\nfn main() -> void {\n' + body + '\n}\n', encoding="utf-8")
            result = execute(source)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_composite_unary_gradient_against_analytic_and_central_difference(self):
        values = [-0.8, 0.2, 1.1]
        function = lambda inputs: math.fsum(math.tanh(value) * value + math.exp(value) for value in inputs)
        numerical = finite_difference(function, values)
        analytic = [math.tanh(value) + value * (1 - math.tanh(value)**2) + math.exp(value) for value in values]
        for exact, approximate in zip(analytic, numerical):
            self.assertAlmostEqual(exact, approximate, places=7)
        self.run_heng(f'let tape=ag_tape();let x=ag_variable(tape,{tensor_literal(values,[3])},true);let y=ag_sum_all(tape,ag_add(tape,ag_mul(tape,ag_tanh(tape,x),x),ag_exp(tape,x)));ag_backward(tape,y,ones([]),true);assert_close(ag_grad(tape,x),{tensor_literal(numerical,[3])},5e-7,5e-7);')

    def test_broadcast_binary_gradient_against_independent_finite_difference(self):
        left, right = [0.4, 1.2], [0.2, 0.6, 1.1]
        def function(values):
            return math.fsum((x * y + x / (y + 2))**2 for x in values[:2] for y in values[2:])
        expected = finite_difference(function, left + right)
        self.run_heng(f'let tape=ag_tape();let x=ag_variable(tape,{tensor_literal(left,[2,1])},true);let y=ag_variable(tape,{tensor_literal(right,[1,3])},true);let denominator=ag_add(tape,y,ag_constant(tape,full([],2.0)));let v=ag_add(tape,ag_mul(tape,x,y),ag_div(tape,x,denominator));let loss=ag_sum_all(tape,ag_pow(tape,v,2.0));ag_backward(tape,loss,ones([]),true);assert_close(ag_grad(tape,x),{tensor_literal(expected[:2],[2,1])},5e-7,5e-7);assert_close(ag_grad(tape,y),{tensor_literal(expected[2:],[1,3])},5e-7,5e-7);')

    def test_nondifferentiable_inputs_and_invalid_seeds_are_rejected(self):
        cases = [
            ('let x=ag_variable(tape,cast(ones([2]),"bool"),true);', '不能请求梯度'),
            ('let x=ag_variable(tape,cast(ones([2]),"i64"),true);', '不能请求梯度'),
            ('let x=ag_variable(tape,ones([2]),true);let y=ag_sum_all(tape,x);ag_backward(tape,y,cast(ones([]),"i64"),false);', '种子必须为浮点'),
            ('let x=ag_variable(tape,ones([2]),true);let y=ag_sum_all(tape,x);let value=ag_jvp(tape,y,[x],[cast(ones([2]),"bool")]);', '种子必须为浮点'),
            ('let x=ag_variable(tape,ones([2]),true);ag_assign(tape,x,cast(ones([2]),"f32"));', '参数替换类型不一致'),
        ]
        for body, message in cases:
            with self.subTest(message=message, body=body), tempfile.TemporaryDirectory(dir=ROOT / ".build") as directory:
                source = Path(directory) / "invalid-gradient.heng"
                source.write_text('import "ops.heng";fn main(){let tape=ag_tape();' + body + '}', encoding="utf-8")
                result = execute(source)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(message, result.stderr)

    def test_all_public_operations_against_independent_reference(self):
        inputs = [0.4, 0.8, 1.3, 1.7, 1.2, 0.4, -0.3, 1.5]
        direction = [0.3, -0.2, 0.5, 0.7, -0.2, 0.6, 0.1, -0.4]
        cases = [
            ("add", "ag_add(tape,x,w)", [2,2], lambda a,b: [x+y for x,y in zip(a,b)]),
            ("sub", "ag_sub(tape,x,w)", [2,2], lambda a,b: [x-y for x,y in zip(a,b)]),
            ("mul", "ag_mul(tape,x,w)", [2,2], lambda a,b: [x*y for x,y in zip(a,b)]),
            ("div", "ag_div(tape,x,w)", [2,2], lambda a,b: [x/y for x,y in zip(a,b)]),
            ("scale", "ag_scale(tape,x,1.7)", [2,2], lambda a,b: [x*1.7 for x in a]),
            ("pow", "ag_pow(tape,x,2.3)", [2,2], lambda a,b: [x**2.3 for x in a]),
            ("exp", "ag_exp(tape,x)", [2,2], lambda a,b: [math.exp(x) for x in a]),
            ("log", "ag_log(tape,x)", [2,2], lambda a,b: [math.log(x) for x in a]),
            ("sqrt", "ag_sqrt(tape,x)", [2,2], lambda a,b: [math.sqrt(x) for x in a]),
            ("tanh", "ag_tanh(tape,x)", [2,2], lambda a,b: [math.tanh(x) for x in a]),
            ("sigmoid", "ag_sigmoid(tape,x)", [2,2], lambda a,b: [1/(1+math.exp(-x)) for x in a]),
            ("relu", "ag_relu(tape,x)", [2,2], lambda a,b: [max(x,0) for x in a]),
            ("abs", "ag_abs(tape,x)", [2,2], lambda a,b: [abs(x) for x in a]),
            ("reshape", "ag_reshape(tape,x,[4])", [4], lambda a,b: list(a)),
            ("transpose", "ag_transpose(tape,x,[1,0])", [2,2], lambda a,b: [a[0],a[2],a[1],a[3]]),
            ("sum", "ag_sum(tape,x,1,false)", [2], lambda a,b: [a[0]+a[1],a[2]+a[3]]),
            ("sum_all", "ag_sum_all(tape,x)", [], lambda a,b: [math.fsum(a)]),
            ("mean", "ag_mean(tape,x,1,false)", [2], lambda a,b: [(a[0]+a[1])/2,(a[2]+a[3])/2]),
            ("matmul", "ag_matmul(tape,x,w)", [2,2], lambda a,b: [math.fsum(a[r*2+k]*b[k*2+c] for k in range(2)) for r in range(2) for c in range(2)]),
            ("softmax", "ag_softmax(tape,x,1)", [2,2], lambda a,b: [math.exp(a[r*2+c]-max(a[r*2:r*2+2]))/math.fsum(math.exp(x-max(a[r*2:r*2+2])) for x in a[r*2:r*2+2]) for r in range(2) for c in range(2)]),
            ("broadcast", "ag_broadcast(tape,x,[2,2,2])", [2,2,2], lambda a,b: list(a)*2),
            ("reduce", "ag_reduce_to_shape(tape,x,[1,2])", [1,2], lambda a,b: [a[0]+a[2],a[1]+a[3]]),
            ("slice", "ag_slice(tape,x,1,0,2,2)", [2,1], lambda a,b: [a[0],a[2]]),
            ("concat", "ag_concat(tape,x,w,0)", [4,2], lambda a,b: list(a)+list(b)),
            ("gather", "ag_gather(tape,x,[1,0,1],0)", [3,2], lambda a,b: [a[2],a[3],a[0],a[1],a[2],a[3]]),
            ("unfold", "ag_unfold2d(tape,ag_reshape(tape,x,[1,1,2,2]),2,2,1,1)", [1,4,9], lambda a,b: [a[(r+kh-1)*2+c+kw-1] if 0<=r+kh-1<2 and 0<=c+kw-1<2 else 0.0 for kh in range(2) for kw in range(2) for r in range(3) for c in range(3)]),
            ("fold", "ag_fold2d(tape,ag_reshape(tape,x,[1,4,1]),[1,1,2,2],2,2,1,0)", [1,1,2,2], lambda a,b: list(a)),
            ("max", "ag_max(tape,x,1,false)", [2], lambda a,b: [max(a[:2]),max(a[2:])]),
            ("log_softmax", "ag_log_softmax(tape,x,1)", [2,2], lambda a,b: [a[r*2+c]-max(a[r*2:r*2+2])-math.log(math.fsum(math.exp(x-max(a[r*2:r*2+2])) for x in a[r*2:r*2+2])) for r in range(2) for c in range(2)]),
        ]
        body = []
        for name, expression, shape, reference in cases:
            values = reference(inputs[:4], inputs[4:])
            cotangent = [(index+1)/len(values) for index in range(len(values))]
            scalar = lambda arguments: math.fsum(x*y for x,y in zip(reference(arguments[:4],arguments[4:]),cotangent))
            reverse = finite_difference(scalar, inputs)
            epsilon = 1e-6
            plus = [x+epsilon*d for x,d in zip(inputs,direction)]
            minus = [x-epsilon*d for x,d in zip(inputs,direction)]
            tangent = [(x-y)/(2*epsilon) for x,y in zip(reference(plus[:4],plus[4:]),reference(minus[:4],minus[4:]))]
            body.append('{' + f'print("{name}");let tape=ag_tape();let x=ag_variable(tape,{tensor_literal(inputs[:4],[2,2])},true);let w=ag_variable(tape,{tensor_literal(inputs[4:],[2,2])},true);let output={expression};assert_close(ag_value(tape,output),{tensor_literal(values,shape)},1e-12,1e-12);let tangent=ag_jvp(tape,output,[x,w],[{tensor_literal(direction[:4],[2,2])},{tensor_literal(direction[4:],[2,2])}]);assert_close(tangent,{tensor_literal(tangent,shape)},2e-7,2e-7);let reverse=ag_vjp(tape,output,{tensor_literal(cotangent,shape)},[x,w],false);assert_close(reverse[0],{tensor_literal(reverse[:4],[2,2])},2e-7,2e-7);assert_close(reverse[1],{tensor_literal(reverse[4:],[2,2])},2e-7,2e-7);' + '}')
        self.run_heng('\n'.join(body))

    def test_jvp_vjp_match_finite_difference_and_adjoint_identity(self):
        inputs = [0.2, -0.4, 0.7, 1.1, 0.3, -0.2]
        weights = [0.4, -0.2, 0.1, 0.6, -0.5, 0.3]
        direction = [0.3, 0.1, -0.4, 0.2, -0.1, 0.5]
        cotangent = [0.2, -0.3, 0.7, 0.4]
        def function(values):
            result = []
            for row in range(2):
                logits = [math.tanh(math.fsum(values[row*3+index] * weights[index*2+column] for index in range(3))) for column in range(2)]
                denominator = math.fsum(math.exp(value - max(logits)) for value in logits)
                result.extend(math.exp(value-max(logits)) / denominator for value in logits)
            return result
        gradients = finite_difference(lambda values: math.fsum(a*b for a,b in zip(function(values),cotangent)), inputs)
        epsilon = 1e-6
        plus = function([value + epsilon * delta for value,delta in zip(inputs,direction)])
        minus = function([value - epsilon * delta for value,delta in zip(inputs,direction)])
        jvp = [(a-b)/(2*epsilon) for a,b in zip(plus,minus)]
        self.run_heng(f'let tape=ag_tape();let x=ag_variable(tape,{tensor_literal(inputs,[2,3])},true);let w=ag_constant(tape,{tensor_literal(weights,[3,2])});let y=ag_softmax(tape,ag_tanh(tape,ag_matmul(tape,x,w)),1);let tangent=ag_jvp(tape,y,[x],[{tensor_literal(direction,[2,3])}]);let reverse=ag_vjp(tape,y,{tensor_literal(cotangent,[2,2])},[x],true)[0];assert_close(tangent,{tensor_literal(jvp,[2,2])},5e-7,5e-7);assert_close(reverse,{tensor_literal(gradients,[2,3])},5e-7,5e-7);assert_close(sum_all(mul(tangent,{tensor_literal(cotangent,[2,2])})),sum_all(mul(reverse,{tensor_literal(direction,[2,3])})),1e-12,1e-12);')


if __name__ == "__main__":
    unittest.main()
