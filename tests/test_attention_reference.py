from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from henglang.__main__ import ROOT, execute
from test_tensor_reference import tensor_literal


def attention_reference(query, key, value, weights, biases, mask):
    def project(values, weight, bias):
        return [
            math.fsum(values[offset + source] * weight[source * 4 + target] for source in range(4)) + bias[target]
            for offset in range(0, len(values), 4)
            for target in range(4)
        ]

    projected_query = project(query, weights[0], biases[0])
    projected_key = project(key, weights[1], biases[1])
    projected_value = project(value, weights[2], biases[2])
    merged = []
    for batch in range(2):
        for query_position in range(2):
            for head in range(2):
                scores = []
                for key_position in range(3):
                    score = math.fsum(
                        projected_query[(batch * 2 + query_position) * 4 + head * 2 + component]
                        * projected_key[(batch * 3 + key_position) * 4 + head * 2 + component]
                        for component in range(2)
                    ) / math.sqrt(2.0)
                    scores.append(score + mask[query_position * 3 + key_position])
                maximum = max(scores)
                exponentials = [math.exp(score - maximum) for score in scores]
                denominator = math.fsum(exponentials)
                probabilities = [exponential / denominator for exponential in exponentials]
                for component in range(2):
                    merged.append(math.fsum(
                        probabilities[key_position]
                        * projected_value[(batch * 3 + key_position) * 4 + head * 2 + component]
                        for key_position in range(3)
                    ))
    return project(merged, weights[3], biases[3])


def central_gradient(function, values, epsilon=1e-6):
    result = []
    for index in range(len(values)):
        plus, minus = list(values), list(values)
        plus[index] += epsilon
        minus[index] -= epsilon
        result.append((function(plus) - function(minus)) / (2.0 * epsilon))
    return result


class AttentionReferenceTests(unittest.TestCase):
    def setUp(self):
        self.query = [0.2, -0.4, 0.7, 1.1, -0.6, 0.3, 0.8, -0.2, 1.2, -0.7, 0.4, 0.9, -0.3, 1.0, -0.5, 0.6]
        self.key = [0.5, -0.8, 0.3, 0.9, -0.4, 0.7, 1.1, -0.2, 0.6, 0.2, -0.9, 0.4, -0.7, 0.8, 0.5, -0.3, 1.0, -0.6, 0.2, 0.7, 0.4, 1.2, -0.5, -0.8]
        self.value = [-0.3, 0.9, 0.5, -0.7, 0.8, -0.2, 1.0, 0.4, 0.6, -0.5, -0.4, 1.1, 0.7, 0.3, -0.8, 0.2, -0.9, 0.6, 0.4, 1.2, 0.5, -0.7, 0.9, -0.1]
        self.weights = [
            [0.8, -0.2, 0.3, 0.1, 0.4, 0.7, -0.1, 0.2, -0.3, 0.5, 0.9, -0.4, 0.2, -0.1, 0.6, 0.8],
            [0.6, 0.3, -0.2, 0.4, -0.5, 0.9, 0.1, -0.3, 0.2, -0.4, 0.7, 0.5, 0.1, 0.6, -0.3, 0.8],
            [0.9, -0.1, 0.4, -0.2, 0.3, 0.8, -0.5, 0.1, -0.4, 0.2, 0.6, 0.7, 0.5, -0.3, 0.1, 0.9],
            [0.7, 0.2, -0.4, 0.3, -0.1, 0.9, 0.5, -0.2, 0.6, -0.3, 0.8, 0.1, -0.2, 0.4, 0.3, 1.0],
        ]
        self.biases = [[0.1, -0.2, 0.05, 0.15], [-0.05, 0.12, -0.08, 0.2], [0.2, -0.1, 0.15, -0.05], [-0.1, 0.07, 0.12, -0.03]]
        self.mask = [0.0, -0.7, 0.25, -1.1, 0.15, -0.3]
        self.cotangent = [0.3, -0.7, 0.2, 0.9, -0.4, 0.6, -0.8, 0.1, 0.5, -0.2, 0.7, -0.6, -0.9, 0.4, 0.8, -0.3]
        self.inputs = self.query + self.key + self.value
        self.direction = [((index * 7) % 19 - 9.5) / 13.0 for index in range(len(self.inputs))]

    def reference(self, inputs):
        return attention_reference(inputs[:16], inputs[16:40], inputs[40:], self.weights, self.biases, self.mask)

    def heng_setup(self):
        body = ['let tape=ag_tape();', 'let attention=nn_multihead_attention(tape,4,2,0.25);']
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            body.append(f'ag_assign(tape,attention.children[{index}].parameters[0],{tensor_literal(weight, [4, 4])});')
            body.append(f'ag_assign(tape,attention.children[{index}].parameters[1],{tensor_literal(bias, [4])});')
        body.extend([
            'nn_eval(attention);',
            'assert(!attention.training && !attention.children[4].training,"注意力求值模式未传播到 Dropout");',
            f'let query=ag_variable(tape,{tensor_literal(self.query, [2, 2, 4])},true);',
            f'let key=ag_variable(tape,{tensor_literal(self.key, [2, 3, 4])},true);',
            f'let value=ag_variable(tape,{tensor_literal(self.value, [2, 3, 4])},true);',
            f'let mask={tensor_literal(self.mask, [1, 1, 2, 3])};',
        ])
        return body

    def run_heng(self, body):
        directory = ROOT / '.build'
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='attention-reference-', dir=directory) as temporary:
            source = Path(temporary) / 'attention.heng'
            source.write_text('import "heng.heng";\nfn main() -> void {\n' + '\n'.join(body) + '\n}\n', encoding='utf-8')
            result = execute(source, timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cross_attention_forward_and_eval_preserve_rng(self):
        expected = tensor_literal(self.reference(self.inputs), [2, 2, 4])
        body = self.heng_setup()
        body.extend([
            'seed(8391);',
            'let before=rng_state();',
            'let output=nn_attention_forward(tape,attention,query,key,value,mask);',
            'assert(rng_state()==before,"注意力求值前向消耗了随机数状态");',
            f'assert_close(ag_value(tape,output),{expected},1e-12,1e-12);',
            'let repeated=nn_attention_forward(tape,attention,query,key,value,mask);',
            'assert(rng_state()==before,"注意力重复求值消耗了随机数状态");',
            f'assert_close(ag_value(tape,repeated),{expected},1e-12,1e-12);',
        ])
        self.run_heng(body)

    def test_cross_attention_gradients_jvp_vjp_and_adjoint_identity(self):
        def objective(inputs):
            return math.fsum(output * seed for output, seed in zip(self.reference(inputs), self.cotangent))

        gradients = central_gradient(objective, self.inputs)
        epsilon = 1e-6
        plus = self.reference([value + epsilon * delta for value, delta in zip(self.inputs, self.direction)])
        minus = self.reference([value - epsilon * delta for value, delta in zip(self.inputs, self.direction)])
        tangent = [(upper - lower) / (2.0 * epsilon) for upper, lower in zip(plus, minus)]
        self.assertAlmostEqual(
            math.fsum(value * seed for value, seed in zip(tangent, self.cotangent)),
            math.fsum(value * delta for value, delta in zip(gradients, self.direction)),
            delta=2e-8,
        )
        for start, end in [(0, 16), (16, 40), (40, 64)]:
            self.assertGreater(max(abs(gradient) for gradient in gradients[start:end]), 1e-3)
        cotangent = tensor_literal(self.cotangent, [2, 2, 4])
        directions = [tensor_literal(self.direction[:16], [2, 2, 4]), tensor_literal(self.direction[16:40], [2, 3, 4]), tensor_literal(self.direction[40:], [2, 3, 4])]
        body = self.heng_setup()
        body.extend([
            'let before=rng_state();',
            'let output=nn_attention_forward(tape,attention,query,key,value,mask);',
            f'assert_close(ag_value(tape,output),{tensor_literal(self.reference(self.inputs), [2, 2, 4])},1e-12,1e-12);',
            f'let tangent=ag_jvp(tape,output,[query,key,value],[{",".join(directions)}]);',
            f'assert_close(tangent,{tensor_literal(tangent, [2, 2, 4])},2e-8,2e-8);',
            f'let reverse=ag_vjp(tape,output,{cotangent},[query,key,value],true);',
        ])
        for index, (name, start, end, shape) in enumerate([
            ('query', 0, 16, [2, 2, 4]), ('key', 16, 40, [2, 3, 4]), ('value', 40, 64, [2, 3, 4]),
        ]):
            body.append(f'assert_close(reverse[{index}],{tensor_literal(gradients[start:end], shape)},2e-8,2e-8);')
        body.extend([
            f'let tangent_inner=sum_all(mul(tangent,{cotangent}));',
            f'let reverse_inner=add(add(sum_all(mul(reverse[0],{directions[0]})),sum_all(mul(reverse[1],{directions[1]}))),sum_all(mul(reverse[2],{directions[2]})));',
            'assert_close(tangent_inner,reverse_inner,1e-12,1e-12);',
            f'ag_backward(tape,output,{cotangent},false);',
        ])
        for index, name in enumerate(['query', 'key', 'value']):
            body.append(f'assert_close(ag_grad(tape,{name}),reverse[{index}],1e-12,1e-12);')
        body.append('assert(rng_state()==before,"注意力求值及自动微分消耗了随机数状态");')
        self.run_heng(body)


if __name__ == '__main__':
    unittest.main()
