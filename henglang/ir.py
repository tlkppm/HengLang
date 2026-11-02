from __future__ import annotations

from copy import deepcopy
import math


def optimize(program, *, enabled: bool = True):
    result = deepcopy(program)
    if not enabled:
        return result
    for function in result.functions.values():
        function.body = transform(function.body)
    return result


def transform(node):
    node.args = [transform(argument) for argument in node.args]
    if node.kind == "call" and node.value == "relu" and len(node.args) == 1:
        addition = node.args[0]
        if addition.kind == "call" and addition.value == "add" and len(addition.args) == 2:
            scaled, offset = addition.args
            if scaled.kind == "call" and scaled.value == "scale" and offset.kind == "call" and offset.value == "full":
                if len(scaled.args) == 2 and len(offset.args) == 2 and offset.args[0].kind == "list" and not offset.args[0].args and scaled.args[1].kind in {"int", "float"} and offset.args[1].kind in {"int", "float"}:
                    node.value = "fused_affine_relu"
                    node.args = [*scaled.args, offset.args[1]]
    if node.kind == "unary" and node.value in {"-", "+", "!"} and node.args[0].kind in {"int", "float", "bool"}:
        operand = node.args[0]
        if node.value == "!" and operand.kind == "bool":
            node.kind, node.value, node.args = "bool", not operand.value, []
        elif node.value in {"-", "+"} and operand.kind in {"int", "float"}:
            value = -operand.value if node.value == "-" else operand.value
            if (operand.kind == "int" and -(2**63) <= value < 2**63) or (operand.kind == "float" and math.isfinite(value)):
                node.kind, node.value, node.args = operand.kind, value, []
    return node
