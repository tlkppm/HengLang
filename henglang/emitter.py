from __future__ import annotations

from pathlib import Path

from .abi import BUILTINS
from .frontend import HengError


def c_string(value: str) -> str:
    if "\0" in value:
        raise HengError("字符串包含不支持的 NUL 字符")
    return '"' + "".join(f"\\{byte:03o}" for byte in value.encode("utf-8")) + '"'


def function_name(value: str) -> str:
    return "hfn_" + value.encode("utf-8").hex()


class FunctionEmitter:
    def __init__(self, program, function, root: Path):
        self.program = program
        self.function = function
        self.root = root
        self.lines: list[str] = []
        self.indent = 1
        self.count = 0
        self.temporary: list[int] = []
        self.scopes: list[dict[str, int]] = [{}]
        self.loop_depths: list[int] = []
        for name, _ in function.params:
            self.scopes[0][name] = self.slot(False)

    def slot(self, temporary: bool = True) -> int:
        result = self.count
        self.count += 1
        if temporary:
            self.temporary.append(result)
        return result

    def line(self, text: str) -> None:
        self.lines.append("    " * self.indent + text)

    def reference(self, index: int) -> str:
        return f"r[{index}]"

    def resolve(self, name: str) -> int:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise HengError(f"生成代码时找不到变量: {name}")

    def location(self, node) -> None:
        source = Path(node.source)
        try:
            display = source.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            display = source.name
        self.line(f"hv_site({c_string(display)}, {node.line}, {node.column});")

    def convert(self, index: int, type_name: str) -> None:
        if type_name == "f64":
            self.line(f"r[{index}] = hv_f64(hv_number(r[{index}]));")
        elif type_name == "f32":
            self.line(f"r[{index}] = hv_f32(hv_number(r[{index}]));")
        elif type_name == "i64":
            self.line(f"r[{index}] = hv_i64(hv_integer(r[{index}]));")

    def expression(self, node) -> int:
        kind = node.kind
        result = self.slot()
        self.location(node)
        if kind == "int":
            value = int(node.value)
            text = "(-INT64_C(9223372036854775807) - INT64_C(1))" if value == -(2**63) else f"INT64_C({value})"
            self.line(f"r[{result}] = hv_i64({text});")
        elif kind == "float":
            self.line(f"r[{result}] = hv_f64({float(node.value).hex()});")
        elif kind == "bool":
            self.line(f"r[{result}] = hv_bool({1 if node.value else 0});")
        elif kind == "string":
            self.line(f"r[{result}] = hv_string({c_string(node.value)});")
        elif kind == "name":
            self.line(f"r[{result}] = r[{self.resolve(node.value)}];")
        elif kind == "enum":
            enum_name, member = node.value
            enum = self.program.enums[enum_name]
            self.line(f"r[{result}] = hv_i64(INT64_C({enum.members.index(member)}));")
        elif kind == "list":
            arguments = [self.expression(argument) for argument in node.args]
            values = "(HValue[]){" + ", ".join(self.reference(value) for value in arguments) + "}" if arguments else "NULL"
            self.location(node)
            self.line(f"r[{result}] = hv_list({len(arguments)}, {values});")
        elif kind == "record":
            arguments = [self.expression(argument) for argument in node.args]
            names = getattr(node, "names", [])
            if not names and isinstance(node.value, dict):
                names = list(node.value["fields"])
            field_names = "(const char *[]){" + ", ".join(c_string(name) for name in names) + "}" if names else "NULL"
            values = "(HValue[]){" + ", ".join(self.reference(value) for value in arguments) + "}" if arguments else "NULL"
            self.location(node)
            self.line(f"r[{result}] = hv_record({c_string(node.value)}, {len(arguments)}, {field_names}, {values});")
        elif kind == "field":
            target = node.args[0]
            if target.kind == "name" and target.value in self.program.enums and target.type == target.value:
                enum = self.program.enums[target.value]
                self.line(f"r[{result}] = hv_i64(INT64_C({enum.members.index(node.value)}));")
            else:
                owner = self.expression(target)
                self.location(node)
                self.line(f"r[{result}] = hv_field(r[{owner}], {c_string(node.value)});")
        elif kind == "index":
            owner = self.expression(node.args[0])
            index = self.expression(node.args[1])
            self.location(node)
            self.line(f"r[{result}] = hv_index(r[{owner}], r[{index}]);")
        elif kind == "unary":
            argument = self.expression(node.args[0])
            self.location(node)
            self.line(f"r[{result}] = hv_unary({c_string(node.value)}, r[{argument}]);")
        elif kind == "binary":
            left = self.expression(node.args[0])
            if node.value in {"&&", "||"}:
                short_value = 0 if node.value == "&&" else 1
                condition = f"hv_truth(r[{left}])" if node.value == "&&" else f"!hv_truth(r[{left}])"
                self.line(f"r[{result}] = hv_bool({short_value});")
                self.line(f"if ({condition}) {{")
                self.indent += 1
                right = self.expression(node.args[1])
                self.line(f"r[{result}] = hv_bool(hv_truth(r[{right}]));")
                self.indent -= 1
                self.line("}")
            else:
                right = self.expression(node.args[1])
                self.location(node)
                self.line(f"r[{result}] = hv_binary({c_string(node.value)}, r[{left}], r[{right}]);")
        elif kind == "call":
            arguments = [self.expression(argument) for argument in node.args]
            self.location(node)
            if node.value in BUILTINS:
                values = "(HValue[]){" + ", ".join(self.reference(value) for value in arguments) + "}" if arguments else "NULL"
                self.line(f"r[{result}] = hv_native({c_string(node.value)}, {len(arguments)}, {values});")
            else:
                function = self.program.functions[node.value]
                for argument, (_, parameter_type) in zip(arguments, function.params):
                    self.convert(argument, parameter_type)
                values = ", ".join(self.reference(value) for value in arguments)
                self.line(f"r[{result}] = {function_name(node.value)}({values});")
        else:
            raise HengError(f"未实现的表达式代码生成: {kind}")
        self.convert(result, getattr(node, "type", ""))
        return result

    def clean_temporaries(self, start: int) -> None:
        for slot in self.temporary[start:]:
            self.line(f"r[{slot}] = hv_void();")

    def clean_scopes(self, start: int) -> None:
        for scope in self.scopes[start:]:
            for slot in scope.values():
                self.line(f"r[{slot}] = hv_void();")

    def block(self, node, initial: dict[str, int] | None = None) -> None:
        self.scopes.append(dict(initial or {}))
        for statement in node.args:
            self.statement(statement)
        self.clean_scopes(len(self.scopes) - 1)
        self.scopes.pop()

    def statement(self, node) -> None:
        start = len(self.temporary)
        self.location(node)
        kind = node.kind
        if kind == "let":
            value = self.expression(node.args[0])
            slot = self.slot(False)
            self.scopes[-1][node.value] = slot
            self.line(f"r[{slot}] = r[{value}];")
            self.convert(slot, node.type)
        elif kind == "expr":
            self.expression(node.args[0])
        elif kind == "assign":
            target, value_node = node.args
            owner = self.expression(target.args[0]) if target.kind in {"field", "index"} else None
            index = self.expression(target.args[1]) if target.kind == "index" else None
            previous = None
            if node.value != "=":
                previous = self.slot()
                if target.kind == "name":
                    self.line(f"r[{previous}] = r[{self.resolve(target.value)}];")
                elif target.kind == "field":
                    self.line(f"r[{previous}] = hv_field(r[{owner}], {c_string(target.value)});")
                else:
                    self.line(f"r[{previous}] = hv_index(r[{owner}], r[{index}]);")
            value = self.expression(value_node)
            self.location(node)
            if previous is not None:
                self.line(f"r[{value}] = hv_binary({c_string(node.value[:-1])}, r[{previous}], r[{value}]);")
            self.convert(value, target.type)
            if target.kind == "name":
                self.line(f"r[{self.resolve(target.value)}] = r[{value}];")
            elif target.kind == "field":
                self.line(f"hv_set_field(r[{owner}], {c_string(target.value)}, r[{value}]);")
            else:
                self.line(f"hv_set_index(r[{owner}], r[{index}], r[{value}]);")
        elif kind == "return":
            if node.args:
                value = self.expression(node.args[0])
                self.convert(value, self.function.return_type)
                self.line(f"HValue returning_{value} = r[{value}];")
                self.line("hv_frame_pop(&frame);")
                self.line("free(r);")
                self.line("--heng_call_depth;")
                self.line(f"return returning_{value};")
            else:
                self.line("hv_frame_pop(&frame);")
                self.line("free(r);")
                self.line("--heng_call_depth;")
                self.line("return hv_void();")
            return
        elif kind == "if":
            condition = self.expression(node.args[0])
            name = f"condition_{self.count}"
            self.line(f"int {name} = hv_truth(r[{condition}]);")
            self.clean_temporaries(start)
            self.line(f"if ({name}) {{")
            self.indent += 1
            self.block(node.args[1])
            self.indent -= 1
            if len(node.args) > 2:
                self.line("} else {")
                self.indent += 1
                self.block(node.args[2])
                self.indent -= 1
            self.line("}")
        elif kind == "while":
            self.line("for (;;) {")
            self.indent += 1
            condition = self.expression(node.args[0])
            self.line(f"if (!hv_truth(r[{condition}])) break;")
            self.clean_temporaries(start)
            self.loop_depths.append(len(self.scopes))
            self.block(node.args[1])
            self.loop_depths.pop()
            self.line("hv_safepoint();")
            self.indent -= 1
            self.line("}")
        elif kind == "for":
            iterable = self.expression(node.args[0])
            variable = self.slot(False)
            counter = f"iteration_{self.count}"
            self.line(f"for (int64_t {counter} = 0; {counter} < hv_length(r[{iterable}]); ++{counter}) {{")
            self.indent += 1
            self.line(f"r[{variable}] = hv_index(r[{iterable}], hv_i64({counter}));")
            self.loop_depths.append(len(self.scopes))
            self.block(node.args[1], {node.value: variable})
            self.loop_depths.pop()
            self.line("hv_safepoint();")
            self.indent -= 1
            self.line("}")
        elif kind in {"break", "continue"}:
            self.clean_scopes(self.loop_depths[-1])
            self.clean_temporaries(start)
            self.line("hv_safepoint();")
            self.line(kind + ";")
            return
        elif kind == "block":
            self.line("{")
            self.indent += 1
            self.block(node)
            self.indent -= 1
            self.line("}")
        else:
            raise HengError(f"未实现的语句代码生成: {kind}")
        self.clean_temporaries(start)
        self.line("hv_safepoint();")

    def generate(self) -> str:
        self.block(self.function.body)
        self.line("hv_frame_pop(&frame);")
        self.line("free(r);")
        self.line("--heng_call_depth;")
        self.line("return hv_void();")
        parameters = ", ".join(f"HValue argument_{index}" for index, _ in enumerate(self.function.params)) or "void"
        head = [f"static HValue {function_name(self.function.name)}({parameters}) {{", '    if (heng_call_depth >= 512) hv_fail("函数调用深度超过安全上限");', "    ++heng_call_depth;", f"    HValue *r = calloc({max(1, self.count)}, sizeof(HValue));", '    if (!r) hv_fail("函数根槽内存分配失败");', "    HFrame frame;", f"    hv_frame_push(&frame, r, {max(1, self.count)});"]
        for index, (_, type_name) in enumerate(self.function.params):
            head.append(f"    r[{index}] = argument_{index};")
            if type_name in {"f64", "f32"}:
                head.append(f"    r[{index}] = hv_{type_name}(hv_number(r[{index}]));")
        return "\n".join(head + self.lines + ["}"])


def emit(program, root: Path, *, entry: str = "main", optimized: bool = True) -> str:
    if entry not in program.functions:
        raise HengError(f"缺少入口函数 {entry}")
    function = program.functions[entry]
    if function.params or function.return_type not in {"void", "i64"}:
        raise HengError("入口函数必须无参数，返回 void 或 i64")
    declarations = ['#include "heng_runtime.h"', '#include <stdint.h>', '#include <stdlib.h>', '#ifdef _WIN32', '#include <windows.h>', '#endif', 'static int heng_call_depth;']
    for name, item in program.functions.items():
        parameters = ", ".join("HValue" for _ in item.params) or "void"
        declarations.append(f"static HValue {function_name(name)}({parameters});")
    implementations = [FunctionEmitter(program, item, root).generate() for item in program.functions.values()]
    main = ["int main(void) {", "#ifdef _WIN32", "    SetConsoleOutputCP(CP_UTF8);", "    SetConsoleCP(CP_UTF8);", "#endif", f"    hv_native(\"set_optimized\", 1, (HValue[]){{hv_bool({int(optimized)})}});", f"    HValue result = {function_name(entry)}();", "    int code = result.tag == H_I64 ? (int)hv_integer(result) : 0;", "    hv_shutdown();", "    return code;", "}"]
    return "\n\n".join(["\n".join(declarations), *implementations, "\n".join(main)]) + "\n"
