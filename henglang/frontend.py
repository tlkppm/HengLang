from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import re

from .abi import BUILTINS


class HengError(Exception):
    def __init__(self, message: str, source: str = '<memory>.heng', line: int = 1, column: int = 1):
        self.message = message
        self.source = source
        self.line = line
        self.column = column
        super().__init__(f'{source}:{line}:{column}: {message}')


@dataclass
class Node:
    kind: str
    value: object = None
    args: list[Node] = field(default_factory=list)
    line: int = 1
    column: int = 1
    source: str = ''
    type: str = ''
    mutable: bool = False
    names: list[str] = field(default_factory=list)
    type_args: list[str] = field(default_factory=list)
    shape: tuple[int, ...] | None = None
    generic_bindings: dict[str, str] = field(default_factory=dict)


@dataclass
class Function:
    name: str
    params: list[tuple[str, str]]
    return_type: str
    body: Node
    generics: list[str] = field(default_factory=list)
    source: str = ''
    line: int = 1
    column: int = 1


@dataclass
class Struct:
    name: str
    fields: list[tuple[str, str]]
    source: str = ''
    line: int = 1
    column: int = 1


@dataclass
class Enum:
    name: str
    members: list[str]
    source: str = ''
    line: int = 1
    column: int = 1


@dataclass(frozen=True)
class Instantiation:
    function: str
    type_args: tuple[str, ...]
    params: tuple[str, ...]
    return_type: str


@dataclass
class Program:
    functions: dict[str, Function] = field(default_factory=dict)
    structs: dict[str, Struct] = field(default_factory=dict)
    enums: dict[str, Enum] = field(default_factory=dict)
    imports: list[Node] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    instantiations: list[Instantiation] = field(default_factory=list)
    checked: bool = False


@dataclass
class Token:
    kind: str
    value: object
    line: int
    column: int
    source: str


class Lexer:
    def __init__(self, source: str, filename: str = '<memory>.heng'):
        self.source = source
        self.filename = filename

    def tokenize(self) -> list[Token]:
        text = self.source
        tokens = []
        position = 0
        line = 1
        column = 1
        while position < len(text):
            start = position
            start_line = line
            start_column = column
            char = text[position]
            if char.isspace():
                position += 1
            elif text.startswith('//', position):
                end = text.find('\n', position)
                position = len(text) if end < 0 else end
            elif text.startswith('/*', position):
                end = text.find('*/', position + 2)
                if end < 0:
                    raise HengError('块注释未结束', self.filename, line, column)
                position = end + 2
            elif char == '"':
                position += 1
                value = ''
                escapes = {'n': '\n', 'r': '\r', 't': '\t', '0': '\0', '"': '"', '\\': '\\'}
                while position < len(text) and text[position] != '"':
                    if text[position] in '\n\r':
                        raise HengError('字符串中不允许直接换行', self.filename, start_line, start_column)
                    if text[position] == '\\':
                        position += 1
                        if position >= len(text):
                            break
                        if text[position] == 'u':
                            digits = text[position + 1:position + 5]
                            if len(digits) != 4 or not re.fullmatch('[0-9a-fA-F]{4}', digits):
                                raise HengError('无效的 Unicode 转义', self.filename, start_line, start_column)
                            codepoint = int(digits, 16)
                            if 0xD800 <= codepoint <= 0xDFFF:
                                raise HengError('字符串包含无效 Unicode 代理码位', self.filename, start_line, start_column)
                            value += chr(codepoint)
                            position += 5
                            continue
                        if text[position] not in escapes:
                            raise HengError('未知的字符串转义', self.filename, start_line, start_column)
                        value += escapes[text[position]]
                    else:
                        value += text[position]
                    position += 1
                if position >= len(text):
                    raise HengError('字符串未结束', self.filename, start_line, start_column)
                position += 1
                tokens.append(Token('string', value, line, column, self.filename))
            elif char.isascii() and char.isdigit():
                match = re.match(r'\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', text[position:])
                spelling = match.group()
                position += len(spelling)
                floating = '.' in spelling or 'e' in spelling.lower()
                value = float(spelling) if floating else int(spelling)
                if floating and not math.isfinite(value):
                    raise HengError('浮点常量超出有限范围', self.filename, line, column)
                tokens.append(Token('float' if floating else 'int', value, line, column, self.filename))
            elif char.isascii() and (char.isalpha() or char == '_'):
                match = re.match(r'[A-Za-z_][A-Za-z_0-9]*', text[position:])
                word = match.group()
                position += len(word)
                tokens.append(Token('id', word, line, column, self.filename))
            else:
                pair = text[position:position + 2]
                if pair in {'->', '==', '!=', '<=', '>=', '&&', '||', '+=', '-=', '*=', '/=', '%='}:
                    position += 2
                    tokens.append(Token('symbol', pair, line, column, self.filename))
                elif char in '{}()[]<>,:;.=+-*/%!':
                    position += 1
                    tokens.append(Token('symbol', char, line, column, self.filename))
                else:
                    raise HengError(f'无法识别的字符 {char!r}', self.filename, line, column)
            consumed = text[start:position]
            if '\n' in consumed:
                line += consumed.count('\n')
                column = len(consumed.rsplit('\n', 1)[1]) + 1
            else:
                column += len(consumed)
        tokens.append(Token('eof', '<eof>', line, column, self.filename))
        return tokens


_RESERVED = {'fn', 'struct', 'enum', 'import', 'let', 'var', 'return', 'if', 'else', 'while', 'for', 'in', 'break', 'continue', 'true', 'false'}
_PRIMITIVES = {'i64', 'f32', 'f64', 'bool', 'str', 'Tensor', 'Sparse', 'void'}
_NUMERIC = {'i64', 'f32', 'f64'}


def _container(type_name: str) -> tuple[str, str] | None:
    match = re.fullmatch(r'(List|Result)<(.+)>', type_name)
    return (match.group(1), match.group(2)) if match else None


def _substitute(type_name: str, mapping: dict[str, str]) -> str:
    if type_name in mapping:
        return mapping[type_name]
    container = _container(type_name)
    return f'{container[0]}<{_substitute(container[1], mapping)}>' if container else type_name


class Parser:
    def __init__(self, source: str | list[Token], filename: str = '<memory>.heng'):
        self.tokens = Lexer(source, filename).tokenize() if isinstance(source, str) else source
        self.position = 0
        self.filename = filename

    @property
    def current(self) -> Token:
        return self.tokens[self.position]

    def fail(self, message: str, token: Token | None = None):
        token = token or self.current
        raise HengError(message, token.source, token.line, token.column)

    def at(self, value: str) -> bool:
        return self.current.value == value

    def take(self) -> Token:
        token = self.current
        if token.kind != 'eof':
            self.position += 1
        return token

    def accept(self, value: str) -> bool:
        if self.at(value):
            self.take()
            return True
        return False

    def expect(self, value: str) -> Token:
        if value == '>' and self.at('>='):
            token = self.current
            self.tokens[self.position] = Token('symbol', '=', token.line, token.column + 1, token.source)
            return Token('symbol', '>', token.line, token.column, token.source)
        if not self.at(value):
            self.fail(f'期望 {value!r}，实际为 {self.current.value!r}')
        return self.take()

    def identifier(self) -> Token:
        if self.current.kind != 'id' or self.current.value in _RESERVED:
            self.fail('期望合法标识符')
        return self.take()

    def node(self, kind: str, token: Token, value=None, args=None, **kwargs) -> Node:
        return Node(kind, value, args or [], token.line, token.column, token.source, **kwargs)

    def type_name(self) -> str:
        name = self.identifier().value
        if self.accept('<'):
            params = [self.type_name()]
            while self.accept(','):
                params.append(self.type_name())
            self.expect('>')
            name += '<' + ','.join(params) + '>'
        return name

    def parse(self) -> Program:
        program = Program(sources=[self.filename])
        while self.current.kind != 'eof':
            if self.accept(';'):
                continue
            token = self.current
            if self.accept('import'):
                if self.current.kind != 'string':
                    self.fail('import 需要字符串形式的相对模块路径')
                path_token = self.take()
                program.imports.append(self.node('import', path_token, path_token.value))
                self.expect(';')
                continue
            if self.accept('fn'):
                name = self.identifier().value
                generics = []
                if self.accept('<'):
                    generics.append(self.identifier().value)
                    while self.accept(','):
                        generics.append(self.identifier().value)
                    self.expect('>')
                self.expect('(')
                params = []
                if not self.at(')'):
                    while True:
                        param = self.identifier().value
                        self.expect(':')
                        params.append((param, self.type_name()))
                        if not self.accept(',') or self.at(')'):
                            break
                self.expect(')')
                return_type = self.type_name() if self.accept('->') else 'void'
                definition = Function(name, params, return_type, self.block(), generics, token.source, token.line, token.column)
                target = program.functions
            elif self.accept('struct'):
                name = self.identifier().value
                self.expect('{')
                fields = []
                while not self.at('}'):
                    key = self.identifier().value
                    self.expect(':')
                    fields.append((key, self.type_name()))
                    if not self.accept(','):
                        break
                self.expect('}')
                definition = Struct(name, fields, token.source, token.line, token.column)
                target = program.structs
            elif self.accept('enum'):
                name = self.identifier().value
                self.expect('{')
                members = []
                while not self.at('}'):
                    members.append(self.identifier().value)
                    if not self.accept(','):
                        break
                self.expect('}')
                definition = Enum(name, members, token.source, token.line, token.column)
                target = program.enums
            else:
                self.fail('顶层只允许 fn、struct、enum 或 import')
            if name in program.functions or name in program.structs or name in program.enums:
                self.fail(f'全局名称重复: {name}', token)
            target[name] = definition
            self.accept(';')
        return program

    def block(self) -> Node:
        token = self.expect('{')
        statements = []
        while not self.at('}'):
            if self.current.kind == 'eof':
                self.fail('代码块未结束')
            statements.append(self.statement())
        self.expect('}')
        return self.node('block', token, args=statements)

    def statement(self) -> Node:
        token = self.current
        if self.at('{'):
            return self.block()
        if self.at('let') or self.at('var'):
            mutable = self.take().value == 'var'
            name = self.identifier().value
            declared_type = self.type_name() if self.accept(':') else ''
            self.expect('=')
            initializer = self.expression()
            self.expect(';')
            node = self.node('let', token, name, [initializer], type=declared_type, mutable=mutable)
            node.declared_type = declared_type
            return node
        if self.accept('return'):
            args = [] if self.at(';') else [self.expression()]
            self.expect(';')
            return self.node('return', token, args=args)
        if self.accept('if'):
            condition = self.expression(allow_record=False)
            then = self.block()
            otherwise = self.node('block', token)
            if self.accept('else'):
                otherwise = self.node('block', self.current, args=[self.statement()]) if self.at('if') else self.block()
            return self.node('if', token, args=[condition, then, otherwise])
        if self.accept('while'):
            condition = self.expression(allow_record=False)
            return self.node('while', token, args=[condition, self.block()])
        if self.accept('for'):
            name = self.identifier().value
            self.expect('in')
            iterable = self.expression(allow_record=False)
            return self.node('for', token, name, [iterable, self.block()])
        if self.at('break') or self.at('continue'):
            kind = self.take().value
            self.expect(';')
            return self.node(kind, token)
        expression = self.expression()
        if self.current.value in {'=', '+=', '-=', '*=', '/=', '%='}:
            operation = self.take().value
            value = self.expression()
            self.expect(';')
            return self.node('assign', token, operation, [expression, value])
        self.expect(';')
        return self.node('expr', token, args=[expression])

    def expression(self, minimum: int = 0, allow_record: bool = True) -> Node:
        token = self.take()
        if token.kind in {'int', 'float', 'string'}:
            left = self.node(token.kind, token, token.value)
        elif token.value in {'true', 'false'}:
            left = self.node('bool', token, token.value == 'true')
        elif token.value in {'!', '-', '+'}:
            left = self.node('unary', token, token.value, [self.expression(7, allow_record)])
        elif token.value == '(':
            left = self.expression()
            self.expect(')')
        elif token.value == '[':
            values = []
            while not self.at(']'):
                values.append(self.expression())
                if not self.accept(','):
                    break
            self.expect(']')
            left = self.node('list', token, args=values)
        elif token.kind == 'id' and token.value not in _RESERVED:
            left = self.node('name', token, token.value)
        else:
            self.fail('期望表达式', token)
        precedence = {'||': 1, '&&': 2, '==': 3, '!=': 3, '<': 4, '<=': 4, '>': 4, '>=': 4, '+': 5, '-': 5, '*': 6, '/': 6, '%': 6}
        while True:
            if self.at('.'):
                self.take()
                member = self.identifier()
                left = self.node('field', member, member.value, [left])
                continue
            if self.accept('['):
                index = self.expression()
                self.expect(']')
                left = self.node('index', token, args=[left, index])
                continue
            type_args = []
            if left.kind == 'name' and self.at('<'):
                saved = self.position
                saved_tokens = self.tokens
                self.tokens = list(self.tokens)
                self.take()
                try:
                    type_args.append(self.type_name())
                    while self.accept(','):
                        type_args.append(self.type_name())
                    self.expect('>')
                    if not self.at('('):
                        type_args = []
                        self.position = saved
                        self.tokens = saved_tokens
                except HengError:
                    type_args = []
                    self.position = saved
                    self.tokens = saved_tokens
            if self.at('('):
                if left.kind != 'name':
                    self.fail('只支持以函数名调用函数')
                self.take()
                values = []
                while not self.at(')'):
                    values.append(self.expression())
                    if not self.accept(','):
                        break
                self.expect(')')
                left = self.node('call', token, left.value, values, type_args=type_args)
                continue
            if allow_record and left.kind == 'name' and self.accept('{'):
                values = []
                names = []
                while not self.at('}'):
                    names.append(self.identifier().value)
                    self.expect(':')
                    values.append(self.expression())
                    if not self.accept(','):
                        break
                self.expect('}')
                left = self.node('record', token, left.value, values, names=names)
                continue
            operation = self.current.value
            priority = precedence.get(operation, -1)
            if priority < minimum:
                break
            self.take()
            right = self.expression(priority + 1, allow_record)
            left = self.node('binary', token, operation, [left, right])
        return left


def parse(source: str, filename: str = '<memory>.heng') -> Program:
    if Path(filename).suffix != '.heng':
        raise HengError('源文件后缀必须为 .heng', filename)
    return Parser(source, filename).parse()


@dataclass
class _Binding:
    type: str
    mutable: bool = False
    shape: tuple[int, ...] | None = None


class _Checker:
    def __init__(self, program: Program):
        self.program = program
        self.scopes: list[dict[str, _Binding]] = []
        self.function: Function | None = None
        self.generics: set[str] = set()
        self.loop_depth = 0

    def fail(self, node, message: str):
        raise HengError(message, node.source or '<memory>.heng', node.line, node.column)

    def validate_type(self, type_name: str, node, generics: set[str] | None = None, allow_void: bool = False):
        generics = self.generics if generics is None else generics
        container = _container(type_name)
        if container:
            self.validate_type(container[1], node, generics)
            return
        if type_name == 'void':
            if not allow_void:
                self.fail(node, 'void 只能用于函数返回类型')
            return
        if type_name not in _PRIMITIVES and type_name not in self.program.structs and type_name not in self.program.enums and type_name not in generics:
            self.fail(node, f'未定义的类型: {type_name}')

    def compatible(self, actual: str, expected: str, builtin: bool = False) -> bool:
        return actual == expected or (builtin and expected == 'Any') or (actual == 'i64' and expected in {'f32', 'f64'}) or (actual == 'f32' and expected == 'f64')

    def require(self, actual: str, expected: str, node, builtin: bool = False):
        if not self.compatible(actual, expected, builtin):
            self.fail(node, f'类型不匹配: 需要 {expected}，实际为 {actual}')

    def lookup(self, name: str, node) -> _Binding:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        self.fail(node, f'未定义的变量: {name}')

    def declare(self, name: str, binding: _Binding, node):
        if name in self.scopes[-1]:
            self.fail(node, f'当前作用域变量重复: {name}')
        self.scopes[-1][name] = binding

    def run(self) -> Program:
        self.program.checked = False
        self.program.instantiations.clear()
        occupied = set()
        for table in (self.program.functions, self.program.structs, self.program.enums):
            for name, definition in table.items():
                if name in occupied or name in BUILTINS or name in _PRIMITIVES or name in {'List', 'Result', 'Any'}:
                    self.fail(definition, f'全局名称重复或保留: {name}')
                occupied.add(name)
        for definition in self.program.structs.values():
            names = [name for name, _ in definition.fields]
            if len(names) != len(set(names)):
                self.fail(definition, f'结构字段重复: {definition.name}')
            for _, type_name in definition.fields:
                self.validate_type(type_name, definition, set())
        for definition in self.program.enums.values():
            if not definition.members:
                self.fail(definition, f'枚举至少需要一个成员: {definition.name}')
            if len(definition.members) != len(set(definition.members)):
                self.fail(definition, f'枚举成员重复: {definition.name}')
        for function in self.program.functions.values():
            generics = set(function.generics)
            if len(generics) != len(function.generics):
                self.fail(function, '泛型参数重复')
            for name in generics:
                if name in _PRIMITIVES or name in {'Any', 'List', 'Result'} or name in occupied:
                    self.fail(function, f'泛型参数与已定义类型或名称冲突: {name}')
            for _, type_name in function.params:
                self.validate_type(type_name, function, generics)
            self.validate_type(function.return_type, function, generics, allow_void=True)
        for function in self.program.functions.values():
            self.function = function
            self.generics = set(function.generics)
            self.scopes = [{}]
            self.loop_depth = 0
            for name, type_name in function.params:
                self.declare(name, _Binding(type_name), function)
            outcomes = self.block(function.body, scoped=False)
            if function.return_type != 'void' and 'normal' in outcomes:
                self.fail(function, f'函数 {function.name} 并非所有路径都返回 {function.return_type}')
        self.expand_instantiations()
        self.program.checked = True
        return self.program

    def block(self, node: Node, scoped: bool = True) -> set[str]:
        if scoped:
            self.scopes.append({})
        outcomes = {'normal'}
        for statement in node.args:
            result = self.statement(statement)
            if 'normal' in outcomes:
                outcomes.remove('normal')
                outcomes.update(result)
        if scoped:
            self.scopes.pop()
        node.type = 'void'
        return outcomes

    def statement(self, node: Node) -> set[str]:
        kind = node.kind
        if kind == 'block':
            return self.block(node)
        if kind == 'let':
            declared = getattr(node, 'declared_type', node.type)
            if declared:
                self.validate_type(declared, node)
            actual = self.expression(node.args[0], declared or None)
            if actual == 'void':
                self.fail(node, '不能将 void 赋给变量')
            if declared:
                self.require(actual, declared, node.args[0])
            node.type = declared or actual
            self.declare(node.value, _Binding(node.type, node.mutable, node.args[0].shape), node)
        elif kind == 'return':
            expected = self.function.return_type
            if not node.args:
                self.require('void', expected, node)
            else:
                if expected == 'void':
                    self.fail(node, 'void 函数不能返回值')
                actual = self.expression(node.args[0], expected)
                self.require(actual, expected, node.args[0])
            node.type = expected
            return {'return'}
        elif kind == 'if':
            self.require(self.expression(node.args[0]), 'bool', node.args[0])
            before = self.capture_shapes()
            first = self.block(node.args[1])
            first_shapes = self.capture_shapes()
            self.restore_shapes(before)
            second = self.block(node.args[2])
            second_shapes = self.capture_shapes()
            for binding, original in before:
                left = dict((id(item), shape) for item, shape in first_shapes).get(id(binding), original)
                right = dict((id(item), shape) for item, shape in second_shapes).get(id(binding), original)
                binding.shape = left if left == right else None
            node.type = 'void'
            return first | second
        elif kind == 'while':
            self.require(self.expression(node.args[0]), 'bool', node.args[0])
            self.invalidate_mutable_shapes()
            self.loop_depth += 1
            outcomes = self.block(node.args[1])
            self.loop_depth -= 1
            self.invalidate_mutable_shapes()
            if node.args[0].kind == 'bool' and node.args[0].value is True and 'break' not in outcomes:
                return {'return'} if 'return' in outcomes else set()
            return {'normal'} | ({'return'} if 'return' in outcomes else set())
        elif kind == 'for':
            actual = self.expression(node.args[0])
            container = _container(actual)
            if not container or container[0] != 'List':
                self.fail(node.args[0], 'for 只能遍历 List<T>')
            node.type = container[1]
            self.invalidate_mutable_shapes()
            self.scopes.append({})
            self.declare(node.value, _Binding(node.type), node)
            self.loop_depth += 1
            self.block(node.args[1], scoped=False)
            self.loop_depth -= 1
            self.scopes.pop()
            self.invalidate_mutable_shapes()
        elif kind in {'break', 'continue'}:
            if not self.loop_depth:
                self.fail(node, f'{kind} 只能用于循环内部')
            node.type = 'void'
            return {kind}
        elif kind == 'expr':
            self.expression(node.args[0])
            node.type = 'void'
        elif kind == 'assign':
            target, value = node.args
            if target.kind == 'name':
                binding = self.lookup(target.value, target)
                if not binding.mutable:
                    self.fail(target, f'let 变量不能重新绑定: {target.value}')
            elif target.kind not in {'field', 'index'}:
                self.fail(target, '赋值目标必须是变量、记录字段或列表索引')
            target_type = self.expression(target)
            if target.kind == 'field' and target.args[0].kind == 'name' and target.args[0].value in self.program.enums and target.args[0].type == target.args[0].value:
                self.fail(target, '枚举成员不能赋值')
            if target.kind == 'index':
                container = _container(target.args[0].type)
                if not container or container[0] != 'List':
                    self.fail(target, '只有列表索引可以赋值')
            actual = self.expression(value, target_type)
            if node.value != '=':
                result = self.binary_type(node.value[0], target_type, actual, node)
                self.require(result, target_type, node)
            else:
                self.require(actual, target_type, value)
            if target.kind == 'name':
                operand_shape = value.shape if actual == 'Tensor' else ()
                binding.shape = value.shape if node.value == '=' else self.broadcast_shapes(target.shape, operand_shape, node)
            node.type = target_type
        else:
            self.fail(node, f'未知语句: {kind}')
        return {'normal'}

    def capture_shapes(self):
        return [(binding, binding.shape) for scope in self.scopes for binding in scope.values()]

    def restore_shapes(self, captured):
        for binding, shape in captured:
            binding.shape = shape

    def invalidate_mutable_shapes(self):
        for scope in self.scopes:
            for binding in scope.values():
                if binding.mutable:
                    binding.shape = None

    def expression(self, node: Node, expected: str | None = None) -> str:
        kind = node.kind
        node.shape = None
        if kind in {'int', 'float', 'bool', 'string'}:
            node.type = {'int': 'i64', 'float': 'f64', 'bool': 'bool', 'string': 'str'}[kind]
            if kind == 'int' and not -(2 ** 63) <= node.value < 2 ** 63:
                self.fail(node, '整数常量超出 i64 范围')
            if kind == 'string' and '\x00' in node.value:
                self.fail(node, '字符串不能包含 NUL 字符')
        elif kind == 'name':
            binding = self.lookup(node.value, node)
            node.type = binding.type
            node.shape = binding.shape
        elif kind == 'list':
            context = _container(expected or '')
            element = context[1] if context and context[0] == 'List' else None
            if not node.args and not element:
                self.fail(node, '空列表需要 List<T> 类型上下文')
            types = [self.expression(value, element) for value in node.args]
            if element:
                for value, actual in zip(node.args, types):
                    self.require(actual, element, value)
            elif types:
                element = types[0]
                for actual in types[1:]:
                    if element in _NUMERIC and actual in _NUMERIC:
                        element = max((element, actual), key=lambda item: ['i64', 'f32', 'f64'].index(item))
                    elif actual != element:
                        self.fail(node, f'列表元素类型不一致: {element} 与 {actual}')
            if element == 'void':
                self.fail(node, '列表元素不能为 void')
            node.type = f'List<{element}>'
        elif kind == 'record':
            definition = self.program.structs.get(node.value)
            if definition is None:
                self.fail(node, f'未定义的结构类型: {node.value}')
            if len(node.names) != len(set(node.names)):
                self.fail(node, '记录字面量字段重复')
            fields = dict(definition.fields)
            missing = fields.keys() - set(node.names)
            extra = set(node.names) - fields.keys()
            if missing or extra:
                self.fail(node, f'记录字段不匹配，缺失: {", ".join(sorted(missing)) or "无"}；多余: {", ".join(sorted(extra)) or "无"}')
            for name, value in zip(node.names, node.args):
                self.require(self.expression(value, fields[name]), fields[name], value)
            node.type = definition.name
        elif kind == 'field':
            target = node.args[0]
            if target.kind == 'name' and target.value in self.program.enums and not any(target.value in scope for scope in self.scopes):
                definition = self.program.enums[target.value]
                if node.value not in definition.members:
                    self.fail(node, f'未定义的枚举成员: {target.value}.{node.value}')
                target.type = definition.name
                node.type = definition.name
            else:
                target_type = self.expression(target)
                definition = self.program.structs.get(target_type)
                if definition is None or node.value not in dict(definition.fields):
                    self.fail(node, f'类型 {target_type} 没有字段 {node.value}')
                node.type = dict(definition.fields)[node.value]
        elif kind == 'index':
            target_type = self.expression(node.args[0])
            self.require(self.expression(node.args[1]), 'i64', node.args[1])
            container = _container(target_type)
            if container and container[0] == 'List':
                node.type = container[1]
            else:
                self.fail(node, f'类型 {target_type} 不支持索引；Tensor 请使用 item 或 get')
        elif kind == 'unary':
            value = node.args[0]
            if node.value == '-' and value.kind == 'int' and value.value == 2 ** 63:
                node.kind = 'int'
                node.value = -(2 ** 63)
                node.args = []
                node.type = 'i64'
                return node.type
            actual = self.expression(value)
            if node.value == '!':
                self.require(actual, 'bool', node)
                node.type = 'bool'
            elif actual in _NUMERIC or actual == 'Tensor':
                node.type = actual
                node.shape = value.shape
            else:
                self.fail(node, f'一元运算 {node.value} 不支持类型 {actual}')
        elif kind == 'binary':
            left = self.expression(node.args[0])
            right = self.expression(node.args[1])
            node.type = self.binary_type(node.value, left, right, node)
            if node.type == 'Tensor':
                left_shape = node.args[0].shape if left == 'Tensor' else ()
                right_shape = node.args[1].shape if right == 'Tensor' else ()
                node.shape = self.broadcast_shapes(left_shape, right_shape, node)
        elif kind == 'call':
            node.type = self.call(node, expected)
        else:
            self.fail(node, f'未知表达式: {kind}')
        return node.type

    def binary_type(self, operation: str, left: str, right: str, node: Node) -> str:
        if operation in {'&&', '||'}:
            self.require(left, 'bool', node)
            self.require(right, 'bool', node)
            return 'bool'
        if operation in {'==', '!='}:
            if (left == right and left not in {'void', 'Tensor', 'Sparse'} and left not in self.generics and not _container(left) and left not in self.program.structs) or (left in _NUMERIC and right in _NUMERIC):
                return 'bool'
            self.fail(node, f'类型 {left} 与 {right} 不支持标量相等比较')
        if operation in {'<', '<=', '>', '>='}:
            if left in _NUMERIC and right in _NUMERIC:
                return 'bool'
            self.fail(node, '关系比较只支持数值标量')
        if operation in {'+', '-', '*', '/'} and ((left == 'Tensor' and right in _NUMERIC | {'Tensor'}) or (right == 'Tensor' and left in _NUMERIC | {'Tensor'})):
            return 'Tensor'
        if operation == '+' and left == right == 'str':
            return 'str'
        if left in _NUMERIC and right in _NUMERIC:
            if operation == '%' and (left != 'i64' or right != 'i64'):
                self.fail(node, '取余运算只支持 i64')
            return max((left, right), key=lambda item: ['i64', 'f32', 'f64'].index(item))
        self.fail(node, f'运算 {operation} 不支持 {left} 与 {right}')

    def unify(self, formal: str, actual: str, generics: set[str], mapping: dict[str, str], node: Node):
        if formal in generics:
            if formal in mapping:
                self.require(actual, mapping[formal], node)
            else:
                if actual == 'void':
                    self.fail(node, '泛型实参不能为 void')
                mapping[formal] = actual
            return
        formal_container = _container(formal)
        actual_container = _container(actual)
        if formal_container and actual_container and formal_container[0] == actual_container[0]:
            if self.contains_generic(formal_container[1], generics):
                self.unify(formal_container[1], actual_container[1], generics, mapping, node)
            return

    def contains_generic(self, type_name: str, generics: set[str]) -> bool:
        container = _container(type_name)
        return type_name in generics or bool(container and self.contains_generic(container[1], generics))

    def call(self, node: Node, expected: str | None) -> str:
        name = node.value
        if name == 'Tensor':
            name = 'tensor'
            node.value = name
        builtin = name in BUILTINS
        if builtin:
            params, result = BUILTINS[name]
            generic_names = ['T'] if any(self.contains_generic(item, {'T'}) for item in params + [result]) else []
        elif name in self.program.functions:
            function = self.program.functions[name]
            params = [type_name for _, type_name in function.params]
            result = function.return_type
            generic_names = function.generics
        else:
            self.fail(node, f'未定义的函数: {name}')
        if len(params) != len(node.args):
            self.fail(node, f'函数 {name} 需要 {len(params)} 个参数，实际为 {len(node.args)}')
        generic_set = set(generic_names)
        mapping = {}
        if node.type_args:
            if len(node.type_args) != len(generic_names):
                self.fail(node, f'函数 {name} 泛型实参数量不匹配')
            for type_name in node.type_args:
                self.validate_type(type_name, node)
            mapping.update(zip(generic_names, node.type_args))
        if expected and self.contains_generic(result, generic_set) and not (result in generic_set and expected in _NUMERIC):
            self.unify(result, expected, generic_set, mapping, node)
        unresolved = []
        for index, (formal, argument) in enumerate(zip(params, node.args)):
            contextual = _substitute(formal, mapping)
            if self.contains_generic(contextual, generic_set - mapping.keys()):
                if argument.kind == 'list' and not argument.args:
                    unresolved.append(index)
                    continue
                if argument.kind == 'call' and argument.value == 'result_err':
                    unresolved.append(index)
                    continue
                contextual = None
            actual = self.expression(argument, contextual if contextual != 'Any' else None)
            self.unify(formal, actual, generic_set, mapping, argument)
        if expected and result in generic_set and result not in mapping:
            mapping[result] = expected
        for index in unresolved:
            contextual = _substitute(params[index], mapping)
            if self.contains_generic(contextual, generic_set - mapping.keys()):
                self.fail(node.args[index], f'无法推断函数 {name} 的泛型实参，请添加类型上下文或显式类型')
            actual = self.expression(node.args[index], contextual)
            self.unify(params[index], actual, generic_set, mapping, node.args[index])
        missing = generic_set - mapping.keys()
        if missing:
            self.fail(node, f'无法推断泛型参数: {", ".join(sorted(missing))}；请添加类型上下文或显式类型')
        resolved_params = [_substitute(formal, mapping) for formal in params]
        for argument, formal in zip(node.args, resolved_params):
            if argument.kind == 'list':
                self.expression(argument, formal)
            self.require(argument.type, formal, argument, builtin)
        if builtin and name == 'len':
            argument_type = node.args[0].type
            container = _container(argument_type)
            if argument_type not in {'str', 'Tensor'} and not (container and container[0] == 'List'):
                self.fail(node, 'len 只支持 List<T>、str 或 Tensor')
        if builtin and name in {'i64', 'f32', 'f64'}:
            if node.args[0].type not in _NUMERIC | {'bool', 'str'}:
                self.fail(node, f'{name} 转换只接受数值、bool 或 str')
        node.generic_bindings = dict(mapping)
        return_type = _substitute(result, mapping)
        if generic_names and not builtin:
            instance = Instantiation(name, tuple(mapping[item] for item in generic_names), tuple(resolved_params), return_type)
            if instance not in self.program.instantiations:
                self.program.instantiations.append(instance)
        if builtin:
            node.shape = self.infer_shape(name, node)
        return return_type

    def expand_instantiations(self):
        def nodes(node):
            yield node
            for child in node.args:
                yield from nodes(child)
        pending = list(self.program.instantiations)
        seen = set(pending)
        count = 0
        while pending:
            instance = pending.pop()
            function = self.program.functions[instance.function]
            if any(self.contains_generic(item, set(function.generics)) for item in instance.type_args):
                continue
            mapping = dict(zip(function.generics, instance.type_args))
            for node in nodes(function.body):
                if node.kind != 'call' or node.value not in self.program.functions:
                    continue
                called = self.program.functions[node.value]
                if not called.generics:
                    continue
                args = tuple(_substitute(node.generic_bindings[name], mapping) for name in called.generics)
                substitutions = dict(zip(called.generics, args))
                derived = Instantiation(called.name, args, tuple(_substitute(type_name, substitutions) for _, type_name in called.params), _substitute(called.return_type, substitutions))
                if derived not in seen:
                    seen.add(derived)
                    pending.append(derived)
                    self.program.instantiations.append(derived)
                    count += 1
                    if count > 1024:
                        self.fail(node, '泛型实例化展开超过限制，可能存在无限递归类型')

    def integer(self, node: Node) -> int | None:
        if node.kind == 'int':
            return node.value
        if node.kind == 'unary' and node.value in {'-', '+'}:
            value = self.integer(node.args[0])
            return (-value if node.value == '-' else value) if value is not None else None
        return None

    def shape_literal(self, node: Node, allow_infer: bool = False) -> tuple[int, ...] | None:
        if node.kind != 'list':
            return None
        values = [self.integer(item) for item in node.args]
        if any(item is None for item in values):
            return None
        if any(item < (-1 if allow_infer else 0) for item in values):
            self.fail(node, '张量维度必须非负，reshape 仅允许一个 -1')
        if allow_infer and values.count(-1) > 1:
            self.fail(node, 'reshape 仅允许一个 -1 推断维度')
        return tuple(values)

    def broadcast_shapes(self, left, right, node):
        if left is None or right is None:
            return None
        result = []
        for offset in range(1, max(len(left), len(right)) + 1):
            first = left[-offset] if offset <= len(left) else 1
            second = right[-offset] if offset <= len(right) else 1
            if first != second and first != 1 and second != 1:
                self.fail(node, f'静态广播形状不兼容: {left} 与 {right}')
            result.append(second if first == 1 else first)
        return tuple(reversed(result))

    def infer_shape(self, name: str, node: Node):
        args = node.args
        if name in {'zeros', 'ones', 'full', 'rand', 'randn'}:
            return self.shape_literal(args[0])
        if name == 'tensor':
            shape = self.shape_literal(args[1])
            if shape is not None and args[0].kind == 'list' and math.prod(shape) != len(args[0].args):
                self.fail(node, 'tensor 数据元素数量与静态形状不匹配')
            return shape
        if name == 'arange':
            size = self.integer(args[0])
            if size is not None:
                if size < 0:
                    self.fail(node, 'arange 长度不能为负')
                return (size,)
        if name in {'add', 'sub', 'mul', 'div', 'equal'}:
            return self.broadcast_shapes(args[0].shape, args[1].shape, node)
        if name == 'matmul':
            left, right = args[0].shape, args[1].shape
            if left is None or right is None:
                return None
            if not left or not right:
                self.fail(node, 'matmul 输入至少需要一维')
            if left[-1] != (right[-2] if len(right) > 1 else right[0]):
                self.fail(node, f'matmul 静态内维不匹配: {left} 与 {right}')
            batch = self.broadcast_shapes(left[:-2] if len(left) > 1 else (), right[:-2] if len(right) > 1 else (), node)
            return batch + (left[-2:-1] if len(left) > 1 else ()) + (right[-1:] if len(right) > 1 else ())
        if name == 'reshape':
            target = self.shape_literal(args[1], allow_infer=True)
            if target is None:
                return None
            original = args[0].shape
            if -1 in target:
                if original is None:
                    return None
                product = math.prod(item for item in target if item != -1)
                total = math.prod(original)
                if product == 0 or total % product:
                    self.fail(node, 'reshape 无法推断静态维度')
                target = tuple(total // product if item == -1 else item for item in target)
            if original is not None and math.prod(original) != math.prod(target):
                self.fail(node, 'reshape 静态元素总数不一致')
            return target
        if name in {'broadcast_to', 'reduce_to_shape'}:
            target = self.shape_literal(args[1])
            original = args[0].shape
            if target is not None and original is not None:
                combined = self.broadcast_shapes(original, target, node)
                if combined != (target if name == 'broadcast_to' else original):
                    self.fail(node, f'{name} 静态目标形状不兼容')
            return target
        if name == 'transpose':
            original = args[0].shape
            permutation = self.shape_literal(args[1])
            if original is not None and permutation is not None:
                if sorted(permutation) != list(range(len(original))):
                    self.fail(node, 'transpose 维度排列无效')
                return tuple(original[index] for index in permutation)
            return None
        if name in {'sum_all'}:
            return ()
        if name in {'sum', 'mean', 'max', 'argmax'}:
            original = args[0].shape
            axis = self.integer(args[1])
            if original is None or axis is None:
                return None
            if not -len(original) <= axis < len(original):
                self.fail(node, f'{name} 静态维度轴越界')
            axis %= len(original)
            if name == 'argmax':
                return original[:axis] + original[axis + 1:]
            if args[2].kind != 'bool':
                return None
            return original[:axis] + ((1,) if args[2].value else ()) + original[axis + 1:]
        if name in {'clone', 'cast', 'scale', 'pow', 'exp', 'log', 'sqrt', 'tanh', 'sigmoid', 'relu', 'abs', 'sign', 'greater', 'softmax', 'fused_affine_relu'}:
            return args[0].shape
        return None


def check(program: Program) -> Program:
    return _Checker(program).run()


def load_program(path: Path) -> Program:
    path = Path(path).resolve()
    if path.suffix != '.heng':
        raise HengError('源文件后缀必须为 .heng', str(path))
    source_root = path.parent
    heng_root = None
    for ancestor in (source_root, *source_root.parents):
        candidate = ancestor / 'heng'
        if candidate.is_dir():
            heng_root = candidate.resolve()
            if not heng_root.is_relative_to(ancestor.resolve()):
                raise HengError('项目 heng 模块根目录越界', str(path))
            break
    allowed_roots = [source_root]
    if heng_root is not None and heng_root not in allowed_roots:
        allowed_roots.append(heng_root)
    merged = Program()
    visited = set()
    active = []

    def allowed(candidate: Path) -> bool:
        return any(candidate.is_relative_to(root) for root in allowed_roots)

    def load(candidate: Path, origin: Node | None = None):
        if candidate in active:
            cycle = ' -> '.join(item.name for item in active + [candidate])
            raise HengError(f'检测到循环导入: {cycle}', origin.source if origin else str(candidate), origin.line if origin else 1, origin.column if origin else 1)
        if candidate in visited:
            return
        try:
            source = candidate.read_text(encoding='utf-8-sig')
        except (OSError, UnicodeError) as error:
            raise HengError(f'无法读取模块 {candidate.name}: {error}', origin.source if origin else str(candidate), origin.line if origin else 1, origin.column if origin else 1) from error
        module = parse(source, str(candidate))
        active.append(candidate)
        for imported in module.imports:
            spelling = imported.value
            relative = Path(spelling)
            if not spelling or relative.is_absolute() or relative.root or relative.drive or '\\' in spelling or ':' in spelling or '\x00' in spelling or relative.suffix != '.heng':
                raise HengError('import 必须使用后缀为 .heng 的相对路径', imported.source, imported.line, imported.column)
            candidates = [(candidate.parent / relative).resolve()]
            if heng_root is not None:
                candidates.append((heng_root / relative).resolve())
            if not allowed(candidates[0]):
                raise HengError('模块导入路径越界', imported.source, imported.line, imported.column)
            valid = [item for item in candidates if allowed(item)]
            selected = next((item for item in valid if item.is_file()), valid[0])
            load(selected, imported)
        for table_name in ('functions', 'structs', 'enums'):
            for name, definition in getattr(module, table_name).items():
                if any(name in getattr(merged, table) for table in ('functions', 'structs', 'enums')):
                    raise HengError(f'跨模块全局名称重复: {name}', definition.source, definition.line, definition.column)
                getattr(merged, table_name)[name] = definition
        merged.imports.extend(module.imports)
        merged.sources.append(str(candidate))
        active.pop()
        visited.add(candidate)

    load(path)
    return check(merged)
