from pathlib import Path
import os
import tempfile
from unittest.mock import patch
import unittest

from henglang.frontend import HengError, Lexer, Parser, Node, Program, check, load_program, parse


class FrontendTests(unittest.TestCase):
    def checked(self, source):
        return check(parse(source, 'case.heng'))

    def rejected(self, source, message):
        with self.assertRaisesRegex(HengError, message) as caught:
            self.checked(source)
        self.assertRegex(str(caught.exception), r'case\.heng:\d+:\d+:')

    def test_lexer_values_and_locations(self):
        tokens = Lexer('let a = 12;\nvar b = 1.25e-2; "a\\n\\u4e2d"', 'lexer.heng').tokenize()
        self.assertEqual(tokens[3].value, 12)
        self.assertEqual((tokens[5].line, tokens[5].column), (2, 1))
        self.assertEqual(tokens[8].value, 0.0125)
        self.assertEqual(tokens[10].value, 'a\n中')
        self.assertEqual(tokens[-1].kind, 'eof')

    def test_lexer_rejects_malformed_input(self):
        for source, message in [('"abc', '字符串未结束'), ('"a\nb"', '直接换行'), ('"\\q"', '未知'), ('/*x', '注释未结束'), ('@', '无法识别'), ('1e999', '浮点'), ('"\\ud800"', '代理码位')]:
            with self.subTest(source=source):
                with self.assertRaisesRegex(HengError, message):
                    Lexer(source).tokenize()

    def test_source_comments_are_lexed(self):
        program = self.checked('/* ignored */ fn main() { let a = 1; // ignored\n print(a); }')
        self.assertEqual(len(program.functions['main'].body.args), 2)

    def test_node_contract_and_parse_only(self):
        program = parse('fn main() { let x = unknown; var y:i64 = 2; y += 1; }')
        self.assertIsInstance(program, Program)
        first, second, third = program.functions['main'].body.args
        self.assertIsInstance(first, Node)
        self.assertEqual(first.kind, 'let')
        self.assertFalse(first.mutable)
        self.assertTrue(second.mutable)
        self.assertEqual(third.kind, 'assign')
        self.assertEqual(third.value, '+=')
        self.assertFalse(program.checked)
        self.assertEqual(Parser('fn main() {}').parse().functions['main'].return_type, 'void')

    def test_suffix_required(self):
        with self.assertRaisesRegex(HengError, '后缀'):
            parse('fn main() {}', 'bad.py')
        with self.assertRaisesRegex(HengError, '后缀'):
            load_program(Path('bad.py'))

    def test_precedence_and_logic(self):
        program = self.checked('fn main() { let x = 1 + 2 * 3; let ok = x > 1 && !false || false; assert(ok, "ok"); }')
        expression = program.functions['main'].body.args[0].args[0]
        self.assertEqual(expression.value, '+')
        self.assertEqual(expression.args[1].value, '*')
        self.assertEqual(program.functions['main'].body.args[1].type, 'bool')

    def test_types_and_explicit_numeric_conversions(self):
        program = self.checked('fn main() { let a:f32 = 1; let b:f64 = a; let c:f32 = f32(1.5); let d = [1, 2.0]; }')
        self.assertEqual([item.type for item in program.functions['main'].body.args], ['f32', 'f64', 'f32', 'List<f64>'])
        self.rejected('fn main() { let a:f32 = 1.0; }', '类型不匹配')
        self.rejected('fn main() { let a:i64 = 1.0; }', '类型不匹配')
        self.rejected('fn main() { let a = f64(zeros([1])); }', '转换只接受')

    def test_integer_bounds(self):
        self.checked('fn main() { let low = -9223372036854775808; let high = 9223372036854775807; }')
        self.rejected('fn main() { let a = 9223372036854775808; }', 'i64 范围')
        self.rejected('fn main() { let a = -9223372036854775809; }', 'i64 范围')

    def test_scopes_and_mutability(self):
        self.checked('fn main() { let x = 1; { let x = 2; print(x); } var y = 1; y += x; }')
        self.rejected('fn main() { let x = 1; x = 2; }', '不能重新绑定')
        self.rejected('fn main() { let x = 1; let x = 2; }', '变量重复')
        self.rejected('fn main() { { let x = 1; } print(x); }', '未定义的变量')
        self.rejected('fn f(x:i64, x:i64) {}', '变量重复')
        self.rejected('fn f(x:i64) { let x = 1; }', '变量重复')

    def test_control_flow_and_loop_contract(self):
        program = self.checked('fn f(x:bool)->i64 { if x { return 1; } else if !x { return 2; } else { return 3; } } fn main() { var sum = 0; for x in [1,2] { if x == 2 { continue; } sum += x; } while sum < 4 { sum += 1; if sum == 4 { break; } } }')
        loop = program.functions['main'].body.args[1]
        self.assertEqual((loop.kind, loop.value, loop.type), ('for', 'x', 'i64'))
        self.assertEqual(program.functions['f'].body.args[0].args[2].kind, 'block')

    def test_return_paths(self):
        self.checked('fn f()->i64 { while true { return 1; } }')
        self.checked('fn f()->i64 { while true { continue; } }')
        self.checked('fn f(x:bool)->i64 { if x { return 1; } return 2; }')
        self.rejected('fn f(x:bool)->i64 { if x { return 1; } }', '所有路径')
        self.rejected('fn f()->i64 { while true { break; } }', '所有路径')
        self.rejected('fn f()->i64 { for x in [1] { return x; } }', '所有路径')
        self.rejected('fn f()->i64 { return; }', '类型不匹配')
        self.rejected('fn f() { return 1; }', 'void 函数')
        self.rejected('fn f()->i64 { return false; }', '类型不匹配')

    def test_loop_and_condition_errors(self):
        self.rejected('fn main() { break; }', '循环内部')
        self.rejected('fn main() { continue; }', '循环内部')
        self.rejected('fn main() { if 1 {} }', '类型不匹配')
        self.rejected('fn main() { while 1 {} }', '类型不匹配')
        self.rejected('fn main() { for x in 1 {} }', '只能遍历')
        self.rejected('fn main() { for x in [1] {} print(x); }', '未定义的变量')

    def test_record_and_enum_contract(self):
        program = self.checked('struct Entry { data:Tensor, parents:List<i64>, op:Op } enum Op { Add, Mul } fn main() { let entry = Entry { parents:[], op:Op.Add, data:zeros([2]) }; entry.data = ones([2]); entry.parents[0] = 2; let same = entry.op == Op.Mul; }')
        record = program.functions['main'].body.args[0].args[0]
        self.assertEqual(record.kind, 'record')
        self.assertEqual(record.value, 'Entry')
        self.assertEqual(record.names, ['parents', 'op', 'data'])
        member = record.args[1]
        self.assertEqual((member.kind, member.value, member.type), ('field', 'Add', 'Op'))
        self.assertEqual(record.args[0].type, 'List<i64>')

    def test_record_errors(self):
        prefix = 'struct Entry { x:i64 } '
        for body, message in [('let e = Entry {};', '缺失: x'), ('let e = Entry { x:1, y:2 };', '多余: y'), ('let e = Entry { x:1, x:2 };', '字段重复'), ('let e = Entry { x:false };', '类型不匹配'), ('let e = Missing { x:1 };', '未定义的结构'), ('let e = Entry { x:1 }; print(e.y);', '没有字段')]:
            with self.subTest(body=body):
                self.rejected(prefix + 'fn main() { ' + body + ' }', message)
        self.rejected('struct X { a:i64, a:i64 }', '字段重复')
        self.rejected('enum X { A, A }', '枚举成员重复')
        self.rejected('enum X {}', '至少需要')
        self.rejected('enum X { A } fn main() { let a = X.B; }', '枚举成员')
        self.rejected('enum X { A } fn main() { X.A = X.A; }', '不能赋值')

    def test_list_context_and_invariance(self):
        program = self.checked('fn use(x:List<f64>) {} fn main() { let a:List<i64> = []; let b:List<List<i64>> = [[],[1]]; use([1,2]); push(a, 2); let c = pop(a); clear(a); }')
        self.assertEqual(program.functions['main'].body.args[1].type, 'List<List<i64>>')
        self.assertEqual(program.functions['main'].body.args[4].type, 'i64')
        self.rejected('fn main() { let x = []; }', '空列表')
        self.rejected('fn main() { let x = [1, false]; }', '列表元素类型')
        self.rejected('fn use(x:List<f64>) {} fn main() { let x = [1]; use(x); }', '类型不匹配')
        self.rejected('fn main() { let x = [1]; push(x, false); }', '类型不匹配')
        self.rejected('fn main() { let x = [print(1)]; }', '不能为 void')

    def test_indexing(self):
        self.checked('fn main() { let x = [1]; x[0] += 2; }')
        self.rejected('fn main() { let x = [1]; let y = x[0.0]; }', '类型不匹配')
        self.rejected('fn main() { let x = 1; let y = x[0]; }', '不支持索引')
        self.rejected('fn main() { let x = zeros([1]); let y = x[0]; }', '不支持索引')
        self.rejected('fn main() { (1 + 2) = 3; }', '赋值目标')

    def test_unknown_and_reserved_types(self):
        for type_name in ['Any', 'Unknown', 'List<Any>', 'List<i64,f64>', 'Result<void>', 'List<void>']:
            with self.subTest(type_name=type_name):
                self.rejected(f'fn f(x:{type_name}) {{}}', '类型|void')
        self.rejected('fn main() { let x:Any = 1; }', '未定义的类型')
        self.rejected('fn print() {}', '全局名称')
        self.rejected('struct i64 {}', '全局名称')
        self.rejected('fn f<T,T>() {}', '泛型参数重复')
        self.rejected('fn f<i64>() {}', '泛型参数')
        self.rejected('fn main() { let x:void = print(1); }', 'void')

    def test_generic_inference_and_explicit_calls(self):
        program = self.checked('fn identity<T>(x:T)->T { return x; } fn first<T>(values:List<T>)->T { return values[0]; } fn main() { let a = identity(1); let b = identity<f64>(2); let c = first([true]); let d:List<i64> = identity([]); }')
        body = program.functions['main'].body.args
        self.assertEqual([node.type for node in body], ['i64', 'f64', 'bool', 'List<i64>'])
        self.assertEqual(body[1].args[0].type_args, ['f64'])
        self.assertEqual(body[1].args[0].generic_bindings, {'T': 'f64'})
        self.assertIn(('identity', ('i64',)), {(item.function, item.type_args) for item in program.instantiations})
        self.assertIn(('first', ('bool',)), {(item.function, item.type_args) for item in program.instantiations})

    def test_nested_generic_instantiations(self):
        program = self.checked('fn identity<T>(x:T)->T { return x; } fn wrapper<U>(x:U)->U { return identity(x); } fn main() { let x = wrapper(1); }')
        self.assertIn(('identity', ('i64',)), {(item.function, item.type_args) for item in program.instantiations})

    def test_generic_errors(self):
        prefix = 'fn identity<T>(x:T)->T { return x; } '
        self.rejected(prefix + 'fn main() { let x = identity<i64,f64>(1); }', '泛型实参数量')
        self.rejected(prefix + 'fn main() { let x = identity<Missing>(1); }', '未定义的类型')
        self.rejected(prefix + 'fn main() { let x = identity<i64>(false); }', '类型不匹配')
        self.rejected('fn make<T>()->List<T> { return []; } fn main() { let x = make(); }', '无法推断')
        self.rejected('fn f<T>(a:T, b:T) {} fn main() { f(1, false); }', '类型不匹配')
        self.rejected('fn main() { print<i64>(1); }', '泛型实参数量')
        self.rejected('fn compare<T>(x:T)->bool { return x==x; }', '相等比较')
        self.rejected('fn add_value<T>(x:T)->T { return x+x; }', '不支持')

    def test_generic_numeric_return_promotion_preserves_container_invariance(self):
        program = self.checked('fn identity<T>(value:T)->T { return value; } fn first<T>(values:List<T>)->T { return values[0]; } fn main() { let values=[1]; let x:f64=pop(values); let y:f64=first(values); let z:f64=identity<i64>(1); let ok:Result<f64>=result_ok(1); }')
        body = program.functions['main'].body.args
        self.assertEqual([node.type for node in body], ['List<i64>', 'f64', 'f64', 'f64', 'Result<f64>'])
        self.assertEqual(body[1].args[0].generic_bindings, {'T':'i64'})
        self.assertEqual(body[4].args[0].generic_bindings, {'T':'f64'})

    def test_result_context(self):
        program = self.checked('fn fail()->Result<i64> { return result_err("失败"); } fn main() { let x:Result<i64> = result_ok(1); let y:Result<f64> = result_err("错误"); let z = result_err<bool>("错误"); let a = result_value(x); let b = result_is_ok(y); let c = result_error(z); }')
        self.assertEqual([node.type for node in program.functions['main'].body.args], ['Result<i64>', 'Result<f64>', 'Result<bool>', 'i64', 'bool', 'str'])
        self.rejected('fn main() { let x = result_err("失败"); }', '无法推断')
        self.rejected('fn main() { let x:Result<i64> = result_ok(false); }', '类型不匹配')

    def test_tensor_list_context_and_static_shapes(self):
        program = self.checked('fn main() { let x = Tensor([1,2,3,4,5,6], [2,3]); let y = ones([3,4]); let z = matmul(x,y); let a = reshape(z, [4,2]); let b = a + 1; let c = 2 * a; let d = transpose(x,[1,0]); let e = sum(x,1,true); let f = sum_all(x); }')
        body = program.functions['main'].body.args
        self.assertEqual(body[0].args[0].value, 'tensor')
        self.assertEqual(body[0].args[0].args[0].type, 'List<f64>')
        self.assertEqual([node.args[0].shape for node in body], [(2,3), (3,4), (2,4), (4,2), (4,2), (4,2), (3,2), (2,1), ()])

    def test_tensor_shape_errors(self):
        for source, message in [('let x = zeros([-1]);', '维度必须'), ('let x = tensor([1],[2]);', '元素数量'), ('let x = matmul(zeros([2,3]),ones([2,4]));', '内维不匹配'), ('let x = zeros([2,3]) + ones([4]);', '广播形状'), ('let x = reshape(zeros([2,3]),[5]);', '元素总数'), ('let x = reshape(zeros([2,3]),[-1,-1]);', '仅允许一个'), ('let x = broadcast_to(zeros([2,3]),[3]);', '目标形状'), ('let x = transpose(zeros([2,3]),[0,0]);', '排列无效'), ('let x = sum(zeros([2,3]),2,false);', '轴越界'), ('let x = arange(-1);', '不能为负')]:
            with self.subTest(source=source):
                self.rejected('fn main() { ' + source + ' }', message)

    def test_reshape_inferred_and_broadcast_zero_dimension(self):
        program = self.checked('fn main() { let x = reshape(zeros([2,3]),[-1,2]); let y = zeros([0,3]) + ones([1,3]); let z = matmul(zeros([3]),ones([3])); }')
        self.assertEqual([node.args[0].shape for node in program.functions['main'].body.args], [(3,2), (0,3), ()])

    def test_dynamic_shapes_remain_unknown(self):
        program = self.checked('fn f(dims:List<i64>) { let x = zeros(dims); let y = matmul(x,ones([2,3])); }')
        self.assertIsNone(program.functions['f'].body.args[1].args[0].shape)

    def test_mutable_shape_updates_and_branches(self):
        program = self.checked('fn f(flag:bool) { var x = zeros([2,3]); if flag { x = zeros([4,3]); } let y = matmul(x,zeros([3,5])); while flag { x = zeros([6,3]); } let z = matmul(x,zeros([3,5])); }')
        body = program.functions['f'].body.args
        self.assertIsNone(body[2].args[0].shape)
        self.assertIsNone(body[4].args[0].shape)
        self.rejected('fn main() { var x = zeros([2,3]); x = zeros([4]); let y = matmul(x,zeros([3,2])); }', '内维不匹配')

    def test_builtin_signatures(self):
        self.checked('fn main() { print(1); print("x"); assert(true,"ok"); let x = len([1]); let y = str(false); let z = len(zeros([1])); }')
        self.rejected('fn main() { assert(true); }', '需要 2 个参数')
        self.rejected('fn main() { let x = len(1); }', 'len 只支持')
        self.rejected('fn main() { missing(); }', '未定义的函数')
        self.rejected('fn main() { let x = true + 1; }', '不支持')
        self.rejected('fn main() { let x = 1.0 % 2.0; }', '取余')
        self.rejected('fn main() { let x = zeros([1]) == zeros([1]); }', '相等比较')

    def test_complete_error_location(self):
        with self.assertRaises(HengError) as caught:
            check(parse('fn main() {\n  let x = missing;\n}', 'nested/source.heng'))
        self.assertEqual((caught.exception.source, caught.exception.line, caught.exception.column), ('nested/source.heng', 2, 11))
        self.assertIn('未定义的变量', str(caught.exception))

    def test_compact_generic_assignment_and_comparison(self):
        program = self.checked('fn main() { let x:List<i64>=[]; let y:List<List<i64>>=[[]]; let a=1; let b=2; let ok=a<b && b>=a; }')
        self.assertEqual(program.functions['main'].body.args[0].type, 'List<i64>')
        self.assertEqual(program.functions['main'].body.args[1].type, 'List<List<i64>>')
        expression = parse('fn main() { let x=a<b>=c; }').functions['main'].body.args[0].args[0]
        self.assertEqual(expression.value, '>=')
        self.assertEqual(expression.args[0].value, '<')

    def test_rechecking_typed_ast(self):
        program = self.checked('fn identity<T>(x:T)->T { return x; } fn main() { let a = identity(1); let b:List<f64> = [1,2]; }')
        original = list(program.instantiations)
        self.assertIs(check(program), program)
        self.assertEqual(program.instantiations, original)
        self.assertTrue(program.checked)

    def test_checked_flag_clears_after_failed_recheck(self):
        program = self.checked('fn main() {}')
        program.functions['main'].return_type = 'Missing'
        with self.assertRaises(HengError):
            check(program)
        self.assertFalse(program.checked)

    def test_minimum_integer_is_lowered_to_signed_literal(self):
        program = self.checked('fn main() { let minimum = -9223372036854775808; }')
        literal = program.functions['main'].body.args[0].args[0]
        self.assertEqual((literal.kind, literal.value, literal.args), ('int', -(2 ** 63), []))
        check(program)

    def test_nul_string_is_rejected_with_source_location(self):
        self.rejected('fn main() { let x = "a\\0b"; }', 'NUL')

    def test_generic_argument_order_and_nested_results(self):
        program = self.checked('fn take<T>(values:List<T>, value:T) {} fn identity<T>(x:T)->T { return x; } fn main() { take([],1); let x:Result<List<i64>> = result_ok([]); let y:Result<i64> = identity(result_err("错误")); }')
        body = program.functions['main'].body.args
        self.assertEqual(body[0].args[0].args[0].type, 'List<i64>')
        self.assertEqual(body[1].type, 'Result<List<i64>>')
        self.assertEqual(body[2].type, 'Result<i64>')

    def test_compound_tensor_shape_and_batched_matmul(self):
        program = self.checked('fn main() { var x = zeros([2,3]); x += 1; let y = x * ones([3]); let z = matmul(zeros([5,2,3]),ones([1,3,4])); }')
        self.assertEqual(program.functions['main'].body.args[2].args[0].shape, (2,3))
        self.assertEqual(program.functions['main'].body.args[3].args[0].shape, (5,2,4))
        self.rejected('fn main() { var x = zeros([2]); x += ones([3]); }', '广播形状')

    def test_local_enum_name_can_be_shadowed(self):
        self.checked('enum Op { Add } struct Entry { value:i64 } fn main() { let Op = Entry { value:1 }; Op.value=2; }')

    def test_parse_errors(self):
        for source in ['let x = 1;', 'fn main( {}', 'fn main() { let x = 1 }', 'fn main() {', 'import bare.heng;', 'fn main() { let x = ; }', 'fn same() {} struct same {}']:
            with self.subTest(source=source):
                with self.assertRaises(HengError):
                    parse(source, 'case.heng')


class ModuleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, relative, source):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8')
        return path

    def test_relative_imports_and_global_merge(self):
        self.write('lib/value.heng', 'struct Entry { x:i64 } fn value()->i64 { return 42; }')
        main = self.write('main.heng', 'import "lib/value.heng"; fn main() { let e = Entry { x:value() }; }')
        program = load_program(main)
        self.assertTrue(program.checked)
        self.assertEqual(set(program.functions), {'main','value'})
        self.assertEqual(len(program.sources), 2)

    def test_shared_dependency_loaded_once(self):
        self.write('common.heng', 'fn common()->i64 { return 1; }')
        self.write('a.heng', 'import "common.heng"; fn a()->i64 { return common(); }')
        self.write('b.heng', 'import "common.heng"; fn b()->i64 { return common(); }')
        main = self.write('main.heng', 'import "a.heng"; import "b.heng"; fn main() {}')
        program = load_program(main)
        self.assertEqual(len(program.sources), 4)

    def test_project_heng_import_root(self):
        self.write('heng/math.heng', 'fn value()->i64 { return 1; }')
        main = self.write('examples/main.heng', 'import "math.heng"; fn main() { print(value()); }')
        self.assertIn('value', load_program(main).functions)
        main.write_text('import "../heng/math.heng"; fn main() {}', encoding='utf-8')
        self.assertIn('value', load_program(main).functions)

    def test_cycles_include_path_and_location(self):
        main = self.write('main.heng', 'import "a.heng"; fn main() {}')
        self.write('a.heng', 'import "main.heng"; fn a() {}')
        with self.assertRaisesRegex(HengError, '循环导入') as caught:
            load_program(main)
        self.assertEqual(caught.exception.source, str(self.root / 'a.heng'))
        self.assertEqual((caught.exception.line, caught.exception.column), (1,8))

    def test_duplicate_cross_module_names(self):
        self.write('a.heng', 'struct Same {}')
        main = self.write('main.heng', 'import "a.heng"; fn Same() {}')
        with self.assertRaisesRegex(HengError, '跨模块全局名称重复'):
            load_program(main)

    def test_path_escape_and_wrong_suffix(self):
        for spelling, message in [('../outside.heng','越界'), ('module.py','相对路径'), ('/absolute.heng','相对路径'), ('C:/absolute.heng','相对路径'), ('sub\\\\file.heng','相对路径')]:
            with self.subTest(spelling=spelling):
                main = self.write('project/main.heng', 'import "' + spelling + '"; fn main() {}')
                with self.assertRaisesRegex(HengError, message):
                    load_program(main)

    def test_resolved_heng_root_cannot_escape_project(self):
        main = self.write('project/main.heng', 'import "external.heng"; fn main() {}')
        self.write('project/heng/internal.heng', 'fn internal() {}')
        outside = self.write('outside/external.heng', 'fn external() {}').parent.resolve()
        real_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            if path.name == 'heng':
                return outside
            return real_resolve(path, *args, **kwargs)

        with patch.object(Path, 'resolve', resolve):
            with self.assertRaisesRegex(HengError, '根目录越界'):
                load_program(main)

    def test_resolved_import_path_is_checked_after_resolution(self):
        outside = self.write('outside.heng', 'fn external() {}').resolve()
        main = self.write('project/main.heng', 'import "link.heng"; fn main() {}')
        self.write('project/link.heng', 'fn external() {}')
        real_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            if path.name == 'link.heng':
                return outside
            return real_resolve(path, *args, **kwargs)

        with patch.object(Path, 'resolve', resolve):
            with self.assertRaisesRegex(HengError, '越界'):
                load_program(main)

    def test_symlink_escape(self):
        outside = self.write('outside/external.heng', 'fn external() {}').parent
        main = self.write('project/main.heng', 'import "link/external.heng"; fn main() {}')
        link = main.parent / 'link'
        if os.name == 'nt':
            import _winapi
            _winapi.CreateJunction(str(outside), str(link))
        else:
            link.symlink_to(outside, target_is_directory=True)
        try:
            with self.assertRaisesRegex(HengError, '越界'):
                load_program(main)
        finally:
            if os.name == 'nt':
                link.rmdir()
            else:
                link.unlink()

    def test_missing_and_invalid_encoding(self):
        main = self.write('main.heng', 'import "missing.heng"; fn main() {}')
        with self.assertRaisesRegex(HengError, '无法读取模块'):
            load_program(main)
        (self.root / 'missing.heng').write_bytes(b'\xff\xfe\x80')
        with self.assertRaisesRegex(HengError, '无法读取模块'):
            load_program(main)

    def test_utf8_bom(self):
        main = self.write('main.heng', '\ufefffn main() {}')
        self.assertTrue(load_program(main).checked)

    def test_imported_type_error_uses_imported_path(self):
        self.write('bad.heng', 'fn bad()->i64 { return false; }')
        main = self.write('main.heng', 'import "bad.heng"; fn main() {}')
        with self.assertRaises(HengError) as caught:
            load_program(main)
        self.assertEqual(caught.exception.source, str(self.root / 'bad.heng'))
        self.assertIn('类型不匹配', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
