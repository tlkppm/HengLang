from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


class BenchmarkError(RuntimeError):
    pass


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise BenchmarkError("命令行参数无效，请使用 --help 查看允许的参数")


def parse_arguments(arguments=None):
    parser = JsonArgumentParser(description="HENG 与已安装框架的 f64 CPU 矩阵乘法对比；不安装依赖、不训练，JSON 输出到 stdout")
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256], help="方阵边长，默认 128 256，允许 2..2048")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数，至少 3")
    parser.add_argument("--repeats", type=int, default=20, help="计时次数，至少 20，取中位数")
    parser.add_argument("--frameworks", choices=["numpy", "torch"], nargs="+", default=["numpy"], help="仅使用已安装的框架，默认 numpy；torch 为可选项")
    parser.add_argument("--timeout", type=float, default=120.0, help="每个 HENG 原生进程的执行超时秒数")
    parser.add_argument("--build-timeout", type=float, default=300.0, help="每次 HENG 构建及执行的总超时秒数")
    args = parser.parse_args(arguments)
    validate_configuration(args.sizes, args.warmup, args.repeats)
    if not all(math.isfinite(value) and value > 0 for value in (args.timeout, args.build_timeout)):
        raise BenchmarkError("超时必须是有限正数")
    if args.build_timeout < args.timeout:
        raise BenchmarkError("构建及执行总超时不得小于执行超时")
    if len(set(args.frameworks)) != len(args.frameworks):
        raise BenchmarkError("框架参数不得重复")
    return args


def validate_configuration(sizes, warmup, repeats):
    if not sizes or any(type(size) is not int or not 2 <= size <= 2048 for size in sizes):
        raise BenchmarkError("矩阵边长必须是 2..2048 的整数")
    if len(set(sizes)) != len(sizes):
        raise BenchmarkError("矩阵边长不得重复")
    if type(warmup) is not int or warmup < 3:
        raise BenchmarkError("至少需要 3 次预热")
    if type(repeats) is not int or repeats < 20:
        raise BenchmarkError("中位数基准至少需要 20 次计时")


def input_parameters(size):
    return 1.0 / (size * size), -0.5, -0.75 / (size * size), 0.25


def input_value(size, index, side):
    if not 0 <= index < size * size or side not in ("left", "right"):
        raise BenchmarkError("输入采样位置或矩阵名称无效")
    left_scale, left_bias, right_scale, right_bias = input_parameters(size)
    scale, bias = (left_scale, left_bias) if side == "left" else (right_scale, right_bias)
    return index * scale + bias


def sample_positions(size):
    return sorted({0, size - 1, (size // 2) * size + size // 2, size * size - size, size * size - 1})


def reference_samples(size):
    return [
        math.fsum(
            input_value(size, (index // size) * size + inner, "left")
            * input_value(size, inner * size + index % size, "right")
            for inner in range(size)
        )
        for index in sample_positions(size)
    ]


def validate_samples(actual, expected, label):
    if len(actual) != len(expected):
        raise BenchmarkError(label + "的校验样本数量不匹配")
    errors = []
    for value, reference in zip(actual, expected):
        if not math.isfinite(value) or not math.isclose(value, reference, rel_tol=1e-10, abs_tol=1e-11):
            raise BenchmarkError(label + "未通过独立数值校验")
        errors.append(abs(value - reference))
    return {"status": "PASS", "samples": actual, "max_absolute_error": max(errors, default=0.0), "rtol": 1e-10, "atol": 1e-11}


def summarize_timings(values, repeats):
    if len(values) != repeats or any(not math.isfinite(value) or value <= 0 for value in values):
        raise BenchmarkError("计时样本数量错误或包含非正、非有限值")
    return {"median_seconds": statistics.median(values), "samples_seconds": values}


def heng_source(size, warmup, repeats):
    validate_configuration([size], warmup, repeats)
    left_scale, left_bias, right_scale, right_bias = input_parameters(size)
    positions = sample_positions(size)
    inputs = "\n".join(f'print(item({side}, {index}));' for side in ("left", "right") for index in positions)
    outputs = "\n".join(f'print(item(result, {index}));' for index in positions)
    return f'''fn trial(left: Tensor, right: Tensor) -> f64 {{
    let start = clock();
    let result = matmul(left, right);
    let elapsed = clock() - start;
    assert(numel(result) == {size * size}, "矩阵乘法输出元素数量错误");
    return elapsed;
}}
fn verify(left: Tensor, right: Tensor) -> void {{
    let result = matmul(left, right);
    assert(dtype(result) == "f64", "矩阵乘法输出必须是 f64");
    print("HENG_OUTPUT_V1");
    {outputs}
}}
fn main() -> void {{
    set_threads(1);
    set_optimized(true);
    print("HENG_THREADS_V1");
    print(get_threads());
    let sequence = arange({size * size});
    let left = reshape(add(scale(sequence, {left_scale!r}), full([], {left_bias!r})), [{size}, {size}]);
    let right = reshape(add(scale(sequence, {right_scale!r}), full([], {right_bias!r})), [{size}, {size}]);
    assert(dtype(left) == "f64" && dtype(right) == "f64", "输入必须是 f64");
    print("HENG_INPUT_V1");
    {inputs}
    verify(left, right);
    var warm: i64 = 0;
    while warm < {warmup} {{
        gc_collect();
        let elapsed = trial(left, right);
        warm += 1;
    }}
    let samples: List<f64> = [];
    var index: i64 = 0;
    while index < {repeats} {{
        gc_collect();
        let elapsed = trial(left, right);
        push(samples, elapsed);
        index += 1;
    }}
    print("HENG_TIMINGS_V1");
    var output: i64 = 0;
    while output < len(samples) {{
        print(samples[output]);
        output += 1;
    }}
    print("HENG_END_V1");
}}
'''


def parse_heng_output(text, size, repeats):
    lines = text.splitlines()
    count = len(sample_positions(size))
    expected_length = 6 + 3 * count + repeats
    if len(lines) != expected_length:
        raise BenchmarkError("HENG 输出协议长度错误")
    if lines[0] != "HENG_THREADS_V1":
        raise BenchmarkError("HENG 线程输出协议标记错误")
    if lines[1] != "1":
        raise BenchmarkError("HENG get_threads() 必须返回整数 1；线程配置无效或与请求不匹配")
    threads = {
        "requested": 1, "num_threads": int(lines[1]), "verified": True, "query_symbol": "get_threads",
        "reason": "实际 get_threads() 返回 1；已审查 matmul 仅在 workers>1 时创建工作线程，配置为 1 时在当前线程计算且不调用外部 BLAS；与 OpenBLAS getter 一样验证计算线程配置，不是全部操作系统线程活动采样",
    }
    lines = lines[2:]
    if (lines[0] != "HENG_INPUT_V1" or lines[1 + 2 * count] != "HENG_OUTPUT_V1"
            or lines[2 + 3 * count] != "HENG_TIMINGS_V1" or lines[-1] != "HENG_END_V1"):
        raise BenchmarkError("HENG 输出协议标记错误")
    try:
        inputs = [float(value) for value in lines[1:1 + 2 * count]]
        outputs = [float(value) for value in lines[2 + 2 * count:2 + 3 * count]]
        timings = [float(value) for value in lines[3 + 3 * count:-1]]
    except ValueError:
        raise BenchmarkError("HENG 输出包含无效数值") from None
    expected_inputs = [input_value(size, index, side) for side in ("left", "right") for index in sample_positions(size)]
    input_check = validate_samples(inputs, expected_inputs, "HENG 输入")
    output_check = validate_samples(outputs, reference_samples(size), "HENG 输出")
    return {**summarize_timings(timings, repeats), "input_validation": input_check, "validation": output_check, "threads": threads}


def configure_threads():
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"


def load_framework(name):
    if name in sys.modules:
        raise BenchmarkError("框架已在设置线程环境前导入，请使用独立 Python 进程运行")
    try:
        return importlib.import_module(name)
    except ImportError:
        raise BenchmarkError("所选框架 " + name + " 未安装或无法导入；请改选已安装框架，本脚本不会安装依赖") from None


def loaded_blas_libraries():
    if os.name != "nt":
        raise OSError("当前备用线程查询仅支持 Windows 已加载 DLL")
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.K32EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    kernel.K32EnumProcessModules.restype = wintypes.BOOL
    kernel.K32GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    kernel.K32GetModuleFileNameExW.restype = wintypes.DWORD
    process = kernel.GetCurrentProcess()
    capacity = 256
    while capacity <= 4096:
        modules = (wintypes.HMODULE * capacity)()
        needed = wintypes.DWORD()
        if not kernel.K32EnumProcessModules(process, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
            raise OSError("无法枚举当前进程已加载模块")
        if needed.value <= ctypes.sizeof(modules):
            break
        capacity = max(capacity * 2, (needed.value + ctypes.sizeof(wintypes.HMODULE) - 1) // ctypes.sizeof(wintypes.HMODULE))
    else:
        raise OSError("已加载模块数量超过查询上限")
    libraries = []
    for handle in modules[:needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        filename = ctypes.create_unicode_buffer(32768)
        length = kernel.K32GetModuleFileNameExW(process, handle, filename, len(filename))
        if not length or length >= len(filename):
            raise OSError("已加载模块名称查询不完整")
        name = Path(filename.value).name
        if any(token in name.lower() for token in ("blas", "mkl", "blis")):
            libraries.append((name, ctypes.CDLL(None, handle=handle)))
    return libraries


def openblas_information():
    try:
        libraries = loaded_blas_libraries()
    except (OSError, AttributeError):
        return {"verified": False, "reason": "threadpoolctl 不可用，已加载 BLAS DLL 枚举失败；实际线程约束未核实", "pools": []}
    pools = []
    symbols = ("openblas_get_num_threads64_", "openblas_get_num_threads", "scipy_openblas_get_num_threads64_",
               "scipy_openblas_get_num_threads", "openblas_get_num_threads_", "scipy_openblas_get_num_threads_")
    for name, library in libraries:
        pool = {"user_api": "blas", "internal_api": "unknown", "library": name, "num_threads": None,
                "version": None, "query_symbol": None, "query_method": "ctypes，已有模块句柄；未加载额外 BLAS DLL"}
        for symbol in symbols:
            function = getattr(library, symbol, None)
            if function is None:
                continue
            function.argtypes = []
            function.restype = ctypes.c_int
            try:
                count = function()
            except OSError:
                pool["query_error"] = "OpenBLAS 导出函数调用失败，实际线程约束未核实"
                break
            pool.update(internal_api="openblas", num_threads=count if 1 <= count <= 65536 else None, query_symbol=symbol)
            break
        if pool["internal_api"] == "openblas":
            prefix = "scipy_openblas" if pool["query_symbol"].startswith("scipy_") else "openblas"
            for suffix in ("_get_config64_", "_get_config", "_get_config_"):
                function = getattr(library, prefix + suffix, None)
                if function is None:
                    continue
                function.argtypes = []
                function.restype = ctypes.c_char_p
                try:
                    config = function()
                except OSError:
                    break
                parts = config.decode("ascii", errors="replace").split() if config else []
                if len(parts) > 1 and parts[0].lower() == "openblas" and all(character.isalnum() or character in ".-_+" for character in parts[1]):
                    pool["version"] = parts[1]
                break
        pools.append(pool)
    verified = bool(pools) and all(pool["internal_api"] == "openblas" and pool["num_threads"] == 1 for pool in pools)
    reason = "通过已加载 OpenBLAS 的实际导出函数查询线程配置；不是操作系统线程活动采样"
    if not verified:
        reason += "；存在未核实或非单线程的 BLAS，不能称公平单线程"
    return {"verified": verified, "reason": reason, "pools": pools}


def threadpool_information():
    try:
        controller = importlib.import_module("threadpoolctl")
    except ImportError:
        return openblas_information()
    pools = [
        {key: pool.get(key) for key in ("user_api", "internal_api", "prefix", "version", "num_threads", "threading_layer")}
        for pool in controller.threadpool_info()
    ]
    verified = bool(pools) and any(pool["user_api"] == "blas" for pool in pools) and all(pool["num_threads"] == 1 for pool in pools)
    return {"verified": verified, "reason": "threadpoolctl 查询已加载线程池的实际配置；不是操作系统线程活动采样", "pools": pools}


def create_inputs(module, name, size):
    left_scale, left_bias, right_scale, right_bias = input_parameters(size)
    if name == "numpy":
        sequence = module.arange(size * size, dtype=module.float64)
    else:
        sequence = module.arange(size * size, dtype=module.float64, device="cpu")
    return ((sequence * left_scale + left_bias).reshape(size, size),
            (sequence * right_scale + right_bias).reshape(size, size))


def framework_benchmark(module, name, size, warmup, repeats):
    left, right = create_inputs(module, name, size)
    positions = sample_positions(size)
    actual_inputs = [float(matrix.reshape(-1)[index]) for matrix in (left, right) for index in positions]
    expected_inputs = [input_value(size, index, side) for side in ("left", "right") for index in positions]
    input_check = validate_samples(actual_inputs, expected_inputs, name + " 输入")
    result = module.matmul(left, right)
    if result.dtype != module.float64 or tuple(result.shape) != (size, size):
        raise BenchmarkError(name + " 输出形状或精度错误")
    expected_outputs = reference_samples(size)
    validation = validate_samples([float(result.reshape(-1)[index]) for index in positions], expected_outputs, name + " 输出")
    del result
    before = threadpool_information()
    times = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for index in range(warmup + repeats):
            gc.collect()
            start = time.perf_counter_ns()
            result = module.matmul(left, right)
            elapsed = (time.perf_counter_ns() - start) * 1e-9
            if index >= warmup:
                times.append(elapsed)
            validate_samples([float(result.reshape(-1)[position]) for position in positions], expected_outputs, name + " 迭代输出")
            del result
    finally:
        if was_enabled:
            gc.enable()
    after = threadpool_information()
    verified = before["verified"] and after["verified"]
    threads = {"verified": verified, "before": before, "after": after}
    if name == "torch":
        threads["torch_intraop"] = module.get_num_threads()
        threads["torch_interop"] = module.get_num_interop_threads()
        threads["verified"] = verified and threads["torch_intraop"] == 1 and threads["torch_interop"] == 1
    return {**summarize_timings(times, repeats), "input_validation": input_check, "validation": validation, "threads": threads}


def blas_metadata(frameworks):
    runtime = threadpool_information()
    libraries = [pool for pool in runtime["pools"] if pool["user_api"] == "blas"]
    numpy = frameworks.get("numpy")
    configuration = getattr(getattr(numpy, "__config__", None), "CONFIG", {})
    build = configuration.get("Build Dependencies", {}).get("blas", {}) if isinstance(configuration, dict) else {}
    return {
        "runtime_libraries": libraries,
        "numpy_build": {key: build[key] for key in ("name", "version") if key in build},
        "version_status": "已核实运行库版本" if libraries and all(pool.get("version") for pool in libraries) else "运行库版本未核实；构建元数据不等于实际运行库版本",
        "detected_loaded_blas": bool(libraries),
    }


def run_heng(directory, size, args):
    source = Path(directory) / f"matmul_{size}.heng"
    source.write_text(heng_source(size, args.warmup, args.repeats), encoding="utf-8", newline="\n")
    command = [sys.executable, "-B", "-m", "henglang", "run", str(source), "--timeout", str(args.timeout)]
    try:
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), capture_output=True,
                                encoding="utf-8", errors="replace", timeout=args.build_timeout)
    except subprocess.TimeoutExpired:
        raise BenchmarkError("HENG 构建及执行超过总超时；未形成有效计时结果") from None
    if result.returncode:
        raise BenchmarkError("HENG 构建或执行失败；为保护本地路径，不转发原始诊断，请单独使用项目 CLI 排查")
    return parse_heng_output(result.stdout, size, args.repeats)


def single_thread_configuration_verified(results):
    return bool(results) and all(
        row["heng"].get("threads", {}).get("verified") is True
        and bool(row["frameworks"])
        and all(result.get("threads", {}).get("verified") is True for result in row["frameworks"].values())
        for row in results
    )


def runtime_identity():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "runtime").glob("*")):
        if path.is_file() and path.suffix in (".c", ".h"):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run(args):
    configure_threads()
    loaded = {name: load_framework(name) for name in sorted(args.frameworks)}
    frameworks = {name: loaded[name] for name in args.frameworks}
    if "torch" in frameworks:
        frameworks["torch"].set_num_threads(1)
        frameworks["torch"].set_num_interop_threads(1)
    from tools.toolchain import VERSION
    identity = runtime_identity()
    report = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "仅 f64 CPU 稠密方阵 matmul；不是完整框架、训练、GPU、能耗或算力节省结论",
        "privacy": {"machine_information": "omitted"},
        "versions": {"frameworks": {name: str(module.__version__) for name, module in frameworks.items()},
                     "heng_runtime_sha256": identity, "heng_compiler": "Zig " + VERSION},
        "parameters": {"sizes": args.sizes, "dtype": "f64", "device": "cpu", "warmup": args.warmup,
                       "repeats": args.repeats, "statistic": "median", "requested_threads": 1,
                       "timeout_seconds": args.timeout, "build_and_run_timeout_seconds": args.build_timeout},
        "method": {"operation": "C = matmul(A, B)，每次调用创建新输出，不使用 out 参数",
                   "inputs": "i = arange(n*n), A = reshape(i*(1/(n*n))-0.5), B = reshape(i*(-0.75/(n*n))+0.25)，C 连续 f64",
                   "reference": "Python math.fsum 独立逐项乘积求和；仅校验记录的少量位置",
                   "timers": {"heng": "运行时 clock()，Windows QueryPerformanceCounter；仅原生调用内计时", "python_frameworks": "time.perf_counter_ns；包括 Python 调用开销"},
                   "excluded": ["输入构造", "导入", "编译", "进程启动", "显式垃圾回收", "校验", "输出销毁"],
                   "allocation": "包含逻辑新输出分配；各框架默认分配器和 HENG 缓冲池保留，不保证每次向操作系统申请存储",
                   "order": "每个 shape 先 HENG，再按 CLI 顺序运行其他框架；无并行计时",
                   "thread_environment": {name: "1" for name in THREAD_VARIABLES}},
        "limitations": ["fair_single_thread 仅表示双方 compute thread configuration 均已核实为 1，不代表 CPU 调度隔离、全部操作系统线程活动采样或完全公平的保证",
                        "CPU 频率、温度、系统负载、内存分配器及语言调用开销未归一化",
                        "单次短调用可能受计时分辨率和系统噪声影响；未进行跨进程重复或置信区间估计",
                        "确定性输入是仿射序列，不代表所有矩阵分布；浮点计算顺序允许不同",
                        "已有构建接口会在 .build 保留标准构建缓存；本次临时 .heng 源码自动清理",
                        "外层超时不能保证终止已启动的编译器子进程；超时后应确认构建已停止再重跑"],
        "results": [],
        "fair_single_thread_meaning": "所有 shape 的 HENG 与全部比较库的计算线程配置均已验证为 1；不保证 CPU 调度隔离或完全公平",
    }
    (ROOT / ".build").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="comparison-", dir=ROOT / ".build") as directory:
        for size in args.sizes:
            heng = run_heng(directory, size, args)
            row = {"shape": [size, size], "sample_flat_indices": sample_positions(size),
                   "reference_samples": reference_samples(size), "heng": heng, "frameworks": {}}
            for name, module in frameworks.items():
                row["frameworks"][name] = framework_benchmark(module, name, size, args.warmup, args.repeats)
                row["frameworks"][name]["ratio_framework_over_heng"] = row["frameworks"][name]["median_seconds"] / heng["median_seconds"]
                row["frameworks"][name]["ratio_interpretation"] = "大于 1 仅表示本次该算子 HENG 更快；线程配置匹配见 fair_single_thread，该标记不保证 CPU 调度隔离或完全公平"
            report["results"].append(row)
    report["fair_single_thread"] = single_thread_configuration_verified(report["results"])
    report["versions"]["blas"] = blas_metadata(frameworks)
    if runtime_identity() != identity:
        raise BenchmarkError("基准期间运行时源码发生变化，结果无效，请等待修改完成后重跑")
    return report


def main(arguments=None):
    try:
        args = parse_arguments(arguments)
        report = run(args)
    except (BenchmarkError, OSError, RuntimeError, ValueError) as error:
        message = str(error) if isinstance(error, BenchmarkError) else "基准环境或执行失败；原始异常已隐藏以保护本地路径"
        print(json.dumps({"schema_version": 1, "status": "FAIL", "error": message}, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
