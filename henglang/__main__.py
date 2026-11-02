from __future__ import annotations

import argparse
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
import threading
import tomllib
import unittest

from .frontend import HengError, load_program
from .emitter import emit
from .ir import optimize
from .formatter import format_source
from tools.toolchain import VERSION as ZIG_VERSION, ZIG, environment

ROOT = Path(__file__).resolve().parents[1]
BUILD_LOCK = threading.Lock()
NATIVE_FLAGS = ["-std=c17", "-g0", "-fno-fast-math", "-Werror=implicit-function-declaration", "-target", "x86_64-windows-gnu"]
LINK_FLAGS = ["-g0", "-s", "-Wl,--build-id=fast"]


def configuration() -> dict:
    path = ROOT / "heng.toml"
    with path.open("rb") as source:
        config = tomllib.load(source)
    if config.get("package", {}).get("source_extension") != ".heng":
        raise HengError("项目源码扩展名必须为 .heng")
    if config.get("build", {}).get("target") != "x86_64-windows-gnu":
        raise HengError("当前交付目标仅支持 Windows x86-64 CPU")
    if config.get("build", {}).get("module_roots") != ["heng"]:
        raise HengError("当前项目模块根目录必须为 heng")
    if config.get("toolchain", {}).get("zig_version") != ZIG_VERSION:
        raise HengError("项目工具链版本与固定版本不一致")
    return config


def runtime_snapshot(*, optimized: bool = True) -> tuple[Path, str]:
    paths = sorted([*(ROOT / "runtime").glob("*.c"), *(ROOT / "runtime").glob("*.h")])
    if not any(path.suffix == ".c" for path in paths):
        raise HengError("原生运行时尚未就绪")
    content = {path.name: path.read_bytes() for path in paths}
    if any(path.read_bytes() != content[path.name] for path in paths):
        raise HengError("运行时源码在构建期间发生变化，请在修改完成后重新构建")
    digest = sha256()
    digest.update(ZIG_VERSION.encode("ascii"))
    digest.update(" ".join(NATIVE_FLAGS).encode("ascii"))
    digest.update(b"-O2" if optimized else b"-O0")
    for name, data in content.items():
        digest.update(name.encode("utf-8"))
        digest.update(data)
    signature = digest.hexdigest()[:20]
    directory = ROOT / ".build" / "runtime" / signature
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in content.items():
        destination = directory / name
        if not destination.is_file() or destination.read_bytes() != data:
            temporary = destination.with_suffix(f".{os.getpid()}.tmp{destination.suffix}")
            temporary.write_bytes(data)
            temporary.replace(destination)
    return directory, signature


def runtime_objects(directory: Path, *, optimized: bool = True) -> list[Path]:
    result: list[Path] = []
    for source in sorted(directory.glob("*.c")):
        destination = source.with_suffix(".obj")
        if not destination.is_file():
            temporary = destination.with_suffix(f".{os.getpid()}.tmp.obj")
            command = [str(ZIG), "cc", *NATIVE_FLAGS, "-O2" if optimized else "-O0", "-I", str(directory), "-c", str(source), "-o", str(temporary)]
            compiled = subprocess.run(command, env=environment(), capture_output=True, encoding="utf-8", errors="replace")
            if compiled.returncode:
                raise HengError("原生运行时编译失败:\n" + compiled.stderr.strip())
            temporary.replace(destination)
        result.append(destination)
    return result


def build(source: Path, *, optimized: bool = True, output: Path | None = None, force: bool = False) -> Path:
    with BUILD_LOCK:
        return build_locked(source, optimized=optimized, output=output, force=force)


def build_locked(source: Path, *, optimized: bool = True, output: Path | None = None, force: bool = False) -> Path:
    config = configuration()
    source = source.resolve()
    program = optimize(load_program(source), enabled=optimized)
    generated = emit(program, ROOT, optimized=optimized)
    if not ZIG.is_file():
        raise HengError("项目原生工具链尚未就绪")
    snapshot, runtime_signature = runtime_snapshot(optimized=optimized)
    digest = sha256()
    digest.update(generated.encode("utf-8"))
    digest.update(runtime_signature.encode("ascii"))
    digest.update(" ".join(LINK_FLAGS).encode("ascii"))
    signature = digest.hexdigest()[:20]
    directory = ROOT / ".build" / "programs" / signature
    directory.mkdir(parents=True, exist_ok=True)
    c_source = directory / "program.c"
    destination = output.resolve() if output else directory / "program.exe"
    if destination.suffix.lower() != ".exe":
        raise HengError("原生程序输出必须使用 .exe 扩展名")
    if destination.exists() and not destination.is_relative_to((ROOT / ".build").resolve()):
        raise HengError("拒绝覆盖构建目录之外的已有文件")
    if output is None and destination.exists() and c_source.exists() and not force:
        return destination
    c_source.write_text(generated, encoding="utf-8", newline="\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".{os.getpid()}.tmp.exe")
    objects = runtime_objects(snapshot, optimized=optimized)
    command = [str(ZIG), "cc", "-std=c17", "-O2" if optimized else "-O0", *LINK_FLAGS, "-fno-fast-math", "-Werror=implicit-function-declaration", "-target", config["build"]["target"], "-I", str(snapshot), str(c_source), *map(str, objects), "-o", str(temporary)]
    result = subprocess.run(command, env=environment(), capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise HengError("原生编译失败:\n" + result.stderr.strip())
    temporary.replace(destination)
    return destination


def execute(source: Path, *, optimized: bool = True, timeout: float = 120) -> subprocess.CompletedProcess:
    executable = build(source, optimized=optimized)
    return subprocess.run([str(executable)], cwd=ROOT, env=environment(), capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="heng", description="衡言 HengLang 原生编译与衡算 HENG 开发工具")
    parser.add_argument("--version", action="version", version="HengLang 1.0.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [("check", "检查语法、类型和模块"), ("build", "生成原生 CPU 可执行程序"), ("run", "编译并执行 HengLang 程序")]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("source", type=Path)
        if name != "check":
            command.add_argument("--no-opt", action="store_true", help="关闭 HengLang 优化与原生优化")
        if name == "build":
            command.add_argument("--output", type=Path)
            command.add_argument("--force", action="store_true", help="重新编译已缓存的构建")
        if name == "run":
            command.add_argument("--timeout", type=float, default=120)
    command = commands.add_parser("fmt", help="格式化 HengLang 源码，默认输出到终端")
    command.add_argument("source", type=Path)
    action = command.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true")
    action.add_argument("--write", action="store_true")
    command = commands.add_parser("test", help="执行全部功能测试，不训练模型")
    command.add_argument("--pattern", default="test_*.py")
    command = commands.add_parser("bench", help="执行 CPU 算子基准，不训练模型")
    command.add_argument("--source", type=Path, default=ROOT / "benchmarks" / "kernels.heng")
    command.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(arguments)
    try:
        configuration()
        if args.command == "check":
            program = load_program(args.source)
            print(f"PASS: 语法、类型和模块检查通过，函数数 {len(program.functions)}")
            return 0
        if args.command == "build":
            result = build(args.source, optimized=not args.no_opt, output=args.output, force=args.force)
            print(result)
            return 0
        if args.command in {"run", "bench"}:
            if args.timeout <= 0:
                raise HengError("执行超时必须大于零")
            result = execute(args.source, optimized=not getattr(args, "no_opt", False), timeout=args.timeout)
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            return result.returncode
        if args.command == "fmt":
            if args.source.suffix != ".heng":
                raise HengError("HengLang 源文件必须使用专属 .heng 扩展名")
            content = args.source.read_bytes()
            original = content.decode("utf-8")
            formatted = format_source(original, str(args.source))
            if b"\r\n" in content and b"\n" not in content.replace(b"\r\n", b""):
                formatted = formatted.replace("\n", "\r\n")
            if not original.endswith(("\r", "\n")):
                formatted = formatted.rstrip("\r\n")
            if args.check:
                if original != formatted:
                    print("FAIL: 源码格式需要调整", file=sys.stderr)
                    return 1
                print("PASS: 源码格式检查通过")
            elif args.write:
                args.source.write_bytes(formatted.encode("utf-8"))
            else:
                sys.stdout.write(formatted)
            return 0
        if args.command == "test":
            suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=args.pattern)
            return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    except (HengError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"HengLang 错误: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
