from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import urllib.request
from urllib.parse import urlsplit
import zipfile


class RequiredProxyHandler(urllib.request.ProxyHandler):
    def proxy_open(self, request, proxy, protocol):
        parsed = urlsplit(proxy)
        if parsed.scheme != "http" or not parsed.hostname or not parsed.port or parsed.username or parsed.password:
            raise RuntimeError("下载代理配置无效")
        request.set_proxy(parsed.netloc, parsed.scheme)
        return None


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.17.0"
PLATFORM = "x86_64-windows"
TOOLS = ROOT / ".tools"
ZIG = TOOLS / f"zig-{PLATFORM}-{VERSION}" / "zig.exe"
INDEX = "https://ziglang.org/download/index.json"
with (ROOT / "heng.toml").open("rb") as configuration_file:
    PROXY = tomllib.load(configuration_file)["toolchain"]["proxy"]
OPENER = urllib.request.build_opener(RequiredProxyHandler({"http": PROXY, "https": PROXY}))
EXPECTED_SHA256 = "b5663f69581dcf391293fbf16c06cb80d81d806545ce618b4d0bab7f0eb8c428"


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env["ZIG_GLOBAL_CACHE_DIR"] = str(ROOT / ".build" / "zig-global-cache")
    env["ZIG_LOCAL_CACHE_DIR"] = str(ROOT / ".build" / "zig-local-cache")
    return env


def install(*, repair: bool = False) -> None:
    if ZIG.is_file():
        verify_version()
        return
    with OPENER.open(INDEX, timeout=60) as response:
        index = json.load(response)
    release = index.get(VERSION, {}).get(PLATFORM)
    if not release or release["shasum"] != EXPECTED_SHA256:
        raise RuntimeError("官方工具链索引与已批准的版本或校验值不一致，已停止安装")
    url = release["tarball"]
    if not url.startswith(f"https://ziglang.org/download/{VERSION}/"):
        raise RuntimeError("工具链下载地址不在批准的官方版本目录")
    TOOLS.mkdir(exist_ok=True)
    archive = TOOLS / f"zig-{PLATFORM}-{VERSION}.zip.download"
    digest = hashlib.sha256()
    received = 0
    if not archive.exists():
        with OPENER.open(url, timeout=120) as response, archive.open("xb") as output:
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > int(release["size"]):
                    raise RuntimeError("工具链下载超过官方声明大小")
                output.write(block)
    with archive.open("rb") as downloaded:
        while block := downloaded.read(1024 * 1024):
            received = downloaded.tell()
            digest.update(block)
    if received != int(release["size"]) or digest.hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("工具链大小或 SHA-256 校验失败，未解压或执行")
    with zipfile.ZipFile(archive) as package:
        total = 0
        for member in package.infolist():
            destination = (TOOLS / member.filename).resolve()
            if not destination.is_relative_to(TOOLS.resolve()) or member.filename.split("/")[0] != ZIG.parent.name:
                raise RuntimeError("工具链压缩包包含越界路径")
            total += member.file_size
            if total > 2 * 1024**3:
                raise RuntimeError("工具链解压体积超过安全上限")
            if destination.exists():
                if member.is_dir() and destination.is_dir():
                    continue
                if destination.is_file():
                    with package.open(member) as original, destination.open("rb") as existing:
                        if hashlib.file_digest(original, "sha256").digest() == hashlib.file_digest(existing, "sha256").digest():
                            continue
                if not repair or not destination.is_file():
                    raise RuntimeError(f"工具链已有文件与压缩包不一致，拒绝覆盖: {destination.relative_to(TOOLS)}")
                print(f"修复本次中断安装的文件: {destination.relative_to(TOOLS)}", flush=True)
                destination.unlink()
            package.extract(member, TOOLS)
    archive.unlink()
    verify_version()
    print("PASS: 项目隔离工具链已安装，官方 SHA-256 校验通过")


def verify_version() -> None:
    result = subprocess.run([str(ZIG), "version"], check=True, capture_output=True, text=True, env=environment())
    if result.stdout.strip() != VERSION:
        raise RuntimeError("本地工具链版本与固定版本不一致")
    print(f"Zig {VERSION}")


def cc(arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    if not ZIG.is_file():
        raise RuntimeError("项目工具链未安装，请运行 python tools/toolchain.py install")
    return subprocess.run([str(ZIG), "cc", *arguments], env=environment(), check=check)


def probe() -> None:
    verify_version()
    build = ROOT / ".build" / "toolchain-probe"
    build.mkdir(parents=True, exist_ok=True)
    source = build / "probe.c"
    program = build / "probe.exe"
    source.write_text('#include <stdio.h>\nint main(void) { puts("HENG_NATIVE_OK"); return 0; }\n', encoding="utf-8")
    cc(["-std=c17", "-O2", "-target", "x86_64-windows-gnu", str(source), "-o", str(program)])
    result = subprocess.run([str(program)], check=True, capture_output=True, text=True, env=environment())
    if result.stdout.strip() != "HENG_NATIVE_OK":
        raise RuntimeError("原生可执行程序输出不符合预期")
    print("PASS: C17 原生编译、独立链接与执行通过")


def main() -> int:
    parser = argparse.ArgumentParser(description="HENG 项目隔离工具链管理")
    parser.add_argument("command", choices=["install", "probe", "version", "cc"])
    parser.add_argument("--repair", action="store_true")
    args, rest = parser.parse_known_args()
    try:
        if args.command == "install":
            install(repair=args.repair)
        elif args.command == "probe":
            probe()
        elif args.command == "version":
            verify_version()
        else:
            return cc(rest, check=False).returncode
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"工具链错误: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
