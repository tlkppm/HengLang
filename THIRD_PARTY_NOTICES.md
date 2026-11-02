## Third-party notices / 第三方声明

### English

#### Scope of this source distribution

The Git source distribution excludes `.tools/`, `.build/`, virtual environments, and installed Python packages. It does not redistribute the downloaded Zig toolchain or a NumPy/PyTorch installation. Apache-2.0 covers the original project material described in [NOTICE](NOTICE), not third-party works.

This list describes the development and optional benchmark components identified for this delivery. It is not a complete inventory of every component a compiler may link into every generated executable.

#### Python

The frontend and development utilities use Python and its standard library. Python is a separate prerequisite, not vendored in the repository. Python distributions carry Python Software Foundation license terms and can include additional third-party notices. Consult the license files of the Python distribution you use. Generated HENG executables do not require the Python interpreter.

#### Zig and bundled libraries

The toolchain manager pins Zig 0.17.0 for Windows x86-64. The downloaded distribution's top-level `LICENSE` identifies the MIT License (Expat) and Copyright (c) Zig contributors. The distribution also contains separately licensed components, including LLVM-related libraries, C/C++ libraries, and platform runtime material. Its top-level MIT license is not a substitute for their individual terms.

Examples of bundled license locations in the installed distribution include `lib/libcxx/LICENSE.TXT`, `lib/libcxxabi/LICENSE.TXT`, `lib/libunwind/LICENSE.TXT`, and `lib/libc/mingw/COPYING`. These are examples, not an exhaustive statement of what any particular program links. Keep and inspect the actual distribution's notices if you redistribute the toolchain. Review linked components and their obligations before distributing generated executables.

The installer verifies the distribution against its pinned SHA-256; this integrity check is not a license-compliance or security certification. The official license source used for this repository's Apache-2.0 text is [the Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0.txt), accessed on 2026-11-03.

#### Optional comparison libraries

HENG's compiler and runtime do not require NumPy or PyTorch. Optional comparison tooling may use an existing local NumPy installation and `threadpoolctl` to inspect numerical-library thread pools. Such packages retain their own license terms. A NumPy distribution may bundle BLAS or LAPACK libraries with separate notices. The comparison output reports the components it actually detects; review your installed package metadata and license files rather than assuming a single license covers every binary component.

Installing additional comparison libraries, redistributing their wheels, or packaging them with HENG requires a separate review of those actual distributions. No third-party benchmark dependency is bundled by this source publication.

### 简体中文

#### 本次源码分发范围

Git 源码发布排除 `.tools/`、`.build/`、虚拟环境和已安装的 Python 包，不再分发下载的 Zig 工具链，也不再分发 NumPy/PyTorch 安装内容。Apache-2.0 适用于 [NOTICE](NOTICE) 中说明的项目原创内容，不替代第三方作品的许可。

这里列出本次交付已识别的开发组件与可选基准组件，不是编译器对每个生成程序可能链接的所有组件的完整清单。

#### Python

语言前端与开发工具使用 Python 及其标准库。Python 是单独的前置环境，不内置于本仓库。Python 发行版适用 Python Software Foundation 的许可条款，并可能包含其他第三方声明。请查阅实际使用的 Python 发行版许可证。生成的 HENG 可执行文件不需要 Python 解释器。

#### Zig 与随附库

工具链管理器固定使用 Windows x86-64 的 Zig 0.17.0。下载发行版的顶层 `LICENSE` 标明 MIT License (Expat)，版权署名为 Copyright (c) Zig contributors。发行版还包含独立许可的 LLVM 相关库、C/C++ 库和平台运行时内容；不能用顶层 MIT 许可证替代这些组件各自的条款。

安装目录中的许可位置示例包括 `lib/libcxx/LICENSE.TXT`、`lib/libcxxabi/LICENSE.TXT`、`lib/libunwind/LICENSE.TXT` 和 `lib/libc/mingw/COPYING`。这些仅为示例，不是任何具体程序实际链接组件的完整结论。若再分发工具链，应保留并核对实际发行版的声明；若分发生成的可执行文件，应检查实际链接组件及对应义务。

安装器按固定 SHA-256 校验发行包，这属于完整性检查，不是许可合规或安全认证。本仓库 Apache-2.0 正文取自 [Apache Software Foundation 官方文本](https://www.apache.org/licenses/LICENSE-2.0.txt)，访问日期为 2026-11-03。

#### 可选比较库

HENG 编译器和运行时不依赖 NumPy 或 PyTorch。可选比较工具可能使用本机已有的 NumPy，以及用于检查数值库线程池的 `threadpoolctl`。这些包各自保留其许可条款。NumPy 发行包可能随附 BLAS 或 LAPACK 库，并包含独立声明。比较程序报告实际检测到的组件；请检查本机安装包的元数据与许可文件，不应假定所有二进制组件共用一个许可证。

额外安装比较库、再分发其 wheel 包或将其与 HENG 打包时，需要针对实际发行包另行审查。本次源码发布不捆绑第三方基准依赖。
