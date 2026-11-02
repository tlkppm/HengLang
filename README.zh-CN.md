## 衡言 HengLang 与衡算 HENG

简体中文 | [English](README.md)

衡言 HengLang 是使用专属 `.heng` 源码扩展名的静态检查编程语言。衡算 HENG 是以该语言实现接口与高层逻辑的 CPU 张量及自动微分框架。当前版本为 1.0.0，交付目标是 Windows x86-64 单机环境。

编译器前端使用 Python 标准库，将程序转换为 C17，再通过项目内隔离安装、固定版本的 Zig C 编译器生成原生程序。生成的可执行文件无需 Python 即可运行。项目编译与运行不依赖 PyTorch 或 NumPy；可选对比基准会使用本机已安装的比较库。

版本号用于标识当前实现，不代表已经达到生产级成熟度、不保证未来版本接口稳定，也不代表与成熟框架功能对等。

### 已实现能力

- 语言：静态类型检查、函数、递归、泛型函数、结构体、枚举、列表、`Result<T>`、模块导入、控制流与源码位置诊断。
- 运行时：`f64`、`f32`、`i64`、`bool` 张量，共享存储视图、广播、索引、归约、矩阵乘法、稳定 softmax、卷积展开和折叠、COO/CSR 稀疏操作。
- 自动微分：显式计算图记录、反向微分、向量与雅可比矩阵乘积、雅可比矩阵与向量乘积、梯度累积、计算图清理与原地修改检查。
- 模块：顺序组合、参数共享、Linear、Embedding、激活函数、Conv2D、平均和最大池化、LayerNorm、BatchNorm、Dropout、基础 RNN 和多头注意力。
- 优化工具：SGD、动量、Nesterov、Adam、AdamW、RMSprop、梯度裁剪，以及阶梯、指数和线性学习率调度。
- 数据及状态：内存张量分批、确定性打乱、模型和优化器及随机数状态保存恢复、异常状态文件校验。
- 开发工具：语法与类型检查、格式化、原生构建、执行、自动化测试和可复现 CPU 基准。

### 环境与安装

项目目标平台为 Windows x86-64，开发与测试使用 Python 3.13，原生工具链固定为 Zig 0.17.0。生成后的程序不需要 Python 解释器。

在仓库根目录的 PowerShell 中执行：

```powershell
python -B tools/toolchain.py install
python -B tools/toolchain.py probe
python -B -m henglang --version
```

安装器读取 `heng.toml`，只下载固定版本的 Zig 官方发行包，在解压和执行前校验固定 SHA-256。下载强制使用配置的 HTTP 代理，初始地址为 `http://127.0.0.1:10808`。请先启动该代理，或在 `heng.toml` 中明确配置其他可用 HTTP 代理；安装器不会回退为直连。这里的本地代理地址是开发配置，不是项目提供的托管服务。

编译器和缓存只保存在 `.tools/`、`.build/`，这两个目录不包含在源码仓库中。首次克隆后，需要先安装工具链，才能执行原生构建和原生测试。无需全局安装编译器，也无需修改系统 `PATH`。

### 第一个原生程序

项目示例计算平方和及其梯度，不训练模型，不下载数据。

```powershell
python -B -m henglang check examples/quickstart.heng
python -B -m henglang run examples/quickstart.heng
python -B -m henglang build examples/quickstart.heng --output .build/quickstart.exe
& '.build/quickstart.exe'
```

输入 `[1, 2, 3]` 时，示例检查结果为 `14`、梯度为 `[2, 4, 6]`。`tests/fixtures/` 中的可执行测试程序覆盖更广泛的框架接口。

### 开发与验证

```powershell
python -B -m henglang test
python -B -m henglang test --pattern test_frontend.py
python -B -m henglang fmt examples/quickstart.heng --check
python -B -m henglang bench
```

`test` 使用 Python 标准库 `unittest` 发现测试。完整验证包括原生编译，可能耗时数分钟。测试包含少量合成输入的前向、反向和优化器单步检查，用于验证正确性，不属于模型训练实验。

`build --no-opt` 同时关闭语言优化与原生编译优化。内部基准使用经过优化的原生构建，再切换运行时内核，因此其内部基线不是关闭编译优化的程序。

### 性能与框架比较

HENG 已实现分块矩阵乘法、有界张量存储复用，以及仿射变换加 ReLU 的融合操作。这些机制可以改善特定负载，但不能仅凭机制存在就认定优于其他框架。

[基准方法与实测结果](BENCHMARKS.md) 记录输入、精度、线程设置、计时范围与限制，不公开测试设备信息。仓库还提供可选比较程序：

```powershell
python -B benchmarks/compare_frameworks.py --help
python -B benchmarks/compare_frameworks.py
```

比较程序使用本机已经安装且受支持的比较库，不自动安装依赖。报告计时前会验证数值样本。对比结果属于算子级测量，不构成整个框架的综合排名。本机已记录的实测中，NumPy 配合 OpenBLAS 在两种矩阵规模上均快于 HENG；HENG 在这些测试中尚未超越该基线。

HENG 全面超越 PyTorch、TensorFlow、JAX 或所有现有框架的说法尚未得到证明。本次交付没有训练模型，也没有建立达到相同模型质量时训练成本更低的证据。

### 当前边界

- 已验证交付目标为 Windows x86-64 单机 CPU；GPU、分布式执行及其他平台不在验证范围内。
- 框架使用自己的接口、张量语义和状态格式，不提供 PyTorch 接口或检查点兼容性。
- 可微矩阵乘法要求张量至少二维；可微切片支持有效正步长。
- Conv2D 暂不支持 groups、dilation。循环模块是基础 RNN，不是 LSTM 或 GRU。
- 浮点参数及叶节点梯度保持声明的数据类型，但混合运算和默认 `f64` 模块参数或常量可能使中间结果提升为 `f64`。
- 检查点覆盖已实现的模型、优化器和随机数状态，不保存调度器对象和数据加载器游标。
- 本地编译器不能作为运行不可信程序的安全沙箱。状态文件检查不构成正式安全审计，也不是身份认证机制。

### 仓库结构

- `henglang/`：语言前端、中间表示、优化、C 代码生成与命令行工具。
- `runtime/`：C17 张量运行时、内存管理、原生内核和二进制状态读写。
- `heng/`：使用衡言实现的框架模块。
- `tools/`：项目隔离工具链管理。
- `tests/`：语言、独立数值参考、框架、资源管理和原生运行时测试。
- `examples/`：可执行入门程序。
- `benchmarks/`：内部内核基准及可选框架对比。

### 许可、贡献与安全

版权署名为 Copyright 2026 tlkppm and HengLang contributors。项目采用 [Apache-2.0](LICENSE)，适用范围见 [NOTICE](NOTICE)。英文许可证原文为准；双语法律说明不替代许可证，也不增加额外使用限制。

请参阅[法律与使用说明](LEGAL.md)、[第三方声明](THIRD_PARTY_NOTICES.md)、[贡献指南](CONTRIBUTING.md)和[安全说明](SECURITY.md)。这些文件区分项目源码许可与第三方工具链、链接运行库的许可义务，属于项目信息，不是法律意见或合规认证。

项目仓库：[tlkppm/HengLang](https://github.com/tlkppm/HengLang)。
