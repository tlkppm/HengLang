## CPU benchmarks / CPU 性能基准

### English

#### Conclusion and scope

HENG's optimized runtime is faster than its previous implementation on the measured workloads, but it does not exceed the tested NumPy/OpenBLAS baseline. PyTorch, TensorFlow, and JAX were not benchmarked. No training, model-quality, equal-quality training-cost, energy, or GPU comparison was performed.

The measurements below were taken on 2026-11-03. HENG used the project's pinned toolchain with a C17 `-O2` build and `-fno-fast-math`. Test-device details and local environment identifiers are not published. No model or dataset was downloaded, and no new comparison package was installed.

These are local observations, not portable performance guarantees. Compilation and performance tests were run sequentially rather than concurrently with each other.

#### Changes measured

- Floating-point matrix multiplication specializes storage reads for all four `f64`/`f32` input combinations, avoiding dtype dispatch inside the multiplication loop.
- Contiguous floating-point `add`, `sub`, `mul`, and `div` use direct storage access when shapes allow matching linear traversal or single-element broadcasting.
- Contiguous `scale`, `relu`, and `fused_affine_relu` avoid per-element dimension conversion and operation-name dispatch.
- Matrix multiplication retains compensated summation and its non-finite behavior. Float32 rounding points, integer overflow checks, strided fallbacks, and thread partitioning remain part of the implementation contract.
- A `get_threads()` builtin exposes HENG's configured compute-worker count for benchmark verification. It is not an operating-system thread activity monitor.

#### Internal before-and-after measurements

Command: `python -B -m henglang bench`.

The benchmark uses `f64`, one configured compute worker, three warmups, and the median of twenty timed calls. Matrix inputs use the same deterministic seed in the plain and optimized paths. The elementwise workload uses 100000 elements and computes `relu(input * 2 + 1)` through separated operations or the explicit fused builtin. The native executable is built with optimization enabled in both kernel modes.

One before-and-after pair from this optimization session recorded:

- Optimized 128 by 128 matrix multiplication: 12.499 ms before, 7.343 ms after; about 1.70 times faster.
- Optimized 256 by 256 matrix multiplication: 101.356 ms before, 63.043 ms after; about 1.61 times faster.
- Separated elementwise path: 16.445 ms before, 0.4475 ms after; about 36.75 times faster.
- Explicit fused elementwise path: 7.956 ms before, 0.21475 ms after; about 37.05 times faster.

A subsequent run of the final runtime recorded 7.765 ms and 63.622 ms for the optimized matrix sizes, and 0.4189 ms and 0.2033 ms for the separated and fused elementwise paths. Its plain matrix kernels took 17.505 ms and 173.823 ms. This illustrates both the improvement and between-run variation; these independent runs do not establish statistical confidence intervals or attribute all variation to code changes.

The final internal run also recorded:

- Elementwise logical allocation counts: 24 for separated operations and 5 for the fused builtin.
- Across 100 tensor-creation cycles: 1000 tracked logical allocations, 901 system allocations, and 99 tensor-storage reuses. These counts include runtime metadata and are not counts of tensors alone.
- Retained buffer storage: 800000 bytes, released to zero by explicit trimming.
- Peak runtime-tracked memory: 11645572 bytes. This is not whole-process resident memory or a comparison with another framework's memory use.

The internal matrix timing program checks output element counts; independent numerical-reference tests supply broader correctness coverage. The elementwise timing program additionally checks values and exact agreement between the separated and fused paths. Allocation reductions are not a claim that computation or training complexity decreased by the same factor.

#### NumPy comparison methodology

Command: `python -B benchmarks/compare_frameworks.py`.

The comparison was run twice in separate driver processes. Each run used HENG and the existing NumPy 2.5.2 installation with OpenBLAS 0.3.34.0.0. For each matrix size, HENG was measured first, then NumPy. Both requested and reported a compute-thread configuration of one.

For a square dimension `n`, both implementations construct the same contiguous `f64` inputs from flattened index `i`:

- Left input: `reshape(i * (1 / (n*n)) - 0.5, [n, n])`.
- Right input: `reshape(i * (-0.75 / (n*n)) + 0.25, [n, n])`.

Every measured call creates a new logical output. There is no `out` parameter. Input creation, imports, compilation, process startup, explicit garbage collection, validation, and output destruction are outside the timing interval. Default allocators and HENG's storage pool remain enabled, so logical output allocation does not imply a fresh system allocation on every call.

HENG uses its native high-resolution `clock()` based on Windows `QueryPerformanceCounter`; NumPy uses Python `time.perf_counter_ns()`, including Python call overhead. Three warmups precede twenty timed calls, summarized by their median.

The driver verifies five output positions per matrix size against independent Python `math.fsum` dot products, and verifies sampled input values. The tolerance is relative `1e-10` and absolute `1e-11`. All recorded samples in these two runs had zero reported absolute error. This sampling is not a full-output proof. HENG keeps compensated summation; OpenBLAS can use a different reduction order, so identical dtype and tolerances do not imply identical internal numerical algorithms.

HENG prints its actual `get_threads()` result. Source inspection confirms that its matrix kernel creates worker threads only when the configured worker count exceeds one; otherwise it executes in the calling thread without external BLAS. NumPy's already-loaded OpenBLAS DLL reports its thread configuration through `scipy_openblas_get_num_threads64_`. Neither mechanism samples all operating-system thread activity. The report's `fair_single_thread: true` means these compute-thread configurations matched; it is not CPU scheduling isolation or a guarantee of complete experimental fairness.

#### Measured NumPy comparison

First independent driver run:

- 128 by 128: HENG 7.2713 ms; NumPy/OpenBLAS 0.19365 ms.
- 256 by 256: HENG 63.0279 ms; NumPy/OpenBLAS 0.9254 ms.

Second independent driver run:

- 128 by 128: HENG 7.8044 ms; NumPy/OpenBLAS 0.1841 ms.
- 256 by 256: HENG 63.42325 ms; NumPy/OpenBLAS 0.92985 ms.

NumPy/OpenBLAS was about 37.5 to 42.4 times faster at 128 and about 68.1 to 68.2 times faster at 256 in these measurements. HENG therefore has substantial matrix-kernel performance work remaining. NumPy is a numerical-library baseline, not a substitute for an end-to-end comparison against a training framework.

#### Limits and reproduction

Test-device identities are withheld for privacy, so the results cannot be reproduced against a fully specified hardware configuration. CPU frequency, temperature, background load, process scheduling, allocator behavior, and language-call overhead were not normalized. No statistical confidence interval was calculated. The deterministic affine-sequence inputs are not representative of all matrix distributions, sizes, layouts, or application workloads. Float32, multi-thread scaling, convolution, attention, full-model training, and inference throughput were not compared with external libraries.

The driver prints structured JSON to standard output and removes its temporary HengLang source directory. Normal compiler caches remain in `.build/`. Its outer timeout cannot guarantee termination of every compiler child process; after a timeout, verify that the previous build has stopped before repeating it. It does not install dependencies or train anything. Use `--help` for configuration. Optional PyTorch support is available only when PyTorch is already installed; it was NOT RUN for this delivery.

### 简体中文

#### 结论与范围

HENG 优化后的运行时在本次测量负载上快于改动前实现，但仍未超越实测的 NumPy/OpenBLAS 基线。PyTorch、TensorFlow、JAX 未执行对比。本次没有训练模型，没有比较模型质量、同等质量下的训练成本、能耗或 GPU 性能。

以下测量于 2026-11-03 完成。HENG 使用项目固定工具链，以 C17 `-O2` 和 `-fno-fast-math` 构建。不公开测试设备详情及本地环境标识。未下载模型或数据集，未安装新的比较库。

这些是本机观察，不是跨机器性能保证。编译与各项性能测试串行执行，未互相并发争用计算资源。

#### 本次改进

- 浮点矩阵乘法针对四种 `f64`、`f32` 输入组合直接读取存储，消除乘法内层循环的数据类型分派。
- 连续浮点 `add`、`sub`、`mul`、`div` 在可线性遍历或单元素广播时直接访问存储。
- 连续 `scale`、`relu`、`fused_affine_relu` 避免逐元素维度换算与操作名称判断。
- 保留矩阵补偿求和及非有限数行为、Float32 舍入位置、整数溢出检查、非连续后备路径和原有线程划分。
- 新增 `get_threads()` 内置函数，供基准读取 HENG 实际配置的计算线程数；它不是操作系统线程活动监测器。

#### 项目内部前后对比

命令：`python -B -m henglang bench`。

配置为 `f64`、1 个计算线程、3 次预热、20 次计时取中位数。矩阵朴素与优化路径使用相同确定性随机种子。逐元素负载为 100000 个元素，以分离操作或显式融合内置函数计算 `relu(input * 2 + 1)`。两种内核模式均使用开启编译优化的原生程序。

本轮优化中的一组前后测量为：

- 128 乘 128 优化矩阵乘法：12.499 ms 降至 7.343 ms，约快 1.70 倍。
- 256 乘 256 优化矩阵乘法：101.356 ms 降至 63.043 ms，约快 1.61 倍。
- 分离逐元素路径：16.445 ms 降至 0.4475 ms，约快 36.75 倍。
- 显式融合逐元素路径：7.956 ms 降至 0.21475 ms，约快 37.05 倍。

随后对最终运行时复测，优化矩阵分别为 7.765 ms、63.622 ms，分离与融合逐元素路径分别为 0.4189 ms、0.2033 ms；朴素矩阵内核分别为 17.505 ms、173.823 ms。这些独立运行既显示改进，也显示批次间波动；它们不构成统计置信区间，也不能把全部波动归因于代码变更。

最终内部复测同时记录：

- 逐元素逻辑分配次数：分离路径 24 次，融合路径 5 次。
- 100 次张量创建循环：逻辑分配 1000 次，系统分配 901 次，张量存储复用 99 次。这些计数包含运行时元数据，不等于张量个数。
- 缓冲池保留 800000 字节，显式清理后为零。
- 运行时跟踪峰值内存为 11645572 字节，不等于整个进程的驻留内存，也不是与其他框架的内存比较。

内部矩阵计时程序检查输出元素数量，更广泛的正确性由独立数值参考测试覆盖。逐元素计时程序还检查数值及分离、融合路径的精确一致性。分配次数下降不代表计算复杂度或训练成本按相同比例下降。

#### NumPy 对比方法

命令：`python -B benchmarks/compare_frameworks.py`。

比较程序在两个独立驱动进程中各运行一次。每次比较 HENG 与本机已有的 NumPy 2.5.2、OpenBLAS 0.3.34.0.0。每个矩阵规模先测 HENG，再测 NumPy。双方均请求并实际返回计算线程配置为 1。

方阵边长为 `n` 时，双方使用相同连续 `f64` 输入，展平索引记为 `i`：

- 左矩阵：`reshape(i * (1 / (n*n)) - 0.5, [n, n])`。
- 右矩阵：`reshape(i * (-0.75 / (n*n)) + 0.25, [n, n])`。

每次计时均产生新的逻辑输出，不使用 `out` 参数。输入构造、导入、编译、进程启动、显式垃圾回收、校验及输出销毁不计入耗时。保留双方默认分配器及 HENG 存储池，因此逻辑分配不表示每次都向操作系统申请新存储。

HENG 使用基于 Windows `QueryPerformanceCounter` 的原生高精度 `clock()`；NumPy 使用 Python `time.perf_counter_ns()`，包含 Python 调用开销。预热 3 次后计时 20 次，报告中位数。

每个规模抽取 5 个输出位置，以独立 Python `math.fsum` 点积作为参考，并校验输入样本。相对容差为 `1e-10`，绝对容差为 `1e-11`。两次运行的所有记录样本报告绝对误差均为零；抽样不等于完整输出证明。HENG 保留补偿求和，OpenBLAS 可以采用不同归约顺序，因此相同精度和容差不代表内部数值算法相同。

HENG 实际打印 `get_threads()` 返回值。源码检查确认，只有配置线程数大于 1 时矩阵内核才创建工作线程，否则直接在调用线程计算，不调用外部 BLAS。NumPy 通过已加载 OpenBLAS DLL 的 `scipy_openblas_get_num_threads64_` 查询线程配置。两种机制都不采样全部操作系统线程活动。报告中的 `fair_single_thread: true` 仅表示计算线程配置匹配，不代表 CPU 调度隔离或完整实验公平性保证。

#### NumPy 实测结果

第一次独立驱动运行：

- 128 乘 128：HENG 为 7.2713 ms，NumPy/OpenBLAS 为 0.19365 ms。
- 256 乘 256：HENG 为 63.0279 ms，NumPy/OpenBLAS 为 0.9254 ms。

第二次独立驱动运行：

- 128 乘 128：HENG 为 7.8044 ms，NumPy/OpenBLAS 为 0.1841 ms。
- 256 乘 256：HENG 为 63.42325 ms，NumPy/OpenBLAS 为 0.92985 ms。

NumPy/OpenBLAS 在 128 规模上约快 37.5 至 42.4 倍，在 256 规模上约快 68.1 至 68.2 倍。HENG 的矩阵内核仍有明显性能差距。NumPy 属于数值库基线，不能替代与训练框架的端到端比较。

#### 限制与复现

为保护隐私，不公开测试设备信息，因此外部无法按完整硬件配置复现这些结果。没有统一 CPU 频率、温度、后台负载、进程调度、分配器行为和语言调用开销，没有计算统计置信区间。确定性的仿射序列输入不代表所有矩阵分布、规模、布局或应用负载。未与外部库比较 Float32、多线程扩展、卷积、注意力、完整模型训练或推理吞吐。

比较程序将结构化 JSON 输出到标准输出，并清理本次临时衡言源码目录；正常编译缓存保留在 `.build/`。外层超时不能保证终止全部编译器子进程，超时后应确认上一次构建停止，再重新执行。程序不自动安装依赖，不执行训练。配置选项见 `--help`。可选 PyTorch 路径要求本机已安装 PyTorch，本次交付对应验证为 NOT RUN。
