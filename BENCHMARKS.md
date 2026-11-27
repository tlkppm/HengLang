## CPU benchmarks / CPU 性能基准

### English

#### Conclusion and scope

The earlier strict-mode runs established a lead for exact binary-fraction inputs. The current measurements cover general `f32` and `f64` matrix inputs, transposed physical layouts, and the full training loop of a character model. The optional fast mode trades compensated accumulation for fused multiply-add on eligible matrices. NumPy remained faster in the matched training experiment.

The original measurements date to 2026-11-03 and the dense-matrix follow-up to 2026-11-26. HENG used the project's pinned toolchain with a C17 `-O2` build and `-fno-fast-math`. Test-device details and local environment identifiers are withheld. The benchmarks used synthetic inputs and an existing NumPy installation.

These are local observations, not portable performance guarantees. Compilation and performance tests were run sequentially rather than concurrently with each other.

#### Changes measured

- Floating-point matrix multiplication specializes storage reads for all four `f64`/`f32` input combinations, avoiding dtype dispatch inside the multiplication loop.
- Contiguous floating-point `add`, `sub`, `mul`, and `div` use direct storage access when shapes allow matching linear traversal or single-element broadcasting.
- Contiguous `scale`, `relu`, and `fused_affine_relu` avoid per-element dimension conversion and operation-name dispatch.
- Matrix multiplication retains compensated summation and its non-finite behavior. Float32 rounding points, integer overflow checks, strided fallbacks, and thread partitioning remain part of the implementation contract.
- A `get_threads()` builtin exposes HENG's configured compute-worker count for benchmark verification. It is not an operating-system thread activity monitor.

#### Original internal before-and-after measurements

Command: `python -B -m henglang bench`.

The benchmark uses `f64`, one configured compute worker, three warmups, and the median of twenty timed calls. Matrix inputs use the same deterministic seed in the plain and optimized paths. The elementwise workload uses 100000 elements and computes `relu(input * 2 + 1)` through separated operations or the explicit fused builtin. The native executable is built with optimization enabled in both kernel modes.

One before-and-after pair from this optimization session recorded:

- Optimized 128 by 128 matrix multiplication: 12.499 ms before, 7.343 ms after; about 1.70 times faster.
- Optimized 256 by 256 matrix multiplication: 101.356 ms before, 63.043 ms after; about 1.61 times faster.
- Separated elementwise path: 16.445 ms before, 0.4475 ms after; about 36.75 times faster.
- Explicit fused elementwise path: 7.956 ms before, 0.21475 ms after; about 37.05 times faster.

A subsequent run during the original optimization recorded 7.765 ms and 63.622 ms for the optimized matrix sizes, and 0.4189 ms and 0.2033 ms for the separated and fused elementwise paths. Its plain matrix kernels took 17.505 ms and 173.823 ms. This illustrates both the improvement and between-run variation; these independent runs do not establish statistical confidence intervals or attribute all variation to code changes.

The last internal run of the original optimization also recorded:

- Elementwise logical allocation counts: 24 for separated operations and 5 for the fused builtin.
- Across 100 tensor-creation cycles: 1000 tracked logical allocations, 901 system allocations, and 99 tensor-storage reuses. These counts include runtime metadata and are not counts of tensors alone.
- Retained buffer storage: 800000 bytes, released to zero by explicit trimming.
- Peak runtime-tracked memory: 11645572 bytes. This is not whole-process resident memory or a comparison with another framework's memory use.

The internal matrix timing program checks output element counts; independent numerical-reference tests supply broader correctness coverage. The elementwise timing program additionally checks values and exact agreement between the separated and fused paths. Allocation reductions are not a claim that computation or training complexity decreased by the same factor.

#### Original NumPy comparison methodology

Command: `python -B benchmarks/compare_frameworks.py`.

The comparison was run twice in separate driver processes. Each run used HENG and the existing NumPy 2.5.2 installation with OpenBLAS 0.3.34.0.0. For each matrix size, HENG was measured first, then NumPy. Both requested and reported a compute-thread configuration of one.

For a square dimension `n`, both implementations construct the same contiguous `f64` inputs from flattened index `i`:

- Left input: `reshape(i * (1 / (n*n)) - 0.5, [n, n])`.
- Right input: `reshape(i * (-0.75 / (n*n)) + 0.25, [n, n])`.

Every measured call creates a new logical output. There is no `out` parameter. Input creation, imports, compilation, process startup, explicit garbage collection, validation, and output destruction are outside the timing interval. Default allocators and HENG's storage pool remain enabled, so logical output allocation does not imply a fresh system allocation on every call.

HENG uses its native high-resolution `clock()` based on Windows `QueryPerformanceCounter`; NumPy uses Python `time.perf_counter_ns()`, including Python call overhead. Three warmups precede twenty timed calls, summarized by their median.

The driver verifies five output positions per matrix size against independent Python `math.fsum` dot products, and verifies sampled input values. The tolerance is relative `1e-10` and absolute `1e-11`. All recorded samples in these two runs had zero reported absolute error. This sampling is not a full-output proof. HENG keeps compensated summation; OpenBLAS can use a different reduction order, so identical dtype and tolerances do not imply identical internal numerical algorithms.

HENG prints its actual `get_threads()` result. Source inspection confirms that its matrix kernel creates worker threads only when the configured worker count exceeds one; otherwise it executes in the calling thread without external BLAS. NumPy's already-loaded OpenBLAS DLL reports its thread configuration through `scipy_openblas_get_num_threads64_`. Neither mechanism samples all operating-system thread activity. The report's `fair_single_thread: true` means these compute-thread configurations matched; it is not CPU scheduling isolation or a guarantee of complete experimental fairness.

#### Original NumPy comparison

First independent driver run:

- 128 by 128: HENG 7.2713 ms; NumPy/OpenBLAS 0.19365 ms.
- 256 by 256: HENG 63.0279 ms; NumPy/OpenBLAS 0.9254 ms.

Second independent driver run:

- 128 by 128: HENG 7.8044 ms; NumPy/OpenBLAS 0.1841 ms.
- 256 by 256: HENG 63.42325 ms; NumPy/OpenBLAS 0.92985 ms.

NumPy/OpenBLAS was about 37.5 to 42.4 times faster at 128 and about 68.1 to 68.2 times faster at 256 before the dense-matrix follow-up.

#### Dense `f64` follow-up

- Contiguous two-dimensional `f64` matrices use packed right-hand panels and CPU vector instructions when the runtime verifies support and a finite accumulation bound. The general vector kernel preserves the order of separately rounded products and compensated accumulation.
- For inputs on suitably bounded binary grids, the runtime proves that every product and partial sum fits exactly in `f64` and uses fused multiply-add for those operations. The proof checks both matrices at runtime, including their magnitude bounds and whether every entry is an exact multiple of a power of two. The sum of the two integer precision budgets and `ceil(log2(k))` is at most 53 bits, where `k` is the inner dimension. The permitted exponent range keeps all nonzero grid products in the normal range.
- The runtime retains its previous kernel for nonfinite inputs, dimensions or layouts outside the packed path, and unsupported instructions. Every result element is written directly, so matrix output storage needs no preliminary zero fill.
- Run `python -B benchmarks/compare_frameworks.py --input affine` to repeat the original input distribution, `--input dyadic` for varied exact binary fractions, and `--input rational` for values rounded from varying integer denominators. The latter two profiles share deterministic integer numerators. The reference validates sampled inputs and outputs for each profile.

The follow-up used the same NumPy/OpenBLAS versions and the same single-thread verification, with three independent process runs per input profile and the original three warmups and twenty timed calls. Times below are HENG / NumPy in milliseconds, listed in run order. All reported samples passed validation.

- `affine`, 128 by 128: `0.12890 / 0.18885`, `0.12990 / 0.19555`, `0.16380 / 0.20235`. HENG ran 1.24 to 1.51 times as fast in all three runs.
- `affine`, 256 by 256: `1.87290 / 1.69520`, `0.92855 / 1.70625`, `0.93890 / 1.21130`. HENG led in two runs, with a 1.29 to 1.84 times speed ratio in those runs.
- `dyadic`, 128 by 128: `0.12640 / 0.20235`, `0.21315 / 0.24320`, `0.23705 / 0.29420`. HENG ran 1.14 to 1.60 times as fast in all three runs.
- `dyadic`, 256 by 256: `0.93185 / 0.98945`, `0.93470 / 0.95875`, `0.93650 / 0.94335`. HENG ran 1.007 to 1.062 times as fast in all three runs. The small difference may vary across additional runs.
- `rational`, 128 by 128: `0.61865 / 0.19230`, `0.76155 / 0.19070`, `0.61780 / 0.18175`. NumPy ran 3.22 to 3.99 times as fast.
- `rational`, 256 by 256: `5.03640 / 0.96505`, `5.49710 / 0.94305`, `5.07720 / 0.90710`. NumPy ran 5.22 to 5.83 times as fast.

At the start of the follow-up, the same `affine` test on the previous commit took 9.7136 and 82.3174 ms in HENG. Compared with that one baseline run, the three follow-up runs were approximately 59 to 75 times as fast at 128 and 44 to 89 times as fast at 256. These ranges describe the observed runs.

#### General floating-point matrix comparison

Run `python -B benchmarks/compare_frameworks.py --input rational --dtype f64 --layout contiguous --matmul-fast`. Change `--dtype` to `f32` for single precision. Set `--layout` to `right-transposed` or `both-transposed` to test a different physical layout with the same logical input values. Omit `--matmul-fast` for the default compensated mode. Eligible `f64` matrices can use packed kernels with either storage layout. The `f32` fast kernel requires contiguous row-major matrices, and it accumulates in `f32`. Unsupported layouts and inputs with unsafe accumulation bounds use the existing path. Fast mode can change results through rounding.

The preceding release's three runs used the same NumPy/OpenBLAS versions, per-run sample counts, and verified single compute-thread configuration. The reported ratio divides NumPy time by HENG time, so a value above one favors HENG.

- Fast `f64`, contiguous: 128 led in 3 of 3 runs, with ratios 1.35 to 1.79. At 256 the ratios were 0.98, 1.04, and 1.81.
- Fast `f64`, right transposed: 128 and 256 both led in 3 of 3 runs, with ratios 1.23 to 1.77 and 1.07 to 1.68, respectively.
- Fast `f64`, both transposed: 128 led in 3 of 3 runs, with ratios 1.09 to 1.61. At 256 HENG led in 1 of 3 runs, with ratios from 0.50 to 1.27.
- Fast `f32`, contiguous: 128 led in 3 of 3 runs, with ratios 1.13 to 2.05. At 256 the ratios were 0.94, 1.16, and 1.89.
- Default `f64`, contiguous: NumPy was faster on both sizes in all three runs.

Every sampled result passed the independent reference check. The largest observed HENG sample errors were `5.56e-16` for `f64` fast mode and `2.80e-7` for `f32` fast mode. Timings varied between runs, particularly at 256.

#### Per-workload performance requirement

Add `--require-win` to the matrix driver to require verified single-thread settings and a strictly lower HENG median for every selected matrix size and library. A tie, a slower result, or unverified threads makes the command fail. The training driver accepts `--numpy --require-win` and also requires matching evaluation scores and generated text. Both commands retain the measured results when the speed requirement fails.

The latest runtime packs transposed matrices in vector tiles and processes more independent outputs together during compensated accumulation. The optional fast mode can skip zero coefficients when the eligible left operand is sparse and the right operand is contiguous. The optimized Adam and AdamW path combines the double-precision moment updates and parameter update in one traversal. Reverse-mode differentiation reuses the validated graph traversal and initializes gradient storage as needed.

The latest three runs retained the two sizes, the previous input profiles and precision modes, and the same numerical tolerances. Every numerical check passed. HENG was faster in 27 of 42 matrix comparisons, including 18 of 24 with fast mode enabled. The general `f64` contiguous fast workload won all six comparisons, with a speed ratio of 1.57 to 1.81 at 128 and 1.05 to 1.81 at 256. NumPy/OpenBLAS won all six general `f64` strict comparisons by factors of 1.87 to 5.62. For the 128 and 256 sizes respectively, HENG won 2 and 2 of 3 right-transposed runs, 3 and 1 of 3 both-transposed runs, and 3 and 1 of 3 fast `f32` runs. The exact binary-fraction profile won 2 and 2 of 3, and the affine profile won 3 and 2 of 3. The all-workload speed requirement failed.

On the same 36-epoch character experiment, the fast-mode HENG training loop took 0.98154, 0.97549, and 0.87229 seconds. The paired NumPy times were 0.42127, 0.24952, and 0.24758 seconds. The NumPy speed ratio was 2.33 to 3.91. NumPy also led all three strict-mode runs by factors of 3.06 to 6.18. The same evaluation scores and greedy completions were reproduced. The held-out accuracy was 93.70%, and accuracy on the 14 unseen context-and-target pairs was 71.43%.

#### Original character model training

Run `python -B benchmarks/train_text_model.py --numpy --heng-matmul fast`. Use `--heng-matmul strict` to repeat with compensated accumulation. The model uses a 12-character context, one 64-unit hidden layer, 17,557 parameters, and Adam with a batch size of 32. It trained on 12 locally generated sentences for 36 epochs and 432 updates. Four sentence combinations were held out. They contain 127 next-character targets, including 14 context-and-target pairs absent from training. The NumPy baseline used the same model, initialization, batches, optimizer, and verified one-thread configuration as HENG.

- Training loss fell from 3.07055 to 0.08910, and accuracy rose from 4.20% to 95.54%.
- Held-out loss fell from 3.07533 to 0.21004, perplexity fell from 21.657 to 1.234, and accuracy rose from 2.36% to 93.70%.
- On the 14 unseen context-and-target pairs, loss fell from 3.04603 to 1.38045, perplexity fell from 21.032 to 3.977, and accuracy rose from 0% to 71.43% (10 of 14).
- HENG and NumPy matched on the accuracy and the three greedy completions. The largest absolute loss difference across the measured runs was below `1.4e-15`.

The generated sentences were `the fox rests near stone walls.`, `the owl waits near old oaks.`, and `the hare hides near still ponds.` All three occur in the training set. The held-out scores cover new combinations of known words in this small corpus.

The two fast-mode training loops took 1.95265 and 1.33276 seconds in HENG, versus 0.36633 and 0.26770 seconds in NumPy. The strict-mode run took 1.57613 seconds in HENG and 0.28343 seconds in NumPy. NumPy ran about 5.0 to 5.6 times as fast in these paired measurements. Timing includes batch extraction, forward and backward passes, and updates. Build, data preparation, evaluation, and generation are outside the timed interval.

The optional `--native-step` mode uses a dedicated native kernel for this exact two-layer character model. It performs the same forward, cross-entropy gradient, and Adam update equations without recording a general-purpose graph. It is a workload-specific optimization, so its timing is reported separately from the general framework path. One matched run of `python -B benchmarks/train_text_model.py --numpy --heng-matmul fast --native-step` took 0.55237 seconds in the HENG path, while the paired NumPy run took 0.43020 seconds. The numerical metrics and generated text matched. The strict performance requirement returned `FAIL` because the specialized path was still slower than NumPy.

#### Limits and reproduction

The test-device details are withheld for privacy, which limits reproduction against a fully specified hardware configuration. CPU frequency, temperature, background load, process scheduling, allocator behavior, and language-call overhead were not normalized. No statistical confidence interval was calculated. The three deterministic input profiles cover specific distributions, sizes, and layouts. Multi-thread scaling, convolution, attention, large-corpus training, inference throughput, and energy use remain outside these comparisons.

The driver prints structured JSON to standard output and removes its temporary HengLang source directory. Normal compiler caches remain in `.build/`. Its outer timeout cannot guarantee termination of every compiler child process; after a timeout, verify that the previous build has stopped before repeating it. It does not install dependencies or train anything. Use `--help` for configuration. Optional PyTorch support is available only when PyTorch is already installed; it was NOT RUN for this delivery.

### 简体中文

#### 结论与范围

前一次严格模式测试已确认精确二进制分数输入可以领先。本轮覆盖一般 `f32` 和 `f64` 矩阵输入、转置存储布局以及字符模型的完整训练循环。可选快速模式对符合条件的矩阵用融合乘加替代补偿累加。使用相同模型和训练配置时，NumPy 的训练仍然更快。

最初测量于 2026-11-03 完成，连续矩阵优化复测于 2026-11-26 完成。HENG 使用项目固定工具链，以 C17 `-O2` 和 `-fno-fast-math` 构建。测试设备及本地环境标识保持私密。基准使用合成输入和已有的 NumPy。

这些是本机观察，不是跨机器性能保证。编译与各项性能测试串行执行，未互相并发争用计算资源。

#### 本次改进

- 浮点矩阵乘法针对四种 `f64`、`f32` 输入组合直接读取存储，消除乘法内层循环的数据类型分派。
- 连续浮点 `add`、`sub`、`mul`、`div` 在可线性遍历或单元素广播时直接访问存储。
- 连续 `scale`、`relu`、`fused_affine_relu` 避免逐元素维度换算与操作名称判断。
- 保留矩阵补偿求和及非有限数行为、Float32 舍入位置、整数溢出检查、非连续后备路径和原有线程划分。
- 新增 `get_threads()` 内置函数，供基准读取 HENG 实际配置的计算线程数；它不是操作系统线程活动监测器。

#### 上一轮项目内部前后对比

命令：`python -B -m henglang bench`。

配置为 `f64`、1 个计算线程、3 次预热、20 次计时取中位数。矩阵朴素与优化路径使用相同确定性随机种子。逐元素负载为 100000 个元素，以分离操作或显式融合内置函数计算 `relu(input * 2 + 1)`。两种内核模式均使用开启编译优化的原生程序。

本轮优化中的一组前后测量为：

- 128 乘 128 优化矩阵乘法：12.499 ms 降至 7.343 ms，约快 1.70 倍。
- 256 乘 256 优化矩阵乘法：101.356 ms 降至 63.043 ms，约快 1.61 倍。
- 分离逐元素路径：16.445 ms 降至 0.4475 ms，约快 36.75 倍。
- 显式融合逐元素路径：7.956 ms 降至 0.21475 ms，约快 37.05 倍。

上一轮运行时复测中，优化矩阵分别为 7.765 ms、63.622 ms，分离与融合逐元素路径分别为 0.4189 ms、0.2033 ms；朴素矩阵内核分别为 17.505 ms、173.823 ms。这些独立运行既显示改进，也显示批次间波动；它们不构成统计置信区间，也不能把全部波动归因于代码变更。

上一轮内部复测同时记录：

- 逐元素逻辑分配次数：分离路径 24 次，融合路径 5 次。
- 100 次张量创建循环：逻辑分配 1000 次，系统分配 901 次，张量存储复用 99 次。这些计数包含运行时元数据，不等于张量个数。
- 缓冲池保留 800000 字节，显式清理后为零。
- 运行时跟踪峰值内存为 11645572 字节，不等于整个进程的驻留内存，也不是与其他框架的内存比较。

内部矩阵计时程序检查输出元素数量，更广泛的正确性由独立数值参考测试覆盖。逐元素计时程序还检查数值及分离、融合路径的精确一致性。分配次数下降不代表计算复杂度或训练成本按相同比例下降。

#### 上一轮 NumPy 对比方法

命令：`python -B benchmarks/compare_frameworks.py`。

比较程序在两个独立驱动进程中各运行一次。每次比较 HENG 与本机已有的 NumPy 2.5.2、OpenBLAS 0.3.34.0.0。每个矩阵规模先测 HENG，再测 NumPy。双方均请求并实际返回计算线程配置为 1。

方阵边长为 `n` 时，双方使用相同连续 `f64` 输入，展平索引记为 `i`：

- 左矩阵：`reshape(i * (1 / (n*n)) - 0.5, [n, n])`。
- 右矩阵：`reshape(i * (-0.75 / (n*n)) + 0.25, [n, n])`。

每次计时均产生新的逻辑输出，不使用 `out` 参数。输入构造、导入、编译、进程启动、显式垃圾回收、校验及输出销毁不计入耗时。保留双方默认分配器及 HENG 存储池，因此逻辑分配不表示每次都向操作系统申请新存储。

HENG 使用基于 Windows `QueryPerformanceCounter` 的原生高精度 `clock()`；NumPy 使用 Python `time.perf_counter_ns()`，包含 Python 调用开销。预热 3 次后计时 20 次，报告中位数。

每个规模抽取 5 个输出位置，以独立 Python `math.fsum` 点积作为参考，并校验输入样本。相对容差为 `1e-10`，绝对容差为 `1e-11`。两次运行的所有记录样本报告绝对误差均为零；抽样不等于完整输出证明。HENG 保留补偿求和，OpenBLAS 可以采用不同归约顺序，因此相同精度和容差不代表内部数值算法相同。

HENG 实际打印 `get_threads()` 返回值。源码检查确认，只有配置线程数大于 1 时矩阵内核才创建工作线程，否则直接在调用线程计算，不调用外部 BLAS。NumPy 通过已加载 OpenBLAS DLL 的 `scipy_openblas_get_num_threads64_` 查询线程配置。两种机制都不采样全部操作系统线程活动。报告中的 `fair_single_thread: true` 仅表示计算线程配置匹配，不代表 CPU 调度隔离或完整实验公平性保证。

#### 上一轮 NumPy 实测

第一次独立驱动运行：

- 128 乘 128：HENG 为 7.2713 ms，NumPy/OpenBLAS 为 0.19365 ms。
- 256 乘 256：HENG 为 63.0279 ms，NumPy/OpenBLAS 为 0.9254 ms。

第二次独立驱动运行：

- 128 乘 128：HENG 为 7.8044 ms，NumPy/OpenBLAS 为 0.1841 ms。
- 256 乘 256：HENG 为 63.42325 ms，NumPy/OpenBLAS 为 0.92985 ms。

连续矩阵优化前，NumPy/OpenBLAS 在这两轮测量中于 128 规模领先约 37.5 至 42.4 倍，于 256 规模领先约 68.1 至 68.2 倍。

#### 连续 `f64` 矩阵复测

- 对于连续二维 `f64` 矩阵，运行时验证指令支持和累加结果的有限范围后，会打包右矩阵分块并使用 CPU 向量指令。通用向量内核保留分别舍入乘积及按原顺序补偿累加的语义。
- 对于满足位宽限制的二进制有理数输入，运行时先证明每个乘积和部分和都能用 `f64` 精确表示，再执行融合乘加。检查覆盖两个矩阵的数值范围，并验证每个元素是否为某个二的幂的整数倍。两个输入的整数位宽预算与 `ceil(log2(k))` 总计最多 53 位，其中 `k` 为内维度。允许的指数范围保证非零乘积都在正规数范围内。
- 非有限输入、超出打包路径适用范围的尺寸或布局、缺少所需指令的情况沿用原有内核。矩阵乘法直接写入每个结果元素，因此可省去输出存储预先清零。
- 使用 `python -B benchmarks/compare_frameworks.py --input affine` 可复测原分布，`--input dyadic` 测量不同的精确二进制分数，`--input rational` 测量由可变整数分母舍入得到的值。后两者采用相同的确定性整数分子。每种分布都会独立核对输入与输出样本。

复测沿用相同的 NumPy/OpenBLAS 版本及单线程核验方式。每种输入分布启动三个独立进程，各预热 3 次，计时 20 次。以下按轮次列出中位耗时，单位为毫秒，每组依次是 HENG 和 NumPy。所有报告样本均通过校验。

- `affine`，128 乘 128：`0.12890 / 0.18885`、`0.12990 / 0.19555`、`0.16380 / 0.20235`。三轮中 HENG 的速度均为 NumPy 的 1.24 至 1.51 倍。
- `affine`，256 乘 256：`1.87290 / 1.69520`、`0.92855 / 1.70625`、`0.93890 / 1.21130`。HENG 领先两轮，这两轮的速度为 NumPy 的 1.29 至 1.84 倍。
- `dyadic`，128 乘 128：`0.12640 / 0.20235`、`0.21315 / 0.24320`、`0.23705 / 0.29420`。三轮中 HENG 的速度均为 NumPy 的 1.14 至 1.60 倍。
- `dyadic`，256 乘 256：`0.93185 / 0.98945`、`0.93470 / 0.95875`、`0.93650 / 0.94335`。三轮中 HENG 的速度均为 NumPy 的 1.007 至 1.062 倍。这一小幅差距可能随复测波动。
- `rational`，128 乘 128：`0.61865 / 0.19230`、`0.76155 / 0.19070`、`0.61780 / 0.18175`。NumPy 的速度约为 HENG 的 3.22 至 3.99 倍。
- `rational`，256 乘 256：`5.03640 / 0.96505`、`5.49710 / 0.94305`、`5.07720 / 0.90710`。NumPy 的速度约为 HENG 的 5.22 至 5.83 倍。

本轮优化开始前，用上一提交重跑 `affine` 基准，HENG 分别耗时 9.7136 和 82.3174 毫秒。相对这一次基线，三轮复测在 128 规模提速约 59 至 75 倍，在 256 规模提速约 44 至 89 倍。该范围反映本轮实测波动。

#### 一般浮点矩阵对比

运行 `python -B benchmarks/compare_frameworks.py --input rational --dtype f64 --layout contiguous --matmul-fast`。把 `--dtype` 改为 `f32` 可测试单精度。把 `--layout` 设为 `right-transposed` 或 `both-transposed`，可在保持逻辑输入值一致的前提下改变物理布局。省略 `--matmul-fast` 即使用默认补偿累加。符合条件的 `f64` 矩阵支持连续与转置存储布局的打包内核。`f32` 快速内核要求行主序连续矩阵，并在 `f32` 中累加。布局或累加安全界不符合条件时沿用原有路径。快速模式可能产生舍入差异。

上一版运行时的三轮测量沿用同版本 NumPy/OpenBLAS、同样的每轮样本数和经过核验的单计算线程配置。以下速度比用 NumPy 耗时除以 HENG 耗时，大于 1 表示 HENG 领先。

- 快速 `f64`，连续布局：128 三轮均领先，速度比为 1.35 至 1.79。256 的速度比为 0.98、1.04、1.81。
- 快速 `f64`，右矩阵转置存储：128 和 256 均三轮领先，速度比分别为 1.23 至 1.77、1.07 至 1.68。
- 快速 `f64`，双矩阵转置存储：128 三轮均领先，速度比为 1.09 至 1.61。256 三轮中领先一轮，速度比为 0.50 至 1.27。
- 快速 `f32`，连续布局：128 三轮均领先，速度比为 1.13 至 2.05。256 的速度比为 0.94、1.16、1.89。
- 默认 `f64`，连续布局：两种规模的三轮测量均由 NumPy 领先。

所有样本均通过独立参考校验。HENG 在 `f64` 快速模式中的最大样本绝对误差为 `5.56e-16`，`f32` 快速模式为 `2.80e-7`。各轮计时存在波动，256 规模尤为明显。

#### 逐项性能验收

矩阵比较程序可加 `--require-win`，要求核验单线程配置，并且 HENG 在每个选择的规模和比较库上中位耗时都严格更低。持平、落后或线程未核实均会使程序返回失败。训练程序支持 `--numpy --require-win`，同时要求评估指标及续写一致。性能验收失败时，两种程序仍保留实测数据。

本轮运行时以向量分块打包转置矩阵，并在补偿累加中同时计算更多独立输出。快速模式可在左矩阵稀疏且右矩阵连续时跳过零系数。优化后的 Adam 与 AdamW 路径把双精度动量和参数更新合到一次遍历。反向自动微分复用已核验的图遍历结果，并按需初始化梯度存储。

本轮沿用原有两种矩阵规模、输入分布、精度模式和误差容限，串行完成三轮测量。所有数值校验均通过。42 次矩阵配对中 HENG 领先 27 次，其中 24 次快速模式配对中领先 18 次。一般 `f64` 连续布局在快速模式下六次全胜，128 规模速度比为 1.57 至 1.81，256 规模为 1.05 至 1.81。一般 `f64` 严格模式六次均由 NumPy/OpenBLAS 领先，差距为 1.87 至 5.62 倍。按 128、256 的顺序，右矩阵转置存储分别领先 2 轮、2 轮，双矩阵转置存储领先 3 轮、1 轮，快速 `f32` 领先 3 轮、1 轮。精确二进制分数输入领先 2 轮、2 轮，仿射输入领先 3 轮、2 轮。全部用例严格领先的验收未通过。

相同的 36 轮字符模型训练中，HENG 快速模式依次用时 0.98154、0.97549、0.87229 秒，NumPy 配对用时为 0.42127、0.24952、0.24758 秒。NumPy 的速度是 HENG 的 2.33 至 3.91 倍。严格模式下 NumPy 也三轮领先，速度为 HENG 的 3.06 至 6.18 倍。复测得到相同的评估指标和贪心续写。完整留出集准确率为 93.70%，14 个未见上下文与目标字符组合的准确率为 71.43%。

#### 原字符模型训练实验

运行 `python -B benchmarks/train_text_model.py --numpy --heng-matmul fast`。用 `--heng-matmul strict` 可以在补偿累加模式下复测。模型使用 12 字符上下文、64 个隐藏单元和 17,557 个参数，Adam 优化的批大小为 32。用 12 句本地合成语料训练 36 轮，共更新 432 步。留出 4 种完整句子组合，共 127 个下一字符目标。其中有 14 个上下文与目标字符的组合在训练中未出现。NumPy 与 HENG 采用相同的模型、初始化、批次和优化器，并核验单线程配置。

- 训练损失从 3.07055 降到 0.08910，准确率从 4.20% 升到 95.54%。
- 完整留出集损失从 3.07533 降到 0.21004，困惑度从 21.657 降到 1.234，准确率从 2.36% 升到 93.70%。
- 14 个未见组合的损失从 3.04603 降到 1.38045，困惑度从 21.032 降到 3.977，准确率从 0% 升到 71.43%，即预测正确 10 个。
- HENG 和 NumPy 的准确率与三条贪心续写一致。各轮最大绝对损失差小于 `1.4e-15`。

生成的句子为 `the fox rests near stone walls.`、`the owl waits near old oaks.` 和 `the hare hides near still ponds.`。三句均出现在训练集。留出集指标反映模型在这个小语料上对已知词语新组合的预测效果。

两轮快速模式训练循环中，HENG 分别用时 1.95265 和 1.33276 秒，NumPy 分别用时 0.36633 和 0.26770 秒。严格模式一轮中，HENG 用时 1.57613 秒，NumPy 用时 0.28343 秒。在这些配对测量中，NumPy 的训练速度约为 HENG 的 5.0 至 5.6 倍。计时包含批次提取、前向、反向及参数更新。构建、数据准备、评估和生成在计时区间外。

可选的 `--native-step` 模式为这个特定的两层字符模型使用专用原生内核。它执行相同的前向、交叉熵梯度和 Adam 更新公式，但不记录通用计算图。这属于针对单一负载的优化，因此耗时与通用框架路径分开报告。一次运行 `python -B benchmarks/train_text_model.py --numpy --heng-matmul fast --native-step` 时，HENG 用时 0.55237 秒，配对的 NumPy 用时 0.43020 秒。数值指标和生成文本一致。严格性能验收返回 `FAIL`，因为这条专用路径仍慢于 NumPy。

#### 限制与复现

为保护隐私，测试设备信息保持私密，因此外部复现缺少完整硬件配置。没有统一 CPU 频率、温度、后台负载、进程调度、分配器行为和语言调用开销，没有计算统计置信区间。三种确定性输入覆盖特定分布、规模与布局。本轮对比范围以外的项目包括多线程扩展、卷积、注意力、大语料训练、推理吞吐和能耗。

比较程序将结构化 JSON 输出到标准输出，并清理本次临时衡言源码目录；正常编译缓存保留在 `.build/`。外层超时不能保证终止全部编译器子进程，超时后应确认上一次构建停止，再重新执行。程序不自动安装依赖，不执行训练。配置选项见 `--help`。可选 PyTorch 路径要求本机已安装 PyTorch，本次交付对应验证为 NOT RUN。
