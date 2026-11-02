## HengLang and HENG

[简体中文](README.zh-CN.md) | English

HengLang (衡言) is a statically checked programming language with the dedicated `.heng` source extension. HENG (衡算) is its CPU tensor and automatic-differentiation framework. Version 1.0.0 targets Windows x86-64 on a single machine.

The compiler frontend uses the Python standard library, generates C17, and invokes a project-local, version-pinned Zig C compiler. Generated executables run natively without Python. HENG does not depend on PyTorch or NumPy for compilation or execution; the optional comparison benchmark uses an already installed comparison library.

The version number identifies this implementation, not a claim of production readiness, API stability across future releases, or parity with established frameworks.

### What is implemented

- Language: static type checking, functions, recursion, generic functions, structures, enumerations, lists, `Result<T>`, module imports, control flow, and source-position diagnostics.
- Runtime: `f64`, `f32`, `i64`, and `bool` tensors, shared-storage views, broadcasting, indexing, reductions, matrix multiplication, stable softmax, convolution unfolding/folding, and COO/CSR sparse operations.
- Differentiation: explicit computation tapes, reverse-mode differentiation, vector-Jacobian products, Jacobian-vector products, gradient accumulation, graph clearing, and in-place modification checks.
- Modules: sequential composition, shared parameters, Linear, Embedding, activations, Conv2D, average/max pooling, LayerNorm, BatchNorm, Dropout, a basic RNN, and multi-head attention.
- Optimization utilities: SGD, momentum, Nesterov, Adam, AdamW, RMSprop, gradient clipping, and step, exponential, and linear learning-rate schedules.
- Data and state: in-memory tensor batching, deterministic shuffling, model/optimizer/random-number state persistence, and validation of malformed state files.
- Tooling: checking, formatting, native builds, execution, automated tests, and reproducible CPU benchmarks.

### Requirements and setup

The project targets Windows x86-64, uses Python 3.13 for development and tests, and pins Zig 0.17.0 as its native toolchain. Generated programs do not require a Python interpreter.

Run these commands from the repository root in PowerShell:

```powershell
python -B tools/toolchain.py install
python -B tools/toolchain.py probe
python -B -m henglang --version
```

The toolchain installer reads `heng.toml`, downloads only the pinned official Zig distribution, and verifies the pinned SHA-256 before extraction and execution. Downloads must use the configured HTTP proxy, initially `http://127.0.0.1:10808`. Start that proxy or explicitly configure another working HTTP proxy in `heng.toml`; there is no direct-connection fallback. The local proxy address is a development setting, not a hosted service supplied by the project.

The compiler and caches remain inside `.tools/` and `.build/`. Neither directory is included in the source repository. A fresh clone must install the toolchain before native builds and native tests. No global compiler installation or system `PATH` modification is required.

### First native program

The included example computes a scalar sum of squares and its gradient; it does not train a model or download data.

```powershell
python -B -m henglang check examples/quickstart.heng
python -B -m henglang run examples/quickstart.heng
python -B -m henglang build examples/quickstart.heng --output .build/quickstart.exe
& '.build/quickstart.exe'
```

For input `[1, 2, 3]`, the example checks a value of `14` and a gradient of `[2, 4, 6]`. Existing executable fixtures in `tests/fixtures/` exercise the broader framework API.

### Development and verification

```powershell
python -B -m henglang test
python -B -m henglang test --pattern test_frontend.py
python -B -m henglang fmt examples/quickstart.heng --check
python -B -m henglang bench
```

`test` discovers the standard-library `unittest` suite. Full verification includes native compilation and can take several minutes. Tests perform small synthetic forward/backward and optimizer-step checks; these are correctness tests, not model-training experiments.

`build --no-opt` disables both language optimizations and native compiler optimization. The benchmark instead uses an optimized native build and switches between runtime kernels, so its internal baseline is not an unoptimized compiler build.

### Performance and comparison claims

HENG includes blocked matrix multiplication, bounded tensor-storage reuse, and an affine-plus-ReLU fused operation. These mechanisms can improve particular workloads; their presence alone does not demonstrate superiority over another framework.

See [benchmark methodology and measured results](BENCHMARKS.md) for inputs, precision, thread settings, timing boundaries, and limitations. Test-device information is kept private. The repository includes an optional comparison driver:

```powershell
python -B benchmarks/compare_frameworks.py --help
python -B benchmarks/compare_frameworks.py
```

The comparison driver requires an already installed supported comparison library and does not install dependencies. It checks numerical samples before reporting timings. Comparisons are operator-level measurements, not whole-framework rankings. In the recorded local tests, NumPy with OpenBLAS was faster than HENG on both tested matrix sizes; HENG did not exceed that baseline.

A general claim that HENG exceeds PyTorch, TensorFlow, JAX, or all existing frameworks is unproven. No model has been trained for this delivery, and no equal-quality training-cost advantage has been established.

### Current boundaries

- Supported delivery target: Windows x86-64, single-machine CPU. GPU, distributed execution, and other platforms are outside the validated scope.
- HENG has its own APIs, tensor semantics, and state format; PyTorch API and checkpoint compatibility are not provided.
- Differentiable matrix multiplication requires rank two or higher. Differentiable slicing supports valid positive steps.
- Conv2D currently has no groups or dilation. The recurrent module is a basic RNN, not an LSTM or GRU.
- Floating-point parameters and leaf gradients preserve their declared dtype, but mixed operations and default `f64` module parameters/constants can promote intermediate results to `f64`.
- Checkpoints cover the implemented model, optimizer, and random-number state; scheduler objects and data-loader cursors are not checkpointed.
- The local compiler is not a sandbox for untrusted programs. State-file checks do not constitute a formal security audit or authentication mechanism.

### Repository layout

- `henglang/`: language frontend, intermediate representation, optimization, C emission, and command-line tools.
- `runtime/`: C17 tensor runtime, memory management, native kernels, and binary state I/O.
- `heng/`: framework modules written in HengLang.
- `tools/`: project-local toolchain management.
- `tests/`: language, numerical-reference, framework, resource, and native-runtime tests.
- `examples/`: executable introductory programs.
- `benchmarks/`: internal kernel benchmarks and optional framework comparisons.

### License, contributions, and security

Copyright 2026 tlkppm and HengLang contributors. The project is licensed under [Apache-2.0](LICENSE), subject to the scope described in [NOTICE](NOTICE). The English license text controls; the bilingual legal guidance does not replace or add restrictions to it.

Read [legal and usage guidance](LEGAL.md), [third-party notices](THIRD_PARTY_NOTICES.md), [contribution guidance](CONTRIBUTING.md), and [security guidance](SECURITY.md). These files distinguish source licensing from third-party toolchain and linked-runtime obligations. They are project information, not a legal opinion or a certification of compliance.

Project repository: [tlkppm/HengLang](https://github.com/tlkppm/HengLang).
