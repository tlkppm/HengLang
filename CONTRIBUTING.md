## Contributing / 贡献指南

### English

Contributions should include a focused change and a reproducible test. Discuss substantial architecture or API changes before implementation. The validated target is Windows x86-64 CPU; GPU, distributed execution, and compatibility with another framework are not implicit requirements.

By intentionally submitting a contribution for inclusion, you offer it under Apache-2.0 as described in Section 5 of [LICENSE](LICENSE), unless you explicitly state otherwise. Submit only material you have the right to license, preserve applicable third-party notices, and identify any third-party material. No separate contributor license agreement or transfer of copyright is required by this guide.

Preserve the existing encoding and line endings. Keep source changes small, avoid unrelated formatting, and follow the project's existing comment-free source style. Do not add credentials, datasets, downloaded tools, build artifacts, or machine-specific configuration.

Useful local checks:

```powershell
python -B -m henglang test
python -B -m henglang bench
```

Native tests require the pinned project-local toolchain. Run relevant tests first and the full suite for changes to runtime semantics, compilation, memory management, or state persistence. Add independent numerical references and boundary tests where appropriate. Fix failures rather than skipping tests or weakening tolerances without a documented numerical contract.

For performance changes, report the workload, compiler/library versions, precision, actual compute-thread settings, warmups, number of samples, median timings, and correctness checks. Keep test-device identities, CPU models, system builds, physical core counts, hostnames, and local paths private. Keep compilation and input generation outside steady-state timings. Separate internal improvements from comparisons with third-party frameworks. State explicitly when training, accuracy, memory usage, or other dimensions were not measured.

Use [the project repository](https://github.com/tlkppm/HengLang) for ordinary questions and reproducible bugs. Follow [SECURITY.md](SECURITY.md) for potential vulnerabilities and avoid publishing exploit details or secrets.

### 简体中文

贡献应包含聚焦的改动与可复现测试。较大的架构或接口变更应先讨论。当前已验证目标为 Windows x86-64 CPU；GPU、分布式执行和其他框架兼容性不属于默认要求。

除非明确另行说明，主动提交用于纳入项目的贡献，按照 [LICENSE](LICENSE) 第 5 条以 Apache-2.0 提供。请仅提交自己有权授权的内容，保留适用的第三方声明，并标明引入的第三方材料。本指南不要求单独签署贡献者许可协议，也不要求转让著作权。

保持现有编码和换行。源码修改应尽量小，避免无关格式化，遵循项目已有的无注释源码风格。请勿加入凭据、数据集、下载的工具、构建产物或机器专属配置。

常用本地检查命令：

```powershell
python -B -m henglang test
python -B -m henglang bench
```

原生测试需要固定版本的项目隔离工具链。先执行相关测试；修改运行时语义、编译、内存管理或状态持久化时，再执行完整套件。需要时增加独立数值参考与边界测试。应修复失败，不通过跳过测试或在缺少明确数值契约时放宽容差来掩盖问题。

性能改动应报告负载、编译器和比较库版本、精度、实际计算线程配置、预热次数、样本数、中位耗时与正确性检查。测试设备标识、处理器型号、系统构建版本、物理核心数量、主机名和本地路径保持私密。稳态计时排除编译和输入构造。区分项目内部改进与第三方框架比较，明确说明是否测量训练、精度、内存等其他维度。

一般问题和可复现缺陷请通过[项目仓库](https://github.com/tlkppm/HengLang)反馈。潜在漏洞遵循 [SECURITY.md](SECURITY.md)，避免公开利用细节或秘密信息。
