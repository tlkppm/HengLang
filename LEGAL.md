## Legal and usage guidance / 法律与使用说明

### English

#### License and authority

Copyright 2026 tlkppm and HengLang contributors. The original material distributed in this repository is offered under [Apache License 2.0](LICENSE), with attribution in [NOTICE](NOTICE). The repository owner's selection of a license does not establish ownership of material belonging to others or grant rights beyond those the relevant contributors can grant.

The unmodified English `LICENSE` is the authoritative license text. This bilingual guidance is explanatory, is not a separate end-user agreement, and does not impose additional restrictions on the Apache-2.0 grant. If this explanation differs from the license, follow the license.

Apache-2.0 permits commercial use, modification, and redistribution subject to its conditions. Redistribution obligations include supplying the license, marking changed files, retaining applicable notices, and preserving applicable NOTICE attribution as described in Section 4. The patent grant is limited to the scope in Section 3 and includes its patent-litigation termination condition. This is not a representation that all possible patents have been cleared.

#### Third-party and generated material

Downloaded compilers and their bundled libraries are separate third-party works. Their licenses are not replaced by Apache-2.0. Generated programs can contain HENG runtime code and compiler-linked third-party components; inspect the actual linked components and applicable license terms before redistributing binaries. See [third-party notices](THIRD_PARTY_NOTICES.md).

No dataset, pretrained model, user input, or independently authored program becomes project-owned merely because it is processed by this software. Obtain the necessary rights for code, data, model weights, and outputs in your own use case. The project makes no blanket statement about ownership or copyright eligibility of generated outputs.

#### Warranty, liability, and suitability

Sections 7 and 8 of `LICENSE` provide the applicable warranty disclaimer and limitation of liability, subject to mandatory law and any separate written agreement. This guidance does not expand those exclusions, waive mandatory rights, or create a support, service-level, security, or fitness guarantee.

The framework is an independently developed CPU implementation. Testing does not prove correctness for all inputs, production suitability, suitability for medical or other safety-critical decisions, or compliance with a particular regulatory regime. Evaluate these matters independently before deployment.

#### Names and performance statements

Apache-2.0 does not grant general trademark rights. Project and third-party names identify their respective software; references to PyTorch, NumPy, TensorFlow, JAX, Python, or Zig do not imply affiliation, endorsement, or certification. This document does not assert that the project names are registered trademarks.

Performance statements apply only to the stated workload, hardware, versions, precision, thread settings, and timing method. Internal kernel improvements do not establish superiority over third-party frameworks. Operator timings do not establish lower training cost at equal model quality. See [benchmark details](BENCHMARKS.md).

#### Privacy, security, and jurisdiction

The supplied command-line build and runtime paths operate locally; the toolchain installer makes explicit network requests through the configured proxy, and optional comparison tools use locally installed libraries. Git hosting, proxies, operating systems, and third-party tools have their own behavior and terms. This document does not claim that every possible dependency or modified installation is telemetry-free.

Do not submit credentials, private datasets, or personal information in public issues or contributions. Treat source programs and state files from others as untrusted. See [security guidance](SECURITY.md).

This document does not select a governing law, require arbitration, or create a separate contractual relationship. It is project information, not legal advice. No independent legal review, ownership audit, patent clearance, or regulatory certification is represented. Seek qualified advice where your distribution or deployment requires it.

### 简体中文

#### 许可证与授权范围

版权署名为 Copyright 2026 tlkppm and HengLang contributors。本仓库发布的原创内容采用 [Apache License 2.0](LICENSE)，归属声明见 [NOTICE](NOTICE)。仓库所有者选择许可证，不代表已经证明对他人内容拥有权利，也不能授予超出相关贡献者有权授予范围的权利。

未修改的英文 `LICENSE` 是许可证正式文本。本双语文件仅作解释，不是独立的最终用户协议，不对 Apache-2.0 授权附加额外限制。说明与许可证不一致时，以许可证为准。

Apache-2.0 在满足其条件的前提下允许商业使用、修改和再分发。再分发义务包括提供许可证、标明被修改的文件、保留适用声明，并按第 4 条保留适用的 NOTICE 归属信息。专利许可受第 3 条规定的范围限制，并含有专利诉讼终止条件。这不表示已经排除全部潜在专利风险。

#### 第三方及生成内容

下载的编译器和随附库属于独立的第三方作品，其许可证不会被 Apache-2.0 替代。生成的程序可能包含 HENG 运行时代码和编译器链接的第三方组件；再分发二进制前，应检查实际链接组件及适用许可。参见[第三方声明](THIRD_PARTY_NOTICES.md)。

数据集、预训练模型、用户输入和独立编写的程序，不会仅因由本软件处理就归本项目所有。请根据自己的用途取得代码、数据、模型权重和输出所需的权利。本项目不对生成内容的所有权或著作权成立条件作统一断言。

#### 保证、责任与适用性

`LICENSE` 第 7、8 条规定适用的保证免责声明与责任限制，同时受强制性法律和另行书面约定约束。本说明不扩大这些免责范围，不排除法定不可放弃权利，也不产生支持、服务等级、安全性或适用性保证。

本框架是独立开发的 CPU 实现。测试不能证明对所有输入均正确、适合生产环境、适合医疗或其他安全关键决策，或符合特定监管制度。部署前应独立评估这些事项。

#### 名称与性能陈述

Apache-2.0 不授予一般商标使用权。项目及第三方名称用于识别相应软件；提及 PyTorch、NumPy、TensorFlow、JAX、Python 或 Zig，不代表存在关联、背书或认证。本文件不声称项目名称已注册为商标。

性能陈述仅适用于明确列出的负载、硬件、版本、精度、线程设置和计时方法。内部内核改进不等于优于第三方框架；算子计时不等于达到同等模型质量时训练成本更低。参见[基准说明](BENCHMARKS.md)。

#### 隐私、安全及法律适用

提供的命令行构建与运行路径在本地工作；工具链安装器通过配置的代理明确发起网络请求，可选比较工具使用本地安装的库。代码托管平台、代理、操作系统和第三方工具各自具有独立行为与条款。本文件不宣称所有可能依赖或修改后的安装都不存在遥测。

请勿在公开问题或贡献中提交凭据、私有数据集或个人信息。其他人提供的源码与状态文件应视为不可信输入。参见[安全说明](SECURITY.md)。

本文件不选择管辖法律，不要求仲裁，也不建立额外合同关系。其内容属于项目信息，不是法律意见，不表示已经完成独立法律审查、权属审计、专利排查或监管认证。分发或部署有相关需要时，应寻求具备资质的专业意见。
