## Security guidance / 安全说明

### English

#### Security scope

This project has not undergone an independent security audit. The current verification target is the repository's Windows x86-64 CPU implementation. There is no promised security-support period, response deadline, or supported historical-version matrix. Check the current repository state when investigating a report.

The compiler generates and executes native machine code. It is a development tool, not an isolation boundary for untrusted programs. Use operating-system isolation appropriate to the risk, and do not run unknown source code with sensitive files, network access, or elevated privileges.

State-file validation checks format, sizes, metadata, and corruption. The state checksum is not a digital signature, message authentication code, encryption, or proof of trustworthy origin. Use only state files from trusted sources, apply operating-system resource limits where necessary, and keep independent backups before overwriting valuable state.

Toolchain downloads require the configured proxy, normal HTTPS verification, a pinned version, and a pinned SHA-256. Do not disable certificate checks or replace a failed download with an unverified executable. A matching hash establishes consistency with the pinned artifact, not the absence of vulnerabilities.

#### Reporting

Start at [the repository's security page](https://github.com/tlkppm/HengLang/security). If GitHub offers private vulnerability reporting for this repository, use it. This document does not assert that the owner has enabled that feature.

If no private channel is available, open an issue containing only a request for a private reporting channel and a non-sensitive high-level description. Wait for an agreed private channel before sending exploit code, vulnerable private data, or operational details. No private contact address or guaranteed response time is advertised here.

Useful report details, once a safe channel exists, include the affected commit, operating-system and compiler versions, minimal synthetic reproduction, expected and observed behavior, impact, and a proposed fix if available. Remove credentials, tokens, usernames, private paths, and personal or proprietary data.

### 简体中文

#### 安全范围

本项目尚未接受独立安全审计。当前验证目标为仓库中的 Windows x86-64 CPU 实现。项目未承诺安全支持期限、响应时限或历史版本支持矩阵。调查问题时应核对仓库当前状态。

编译器生成并执行原生机器码，属于开发工具，不是运行不可信程序的隔离边界。请根据风险采用适当的操作系统隔离措施，不要让未知源码接触敏感文件、网络访问权限或高权限环境。

状态文件校验检查格式、大小、元数据与损坏情况。文件校验和不是数字签名、消息认证码、加密或来源可信证明。仅使用可信来源的状态文件，必要时施加操作系统资源限制，覆盖有价值的状态前保留独立备份。

工具链下载要求配置代理、正常 HTTPS 验证、固定版本和固定 SHA-256。请勿关闭证书校验，也不要以未经验证的可执行文件替代失败的下载。哈希匹配仅证明与固定发行包一致，不代表不存在漏洞。

#### 问题报告

请先查看[仓库安全页面](https://github.com/tlkppm/HengLang/security)。若 GitHub 为本仓库提供私密漏洞报告入口，请使用该入口。本文件不表示仓库所有者已经启用这一功能。

如果没有可用私密渠道，可创建仅包含私密联系渠道请求和非敏感概述的问题。在双方确定私密渠道前，请勿发送利用代码、存在漏洞的私有数据或运行细节。本文件不公布未经确认的私密邮箱，也不承诺响应时间。

确定安全渠道后，有用的信息包括受影响提交、操作系统与编译器版本、最小合成复现、预期与实际行为、影响及可选修复建议。请移除凭据、令牌、用户名、私有路径、个人信息和专有数据。
