# TaskAudit 核心软件

本仓库只含原创数值核心、公开接口和合成字母标签示例，版本为 0.1.0，
与早期任务 manifest 工具的 0.2.0 版本不同。作者为 Pengyuan Zou 和 Liming Liu。
三个冻结核心保留原字节：H 类别细化、完整 F 成对区分，以及带独立 H/F
分离器的 robust F-floor 优化。公开接口和安装配置不会改写核心。

在 Python 3.11+ 环境执行：

```text
python verify_publication.py
python -m pip install .
python -m taskaudit_core.demo
python -m unittest discover -s tests -v
```

请先对干净源码验包再安装。构建工具可能创建被忽略的 build 或 *.egg-info
目录；它们不在发布白名单中，之后再做严格源码检查会被指出。上传源码时
不要混入这些生成文件。

示例应输出 H=1、上界=1、主目标已认证，并通过独立 F-floor 不可行示例。
首次安装需取得 NumPy/SciPy；之后示例和检查不下载任何数据。
具体 API、输入契约和时限口径见英文 README。

这里不包含任何真实化合物注释表、论文实验结果、论文稿件或外部证据缓存，
不能据此宣称论文全部实验已经公开可复现。完整实证材料仍有来源权利与访问
问题待澄清。本包只验证给定注释任务和预算下的有限计算，不证明生物效果。
Gamma 仅支持 0、1、2，是敏感性预算而非校准后的未知错误率。

依赖版本和实际测试平台在英文 README 与 requirements-lock.txt 中列明；
不宣称已做跨平台测试。原始合成测试枚举完整缺失 cell（含全零块）建立独立
期望值，不用被测 oracle 的输出当“真值”。

发布应严格按 PUBLISH_ALLOWLIST.txt，仅包含清单成员。原始核心哈希见
FROZEN_CORE_MANIFEST.json，全部发布文件哈希见 PUBLICATION_MANIFEST.json。
MIT 仅适用于本仓库的原创代码、文档和合成示例，依赖不随仓库分发。

