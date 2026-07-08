# AIGC Reduce

降低学术论文 AIGC 查重率的 Claude Code Skill，同时增量兼容 Codex。

基于知网 3.0、万方、PaperPass、PaperPure 等主流检测器的技术原理，结合 [humanizer skill](https://github.com/blader/humanizer)（源自 Wikipedia "Signs of AI writing"）的 AI 痕迹识别方法论，通过**三轮降重协议**（去除痕迹 → 注入书面学术特征 → 自检审计）打破 AI 的模板化痕迹。

> **核心立场：降重 ≠ 口语化。** 去 AI 味的默认方向会漂向口语化（因为口语色彩能骗过统计检测器），但学术论文过了机审还要过人审。本 skill 把**保持学术书面语体设为硬底线**，优先级高于修改率目标——降重靠打破模板化句式结构，而不是把论文写得口语、写差一点。

## 安装

### Claude Code

Claude Code 是本 skill 的原始目标环境，推荐安装到：

```bash
git clone https://github.com/xiaofenggan01/aigc-reduce.git ~/.claude/skills/aigc-reduce
```

安装后，当你在 Claude Code 中提到「降 AI 味」「降 AIGC 率」「去 AI 检测」等关键词时，该 skill 会自动触发。

### Codex

Codex 可安装到 Codex 可发现的 skills 目录：

```bash
git clone https://github.com/xiaofenggan01/aigc-reduce.git ~/.agents/skills/aigc-reduce
```

如果你的环境使用 `$CODEX_HOME/skills`：

```bash
git clone https://github.com/xiaofenggan01/aigc-reduce.git "$CODEX_HOME/skills/aigc-reduce"
```

Codex 通过 `SKILL.md` frontmatter 识别触发条件，通过 `agents/openai.yaml` 读取 UI 元数据。Claude Code 可以忽略 `agents/openai.yaml`。

## 为什么需要这个工具？

AIGC 检测器已经在各大高校强制使用。但检测技术存在根本性局限：

- **写得好 = 判 AI？** 高质量流畅论文假阳性率 6.5%，有口语化瑕疵的论文仅 1.8%
- **跨平台结果不一致：** 同一篇论文知网 AI 率 35%，万方仅 12%
- **术语密集型论文受害：** 术语密度 >20% 时假阳性率 8-12%
- **检测器依赖浅层特征：** DPO 训练可使检测准确率下降 60%

核心思路：不是帮人作弊，而是帮**本来是人写的论文**降低被误判的风险。

## 四条铁律

1. **禁止 AI 全量重写** — AI 重写 AI 文本 = 叠加 AI 指纹（PaperPure 实测：重写后检测率升至 100%）
2. **修改率 >40%，但只能靠限定手段达标** — 贡献只能来自结构性改写（语序、句式、主被动、长句拆分）和模板句去除；**禁止靠口语化、情绪化、无意义换词凑数**。当"达 40%"与"保语体"冲突时，语体优先。
3. **确定性替换** — 每次只改一小处，不经过 LLM 的 token 采样
4. **保持学术语体（硬底线）** — 降重后绝不出现网络用语、情绪词、口语表达、破折号泛滥；绝不为降重编造原文没有的事实、数据或文献。正确方向是打破模板化结构，而非破坏语体。

## 三轮降重协议

### 第一轮：去除 AI 痕迹（减法）

扫描文本 → 词级替换 → 句级重构 → 段落调整，确保修改率 >40%。

替换操作覆盖三个层级：
- **词级（10-15%）**：模板词、72 个中文 AI 高频词、动词的系统性替换
- **句级（15-20%）**：语序重组、长句拆分、被动改主动、主语更换
- **段落级（10-15%）**：对称段长打破、段间过渡、语义注入

### 第二轮：注入书面学术特征（加法）

> 只去除 AI 痕迹不够，句长过匀的干净文本仍像 AI。但注入的必须是**书面学术**特征，不是口语特征——注入方向错了就会把论文改成说明文。

- **节奏工程**：长短句自然交错（目标 CV ≈ 0.45），但每句仍是完整书面陈述句，不拆成电报体碎句
- **审慎推断（书面）**："这一结果可能源于…""该趋势有待更多样本验证"；禁止"我觉得""说白了"类口语表达
- **操作细节补充**：只据实补全原文已含的观察/参数，绝不凭空编造
- **声音校准**：如果用户提供参考文本，分析并匹配其写作风格（参考文本正式则保持正式）

### 第三轮：Anti-AI 审计（自检）

审视文本，回答"这段话为什么还像 AI 写的？"，逐项排查深度 AI 痕迹模式：

1. 重要性膨胀（"至关重要""不可忽视"）
2. 同义词轮换（同一概念被叫了 3 个以上名字）
3. 三板斧强迫症（连续出现"A、B和C"三件套）
4. 系词回避（AI 不用"是"，偏用"作为/体现/代表"）
5. 模糊归因（"有研究表明"无引用）
6. 公式化挑战段（"尽管…但仍存在局限性"）
7. 悬浮式分析（一句话挂了 2+ 个"从而/进而"）
8. 空洞结论（"具有良好的应用前景"）
9. 破折号过度使用
10. 虚假范围（"从宏观到微观"）
11. 成对转折收束（连续使用"不是…而是…"、"但至少…"、"不代表…"）

并新增三项**语体守门**自检（命中即改回书面学术表达，优先于修改率）：

- □ 是否出现口语化 / 网络用语 / 情绪词？
- □ 破折号是否每段 ≤1 个？
- □ 是否为降重补进了原文没有的事实、数据或文献？

## 自动化扫描

内置 `aigc_scan.py` 脚本，自动扫描文本的 9 个维度（前 7 维查 AI 痕迹，后 2 维查"降重过度"）：

```bash
python3 scripts/aigc_scan.py your_paper.txt
```

| 维度 | 对应检测器 / 用途 |
|------|-----------|
| 模板句式密度 | 知网语义指纹 |
| 突发性 CV | GPTZero 核心算法 |
| 段落对称性 | 知网格式规范性校验 |
| 嵌套编号 | 统计检测 |
| 冒号并列 | 知网逻辑断层检测 |
| 被动语态 | Turnitin 句式特征分析 |
| 标点规律 | 万方 BERT 语义分析 |
| **口语化 / 网络用语** | **降重过度预警（守学术语体）** |
| **破折号密度** | **降重过度预警（每段 ≤1 个）** |

> 后两个维度不是 AI 痕迹，而是"降重把语体改坏"的门禁：命中即提示对照 `positive-style-academic.md` 改回书面学术表达。

## 文件结构

```
aigc-reduce/
├── SKILL.md                            # Skill 核心（渐进式披露入口）
├── README.md                           # 本文件
├── agents/
│   └── openai.yaml                     # Codex UI 元数据，Claude Code 可忽略
├── references/
│   ├── positive-style-academic.md      # 降重正向标准 + 三级对照（守语体核心）
│   ├── protected-spans.md              # 受保护片段预检（引用/公式/数据/术语）
│   ├── replacement-tables.md           # 替换表 + AI 高频词 + 口语化负面清单
│   ├── ai-patterns.md                  # 深度 AI 痕迹识别模式
│   └── detection-principles.md         # 检测器技术原理与弱点
├── evals/
│   └── benchmark.md                    # 双向 benchmark（SF 该改 / SNF 不该误杀）
├── scripts/
│   └── aigc_scan.py                    # 自动化扫描脚本（9 维度）
└── tests/
    └── test_aigc_scan.py               # 扫描脚本单元测试
```

## 检测器参考数据

### 国内产品

| 产品 | 综合准确率 | 假阳性率 | 核心检测维度 |
|------|-----------|---------|-------------|
| 知网 3.0 | 98.6% | 1.2% | 语义指纹、格式规范性 |
| 万方 | ~95% | ~3% | 多模型融合 |
| PaperPass | ~97% | 0.3% | 语言风格一致性 |

### 国际产品

| 产品 | 综合准确率 | 假阳性率 | 中文支持 |
|------|-----------|---------|---------|
| Turnitin | 98%（英文）/ 85%（中文） | 2.1% | 一般 |
| GPTZero | 90-99% | 16% | 良好 |
| Copyleaks | 99%+ | 未公开 | 良好 |

## 致谢

方法论综合了以下来源的实证研究：

- 《论文AIGC查重检测方法与原理深度研究报告》(2026-05)
- [Wikipedia "Signs of AI writing"](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)（WikiProject AI Cleanup 维护）
- [Humanizer skill](https://github.com/blader/humanizer) — AI 痕迹识别 + 二次自检方法论
- LINUX DO 论坛社区验证的 Prompt 工程
- 微博博主"厄加特特"的句式变换方法论
- 火山引擎开发者社区发布的降重 Prompt
- GitHub 开源项目：BypassAIGC、Humanizer-zh、humanize-chinese
- [shuorenhua（说人话）](https://github.com/MrGeDiao/shuorenhua) — 正向风格合同、受保护片段、双向 benchmark 的设计思路
- PaperPure 实测反馈

## License

MIT
