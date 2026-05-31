# CCL25-Eval

古典诗词理解与推理评测系统。基于 Qwen3-8B + QLoRA 渐进式微调，配合 Reasoner-Formatter 两阶段 Harness 实现结构化推理，参加 CCL25 竞赛。

## 当前最佳成绩

**模型**：BC8-v2（Qwen3-8B + 4-bit NF4 QLoRA，混合蒸馏微调，43.6M 可训练参数 / 8.2B 总参数）

**官方评测**（`submit.json`，327 样本）：

| score | sim_sents | emo_acc | bleu_sents | taskA | sim_words | taskB | bleu_words |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **0.6915** | 0.9100 | 0.8130 | 0.2530 | 0.8130 | 0.8700 | 0.5690 | 0.2450 |

- **Task A**（词义解释）：0.813，由 `sim_words + bleu_words` 组合计算
- **Task B**（句子翻译）：0.569，由 `sim_sents + bleu_sents` 组合计算
- **情感准确率**：0.813（A/B/C/D 四选一）
- **总分** = 0.5 × taskA + 0.5 × taskB = 0.6915

---

## 1. 核心设计

### 1.1 结构化输出范式

拒绝让模型在多选题面前进行自由发散的长链推理（Free-form CoT）。所有解题过程被强制标准化为三步：**抽取词义、翻译句子、抽象情感判断**。每个步骤的输出都是结构化短字段（`meaning`、`translation`、`rationale` 等），可审计、可映射回原题，不包含完整推理链。

> **引用**：[`docs/spec/data-schema.md`](docs/spec/data-schema.md) §3 定义了 Reasoner 中间输出 Schema；[`docs/contract/teacher-data.md`](docs/contract/teacher-data.md) §8 详述禁止自由长 CoT 的约束与理由。

### 1.2 Two-Stage Harness：做题者与判卷者分离

如果模型在推理阶段直接看到 A/B/C/D 选项，极易为了迎合选项而扭曲自己的诗词理解（Local-cue Overfit，局部线索过拟合）。因此，采用"真正的评测台（True Harness）"双阶段隔离设计：

```
Reasoner（Stage 1）                 Formatter / Local Mapper（Stage 2）
─────────────────────              ───────────────────────────
输入：题目（不含选项）               输入：Reasoner 输出 + 选项
输出：evidence + sentiment         输出：最终 JSON（含 choose_id）
      + draft_answer
      不输出 choose_id
```

- **Reasoner**：只负责写结构化证据（词义线索、句意骨架、情感判断）和生成草稿答案，**绝对不自己输出 `choose_id`**
- **Formatter / Local Mapper**：根据 Reasoner 的 `sentiment` 分析与原始选项进行语义匹配，生成最终的 `choose_id`

两条执行路径：

| 模式 | 流程 | 适用场景 |
| --- | --- | --- |
| **H1** | Reasoner → 规则后处理（`fallback_final`） | 无 formatter 模型，轻量级 |
| **H2** | Reasoner → Formatter 模型重写 → 规则兜底 | 有额外模型协助修正格式和映射情感 |

**结论**：H2 formatter 对所有 checkpoint 的 net_gain 均为 0，选择 **H1**（规则后处理）即可。

> **引用**：[`docs/spec/harness.md`](docs/spec/harness.md) 定义完整的 Reasoner→Formatter 契约、Local Validator 规则和 Fallback 策略；[`src/harness.py`](src/harness.py) 是核心实现（`run_harness_once`、`validate_reasoner_output`、`fallback_final`、`map_sentiment_to_choice` 等）。

### 1.3 渐进式微调策略

微调不是简单的一波流，而是三个相互承接的阶段：

```
B8 (格式对齐) ──→ BC8 (能力提升) ──→ BC8-final (格式修复)
```

- **B8**：纯 answer-only 数据，让基座模型强行对齐目标 JSON 边界，学会输出合法、完整、简洁的 JSON
- **BC8**：混合蒸馏（50% answer-only / 25% short-evidence / 25% teacher-critique），在保持格式稳定性的同时注入结构化思维与纠错能力
- **BC8-final**：短轮次 answer-only replay，压制混合蒸馏后可能产生的格式漂移

> **引用**：[`docs/plans/training-plan.md`](docs/plans/training-plan.md) 定义各阶段训练目标、数据配方、超参和验收标准。

---

## 2. 数据 Schema

本项目所有模块（prompt baseline、teacher 数据生成、训练、harness、评测）共用一套统一数据契约。

### 2.1 统一输入 Schema

```json
{
  "idx": 0,
  "title": "李端公",
  "author": "卢纶",
  "content": "故关衰草遍，离别自堪悲。...",
  "qa_words": ["衰草", "故关", "风尘"],
  "qa_sents": ["故关衰草遍，离别自堪悲", "掩泪空相向，风尘何处期。"],
  "choose": { "A": "欢快的重逢", "B": "仕途的无奈", "C": "对未来的期待", "D": "惜别的感伤" }
}
```

### 2.2 最终输出 Schema

```json
{
  "idx": 0,
  "ans_qa_words": { "衰草": "枯黄的草，烘托荒凉离别的氛围" },
  "ans_qa_sents": { "故关衰草遍，离别自堪悲": "旧关一带长满枯草，分别本就令人悲伤" },
  "choose_id": "D"
}
```

### 2.3 训练与评测输入的差异

- **训练集**（`data/train-data/`，164 首）：`choose = {}`，`choose_id` 不作为训练目标。模型只学习词义、句译和情感分析（sentiment），不学习选项选择。
- **评测集**（`data/eval_data.json`，327 首）：包含完整 `choose` 选项（A/B/C/D），但无标准答案。

这决定了本项目的两阶段情感管线：Reasoner 分析情感（不输出 `choose_id`），Formatter 将情感映射到选项。

### 2.4 情感标签受控词汇表

`sentiment.primary` 和 `sentiment.secondary` 必须从以下受控词汇表中选择，不得自由发挥。共 8 大类 24 小类 + "其他"：

| 类别 | 可用标签 |
| --- | --- |
| 离别 | 惜别感伤、送别不舍、离别愁绪 |
| 思乡 | 思乡怀远、羁旅思归、故园之思 |
| 忧国 | 忧国伤时、报国壮志、兴亡之叹 |
| 山水田园 | 山水闲适、田园之乐、隐逸情怀 |
| 怀古 | 怀古伤今、历史沧桑、昔盛今衰 |
| 爱情闺怨 | 相思闺怨、爱情甜蜜、相思之苦 |
| 人生感慨 | 人生无常、时光易逝、仕途失意 |
| 边塞战争 | 边塞征战、将士艰辛、厌战思归 |

> **引用**：[`docs/spec/data-schema.md`](docs/spec/data-schema.md) 完整定义了统一输入、最终输出、Reasoner 中间输出三种 Schema，以及字段映射规则、边界字段处理（空字段、重复词句、缺失选项等）；[`src/schema.py`](src/schema.py) 实现了 `normalize_input`、`validate_input`、`validate_output` 等核心校验函数。

---

## 3. 评测体系

### 3.1 JSON 错误分类

| 类别 | 定义 | 典型错误 |
| --- | --- | --- |
| **Core Error**（输出不可用） | 无法用于下游评分或提交 | `parse_error`、`missing_top_field`、`idx_mismatch`、`missing_word_key`、`missing_sentence_key`、`empty_required_answer`、`invalid_choose_id` |
| **Format Error**（输出不干净） | Parser 已成功剥离，JSON 内容可用 | `extra_text`、`thinking_trace_leak`、`extra_top_field`、`overlong_word_answer`、`overlong_sentence_answer` |

汇总指标：
- `json_error_rate`：出现任一 Core 错误的样本比例
- `hard_json_error_rate`：出现 parse_error / missing_top_field / idx_mismatch / wrong_field_type / invalid_choose_id 的样本比例
- `format_style_error_rate`：出现任一 Format 错误的样本比例

### 3.2 两阶段情感评估

情感评估拆分为两个独立指标：
- **Reasoner 情感分析准确率**：`sentiment.primary` 是否正确反映诗歌情感（用于诊断和消融）
- **Formatter 情感映射准确率**：sentiment → choose_id 映射是否正确（计入最终提交得分）

### 3.3 Formatter 改坏率

统计 Formatter 是否把本来正确的草稿改成错误最终输出。核心指标：
- `formatter_regression_rate`：任一子任务或总分退化的样本比例
- `formatter_net_gain`：`mean(final_total_score - draft_total_score)`

> **引用**：[`docs/contract/eval-plan.md`](docs/contract/eval-plan.md) 完整定义了评测原则、实验结果表字段、JSON 错误分类、Formatter 改坏率统计、分任务错误分析模板和消融实验设计；[`src/eval.py`](src/eval.py) 实现了 `parse_json_object`、`classify_json_errors`、`compute_json_error_rates`、`compute_formatter_regression` 等核心评测函数。

---

## 4. 数据管线

### 4.1 数据源

| 数据源 | 路径 | 样本数 | 词义标签 | 句译标签 | 情感选项 |
| --- | --- | ---: | --- | --- | --- |
| 训练原始数据 | `data/train-data/` | 164 | keywords（高置信） | 无句级对齐，不可直接使用 | 无 |
| 评测数据 | `data/eval_data.json` | 327 | 无 gold label | 无 gold label | 有（A/B/C/D） |
| Dev Split（评测） | `data/splits/eval50.json` | 50 | — | — | 有 |
| Dev Split（训练） | 从训练数据划分 | ~30 | — | — | 无 |

### 4.2 Teacher 数据生成

使用 DeepSeek-V4-Flash API 为训练集生成 teacher 数据。两种类型：

| 类型 | 内容 | 用途 |
| --- | --- | --- |
| **Short-evidence** | `evidence` + `sentiment` + `draft_answer` | 学习词义分析、句译骨架和情感判断 |
| **Teacher-critique** | `critique` + `correction_evidence` + `corrected_answer` | 学习识别和修正常见错误（词义误解、情感标签偏差等） |

训练集 Teacher 输出不含 `choose_id`（因为训练集原本无选项），仅生成 `sentiment` 情感分析。

> **引用**：[`docs/contract/teacher-data.md`](docs/contract/teacher-data.md) 定义了 Teacher prompt 模板、输出 Schema、自动过滤规则和人工抽查 checklist。

### 4.3 自动过滤

生成后经 `filter_teacher_data.py --strict` 过滤：
- JSON 可解析，必填字段完整
- `sentiment.primary` 在受控词汇表内（100% 合规）
- `ans_qa_words` 和 `ans_qa_sents` key 覆盖 100% 目标词/句
- 长度不超标（词义 ≤ 80 字，句译 ≤ 180 字）
- 无自由 CoT 残留、Markdown 或提示词泄漏

> **引用**：[`src/cli/generate_teacher_data.py`](src/cli/generate_teacher_data.py) 调用 DeepSeek API 生成 teacher 数据；[`src/cli/filter_teacher_data.py`](src/cli/filter_teacher_data.py) 执行自动过滤。

### 4.4 训练数据组装

| 数据集 | 内容 | 样本数 | 比例 |
| --- | --- | --- | --- |
| `b8-answer-only.jsonl` | 输入题目 → 最终 JSON | 164 | 100% answer-only |
| `bc8-mixed/train.jsonl` | 混合蒸馏 | 219 | 50% answer-only / 25% short-evidence / 25% teacher-critique |

> **引用**：[`docs/plans/data-pipeline.md`](docs/plans/data-pipeline.md) 定义了从原始数据到训练就绪数据集的完整管线；[`src/cli/build_training_data.py`](src/cli/build_training_data.py) 组装训练数据。

---

## 5. 训练流程

### 5.1 Prompt Baseline（P8 / P14）

在微调前评估多个 prompt-only 模型（不微调），考察纯 Prompt 驱动下模型的理解上限与格式服从度，为后续微调提供参照。

| 实验 | JSON Error Rate（eval50） |
| --- | --- |
| P14-qwen3-14b-bf16-nothink-zero | 0% |
| P14-fast-qwen3-14b-awq4-nothink-zero | 0% |
| P8-qwen3-8b-awq4-nothink-zero | 0% |
| P8-qwen3-8b-bf16-nothink-zero | 4% |
| P8-qwen3-8b-bf16-think-zero | 4% |

**选型决断**：确定以 **Qwen3-8B** 为主力基座（具备良好的基础古文素养，在严格 JSON 约束下表现出一定脆弱性，这正是微调需要解决的）。

> **引用**：[`docs/contract/prompt-baseline.md`](docs/contract/prompt-baseline.md) 定义 P14/P8/FMT baseline 实验协议和 Prompt 模板；[`src/cli/run_baseline_matrix.py`](src/cli/run_baseline_matrix.py) 运行 prompt baseline 矩阵评测；[`src/cli/eval_p14.py`](src/cli/eval_p14.py) 运行 P14 全量评测。

### 5.2 B8：Answer-only QLoRA（格式适配）

**目标**：将 Qwen3-8B 适配到 CCL25 的最终输出格式，保证合法、完整、简洁的 JSON。

- 164 个 answer-only 样本，plain-text prompt + JSON target + EOS
- QLoRA 4-bit NF4，LoRA r=16 alpha=32
- 学习率 5e-5，2 epochs，有效 batch=16，max_steps=22

**结果**：eval50 JSON error 2%，0% hard error。但每个样本都带 `extra_text`（思考文本泄漏），format_style_error_rate = 100%。

**B8 条件执行规则**：B8 不是必经阶段，P8 baseline 完成后根据 Core 错误率决定：

| P8 Core 错误率 | B8 执行策略 |
| ---: | --- |
| < 5% | B8 缩短为格式确认（少量 steps / 0.3 epoch） |
| 5% – 15% | 执行完整 B8 |
| > 15% | 完整 B8，提高 answer-only 数据比例和 epoch 数 |

> **引用**：[`src/cli/train_b8.py`](src/cli/train_b8.py) B8 训练脚本。

### 5.3 BC8：Mixed Distillation（能力提升）

**目标**：在 B8 基础上提升古诗词理解和情感分析能力，同时不牺牲格式稳定性。

- 从 B8 checkpoint 继续训练
- 50% answer-only + 25% short-evidence + 25% teacher-critique
- 加权采样，pre-generate `total_steps × effective_batch` 样本对
- 学习率 5e-5，2 epochs，max_steps=33

**结果**：eval50 JSON error 2%，0% hard error，0% format error。输出干净，无思考文本泄漏。

### 5.4 BC8-final：Answer-only Replay（格式修复）

**目标**：在最佳 BC8 checkpoint 上做短周期 answer-only 重训，清理格式错误，逼近 0% JSON error。

- 从 BC8-v2 adapter 继续训练
- 纯 answer-only 数据，低学习率 1e-5
- max_steps=33（164 样本，3 epochs）

**结果**：BC8-final-v3 JSON error 上升至 6%（eval50 Harness H1）。Replay 反而不如 BC8-v2——模型对 harness 的 zero-shot prompt 更敏感，泛化能力下降。最终提交选用 **BC8-v2**。

> **引用**：[`docs/plans/training-plan.md`](docs/plans/training-plan.md) 详细定义各阶段 QLoRA 超参、数据混合比例、输出格式约束和早停策略；[`src/cli/train_bc8.py`](src/cli/train_bc8.py)、[`src/cli/train_replay.py`](src/cli/train_replay.py) 分别为 BC8 和 replay 训练脚本。

---

## 6. Harness 推理架构

### 6.1 架构概览

```
Reasoner 输出
    │
    ▼
Local Validator（确定性规则校验）
    │
    ├── 输出合法且满足跳过条件 ──→ 规则 postprocess ──→ Final Validator ──→ 最终 JSON
    │
    ├── 可解析但有缺项/超长/冲突 ──→ Formatter（H2 模式）──→ Final Validator ──→ 最终 JSON
    │
    └── 无法解析或关键字段缺失 ──→ Retry Reasoner（1次）
            │
            ├── 成功 ──→ 继续正常流程
            └── 仍失败 ──→ Fallback（规则兜底：提取可解析字段 + 空占位 + sentiment 关键词匹配 choose_id）
```

### 6.2 跳过 Formatter 的条件

- Reasoner 输出是合法 JSON
- `idx` 正确，`draft_answer` 字段完整
- 目标词/句 key 100% 覆盖，无空值
- 无超长字段、无冲突标记
- 任务无 `choose` 选项（此时无需情感映射）

### 6.3 eval50 Harness 结果

| Checkpoint | H1 Core Error | H2 Final Error | Formatter net_gain |
| --- | --- | --- | --- |
| **BC8-v2** | 2% (1/50) | 2% (1/50) | 0.00 |
| B8-v2 | 0% | 0% | 0.00 |
| BC8-final-v3 | 6% (3/50) | 4% (2/50) | +0.02 |

**结论**：H2 formatter 对所有 checkpoint 的 net_gain 均为 0 → 选择 **H1**（规则后处理）即可。

> **引用**：[`docs/spec/harness.md`](docs/spec/harness.md) 完整定义 Reasoner 输出 Schema、Local Validator 规则、Fallback 策略和重试策略；[`src/harness.py`](src/harness.py) 实现核心 Harness 逻辑（`run_harness_once`、`validate_reasoner_output`、`decide_next_action`、`fallback_final`、`map_sentiment_to_choice`）；[`src/cli/run_harness.py`](src/cli/run_harness.py) 和 [`src/cli/run_submission_eval.py`](src/cli/run_submission_eval.py) 分别为 True Harness 和 Submission Runner 的 CLI 入口。

---

## 7. 全量评测与提交

### 7.1 全量评测（327 samples, eval_data.json）

| Checkpoint | Core Error | Hard Error | Format Error | 速度 |
| --- | --- | --- | --- | --- |
| **BC8-v2** | **1.83% (6/327)** | **0.61% (2/327)** | **2.14% (7/327)** | 8.8s/sample |
| B8-v2 | 2.14% (7/327) | 0.92% (3/327) | 100% (327/327) | 69s/sample |

B8 的 format error 100% 是因为模型生成了大量思考文本，parser 虽能成功剥离 JSON，但不干净。BC8 混合蒸馏后输出干净。

### 7.2 提交

`generate_submit.py` 用 BC8-v2 对 327 样本推理，生成 `submit.json`。2 个解析失败的样本（idx=89, 255）手动从 raw output 中修复 JSON 结构后补全。

> **引用**：[`src/cli/generate_submit.py`](src/cli/generate_submit.py) 生成最终提交格式。

---

## 8. 代码结构

```
src/
├── schema.py                 # 统一输入/输出 Schema 校验与归一化
├── eval.py                   # 错误分类、评测记录、Formatter 改坏率计算
├── harness.py                # Harness 核心（Reasoner 校验、Formatter 调度、Fallback）
├── baseline.py               # Prompt 渲染与 baseline 实验记录
└── cli/
    ├── train_b8.py           # B8 answer-only QLoRA 训练
    ├── train_bc8.py          # BC8 混合蒸馏训练
    ├── train_replay.py       # BC8-final answer-only replay
    ├── run_baseline_matrix.py # Prompt baseline 矩阵评测
    ├── run_baseline_smoke.py  # Baseline smoke test
    ├── run_harness.py         # True Harness 推理（evidence+sentiment+draft）
    ├── run_submission_eval.py # Submission runner 评测
    ├── generate_teacher_data.py # Teacher 数据生成（DeepSeek API）
    ├── filter_teacher_data.py   # Teacher 数据自动过滤
    ├── build_training_data.py   # 训练数据集组装
    ├── generate_candidates.py   # 合成扰动生成候选错误样本
    ├── generate_submit.py       # 生成 submit.json
    ├── eval_p14.py              # P14 prompt-only 14B 全量评测
    ├── eval_b8.py / eval_bc8.py / eval_bc8_final.py  # 各阶段 checkpoint 评测
    └── inspect_b8_results.py    # B8 结果检查
tests/
├── test_schema.py            # Schema 校验单元测试
├── test_eval.py              # 评测函数单元测试
├── test_harness.py           # Harness 逻辑单元测试
├── test_baseline.py          # Baseline 工具单元测试
├── test_filter_teacher_data.py # Teacher 数据过滤测试
└── test_sync_protection.py   # 同步保护测试
```

---

## 9. 文档导航

### `docs/spec/` — Schema 与架构契约
- [`data-schema.md`](docs/spec/data-schema.md) — 统一输入/输出/Reasoner 中间输出 Schema 定义
- [`harness.md`](docs/spec/harness.md) — Reasoner-Formatter 两阶段 Harness 契约

### `docs/contract/` — 评测协议与数据规范
- [`eval-plan.md`](docs/contract/eval-plan.md) — 评测原则、错误分类、Formatter 改坏率、消融实验设计
- [`prompt-baseline.md`](docs/contract/prompt-baseline.md) — P14/P8/FMT Prompt Baseline 实验协议
- [`teacher-data.md`](docs/contract/teacher-data.md) — Teacher 数据生成 Prompt、Schema、过滤规则

### `docs/plans/` — 执行计划与路线图
- [`execution-plan.md`](docs/plans/execution-plan.md) — 从零到提交的完整分步执行计划
- [`training-plan.md`](docs/plans/training-plan.md) — B8/BC8/Replay 训练配置与超参
- [`data-pipeline.md`](docs/plans/data-pipeline.md) — 数据构造管线总览
- [`bcd-plan.md`](docs/plans/bcd-plan.md) — BCD 循环 Block 消融计划
- [`agent-task-list.md`](docs/plans/agent-task-list.md) — Agent 任务拆解列表

### `docs/reports/` — 实验报告与调研
- [`model-selection-report.md`](docs/reports/model-selection-report.md) — 模型选型调研与决断
- [`baseline-smoke-results.md`](docs/reports/baseline-smoke-results.md) — Baseline smoke test 结果
- [`data-inspection.md`](docs/reports/data-inspection.md) — 数据 Schema 规范性校验报告
- [`e3-dev50-results.md`](docs/reports/e3-dev50-results.md) — E3 dev50 评测结果
- [`e4-gate-report.md`](docs/reports/e4-gate-report.md) — E4 Gate 决策报告
- [`model-research.md`](docs/reports/model-research.md) — 候选模型调研笔记

### `docs/agents/` — Agent 工作流
- [`domain.md`](docs/agents/domain.md) — 项目领域上下文
- [`triage-labels.md`](docs/agents/triage-labels.md) — Issue 分类标签规范
- [`issue-tracker.md`](docs/agents/issue-tracker.md) — Issue 追踪配置

### `docs/workspace-state.md`
**高频变动文档**。追踪当前远程服务器上的 checkpoint、数据制品和已知漂移。开发与验证前建议优先查阅。

---

## 10. 关键决策记录

- **BC8-v2 > BC8-final-v3**：Replay 训练使 eval50 harness 错误率从 2% 升至 6%。混合蒸馏的泛化能力优于纯 answer-only replay。
- **H1 > H2**：Formatter 对所有 checkpoint 的 net_gain = 0。规则后处理更简单、同样有效。
- **Prompt 格式一致很重要**：训练用 `render_prompt_text`（plain-text + EOS），harness 用不同的 zero-shot prompt，导致 BC8-final-v3 产生 4pp 的误差差距。
- **单 GPU 推理更快**：`device_map={"": "cuda:0"}` 比双卡 auto 分配快，BC8-v2 推理 8.8s/sample vs B8 的 69s/sample（后者因生成长段思考文本更慢）。
- **B8 不宜直接提交**：虽然 B8 的 core error 更低（0% vs 2% harness H1），但 100% format error（思考文本泄漏）使输出不干净；BC8 的输出格式干净得多。

---

## 11. 环境与运行

### 包管理

使用 `uv` 作为包管理器：
- 轻量级/跨平台依赖：`uv add <包名>`，然后 `python3 remote_run.py uv pip install -e .` 同步到服务器
- GPU 依赖（vllm, torch, transformers 等）：写在 `requirements-remote.txt` 中，通过 `python3 remote_run.py uv pip install -r requirements-remote.txt` 安装
- **禁止** `remote_run.py uv sync`：会删除远程 GPU 包

### 远程执行

所有 GPU 代码通过 `python3 remote_run.py <command>` 在 Linux 服务器上运行。代码通过 `sync.sh` 自动同步。

### 本地运行

```bash
# 运行单元测试
uv run python -m pytest tests/

# 运行特定测试
uv run python -m unittest tests.test_harness
```

> **引用**：[`CLAUDE.md`](CLAUDE.md) 包含完整的开发环境配置和项目约定。

---

## 12. 后续计划

- **翻译任务提升**：TaskB = 0.569 是当前瓶颈，需更多高质量翻译训练数据或更大 teacher 模型
- **BCD 消融**（[`docs/plans/bcd-plan.md`](docs/plans/bcd-plan.md)）：中间 1/3 Transformer Block 循环，提升推理深度。前置条件：BC8-final 稳定、harness 评测完备
