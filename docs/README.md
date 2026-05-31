# CCL25-Eval: 古诗词理解评测与微调架构

本项目旨在为 **CCL25 古诗词理解** 挑战提供一整套高度模块化、防退化且可复现的 LLM 评测与微调架构。针对大语言模型在严苛格式要求下的 JSON 崩溃问题，以及在多项选择题中容易“生搬硬套”选项文本（Local-cue Overfit）的缺陷，本项目提出并实现了一套从数据合成、模型选型到混合蒸馏与两段式评测的工业级 Pipeline。

---

## 1. 核心架构与设计哲学 (Core Architecture & Design Philosophy)

### 1.1 摒弃长 CoT，走向结构化输出
我们拒绝让模型在多选题面前进行无法收敛的“自由散漫长推理（Free-form CoT）”。所有的解题过程被强制标准化为三个步骤：**抽取词义、翻译句子、抽象情感判断**。
*   **文档引用**：[`docs/spec/data-schema.md`](spec/data-schema.md) 定义了严格的输入与中间态 JSON 格式。

### 1.2 Two-Stage True Harness：做题者与判卷者分离
如果模型在推理阶段直接看到 A/B/C/D 选项，极易为了迎合选项而扭曲自己的诗词理解（局部线索过拟合）。因此，我们采用了“真正的评测台（True Harness）”双阶段隔离设计：
1.  **Stage 1 (Reasoner)**：模型只负责写结构化证据（短句解析）并生成**抽象的情感分析（Sentiment）**，绝对不自己吐出 `choose_id`。
2.  **Stage 2 (Formatter)**：本地轻量级规则或 Formatter 根据提取出的情感特征与原始选项进行映射，输出最终的 JSON 和选项 ID。
*   **文档引用**：[`docs/spec/harness.md`](spec/harness.md)
*   **代码实现**：[`src/harness.py`](../src/harness.py), [`src/cli/run_harness.py`](../src/cli/run_harness.py)

---

## 2. 基线评测与模型选型 (Baseline Testing & Model Selection)

在微调前，项目进行了充分的基线评测，考察了纯 Prompt 驱动下模型的理解上限与格式服从度。
*   **基线测试矩阵**：在 P8（8B 级参数）与 P14（14B 级参数）区间进行了零样本/少样本能力试探。
*   **选型决断**：通过 `model-selection-report.md`，我们确定以 **8B 量级模型（如 Qwen2.5-8B）** 为主力基座。其具备良好的基础古文素养，但在严格 JSON 约束下表现出了一定的脆弱性。这直接催生了我们后续的 QLoRA 格式对齐与微调计划。
*   **相关入口**：
    *   [`docs/reports/model-selection-report.md`](reports/model-selection-report.md)
    *   [`docs/contract/prompt-baseline.md`](contract/prompt-baseline.md)

---

## 3. 数据合成与流水线 (Data Synthesis & Pipeline)

由于开源社区极缺高质量的“诗词理解中间态推理过程”，本工程构建了一套无损的数据蒸馏与合成管线：
1.  **强模型蒸馏**：通过调用诸如 DeepSeek-V4-Flash 的 API，让其充当 Teacher 模型，对题库输出 `short-evidence` (短推理证据) 以及针对干扰项的 `teacher-critique` (纠错批评)。
2.  **合成微调集**：加入正负样本扰动，强迫模型学习识别错误答案。
3.  **清洗过滤**：剔除 Teacher 模型产生的幻觉与格式错乱数据。
*   **相关入口**：
    *   [`docs/contract/teacher-data.md`](contract/teacher-data.md) 详细定义了合成范式。
    *   [`src/cli/generate_teacher_data.py`](../src/cli/generate_teacher_data.py), [`src/cli/filter_teacher_data.py`](../src/cli/filter_teacher_data.py) 是数据管线的核心抓手。

---

## 4. 渐进式模型微调 (Progressive Fine-Tuning)

微调没有采用简单的一波流，而是分为了三个相互承接的阶段：
1.  **B8 阶段 (Format Alignment)**：通过纯答案 (answer-only) 数据，让基座模型强行对齐目标 JSON 边界，学习闭嘴。
2.  **BC8 阶段 (Mixed Distillation)**：采用 `60%纯答案 : 30%短证据 : 10%错误批评` 的比例进行混合微调。既保证了直接回答的能力，又注入了结构化思维与找错能力。
3.  **BC8-final 阶段 (Answer-only Replay)**：短轮次数据回放，专门压制经过大量思考训练后产生的格式漂移或 markdown 泄露。
*   **结果**：在此管线下训练出的 `BC8-final-v3` 模型，在 Harness 环境下的格式崩溃率被成功压制到 **4%**，达成阶段性最优。
*   **相关入口**：
    *   [`docs/plans/training-plan.md`](plans/training-plan.md)
    *   [`src/cli/train_bc8.py`](../src/cli/train_bc8.py), [`src/cli/train_replay.py`](../src/cli/train_replay.py)

---

## 5. 后续设计与消融计划 (Future Design & Ablations)

项目设计不仅止步于单纯的打榜得分，后续工作重点转向结构消融，以量化各个组件对模型智能的真实贡献：
*   **BCD (Block-Coordinate Descent) 消融计划**：在基线与最佳 checkpoint（BC8-final-v3）之间，通过控制变量法，分别剥离“词义提取”、“情感打标”等特征，分析是哪一个特征在阻碍或拉升模型能力。
*   这部分的系统设计将在未来的迭代中通过修改构建管线来落地。
*   **相关入口**：[`docs/plans/bcd-plan.md`](plans/bcd-plan.md)

---

## 6. 开发者导览 (Documentation Structure)

本项目所有设计、契约与实验追踪均进行了系统化的分类管理：

### `spec/`
包含了最严格的 Schema 定义以及核心架构层面的接口契约。
- [data-schema.md](spec/data-schema.md): 核心输入/输出 JSON 数据结构定义。
- [harness.md](spec/harness.md): True Harness 的分离契约（Reasoner -> Formatter）定义。

### `contract/`
包含了评测指标准则、标准 Prompt 协议以及数据合成规则。
- [eval-plan.md](contract/eval-plan.md): 包含 Core Error / Format Error 评估指标追踪以及错误分类。
- [prompt-baseline.md](contract/prompt-baseline.md): 基线评测的 Prompt 配置与测试协议。
- [teacher-data.md](contract/teacher-data.md): 教师模型数据生成的硬性规则。

### `plans/`
项目的路线图与任务执行计划。
- [execution-plan.md](plans/execution-plan.md): 从零准备数据直至远程命令执行的分步记录。
- [training-plan.md](plans/training-plan.md): 微调训练（B8, BC8, Replay）阶段定义与配置。
- [bcd-plan.md](plans/bcd-plan.md): 后续 BCD 消融实验的演进规划。
- [data-pipeline.md](plans/data-pipeline.md): 数据构建流水线说明。
- [agent-task-list.md](plans/agent-task-list.md): 专供智能体分配使用的 Task 拆解列表。

### `reports/`
所有的真实测评、探路测试以及 Milestone 产出。
- [model-selection-report.md](reports/model-selection-report.md): 早期基线测试与选型决定。
- [baseline-smoke-results.md](reports/baseline-smoke-results.md): 基本 Smoke 测试输出。
- [data-inspection.md](reports/data-inspection.md): 数据管线中产生的 Schema 规范性校验报告。
- [e3-dev50-results.md](reports/e3-dev50-results.md), [e4-gate-report.md](reports/e4-gate-report.md): 阶段性的远端跑分成果。

### `agents/`
存放与 AI Agent 运作流程相关的元数据和提示词规则。

### Top-Level State
- **[workspace-state.md](workspace-state.md)**: **高频变动！** 追踪当前与设计存在漂移的地方，记录了远端服务器 (`checkpoints/` 与 `data/`) 的最新进度与命令入口，开发与验证前请务必优先查阅本文件。
