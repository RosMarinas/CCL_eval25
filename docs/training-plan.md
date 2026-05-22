# B8 / BC8 QLoRA 训练计划

本文定义 B8 answer-only QLoRA、BC8 mixed distillation、answer-only replay 与实验命名规则。训练目标服务于 `plan.md` 的主路线：

```text
最佳 8B 级 reasoner，优先 Qwen3-8B
-> [B8 answer-only QLoRA，条件执行]
-> BC8 mixed distillation
-> BC8-final answer-only replay
```

具体基座模型不在本文硬编码，必须由 P8 baseline 与专项模型调研确定。候选应记录完整模型名、参数规模、量化方式、推理后端、thinking/non-thinking 模式和主要训练参数。

## 1. 训练目标总览

| 阶段 | 输入 | 训练输出 | 用途 |
| --- | --- | --- | --- |
| B8 answer-only | 统一输入题目 | 最终 JSON | 学习提交 schema、字段覆盖、答案长度和选项合法性 |
| BC8 answer-only 子集 | 统一输入题目 | 最终 JSON | 保持 B8 的格式稳定性 |
| BC8 short-evidence 子集 | 统一输入题目 | `structured evidence + draft_answer` | 学习词义、句译骨架和情感短证据 |
| BC8 teacher-critique 子集 | 统一输入题目 + 候选错误答案 | `structured critique + correction_evidence + corrected_answer` | 学习识别相近情感选项、词义误解、缺项和过长答案 |
| BC8-final replay | 统一输入题目 | 最终 JSON | 降低混合蒸馏后的 JSON 错误率和输出漂移 |

最终提交仍只使用 `docs/data-schema.md` 的最终 JSON。`structured evidence`、`draft_answer`、`critique` 和 `correction_evidence` 只用于训练和 harness 中间态，不进入最终提交。

## 2. B8 answer-only QLoRA 配置

B8 的目标是把最佳 8B 级 reasoner 适配到 CCL25 的最终输出格式，优先保证合法、完整、简洁的 JSON，而不是训练推理文本。

**B8 条件执行规则：** B8 不再是必经阶段。P8 baseline 完成后根据 Core 错误率决定：

| P8 Core 错误率 | B8 执行策略 |
| ---: | --- |
| < 5% | B8 缩短为格式确认（少量 steps / 0.3 epoch），或直接从基座进入 BC8 |
| 5% – 15% | 执行完整 B8，以 `json_error_rate` 和 `missing_word_key` 率早停 |
| > 15% | 完整 B8，且提高 answer-only 数据比例和 epoch 数 |

判断依据以 `json_error_rate`（Core 错误）为准，不包含 `extra_text` 和 `thinking_trace_leak`（这两类由 parser 统一处理，不应迫使模型额外训练）。若主要错误集中在 `missing_word_key` / `missing_sentence_key` / `empty_required_answer`，B8 仍有价值；若仅剩 format 类错误，B8 可跳过。

### 2.1 数据

- 数据来源：高置信 answer-only 样本，标签为最终 JSON。
- 输入：`idx/title/author/content/qa_words/qa_sents/choose` 的统一输入 JSON。
- 输出：只包含 `idx/ans_qa_words/ans_qa_sents/choose_id` 的最终 JSON。
- 不混入 `short_evidence`、`teacher_critique`、自由 CoT 或 Markdown。
- 样本过滤复用 `docs/data-schema.md` 与 `docs/teacher-data.md` 的覆盖、选项合法性和长度规则。

### 2.2 推荐 QLoRA 超参

| 项目 | 推荐值 | 说明 |
| --- | --- | --- |
| 基座 | 最佳 8B 级 reasoner，优先 Qwen3-8B | 由 P8 baseline / 专项模型调研确定 |
| 量化 | 4-bit NF4，double quant，bf16 compute | 训练期 QLoRA；不等同于推理量化结论 |
| LoRA 目标模块 | attention + MLP linear 层 | 至少覆盖 `q_proj/k_proj/v_proj/o_proj`；显存允许时加入 `gate_proj/up_proj/down_proj` |
| LoRA rank | 16 起步，必要时试 32 | 小数据优先 16；欠拟合再升 rank |
| LoRA alpha | 32 或 64 | 通常取 `2 * rank` |
| LoRA dropout | 0.05 | 数据少时可升至 0.1 防过拟合 |
| 序列长度 | 2048 起步 | 若长诗或多目标句截断，再升到 4096 |
| 学习率 | `1e-4` 到 `2e-4` | 首轮推荐 `1e-4`，格式仍不稳再延长训练而非先升 LR |
| scheduler | cosine 或 linear warmup | warmup ratio `0.03-0.05` |
| epoch | 2-3 | 以 dev JSON 错误率和子任务分数早停 |
| 有效 batch | 64-128 条样本 | 用 gradient accumulation 达成 |
| weight decay | 0.0-0.01 | LoRA 训练不做复杂正则 |
| packing | 默认关闭 | 避免多题拼接导致 JSON 边界学习混乱；若实现能严格加边界 token，可单独消融 |

### 2.3 B8 验收指标

若 B8 被执行，验收标准如下：

- JSON 严格解析错误率低于 P8 prompt baseline。
- `idx` 原样传递，`ans_qa_words` 与 `ans_qa_sents` 覆盖去重后的所有 key。
- `choose_id` 只来自输入 `choose` 的合法选项。
- 词义和句译不过度赏析，长度分布接近 teacher-data 的建议阈值。

## 3. BC8 mixed-distillation 配置

BC8 在 B8 基础上继续 QLoRA，目标是提升古诗词理解能力，同时不牺牲最终 JSON 格式稳定性。

### 3.1 数据混合比例

初始混合比例固定为：

```text
answer-only：60%
short-evidence：30%
teacher-critique：10%
```

采样以训练 step 为单位做比例控制，而不是简单拼接后随机打乱。每个 batch 内可以混合不同目标，但每条样本必须带明确的训练目标标识，例如 `target=final_json`、`target=evidence_draft`、`target=critique_correction`，避免模型混淆输出格式。

### 3.2 BC8 推荐超参

| 项目 | 推荐值 | 说明 |
| --- | --- | --- |
| 初始化 | B8 checkpoint | 不直接从 base 训练 BC8 |
| 量化与 LoRA | 沿用 B8 | 只在 B8 欠拟合时提高 rank |
| 学习率 | `5e-5` 到 `1e-4` | 推荐低于 B8，减少格式遗忘 |
| epoch | 1-2 | 优先短训，多做 dev 检查 |
| 有效 batch | 64-128 | 保持与 B8 接近 |
| 序列长度 | 2048 起步，必要时 4096 | `teacher_critique` 更长，需监控截断 |
| loss 权重 | 默认等权 | 若 critique 过长，可降低 critique loss 权重到 `0.5x` |
| 早停依据 | 总分、JSON 错误率、情感分 | 任一关键格式指标明显回退即停止或 replay |

### 3.3 输出格式约束

BC8 不是自由 CoT 训练。允许的非最终 JSON 输出只限结构化短字段：

```json
{
  "evidence": {
    "words": {},
    "sentences": {},
    "emotion": []
  },
  "draft_answer": {
    "ans_qa_words": {},
    "ans_qa_sents": {},
    "choose_id": "D"
  }
}
```

`teacher_critique` 训练目标使用：

```json
{
  "critique": {
    "word_errors": [],
    "sentence_errors": [],
    "emotion_error": {}
  },
  "correction_evidence": {
    "words": {},
    "sentences": {},
    "emotion": []
  },
  "corrected_answer": {
    "idx": 0,
    "ans_qa_words": {},
    "ans_qa_sents": {},
    "choose_id": "D"
  }
}
```

其中 `draft_answer` 与 `docs/harness.md` 的 reasoner 输出保持一致，不包含 `idx`，由外层样本或 harness 注入；`corrected_answer` 是可直接评测的最终答案，必须包含 `idx` 并符合最终答案 schema。

## 4. 数据比例调整条件

先使用 `60/30/10` 作为唯一主线比例，只有 dev 指标触发时才调整。

| 触发条件 | 调整 |
| --- | --- |
| JSON 解析错误率、缺项率、非法 `choose_id` 高于 B8 | 提高 answer-only 到 70%-80%，降低 short-evidence 和 critique |
| 输出开始夹带 evidence、Markdown 或额外字段 | 提高 answer-only，并缩短 BC8 训练 epoch；必要时提前进入 replay |
| 词义或句译分低，格式仍稳定 | 提高 short-evidence 到 35%-40%，answer-only 不低于 50% |
| 情感选择弱，尤其相近选项混淆 | 提高 teacher-critique 到 15%-20%，优先采样情感错误 critique |
| critique 导致模型默认批改而不是答题 | critique 降回 5%-10%，并强化训练目标标识 |
| 过拟合 teacher 表达，答案变长 | 提高 answer-only，过滤长答案样本，降低 short-evidence loss 或采样率 |

任何比例调整都要记录为独立实验，不覆盖主线 BC8。

## 5. Answer-only replay 方案

Replay 在 BC8 mixed distillation 后执行，产物命名为 `BC8-final`。

1. 初始化：从最佳 BC8 checkpoint 继续训练。
2. 数据：只使用高置信 answer-only 样本，包括原始 answer-only 标签，以及从 `short_evidence.final_answer`、`teacher_critique.corrected_answer` 抽取且通过过滤的最终 JSON。
3. 比例：answer-only replay 内部不再混入 evidence 或 critique；若使用抽取数据，原始 gold / teacher 抽取样本建议按 `70/30` 起步。
4. 学习率：使用 BC8 学习率的 `0.3x-0.5x`，推荐 `2e-5` 到 `5e-5`。
5. 训练量：`0.3-1` epoch 或固定少量 steps，以 dev JSON 错误率最低点早停。
6. 目标：输出最终 JSON，不输出 `evidence`、`draft_answer`、`critique` 或额外字段。
7. 验收：若 replay 提升格式但明显降低词义、句译或情感分，应保留 replay 前 BC8 作为 harness reasoner 候选，同时保留 `BC8-final` 作为直接提交候选。

## 6. Checkpoint 命名和实验编号

### 6.1 Checkpoint 目录命名

checkpoint 名称必须可追溯到模型、量化、训练阶段和关键参数：

```text
{exp_id}__ckpt-{step}__dev-json{jsonerr}__dev-total{score}
```

示例：

```text
B8-qwen3-8b-nf4-peft-nothink-finaljson-stage-b8-lr1e4-r16-a32-seq2048__ckpt-1200__dev-json0.018__dev-total0.742
BC8-qwen3-8b-nf4-peft-nothink-mix60-30-10-stage-bc8-lr5e5-r16-a32-seq2048__ckpt-1800__dev-json0.024__dev-total0.781
BC8-final-qwen3-8b-nf4-peft-nothink-replay-stage-final-lr2e5-r16-a32-seq2048__ckpt-2100__dev-json0.010__dev-total0.776
```

### 6.2 实验 ID 字段

实验 ID 兼容 `plan.md` 推荐字段，并补充训练阶段参数：

```text
{group}-{model_family}-{size}-{quant}-{backend}-{mode}-{objective}-{stage}-{train_params}
```

字段说明：

| 字段 | 示例 | 说明 |
| --- | --- | --- |
| `group` | `B8` / `BC8` / `BC8-final` | `BC8-final` 表示 answer-only replay 后的产物；文件名需要短名时可用 `BC8F` |
| `model_family` | `qwen3` / `gemma` | 使用规范简称，完整模型名另存 metadata |
| `size` | `8b` | 参数规模 |
| `quant` | `nf4` / `bf16` / `awq4` | 训练期 QLoRA 用 `nf4`；推理评测另记实际量化 |
| `backend` | `peft` / `trl` / `vllm` | 训练实验可记训练栈，推理实验记推理后端 |
| `mode` | `nothink` / `think` | 默认按 P8 结论；不得省略 |
| `objective` | `finaljson` / `mix60-30-10` / `replay` | 对应训练目标 |
| `stage` | `stage-b8` / `stage-bc8` / `stage-final` | 训练阶段或 replay |
| `train_params` | `lr1e4-r16-a32-seq2048` | 主要训练参数 |

完整 metadata 还必须记录：

- 模型全名和 revision。
- 参数规模统计。
- 训练数据版本和过滤版本。
- shot 数：训练实验填 `shot-train`，prompt baseline 填 `zero`、`few3` 等。
- 主要解码参数：训练评测时至少记录 temperature、top_p、max_new_tokens。
- LoRA target modules、rank、alpha、dropout。
- 是否经过 answer-only replay。

## 7. 训练目标输出契约

| 目标名 | 训练阶段 | 是否输出最终 JSON | 是否输出 `structured evidence + draft_answer` | 说明 |
| --- | --- | --- | --- | --- |
| `final_json` | B8、BC8 answer-only、BC8-final replay | 是 | 否 | 根对象必须是最终 JSON，无额外字段 |
| `evidence_draft` | BC8 short-evidence | 否 | 是 | 根对象包含 `idx`、`evidence` 与 `draft_answer`；`draft_answer` 不含 `idx`，其余字段符合最终 JSON schema |
| `critique_correction` | BC8 teacher-critique | 通过 `corrected_answer` 嵌套输出 | 否 | 根对象包含 `critique/correction_evidence/corrected_answer`，不输出自由 CoT |

推理时如果需要直接提交，使用 `final_json` 目标或 replay 后的 `BC8-final`。如果使用 harness，reasoner 使用 `evidence_draft` 输出，由 formatter 或规则 postprocess 生成最终 JSON。

## 8. 待确认问题

1. P8 baseline 与专项模型调研尚未确定最终 8B 级 reasoner；本文只规定选择后如何训练。
2. 官方训练数据是否有可靠句级译文和正确情感选项仍需数据检查；不可靠字段不得直接进入 answer-only。
3. 实际显存会影响 LoRA target modules、rank、序列长度和 batch；首轮应优先稳定复现实验，再扩 rank 或 seq length。
4. `teacher_critique` 对最终分数的净收益需要通过 BC8 消融确认；若只增加格式漂移，应降低比例或只用于情感专项实验。
