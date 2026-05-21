# CCL25 古诗词理解与推理任务技术路线

## 0. 结论

最终推理系统按所有参与模型的**总参数量小于 20B**计算。任务不要求端到端单模型输出，因此主线应是：

```text
Prompt baseline
-> 8B/9B BC 微调
-> BC + harness / formatter
-> BCD 循环 block 消融
```

优先级：

1. **BC 是主训练路线**：answer-only QLoRA + short-evidence 蒸馏 + answer-only replay。
2. **Harness 是主推理路线**：reasoner 生成短证据和草稿答案，formatter/verifier 生成最终 JSON。
3. **BCD 是后续消融**：循环 block 工程风险高，只有 BC 稳定后再试。

核心目标：稳定输出合法、完整、简洁的 JSON；训练 structured evidence，不训练自由长 CoT。

---

## 1. 任务输出

最终输出统一为：

```json
{
  "idx": 0,
  "ans_qa_words": {
    "衰草": "枯黄的草，烘托荒凉离别的氛围"
  },
  "ans_qa_sents": {
    "故关衰草遍，离别自堪悲": "旧关一带长满枯草，分别本就令人悲伤"
  },
  "choose_id": "D"
}
```

中间证据、草稿和修正意见都不进入最终提交。

---

## 2. 模型分工

| 角色 | 候选 | 用途 |
| --- | --- | --- |
| 教师模型 | DeepSeek-V4-Pro / Flash 等强开源模型 | 离线生成 short-evidence、critique、伪标签 |
| Reasoner / 主学生模型 | 多个 8B 级候选，优先 Qwen3-8B | 第一轮横向比较后，选择最强者微调 |
| Prompt baseline | Qwen3-14B、多个 8B 级候选 | 不微调，测 prompt 上限 |
| Formatter / verifier | Qwen3-8B、Gemma 4 E4B | 根据草稿生成最终 JSON，并做轻量校验 |

第一轮不使用 Qwen3.5-9B。`P4` 不再作为实验组名，改为 `FMT`，表示 formatter / verifier baseline。Gemma 4 E4B 的参数量按官方 config 或实际加载参数统计；若与 Qwen3-8B 组合，仍需确认总参数小于 20B。

实验 ID 必须记录模型全名、参数规模、量化方式、推理后端、thinking/non-thinking 模式、prompt 类型、shot 数和主要解码参数。推荐格式：

```text
{group}-{model_family}-{size}-{quant}-{backend}-{mode}-{prompt}
```

示例：

```text
P14-qwen3-14b-bf16-vllm-nothink-few3
P8-qwen3-8b-bf16-vllm-nothink-few3
P8-qwen3-8b-awq4-vllm-nothink-few3
FMT-gemma4-e4b-bf16-transformers-jsonfix
FMT-qwen3-8b-bf16-vllm-jsonfix
```

---

## 3. BC 训练

训练数据保留三类：

| 类型 | 格式 | 作用 |
| --- | --- | --- |
| Answer-only | 输入题目 -> 最终 JSON | 固定 schema、答案长度和提交格式 |
| Short-evidence | 输入题目 -> 短证据 -> 最终 JSON | 学习词义、翻译骨架、情感线索 |
| Teacher-critique | 输入题目 + 错误答案 -> 错误原因 -> 修正 JSON | 区分相近情感选项 |

主模型训练阶段：

```text
1. base 8B/9B -> answer-only QLoRA -> B8
2. B8 -> mixed distillation -> BC8
3. BC8 -> answer-only replay -> BC8-final
```

混合比例初始设为：

```text
answer-only：60%
short-evidence：30%
teacher-critique：10%
```

若格式错误率升高，提高 answer-only；若情感题弱，提高 short-evidence 和 critique。

可选拆分微调：

| 模型 | 训练目标 | 重点 |
| --- | --- | --- |
| Reasoner | 输入题目 -> structured evidence + draft_answer | 做题能力、词义解释、翻译骨架、情感判断 |
| Formatter | 原题 + evidence + draft_answer -> 最终 JSON | schema、缺项、长度、选项合法性 |

不训练“自由 CoT -> JSON”。Reasoner 至少要输出 `draft_answer`，避免 formatter 从推理文本里重新做题。

---

## 4. Harness 推理

推荐最小两阶段架构：

```text
Stage 1: reasoner 生成 structured evidence + draft_answer
Stage 2: formatter / verifier 输出最终 JSON
```

Stage 1 可输出：

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

Stage 2 只输出最终 JSON。Formatter / verifier 只负责：

1. 修复非法 JSON；
2. 检查 `idx`；
3. 检查指定词语和句子是否缺项；
4. 检查 `choose_id` 是否为 A/B/C/D；
5. 检查答案是否过长或变成赏析；
6. 检查情感选项是否与证据明显冲突。

不要让 formatter 默认重做题；高置信合法输出可以跳过第二模型，只走规则 postprocess。

---

## 5. BCD 消融

BCD 只在 `BC8-final` 稳定后尝试。

推荐方案：

```text
前 1/3 层：不循环
中间 1/3 层：重复一次
后 1/3 层：不循环
```

实验：

| 编号 | 方案 | 目的 |
| --- | --- | --- |
| BCD0 | 推理时直接循环 | 快速检查是否严重破坏 |
| BCD1 | 循环结构 + 继续 QLoRA | 主结构消融 |
| BCD2 | 门控循环 + 继续 QLoRA | BCD1 不稳定时尝试 |

若 BCD 不能稳定超过 `BC8-final` 或 `BC8-final + harness`，不进入最终候选。

---

## 6. 实验矩阵

| 编号 | 方案 | 目的 |
| --- | --- | --- |
| P14 | Qwen3-14B prompt baseline | 14B prompt 上限 |
| P14-fast | Qwen3-14B-AWQ prompt baseline | 14B 量化速度/效果对照 |
| P8 | 多个 8B 级模型 prompt baseline | 选择 reasoner 微调基座 |
| FMT | Qwen3-8B / Gemma 4 E4B formatter baseline | 选择 formatter / verifier |
| B8 | 最佳 8B 级 reasoner answer-only QLoRA | 格式适配 |
| BC8 | B8 + mixed distillation + replay | 主学生模型 |
| H1 | BC8 + 规则 postprocess | 纯规则修复收益 |
| H2 | BC8 reasoner + Gemma formatter | 主推 harness |
| H3 | BC8 reasoner + 8B formatter | formatter 强度消融 |
| H4 | 14B prompt reasoner + formatter | 不微调协作上限 |
| BCD1 | BC8 + 循环 block 继续训练 | 结构消融 |
| BCD-H | BCD1 + formatter | 仅在 BCD1 明显更强时尝试 |

最可能胜出的方案：

```text
BC8 reasoner + Gemma formatter
或
BC8 + 规则 postprocess
```

---

## 7. 评测

每个实验记录：

| 实验 | 词义分 | 翻译分 | 情感分 | JSON 错误率 | 平均延迟 | 总分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P14 | | | | | | |
| P8 | | | | | | |
| FMT | | | | | | |
| BC8 | | | | | | |
| H2 | | | | | | |
| BCD1 | | | | | | |

重点看：

1. BC 是否提升词义和情感；
2. replay 是否降低格式错误；
3. formatter 是否减少非法 JSON 和缺项；
4. formatter 是否改坏原本正确答案；
5. BCD 是否损害翻译或格式；
6. 多模型延迟是否可接受。

---

## 8. 执行顺序

```text
1. 跑 P14 / P14-fast / P8 / FMT baseline
2. 选择最佳 8B 级 reasoner，构造 answer-only 数据，训练 B8
3. 用教师模型生成 short-evidence 和 critique
4. 训练 BC8，并做 answer-only replay
5. 加规则 postprocess，得到 H1
6. 在 Qwen3-8B 与 Gemma 4 E4B 中选择 formatter，得到 H2
7. 对比另一种 8B formatter 和 14B prompt + formatter
8. 若仍有时间，再做 BCD1 / BCD-H
```

最终选择规则：

```text
优先选总分高、JSON 错误率低、延迟可接受的方案。
若 harness 只修格式不提分，提交 BC8 + 规则 postprocess。
若 BCD 不稳定或工程复杂，提交 BC8 / H2。
```

一句话总结：

> 先用 8B/9B 做出稳定 reasoner，再用轻量 formatter 组成总参数小于 20B 的 harness；BCD 只作为后续结构消融。
