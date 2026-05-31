# Improvement Plan

## Current Baseline

BC8-v2 (Qwen3-8B + 4-bit NF4 QLoRA, mixed distillation):

| score | sim_sents | emo_acc | bleu_sents | taskA | sim_words | taskB | bleu_words |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6915 | 0.9100 | 0.8130 | 0.2530 | 0.8130 | 0.8700 | 0.5690 | 0.2450 |

**Diagnosis**: sim 高 + bleu 极低 = 模型理解正确但措辞与参考不同。核心瓶颈在 translation (taskB=0.569)，词义 bleu (0.245) 也有提升空间。

**Root cause**: 训练数据 `keywords` 和 `trans`（参考风格）被 Teacher 模型间接转换，引入了风格偏移。当前 Teacher prompt 从未见过参考译文。

**Data available**: 164 首训练诗，全部有 `trans`（全诗译文），162 首有 `keywords`（词义解释）。

---

## P14 Baseline (completed)

| Item | Choice |
| --- | --- |
| Model | `Qwen/Qwen3-14B-AWQ` |
| Script | `src/cli/eval_p14.py` |
| Harness | H1 rule-only |

---

## Priority 0: 直接风格对齐（不改架构，只改数据流）

### P0-1: `keywords` 缩短后直接做 `ans_qa_words`

**改动**: `src/data/builder.py` — `build_answer_only()`

训练数据 `keywords` 的词义太长（"和：即和诗，是用来和答他人诗作的诗，依照别人诗词的格律或内容作诗词..."），但语义精确。用规则（取逗号/分号后首个短句）或 Teacher 缩写，生成 6-25 字的简洁词义。

当前 answer-only 的 `ans_qa_words` 来自 Teacher 重写，替换为直接从 `keywords` 缩短的版本后，BLEU_words 应显著提升（参考风格直接注入）。

**验证**: 对比 B8 answer-only 重新训练后的词义 BLEU。

### P0-2: Teacher 做对齐而非翻译（`trans` → `ans_qa_sents`）

**改动**: `src/data/teacher.py` — `build_prompt()`

在 Teacher prompt 中注入全诗参考译文和目标句子，让 Teacher 从参考译文中**定位/抽取**对应片段，而非凭空翻译。只有无精确对应时才轻微改写。

```
## 全诗参考译文（请基于此译文定位目标句子，而非重新翻译）：
{reference_trans}
```

**验证**: 对比 Teacher 生成译文与参考译文的 BLEU，确认风格偏差缩小。

### P0-3: Teacher 批量处理（2-3 samples / request）

**改动**: `src/data/teacher.py` — `process_batch()`, `build_prompt()`

将 2-3 个 sample 的任务合并到一个 API 请求中。Prompt 中区分各 sample，要求 Teacher 输出一个 JSON 数组：

```
## 以下有 3 个独立题目，请为每个题目分别输出 JSON，用数组包裹：
[
  {题目1的输出},
  {题目2的输出},
  {题目3的输出}
]
```

**收益**: API 调用次数减少 2-3x，成本降低，同时给 Teacher 更多上下文（相邻诗可作为风格参考）。

---

## Priority 1: 训练流程优化

### P1-1: 混合训练数据直接注入参考风格

**改动**: `src/data/builder.py` — `build_bc8_mixed()`

BC8 mixed 数据配方：
```
50% answer-only（P0-1 的直接 labels）
25% short-evidence（Teacher 生成，含 P0-2 风格对齐后的句译）
25% teacher-critique
```

### P1-2: Few-shot 注入训练样例到 Prompt

**改动**: `src/training/__init__.py` — `ZERO_SHOT_PROMPT`

在推理 prompt 中增加 1-2 个来自训练数据的参考样例，让 8B 在推理时直接看到目标风格。

### P1-3: 测试砍掉 short-evidence 训练分支

**验证实验**: 在 eval50 上比较 BC8-v2 有无 short-evidence 训练分支的 JSON 错误率。如果无差异，砍掉 25% short-evidence 分支，用更多 answer-only 数据替代。

---

## Priority 2: 架构增强

### P2-1: Self-Critique Loop

8B Reasoner 生成 → Harness Validator 校验 → 提取格式/覆盖错误样本 → 自动生成 critique 训练数据 → fine-tune 8B。

无需 Teacher 介入的闭环改进。

### P2-2: BCD 消融（条件执行）

前置条件全部满足后才启动：
- [ ] BC8 结果稳定（当前 0.6915）
- [ ] P0 改进完成，翻译 BLEU 显著提升
- [ ] H1/H2 harness 结果完备
- [ ] 200+ 样本的 dev_main 建立

详见 `docs/plans/bcd-plan.md`。

---

## Files to modify

| Priority | File | Change |
| --- | --- | --- |
| P0-1 | `src/data/builder.py` | `keywords` 直接缩短做 `ans_qa_words` |
| P0-2 | `src/data/teacher.py` | `build_prompt()` 注入参考译文 |
| P0-3 | `src/data/teacher.py` | 2-3 样本批量处理 |
| P1-1 | `src/data/builder.py` | 更新 BC8 数据配方 |
| P1-2 | `src/training/__init__.py` | Few-shot 注入 |
| P1-3 | — | 实验验证，不改架构 |
| P2-1 | 新文件 | Self-Critique Loop |
| P2-2 | `docs/plans/bcd-plan.md` | 已有 |
