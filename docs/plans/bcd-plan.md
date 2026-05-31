# BCD 循环 Block 消融方案

本文定义 `BCD0`、`BCD1`、`BCD2` 与 `BCD-H` 的最小可执行设计。BCD 是后续结构消融，不是主路线；只有 `BC8-final`、`H1/H2` harness 结果和 `docs/contract/eval-plan.md` 的评测协议稳定后才启动。

本文只做设计，不写训练代码、不引入新依赖、不修改 harness。若未来实现需要大规模改推理框架、vLLM 适配或多轮 agent，直接放弃 BCD，优先提交 `BC8-final`、`BC8-final + 规则 postprocess` 或 `BC8-final + harness`。

## 1. 启动前置条件

启动 BCD 前必须同时满足：

1. 已完成 `BC8-final`，并在固定 `dev_main` 上记录词义分、翻译分、情感分、总分、JSON 错误率和延迟。
2. `H1` 与 `H2` 至少完成 `Harness decision` 级别评测，formatter 改坏率和调用率可追踪。
3. 已确认 `BC8-final` 的输出目标：直接提交时只输出最终 JSON；作为 harness reasoner 时输出 `structured evidence + draft_answer`。
4. 已保存可回退的 `BC8-final` checkpoint 与 LoRA 配置。

不满足以上条件时，不做 BCD。

## 2. 中间 1/3 Transformer Block 循环策略

BCD 只改模型前向路径中的 block 调度，不改变 tokenizer、数据 schema、训练样本格式或最终输出 schema。

设模型共有 `L` 个 Transformer block，按索引划分：

```text
前 1/3：0 到 floor(L/3)-1，不循环
中间 1/3：floor(L/3) 到 floor(2L/3)-1，循环一次
后 1/3：floor(2L/3) 到 L-1，不循环
```

默认循环深度为 `1`，即中间 blocks 连续执行两遍：

```text
front_blocks -> middle_blocks -> middle_blocks -> back_blocks
```

选择中间 1/3 的原因：

- 前层更偏词形、位置和浅层语义，直接循环容易破坏输入表示。
- 后层更贴近输出分布和 JSON 格式，直接循环容易增加格式漂移。
- 中间层更可能影响词义、句译和情感证据整合，适合作为结构消融入口。

首轮不试多次循环、不做逐层搜索、不做动态循环次数。若一次循环都不能稳定收益，多循环没有进入主线的价值。

## 3. BCD0：推理时直接循环

`BCD0` 是快速风险检测，不作为最终候选。

做法：

1. 从 `BC8-final` 加载相同权重。
2. 只在推理 forward 中把中间 1/3 blocks 执行两遍。
3. 不继续训练、不改 prompt、不改 decoding 参数。
4. 在 smoke 样本上与 `BC8-final` 使用同一评测脚本比较。

观察指标：

- hard JSON error 是否明显升高。
- 翻译是否变长、重复、出现赏析化。
- `choose_id` 是否出现非法选项或情感选择大幅漂移。
- 平均延迟是否接近增加一个中间段 block 的成本。

`BCD0` 的结论只用于判断是否值得继续：

- 若 JSON 错误率明显升高、输出重复严重或翻译分大幅下降，直接放弃 BCD。
- 若任务分没有明显崩坏且延迟仍可接受，才进入 `BCD1`。

## 4. BCD1：循环结构 + 继续 QLoRA

`BCD1` 是主结构消融，实验命名沿用 `plan.md` / `docs/contract/eval-plan.md` 的 `BCD1`。

做法：

1. 初始化：使用 `BC8-final` 权重和 LoRA 配置。
2. 结构：保持中间 1/3 blocks 执行两遍。
3. 训练：继续 QLoRA，训练目标优先使用 answer-only replay；如需要 harness reasoner，再小比例混入 `structured evidence + draft_answer`，但不得重新扩大为复杂混合训练。
4. 学习率：低于 `BC8-final` replay，建议从 replay 学习率的 `0.3x-0.5x` 起步。
5. 训练量：短训，优先固定少量 steps 或不超过 `0.3` epoch，用 dev JSON 错误率和总分早停。

风险控制：

- 每个 checkpoint 先跑 smoke，再进入 `BCD gate` 样本量。
- 不因为单轮总分小涨就扩训练；必须同时看 JSON 错误率、翻译分和延迟。
- 若输出开始夹带 evidence、额外字段或长解释，立即停止并回退到上一个 checkpoint。

进入后续比较的最低条件：

- `BCD1` hard JSON error 不高于 `BC8-final`。
- 翻译分无明显回退。
- 总分至少在两次相同 dev 协议下稳定不低于 `BC8-final`。
- 端到端延迟与部署复杂度仍可接受。

## 5. BCD2：门控循环 + 继续 QLoRA

`BCD2` 是内部消融 ID，只在 `BCD1` 有潜在收益但不稳定时尝试。

最小设计：

```text
front_blocks -> middle_blocks -> gate * middle_blocks(second pass) -> back_blocks
```

门控目标不是做复杂动态推理，而是降低第二遍 middle blocks 对输出格式的破坏。优先采用最小静态门控：

- 每层或每个 hidden state 使用一个可训练标量 gate，初始值接近 0。
- 第二遍输出以残差方式合入第一遍结果。
- 只训练 gate 和现有 LoRA 参数，不新增复杂控制器。

训练策略与 `BCD1` 相同：从 `BC8-final` 或最佳 `BCD1` 初始化，短训 QLoRA，优先 answer-only replay，按 dev 指标早停。

放弃 `BCD2` 的条件更严格：

- 若需要新增复杂模型模块、额外依赖或改 harness 才能跑通，放弃。
- 若 gate 学不到稳定收益，或只是在修复 `BCD1` 自己引入的问题，放弃。
- 若部署需要自定义 vLLM kernel、复杂权重合并或不兼容现有推理流程，放弃。

## 6. BCD-H

`BCD-H` 表示 `BCD1 + formatter`，只在 `BCD1` 明显强于 `BC8-final` 时尝试。

规则：

- 不为 BCD 新设计 harness。
- 直接复用已稳定的 `H1/H2` 协议、validator、formatter prompt 和评测字段。
- 只比较 `BCD1` 作为 reasoner 后，formatter 是否带来净收益。

若 `BCD1` 本身没有超过 `BC8-final`，不做 `BCD-H`。

## 7. 与 BC8-final / BC8-final + harness 的对比方式

BCD 的核心对照不是 prompt baseline，而是主路线最终候选：

| 实验 | 配置 | 对照对象 | 判断重点 |
| --- | --- | --- | --- |
| `BC8-final` | replay 后单模型最终 JSON | 主基线 | 总分、JSON 错误率、延迟 |
| `H1` | `BC8-final + 规则 postprocess` | `BC8-final` | 纯规则修复收益 |
| `H2` | `BC8-final reasoner + formatter` | `H1` | formatter 净收益、改坏率、参数与延迟 |
| `BCD0` | 推理时直接循环 | `BC8-final` | 快速风险检测，不进最终候选 |
| `BCD1` | 循环结构 + 继续 QLoRA | `BC8-final` | 结构是否带来稳定净收益 |
| `BCD-H` | `BCD1 + formatter` | `BCD1` 与 `H2` | BCD 在 harness 下是否仍有净收益 |

对比要求：

1. 使用同一 `dev_split_id`、同一评分脚本、同一 decoding 参数和同一 JSON 错误分类。
2. BCD 至少先跑 smoke 20 条，再跑 `BCD gate` 150 条；进入主结论时使用完整 `dev_main`。
3. 若 `BCD1` 与 `BC8-final` 总分差小于 `0.02`，不认为 BCD 有稳定收益，除非 JSON 错误率和延迟同时不变且人工抽查支持。
4. 若 `BCD-H` 相比 `H2` 没有总分净收益，或 formatter 改坏率升高，不进入最终候选。
5. 最终报告必须列出部署风险，不能只报告任务分。

## 8. 放弃条件

任一条件触发即停止 BCD，并记录原因：

- `BCD0` 出现明显 JSON 崩坏、重复输出、翻译赏析化或情感选择大幅漂移。
- `BCD1` 的 hard JSON error、coverage error 或 format style error 高于 `BC8-final`，且短训 replay 无法修复。
- `BCD1` 翻译分明显低于 `BC8-final`，尤其出现重复句、漏译或过长解释。
- `BCD1` 总分不能稳定超过 `BC8-final`，或收益只出现在单次小样本评测。
- `BCD-H` 不能超过 `H2`，或 formatter 改坏率、延迟、fallback 率升高。
- 循环结构导致 vLLM / transformers 部署复杂度明显高于主路线，影响最终提交可靠性。
- 需要新增依赖、复杂多轮 agent、harness 改造或长期训练预算。

放弃 BCD 不视为失败。BCD 的作用是验证循环 block 是否值得进入候选；若风险高或收益不稳，最终路线回到 `BC8-final`、`H1` 或 `H2`。

## 9. 记录字段

BCD 实验记录复用 `docs/contract/eval-plan.md` 字段，并补充：

| 字段 | 说明 |
| --- | --- |
| `loop_block_range` | 中间 1/3 block 的起止索引。 |
| `loop_count` | 默认 1，表示第二遍 middle blocks。 |
| `loop_type` | `direct`、`continued-qlora`、`gated-continued-qlora`。 |
| `init_checkpoint` | `BC8-final` 或最佳 `BCD1` checkpoint。 |
| `gate_type` | 仅 `BCD2` 填写，例如 `scalar_residual_gate`。 |
| `deployment_risk` | `low`、`medium`、`high`，并写明原因。 |
| `abandon_reason` | 若停止，记录触发的放弃条件。 |

## 10. 待确认问题

1. 最终 8B 级 reasoner 的具体模型和 block 数尚未确定；中间 1/3 的实际索引需在模型选型后填写。
2. 现有推理后端是否允许以最小改动插入 block loop 需要实现前确认；若需要改 vLLM 内核，直接放弃 BCD。
3. `BCD1` 是否用纯 answer-only replay，还是保留少量 `structured evidence + draft_answer`，应由 `BC8-final` 的直接提交表现和 `H2` 的 reasoner 表现决定。
