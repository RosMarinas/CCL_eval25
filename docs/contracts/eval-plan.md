# CCL25 评测与消融跟踪方案

本文定义 P14 / P14-fast / P8 / FMT、B8 / BC8、H1-H4、BCD1 / BCD-H 的统一评测记录方式，目标是让 prompt baseline、BC 训练、harness 和 BCD 消融结果可横向比较。

## 1. 评测原则

- 输入和输出 schema 以 `docs/contracts/data-schema.md` 为准，最终输出只评测 `idx`、`ans_qa_words`、`ans_qa_sents`、`choose_id`。
- 评分规则延续已确认策略：优先使用官方评测脚本。官方脚本缺失时，内部临时总分按官方 Task1:Task2 = 0.5:0.5 权重计算：`0.25 * 词义分 + 0.25 * 翻译分 + 0.50 * 情感分`。Task1 内部词义与翻译等权分割为暂定假设，待官方评测脚本发布后以官方为准。
- 情感分采用两阶段分解评估：(a) **Reasoner 情感分析准确率**——`sentiment.primary` 是否正确反映诗歌情感；(b) **Formatter 情感映射准确率**——sentiment→choose_id 映射是否正确。仅映射准确率（即 choose_id 准确性）计入最终提交得分；Reasoner 情感分析准确率用于诊断和消融分析，不计入提交总分。
- JSON 错误率、formatter 改坏率和平均延迟独立记录，不并入内部临时总分。
- 所有实验必须记录完整实验 ID：模型全名、参数规模、量化方式、推理后端、thinking/non-thinking 模式、prompt 类型、shot 数和主要解码参数。
- 同一轮比较必须使用同一 dev split、同一 few-shot 池、同一评分脚本或同一人工评分规则。

## 2. 实验结果表字段

主结果表每行对应一个完整实验配置，不用同一个实验名覆盖 retry、不同 shot 数或不同 formatter。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `experiment_id` | string | 完整实验 ID，例如 `P8-qwen3-8b-bf16-vllm-nothink-few3`。 |
| `group` | enum | `P14`、`P14-fast`、`P8`、`FMT`、`B8`、`BC8`、`BC8-final`、`H1`、`H2`、`H3`、`H4`、`BCD1`、`BCD-H`。 |
| `model_role` | string | `prompt-reasoner`、`reasoner`、`formatter`、`harness`、`bcd-reasoner` 等。 |
| `reasoner_model` | string | reasoner 或 prompt-only 模型全名；无则填空。 |
| `formatter_model` | string | formatter 模型全名；未使用 formatter 时填空。 |
| `param_total_b` | number | 参与推理模型总参数量，单位 B，用于检查小于 20B 约束。 |
| `quantization` | string | `bf16`、`awq4`、`qlora` 等。 |
| `backend` | string | `vllm`、`transformers` 等。 |
| `mode` | string | `think`、`nothink` 或模型实际推理模式。 |
| `prompt_type` | string | `zero`、`few`、`formatter-jsonfix`、`reasoner-evidence` 等。 |
| `shot_count` | integer | few-shot 数；无 few-shot 时为 0。 |
| `decode_params` | object/string | temperature、top_p、max_new_tokens、stop 等主要参数。 |
| `dev_split_id` | string | 使用的 dev split 名称和版本。 |
| `sample_count` | integer | 计入汇总的样本数。 |
| `word_score` | number | `ans_qa_words` 平均分。 |
| `translation_score` | number | `ans_qa_sents` 平均分。 |
| `emotion_score` | number | `choose_id` 平均分（经 Formatter 映射后的最终情感选择准确率，即情感映射准确率）。 |
| `sentiment_primary_accuracy` | number | Reasoner `sentiment.primary` 准确率（是否匹配诗歌实际情感，需要 gold sentiment label 或人工评判）。 |
| `sentiment_mapping_accuracy` | number | Formatter 将 sentiment→choose_id 映射的准确率（分母为 Reasoner 情感分析正确的样本数，或全部样本）。 |
| `total_score` | number | 官方总分；无官方脚本时用内部临时权重。 |
| `json_error_rate` | number | final 输出出现 Core 错误（输出不可用）的样本比例。 |
| `avg_latency_ms` | number | 单样本端到端平均延迟。 |
| `p95_latency_ms` | number | 单样本端到端 P95 延迟。 |
| `formatter_call_rate` | number | harness 中调用 formatter 的样本比例；非 harness 为空。 |
| `formatter_regression_rate` | number | formatter 改坏率；非 formatter / harness 实验为空。 |
| `retry_rate` | number | 触发 reasoner retry 的样本比例；无 retry 时为 0。 |
| `fallback_rate` | number | 使用规则兜底的样本比例。 |
| `notes` | string | 数据异常、人工评分说明、失败原因摘要。 |

逐样本明细表至少保留：

| 字段 | 说明 |
| --- | --- |
| `experiment_id`、`idx`、`raw_output`、`parsed_json` | 复现单样本结果。 |
| `json_valid`、`json_error_categories` | JSON 与 schema 校验结果。 |
| `core_valid` | `bool` | 无 Core 错误，输出可用于下游评分。 |
| `word_score`、`translation_score`、`emotion_score`、`total_score` | 单样本任务分。 |
| `reasoner_sentiment` | Reasoner 输出的 `sentiment` 内容（primary + secondary + rationale）。 |
| `sentiment_correct` | Reasoner 情感分析是否正确（sentiment.primary 是否匹配诗歌实际情感）。 |
| `mapping_correct` | Formatter 映射是否正确（Reasoner 情感分析正确的情况下，choose_id 是否选对）。 |
| `latency_ms`、`reasoner_latency_ms`、`formatter_latency_ms` | 延迟拆分。 |
| `formatter_called`、`reasoner_retried`、`fallback_used` | harness 决策路径。 |
| `draft_answer`、`final_answer` | 仅 harness / formatter 实验必填，用于计算改坏率。 |

## 3. JSON 错误分类

JSON 错误分为两类，一个样本可同时有 core 和 format 错误。

### Core 错误（输出不可用）

输出存在以下错误时无法用于下游评分或提交。

| 错误类别 | 触发条件 |
| --- | --- |
| `parse_error` | 原始输出经剥离（think tags、fence、extra text）后仍无法提取为单个 JSON 对象。 |
| `missing_top_field` | 解析后的 JSON 缺少 `idx`、`ans_qa_words`、`ans_qa_sents` 或 `choose_id` 之一。 |
| `idx_mismatch` | 输出 `idx` 与输入 `idx` 不一致。 |
| `wrong_field_type` | 顶层字段或答案字段类型错误，例如答案对象输出成数组。 |
| `missing_word_key` | `ans_qa_words` 未覆盖 `qa_words` 去重后的目标词。 |
| `missing_sentence_key` | `ans_qa_sents` 未覆盖 `qa_sents` 去重后的目标句。 |
| `empty_required_answer` | 目标词或目标句答案为空字符串、`null` 或空对象。 |
| `invalid_choose_id` | 最终输出 `choose_id` 不属于原题 `choose` 的选项 ID（仅校验 final 输出，Reasoner 中间输出不含 choose_id）。 |
| `missing_sentiment` | Reasoner 中间输出缺少 `sentiment` 字段（仅 Reasoner 输出校验）。 |
| `invalid_sentiment_primary` | Reasoner `sentiment.primary` 不在受控词汇表中（参见 docs/contracts/data-schema.md 第 3.2 节；仅 Reasoner 输出校验）。 |
| `non_chinese_or_unusable` | 答案主体不是中文，或明显不可用于提交。 |

### Format 错误（输出不干净）

原始输出包含不符合格式规范的文本，但 parser 已成功剥离、JSON 内容可用。

| 错误类别 | 触发条件 |
| --- | --- |
| `extra_text` | JSON 前后含解释、Markdown fence、代码块或其他非 JSON 文本。 |
| `thinking_trace_leak` | 输出包含 `<think>` 标签或显式推理过程标记。 |
| `extra_top_field` | 解析后的 JSON 包含 `evidence`、`draft_answer`、`analysis` 等额外顶层字段。 |
| `extra_word_key` | `ans_qa_words` 出现不来自 `qa_words` 的 key。 |
| `extra_sentence_key` | `ans_qa_sents` 出现不来自 `qa_sents` 的 key。 |
| `overlong_word_answer` | 词义答案超过 40 个中文字符。 |
| `overlong_sentence_answer` | 句子翻译超过 80 个中文字符。 |

### 汇总指标

- `json_error_rate`：出现任一 **Core 错误**的样本比例。回答「输出能不能用」。
- `hard_json_error_rate`：`parse_error`、`missing_top_field`、`idx_mismatch`、`wrong_field_type`、`invalid_choose_id` 中任一出现的比例。Core 错误的严重子集。
- `format_style_error_rate`：出现任一 **Format 错误**的样本比例。回答「输出干不干净」。

## 4. Formatter 改坏率统计方式

Formatter 改坏率只在 FMT、H2、H3、H4、BCD-H 等存在 formatter 的实验中统计。核心问题是：formatter 是否把本来正确的草稿改成错误 final。

对每个样本同时评测 `draft_answer` 和 `final_answer`：

1. 将 `draft_answer` 注入 `idx` 后按最终输出 schema 校验。
2. 分别计算 draft 与 final 的 `word_score`、`translation_score`、`emotion_score`、`total_score`。
3. 若 draft 在某一子任务满分或达到人工确认的正确阈值，而 final 在同一子任务低于该阈值，记为该子任务 formatter regression。对于情感子任务，draft（Reasoner 输出）不含 `choose_id`，因此情感回归定义为：(a) Reasoner 情感分析正确但 Formatter 映射到错误选项（映射回归），或 (b) Formatter 修改了原本正确的 `sentiment` 分析（情感回归，仅 formatter 允许修改 sentiment 时适用）。
4. 若 `draft_total_score > final_total_score` 且差值不小于 `0.05`，记为总分 regression。
5. 若 draft JSON 合法但 final 产生任一 hard JSON error，也记为 regression。

推荐汇总字段：

| 字段 | 定义 |
| --- | --- |
| `formatter_regression_rate` | 出现任一子任务 regression、总分 regression 或 hard JSON regression 的样本比例。 |
| `formatter_word_regression_rate` | 词义从正确变错的样本比例。 |
| `formatter_translation_regression_rate` | 翻译从正确变错的样本比例。 |
| `formatter_emotion_regression_rate` | 情感选择（choose_id）从正确变错的样本比例。draft 无 choose_id 时定义为：Reasoner 情感分析正确但 Formatter 映射到错误选项的比例（映射回归）。 |
| `formatter_sentiment_regression_rate` | Reasoner 情感分析原本正确，Formatter 改变 `sentiment` 内容后变错的样本比例（仅 Formatter 能够改 sentiment 时适用）。 |
| `formatter_json_regression_rate` | draft 合法但 final 出现 hard JSON error 的样本比例。 |
| `formatter_fix_rate` | draft 有 JSON / coverage 错误，final 修复且未降低总分的样本比例。 |
| `formatter_net_gain` | `mean(final_total_score - draft_total_score)`。 |

结论规则：

- 若 `formatter_fix_rate` 低且 `formatter_regression_rate` 高，H2 / H3 / H4 不应覆盖对应 reasoner 的 H1 或规则 postprocess 结果。
- 若 formatter 只降低 JSON 错误率但 `formatter_net_gain <= 0`，最终候选优先考虑 BC8-final + 规则 postprocess。
- 若 formatter 显著提高 JSON 合法率且 `formatter_regression_rate <= 2%`，可进入主候选。

## 5. 分任务错误分析模板

每轮主实验后，对总分最低、formatter regression、JSON 错误和情感错误样本分别抽样分析。模板如下。

### 5.1 词义错误分析

| 字段 | 内容 |
| --- | --- |
| `idx` | 样本 ID。 |
| `experiment_id` | 实验 ID。 |
| `target_word` | 目标词。 |
| `gold_or_reference` | 标准或人工参考答案。 |
| `prediction` | 模型答案。 |
| `error_type` | `missing`、`literal_only`、`context_misread`、`over_explained`、`wrong_sense`、`format_only`。 |
| `evidence_issue` | 是否缺少或误用 short-evidence。 |
| `fix_hint` | prompt、BC 数据、formatter 或人工规则层面的修复建议。 |

### 5.2 句子翻译错误分析

| 字段 | 内容 |
| --- | --- |
| `idx` | 样本 ID。 |
| `target_sentence` | 目标句。 |
| `gold_or_reference` | 标准或人工参考译文。 |
| `prediction` | 模型译文。 |
| `error_type` | `missing`、`partial_translation`、`syntax_misread`、`imagery_loss`、`added_appreciation`、`too_long`、`wrong_subject`。 |
| `affected_score` | 翻译分损失估计。 |
| `fix_hint` | 是否需要增加句式样例、压缩答案或调整训练样本。 |

### 5.3 情感选择错误分析

| 字段 | 内容 |
| --- | --- |
| `idx` | 样本 ID。 |
| `choose` | 原题选项。 |
| `gold_choose_id` | 正确选项。 |
| `pred_choose_id` | 模型选项。 |
| `error_type` | `opposite_emotion`、`near_option_confusion`、`local_cue_overfit`、`ignored_title_author`、`formatter_changed`、`invalid_option`、`sentiment_misanalysis`、`sentiment_mapping_error`、`sentiment_vocab_mismatch`。 |
| `key_evidence` | 支持正确选项的关键词或句子。 |
| `model_evidence` | reasoner evidence 或模型输出中的依据。 |
| `fix_hint` | 是否需要 teacher-critique、情感 few-shot 或 formatter 冲突规则。 |

新增两阶段情感错误类型说明：
- `sentiment_misanalysis`：Reasoner 情感分析本身错误，`sentiment.primary` 标签不匹配诗歌实际情感。
- `sentiment_mapping_error`：Reasoner 情感分析正确但 Formatter 将其映射到错误选项（映射回归）。
- `sentiment_vocab_mismatch`：`sentiment.primary` 不在受控词汇表中（参见 docs/contracts/data-schema.md 第 3.2 节）。

### 5.4 JSON / Harness 错误分析

| 字段 | 内容 |
| --- | --- |
| `idx` | 样本 ID。 |
| `json_error_categories` | 第 3 节错误标签。 |
| `stage` | `reasoner`、`validator`、`formatter`、`final_validator`、`fallback`。 |
| `raw_output_excerpt` | 原始输出摘要。 |
| `validator_report` | 本地校验摘要。 |
| `formatter_called` | 是否调用 formatter。 |
| `fallback_used` | 是否使用兜底。 |
| `fix_hint` | prompt 约束、validator 规则或 formatter prompt 的修复建议。 |

## 6. 消融表

### 6.1 Baseline 消融表

| 实验 | 配置 | 目标 | 词义分 | 翻译分 | 情感分 | 情感分析准确率 | 总分 | JSON 错误率 | 平均延迟 | 备注 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P14 | Qwen3-14B prompt baseline | 14B prompt 上限 | | | | | | | | |
| P14-fast | Qwen3-14B-AWQ prompt baseline | 14B 量化速度/效果对照 | | | | | | | | |
| P8 | 多个 8B 级模型 prompt baseline | 选择 reasoner 微调基座 | | | | | | | | |
| FMT | Qwen3-8B / Gemma 4 E4B formatter baseline | 选择 formatter / verifier | | | | | | | | |

### 6.2 BC 训练消融表

| 实验 | 训练目标 | 数据组成 | 对照对象 | 词义分 | 翻译分 | 情感分 | 情感分析准确率 | 情感映射准确率 | 总分 | JSON 错误率 | 备注 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P8 | prompt-only 8B 基座 | 无训练 | baseline | | | | | | | | |
| B8 | answer-only QLoRA | 输入题目 -> 最终 JSON | P8 | | | | | | | | |
| BC8 | mixed distillation | answer-only 50% / short-evidence 25% / teacher-critique 25% | B8 | | | | | | | | |
| BC8-final | answer-only replay | 从最佳 BC8 继续短训，只输出最终 JSON | BC8 | | | | | | | | |

重点判断：

- B8 是否显著降低 JSON 错误率。
- BC8 是否提升词义、翻译和情感；BC8-final 是否在 replay 后降低格式错误且不明显损害任务分。
- 若 BC8 情感分低于 B8，优先检查 teacher-critique 质量和情感样本比例。

### 6.3 Harness 消融表

| 实验 | 配置 | 对照对象 | 词义分 | 翻译分 | 情感分 | 情感分析准确率 | 情感映射准确率 | 总分 | JSON 错误率 | formatter 改坏率 | 平均延迟 | 备注 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BC8 | 单模型 final JSON | P8 / B8 | | | | | | | | | |
| H1 | BC8-final + 规则 postprocess | BC8-final | | | | | | | | | |
| H2 | BC8-final reasoner + Gemma formatter | H1 | | | | | | | | | |
| H3 | BC8-final reasoner + 8B formatter | H2 | | | | | | | | | |
| H4 | 14B prompt reasoner + formatter | P14 / H2 | | | | | | | | | |

重点判断：

- H1 衡量纯规则修复收益。
- H2 是主推 harness，必须同时看总分、JSON 错误率、formatter 改坏率和延迟。
- H3 用于判断 formatter 强度是否值得增加参数和延迟。
- H4 用于估计不微调协作上限，不直接替代 BC8 路线。

### 6.4 BCD 消融表

| 实验 | 配置 | 对照对象 | 词义分 | 翻译分 | 情感分 | 情感分析准确率 | 情感映射准确率 | 总分 | JSON 错误率 | 平均延迟 | 部署风险 | 备注 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| BC8-final | 主学生模型 replay 产物 | P8 / B8 / BC8 | | | | | | | | | 低 | |
| H2 | BC8-final reasoner + Gemma formatter | BC8-final / H1 | | | | | | | | | 中 | |
| BCD1 | BC8-final + 循环 block 继续训练 | BC8-final | | | | | | | | | 高 | |
| BCD-H | BCD1 + formatter | BCD1 / H2 | | | | | | | | | 高 | |

BCD 放弃条件：

- BCD1 总分未稳定超过 BC8-final，或任一主任务分明显下降。
- BCD1 JSON 错误率高于 BC8-final，且无法通过 replay 或规则 postprocess 修复。
- BCD-H 相比 H2 没有净收益，或 formatter 改坏率升高。
- 循环 block 导致 vLLM / transformers 部署复杂度不可接受。

## 7. Dev Split 与最小样本量要求

### 7.1 固定 dev split

推荐建立一个固定 `dev_main`，用于所有主实验横向比较：

- 样本来源只能是训练集划出的开发样本或显式允许使用的 dev 数据，不得使用测试集或待提交样本答案。
- `dev_main` 至少 200 条样本；若数据总量不足，至少使用可用带答案数据的 20%，但不得少于 80 条。
- 情感选项 A/B/C/D 尽量均衡；每个选项至少 20 条，若原始分布不足则记录偏差。
- 覆盖无词语、多词语、无句子、多句子、长诗、短诗、重复词句等 schema 边界。
- few-shot 样例池必须从 `dev_main` 外选择，避免样例泄漏到评测样本。

### 7.2 分阶段最小样本量

| 阶段 | 适用实验 | 最小样本量 | 用途 |
| --- | --- | ---: | --- |
| Smoke | P14 / P14-fast / P8 / FMT / B8 / BC8 / H1-H4 / BCD1 / BCD-H | 20 | 快速发现 parse error、缺字段、延迟异常。 |
| Candidate | P14 / P14-fast / P8 / FMT | 80 | 选择 reasoner 和 formatter 候选。 |
| Training checkpoint | B8 / BC8 | 100 | 比较训练 checkpoint，观察格式和任务分趋势。 |
| Harness decision | H1 / H2 / H3 / H4 | 150 | 稳定估计 formatter 改坏率和修复率。 |
| Main dev | 所有进入主比较的实验 | 200 或全部 `dev_main` | 最终横向比较和消融结论。 |
| BCD gate | BCD1 / BCD-H | 150 起，主结论用 `dev_main` | 先排除结构破坏，再与 BC8-final / H2 比较。 |

若两组总分差小于 0.02，不能仅凭一次 dev 结果判断优劣；需要查看子任务分、JSON 错误率、formatter 改坏率、延迟，并增加人工抽查或扩大样本量。

## 8. 最终选择记录

最终候选必须填写以下摘要：

| 字段 | 内容 |
| --- | --- |
| `selected_experiment_id` | 被选中的提交方案。 |
| `compared_against` | 至少包含 P14、P8、BC8、BC8-final、H1、H2；若做了 BCD，也包含 BCD1 / BCD-H。 |
| `selection_reason` | 总分、JSON 错误率、延迟、formatter 改坏率的综合理由。 |
| `known_risks` | 数据偏差、人工评分不确定性、部署风险。 |
| `not_selected_reason` | 对未选 H2、BCD-H 等关键方案说明放弃原因。 |

优先选择总分高、JSON 错误率低、延迟可接受且 formatter 改坏率低的方案。若 harness 只修格式但不提分，选择 BC8-final + 规则 postprocess；若 BCD 不稳定或工程复杂，选择 BC8-final / H2。
