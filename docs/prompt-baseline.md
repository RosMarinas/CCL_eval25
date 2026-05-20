# P14 / P8 / P4 Prompt Baseline 实验设计

## 1. 范围与依赖

本文设计不微调模型的 prompt-only baseline，用于评估 14B、Qwen 8B/9B、Gemma E4B 在统一 schema 下的直接作答能力。

实验编号：

| 实验 | 模型候选 | 目的 |
| --- | --- | --- |
| P14 | 14B 指令模型 | 评估 prompt-only 上限 |
| P8 | Qwen 8B/9B 指令模型 | 评估主学生模型微调前基线 |
| P4 | Gemma E4B 指令模型 | 评估小模型 prompt-only 基线 |

依赖 Task 1 的统一数据 schema。本文使用 `docs/data-schema.md` 中定义的字段：`idx/title/author/content/qa_words/qa_sents/choose`。

统一输入 schema：

```json
{
  "idx": 0,
  "title": "诗题",
  "author": "作者",
  "content": "诗歌正文",
  "qa_words": ["衰草"],
  "qa_sents": ["故关衰草遍，离别自堪悲"],
  "choose": {
    "A": "选项 A",
    "B": "选项 B",
    "C": "选项 C",
    "D": "选项 D"
  }
}
```

最终输出 schema：

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

## 2. 通用约束

所有 P14 / P8 / P4 prompt 使用同一任务说明，只调整 few-shot 样例数量。模板不使用模型专属特殊 token，以兼容 14B、Qwen 8B/9B、Gemma E4B 的普通 chat 或 instruction 接口。

生成约束：

1. 只输出最终 JSON。
2. 不输出 Markdown 代码块。
3. 不输出解释、分析、证据、草稿或多余文本。
4. `idx` 必须与输入一致。
5. `ans_qa_words` 的 key 必须覆盖所有 `qa_words` 去重后的词语。
6. `ans_qa_sents` 的 key 必须覆盖所有 `qa_sents` 去重后的句子。
7. `choose_id` 必须从 `choose` 的选项 ID 中选择；标准样本应为 `A`、`B`、`C`、`D` 之一。
8. 词义和句意回答保持简洁，避免扩写成赏析。

建议解码参数：

| 字段 | 建议值 |
| --- | --- |
| temperature | 0.1-0.3 |
| top_p | 0.8-0.95 |
| max_new_tokens | 512-768 |
| stop | 无；以完整 JSON 解析为准 |

## 3. Zero-shot Prompt 模板

```text
你需要完成古诗词理解任务。请根据输入诗歌、目标词语、目标句子和情感选项，直接生成最终答案 JSON。

输出要求：
- 只输出一个合法 JSON 对象。
- 不要输出 Markdown 代码块。
- 不要输出解释、分析、证据、草稿或任何 JSON 之外的文字。
- JSON 字段必须且只能包含：idx、ans_qa_words、ans_qa_sents、choose_id。
- idx 必须与输入 idx 完全一致。
- ans_qa_words 是对象，key 必须使用 qa_words 中的原词；重复词语只输出一个 key；value 是该词在诗中的简洁解释。
- ans_qa_sents 是对象，key 必须使用 qa_sents 中的原句；重复句子只输出一个 key；value 是该句的简洁现代汉语翻译。
- choose_id 必须从 choose 的选项 ID 中选择一个最符合全诗情感的选项。

输入：
{{input_json}}

现在只输出最终 JSON：
```

## 4. Few-shot Prompt 模板

Few-shot 模板在通用说明后加入 2-5 个完整示例。示例的输出仍是最终 JSON，不展示推理过程。

```text
你需要完成古诗词理解任务。请根据输入诗歌、目标词语、目标句子和情感选项，直接生成最终答案 JSON。

输出要求：
- 只输出一个合法 JSON 对象。
- 不要输出 Markdown 代码块。
- 不要输出解释、分析、证据、草稿或任何 JSON 之外的文字。
- JSON 字段必须且只能包含：idx、ans_qa_words、ans_qa_sents、choose_id。
- idx 必须与输入 idx 完全一致。
- ans_qa_words 是对象，key 必须使用 qa_words 中的原词；重复词语只输出一个 key；value 是该词在诗中的简洁解释。
- ans_qa_sents 是对象，key 必须使用 qa_sents 中的原句；重复句子只输出一个 key；value 是该句的简洁现代汉语翻译。
- choose_id 必须从 choose 的选项 ID 中选择一个最符合全诗情感的选项。

示例 1 输入：
{{shot_1_input_json}}

示例 1 输出：
{{shot_1_output_json}}

示例 2 输入：
{{shot_2_input_json}}

示例 2 输出：
{{shot_2_output_json}}

{{optional_more_shots}}

待作答输入：
{{input_json}}

现在只输出待作答输入的最终 JSON：
```

建议 shot 数：

| 实验 | 默认 shot 数 | 说明 |
| --- | ---: | --- |
| P14 | 5 | 上下文和能力较强，优先覆盖更多题型 |
| P8 | 3 | 平衡格式示范和延迟 |
| P4 | 2 | 降低上下文干扰和延迟 |

## 5. Few-shot 样例选择策略

样例必须只来自训练集或显式允许使用的开发样例，不能从测试集或待预测样本泄漏答案。

选择优先级：

1. 覆盖字段形态：至少包含 1 个单词语、1 个多词语、1 个单句翻译、1 个多句翻译样例。
2. 覆盖情感选项：在固定 shot 池中尽量让 `choose_id` 分布均衡，避免全部集中到同一选项。
3. 覆盖诗歌类型：优先混合送别、怀古、咏物、羁旅、山水等常见主题。
4. 控制答案长度：选择答案简洁且 JSON 完全合法的样例，避免模型模仿长篇赏析。
5. 控制相似度：允许使用同主题样例，但不得选择同一首诗、同一 `idx` 或明显来自同一原题的样例。
6. 固定 shot 池：每个实验先使用同一套固定样例，便于 P14 / P8 / P4 横向比较。

推荐准备两个 shot 池：

| shot 池 | 用途 | 规模 |
| --- | --- | ---: |
| `balanced_static` | 主 baseline，固定顺序，保证可复现 | 5 |
| `short_static` | P4 或上下文受限模型 | 2 |

若后续需要相似检索 few-shot，应单独记录为新实验，不混入 P14 / P8 / P4 主 baseline。

## 6. Baseline 推理流程

1. 将官方原始样本转换为 Task 1 统一输入 schema。
2. 按实验编号选择模板：zero-shot 或 few-shot。
3. 渲染 prompt，保留完整输入 JSON 和固定 shot 示例。
4. 调用对应模型生成一次答案，记录开始时间、结束时间和原始输出。
5. 解析原始输出为 JSON。
6. 执行本地校验：字段集合、`idx`、目标词 key、目标句 key、`choose_id` 合法性。
7. 不调用 formatter，不做语义修正，不补写缺失答案；非法或缺字段样本计入 JSON 错误。
8. 对合法 JSON 进入评测，分别计算词义分、翻译分、情感分和总分。
9. 汇总每个实验的 JSON 错误率、平均延迟和任务分数。

为保证 prompt-only baseline 可比较，默认不做重试。若必须评估 retry 收益，应另起实验名，例如 `P8-retry1`，不能覆盖 `P8` 主结果。

## 7. 结果记录表字段

主表字段固定如下：

| 实验 | 模型 | Prompt | Shot 数 | 词义分 | 翻译分 | 情感分 | JSON 错误率 | 平均延迟 | 总分 | 备注 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P14-zero | 14B | zero-shot | 0 | | | | | | | |
| P14-few | 14B | few-shot | 5 | | | | | | | |
| P8-zero | Qwen 8B/9B | zero-shot | 0 | | | | | | | |
| P8-few | Qwen 8B/9B | few-shot | 3 | | | | | | | |
| P4-zero | Gemma E4B | zero-shot | 0 | | | | | | | |
| P4-few | Gemma E4B | few-shot | 2 | | | | | | | |

字段定义：

| 字段 | 含义 |
| --- | --- |
| 词义分 | `ans_qa_words` 的自动或人工评分均值 |
| 翻译分 | `ans_qa_sents` 的自动或人工评分均值 |
| 情感分 | `choose_id` 与标准答案匹配的得分 |
| JSON 错误率 | 无法解析、字段缺失、key 不匹配或 `choose_id` 非法的样本比例 |
| 平均延迟 | 单样本从请求开始到生成结束的平均耗时 |
| 总分 | 按比赛或内部评测规则聚合后的总分 |

建议同时保留逐样本明细字段：`experiment_id`、`idx`、`model_name`、`prompt_type`、`shot_pool`、`raw_output`、`parsed_json`、`json_valid`、`json_error_type`、`latency_ms`、`word_score`、`sent_score`、`emotion_score`、`total_score`。

## 8. 需要人工确认的问题

1. 词义分和翻译分使用人工评分、规则匹配、LLM judge，还是官方评测脚本。
   A：未知
   建议处理：优先查找并使用官方评测脚本；若官方脚本暂缺，内部 baseline 临时采用 `LLM judge + 小规模人工抽查校准`。规则匹配只用于 JSON、key 覆盖和选项合法性检查，不作为词义/翻译语义分的主评分方法。
2. 总分的权重是否由官方规则确定；若无官方权重，内部 baseline 是否需要固定临时权重。
   A：未确定
   建议处理：官方规则确定前，内部临时总分使用 `0.30 * 词义分 + 0.40 * 翻译分 + 0.30 * 情感分`，同时所有实验表必须保留三项分数，避免临时总分掩盖具体退化。
3. P14 / P8 / P4 的具体模型名称、量化方式和推理后端是否需要写入实验 ID。
   A：具体模型名称、方案需要在实施时讨论
   建议处理：实施前单独做模型选择调研。实验 ID 至少记录模型全名、参数规模、量化方式、推理后端、prompt 类型、shot 数和主要解码参数；推荐格式为 `{group}-{model_family}-{size}-{quant}-{backend}-{prompt}`，例如 `P8-qwen3-8b-int4-vllm-few3`。
4. few-shot 样例池是否允许使用开发集答案，还是必须仅使用训练集答案。
   A：仅使用训练集答案
5. baseline 主结果是否严格禁止 retry；如果允许 retry，是否作为独立实验记录。
   A：有输出情况下不允许retry
