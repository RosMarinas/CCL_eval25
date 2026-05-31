# Reasoner-to-Formatter Harness 设计

## 目标和边界

本 harness 用于最小两阶段推理：

```text
reasoner -> local validator -> formatter -> final validator
```

目标是稳定得到合法、完整、简洁的最终 JSON。Reasoner 负责做题，输出结构化证据和草稿答案；formatter 默认不重新做题，只做 JSON 化、缺项补齐和轻量一致性检查。

本文区分两个容易混淆的执行路径：

- **Submission runner**：模型直接生成最终提交 JSON，用于最终答案指标、JSON 错误率和延迟评测。当前 `src/cli/run_submission_eval.py` 属于这一类。
- **True harness**：Reasoner 输出 `evidence`、`sentiment` 和 `draft_answer`，其中 `draft_answer` 不含 `choose_id`；Formatter 或本地 mapper 根据 `sentiment + task.choose` 生成最终 `choose_id`。当前 `src/cli/run_harness.py` 以这一契约为目标。本文的 schema 和 validator 规则以 true harness 为准。

本文使用 `docs/spec/data-schema.md` 中的统一输入 schema：

```json
{
  "idx": 0,
  "title": "诗题",
  "author": "作者",
  "content": "原诗文本",
  "qa_words": ["衰草"],
  "qa_sents": ["故关衰草遍，离别自堪悲"],
  "choose": {
    "A": "选项文本",
    "B": "选项文本",
    "C": "选项文本",
    "D": "选项文本"
  }
}
```

## Reasoner 输出 schema

Reasoner 必须输出一个 JSON 对象，不输出自由长 CoT：

```json
{
  "idx": 0,
  "evidence": {
    "words": {
      "衰草": {
        "meaning": "枯黄的草",
        "text_clue": "故关衰草遍",
        "rationale": "衰草烘托荒凉离别的氛围"
      }
    },
    "sentences": {
      "故关衰草遍，离别自堪悲": {
        "translation": "旧关一带遍布枯草，离别本就令人悲伤",
        "key_images": ["故关", "衰草", "离别"],
        "rationale": "以衰败景象写离别之悲"
      }
    },
    "emotion": [
      "衰草、离别、悲等词指向伤感",
      "诗句整体不是昂扬或闲适情绪"
    ]
  },
  "sentiment": {
    "primary": "惜别感伤",
    "secondary": ["羁旅漂泊"],
    "rationale": "衰草、离别、悲、掩泪、风尘何处期等意象共同指向离别之伤感和前路茫茫"
  },
  "draft_answer": {
    "ans_qa_words": {
      "衰草": "枯黄的草，烘托荒凉离别的氛围"
    },
    "ans_qa_sents": {
      "故关衰草遍，离别自堪悲": "旧关一带长满枯草，分别本就令人悲伤"
    }
  }
}
```

字段约束：

- `idx` 必须等于原题 `idx`。
- `evidence.words` 的 key 覆盖 `qa_words` 去重后的词语；value 为嵌套对象，含 `meaning`（简明词义）、`text_clue`（诗中依据）和 `rationale`（不超过 40 字）。
- `evidence.sentences` 的 key 覆盖 `qa_sents` 去重后的句子；value 为嵌套对象，含 `translation`（现代汉语直译/意译）、`key_images`（意象列表）和 `rationale`（不超过 40 字）。
- `evidence.emotion` 是短证据数组，每条只写判断依据，不写完整推理过程。
- `sentiment` 是结构化情感分析，包含 `primary`（主要情感标签）、`secondary`（次要情感标签列表，可选）和 `rationale`（判断依据）。
- `sentiment.primary` 必须从情感标签受控词汇表中选择（参见 `docs/spec/data-schema.md` 第 3.2 节），2-8 个中文字符。
- `sentiment.rationale` 不超过 80 字，引用诗中关键词和意象。
- `draft_answer` 必须包含 `ans_qa_words`、`ans_qa_sents`，**不包含 `choose_id`**。
- `draft_answer` 不包含 `idx`，避免和最终提交字段混淆；最终 `idx` 由 harness 注入。
- `choose_id` 由 Formatter（Stage2）根据 `sentiment` + 选项映射生成，Reasoner 不输出。

## Formatter 输入 schema

Formatter 接收原题、reasoner 输出和本地验证结果：

```json
{
  "task": {
    "idx": 0,
    "title": "诗题",
    "author": "作者",
    "content": "原诗文本",
    "qa_words": ["衰草"],
    "qa_sents": ["故关衰草遍，离别自堪悲"],
    "choose": {
      "A": "选项文本",
      "B": "选项文本",
      "C": "选项文本",
      "D": "选项文本"
    }
  },
  "reasoner_output": {
    "idx": 0,
    "evidence": {
      "words": {},
      "sentences": {},
      "emotion": []
    },
    "sentiment": {
      "primary": "",
      "secondary": [],
      "rationale": ""
    },
    "draft_answer": {
      "ans_qa_words": {},
      "ans_qa_sents": {}
    }
  },
  "validator_report": {
    "valid_json": true,
    "missing_fields": [],
    "missing_words": [],
    "missing_sentences": [],
    "sentiment_primary_missing": false,
    "sentiment_not_in_vocab": [],
    "invalid_choose_id": false,
    "overlong_fields": [],
    "suspected_conflicts": []
  }
}
```

Formatter 只输出最终提交 JSON：

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

## Formatter prompt

```text
你是古诗词理解任务的 formatter / verifier。

输入包括原题 task、reasoner_output 和 validator_report。
你的任务是把 reasoner 的 draft_answer 整理成最终提交 JSON，并根据 sentiment 分析生成 choose_id。

必须遵守：
1. 默认相信 reasoner 的 draft_answer，不要重新做题。
2. 不要输出推理过程、解释、Markdown 或代码块，只输出一个 JSON 对象。
3. 最终 JSON 只能包含 idx、ans_qa_words、ans_qa_sents、choose_id 四个字段。
4. idx 必须使用 task.idx。
5. ans_qa_words 的 key 必须覆盖 task.qa_words 去重后的词语。
6. ans_qa_sents 的 key 必须覆盖 task.qa_sents 去重后的句子。
7. choose_id 由你根据 reasoner 的 sentiment 分析生成，必须来自 task.choose 的选项 ID；标准样本应为 A、B、C、D 之一。
8. 答案应简洁：词义答案通常不超过 40 个中文字符，句子翻译通常不超过 80 个中文字符。
9. 情感映射：根据 reasoner 的 sentiment 分析和 task.choose 的选项，选择最匹配全诗情感的选项 ID。优先根据 sentiment.primary 与选项文本的语义匹配程度判断；如有歧义，参考 sentiment.secondary 和 sentiment.rationale。
10. 只在以下情况允许轻微修改 draft_answer：
    - JSON 不合法或字段名错误；
    - 缺少目标词、目标句或 idx；
    - 答案明显过长、变成赏析，需压缩；
    - draft_answer 与 evidence 或选项存在明显直接冲突；
    - sentiment 与选项的映射需要调整 draft 的细节（例如占位答案不足以覆盖选项情感）。
11. 如果缺项无法根据 draft_answer 和 evidence 补齐，使用最短占位式答案，不要发明长解释。

输出最终 JSON。
```

## Local validator 规则

本地验证器不调用模型，只做确定性检查。

Reasoner 后验证：

- JSON 解析：无法解析为对象，记为 `valid_json=false`。
- 必需字段：检查 `idx`、`evidence`、`sentiment`、`sentiment.primary`、`sentiment.rationale`、`draft_answer`、`draft_answer.ans_qa_words`、`draft_answer.ans_qa_sents`。
- `idx`：必须等于原题 `idx`。
- 情感合法性：`sentiment.primary` 必须为非空字符串；`sentiment.primary` 必须属于受控词汇表（参见 `docs/spec/data-schema.md` 第 3.2 节）。不合法时标记 `sentiment_primary_missing` 或 `sentiment_not_in_vocab`。
- 目标覆盖：`ans_qa_words` 必须包含所有 `qa_words` 去重后的词语；`ans_qa_sents` 必须包含所有 `qa_sents` 去重后的句子。
- 长度：词义答案建议不超过 40 个中文字符；句子翻译建议不超过 80 个中文字符。超限不直接判失败，但标记 `overlong_fields`。`sentiment.rationale` 建议不超过 80 个中文字符。
- 空值：空字符串、`null`、空对象中的目标 key 视为缺项。
- 证据完整性：`evidence.words` 和 `evidence.sentences` 缺 key 时标记，但不一定阻止生成最终答案。
- 明显冲突：若 reasoner 输出了 `draft_answer.choose_id`（不应出现），标记 `suspected_conflicts` 并拦截该字段（使用 `choose_id = ""` 占位交由 Formatter 处理）。

Final 后验证：

- 最终输出必须是单个 JSON 对象。
- 只能包含 `idx`、`ans_qa_words`、`ans_qa_sents`、`choose_id` 四个顶层字段。
- `idx`、目标覆盖、选项合法、空值和长度规则同上。
- 最终 JSON 中不得包含 `evidence`、`draft_answer`、`reasoning`、`analysis` 等中间字段。

## 跳过 formatter 的条件

满足以下条件时可跳过 formatter，只用规则 postprocess 生成最终 JSON：

- Reasoner 输出是合法 JSON。
- `idx` 正确。
- `draft_answer` 包含 `ans_qa_words` 和 `ans_qa_sents` 两个必需字段。
- `ans_qa_words` 覆盖所有目标词，且没有空值。
- `ans_qa_sents` 覆盖所有目标句，且没有空值。
- 没有答案超长字段。
- 没有 `suspected_conflicts`。
- 任务没有 `choose` 选项，即 `task.choose = {}`（此时无需情感映射；若存在选项，必须经过 Formatter 完成 sentiment→choose_id 映射）。

跳过时的规则 postprocess：

```text
final.idx = task.idx
final.ans_qa_words = reasoner_output.draft_answer.ans_qa_words
final.ans_qa_sents = reasoner_output.draft_answer.ans_qa_sents
final.choose_id = ""
```

## Retry / fallback policy

推荐最多三步，不做复杂多轮 agent：

1. 第一次调用 reasoner。
   - 若输出合法且满足跳过条件，直接规则 postprocess 后进入 final validator。
   - 若输出可解析但有缺项、超长或轻微冲突，调用 formatter。
2. 重试 reasoner 一次。
   - 仅当 reasoner 输出无法解析为 JSON，或缺少 `draft_answer`，或 `sentiment` 字段缺失/非法时触发。
   - 重试 prompt 应强调”只输出指定 JSON schema”，不改变题目内容。
3. Formatter 和规则兜底。
   - 若重试后仍有结构问题，先尝试从文本中提取最外层 JSON；能提取则送 formatter。
   - Formatter 输出通过 final validator，则采用。
   - Formatter 输出仍非法时，使用规则兜底：注入 `idx`，保留可解析的草稿字段；缺失目标词或目标句填入空字符串或最短占位答案；`choose_id` 非法时填入空字符串并记录错误，不猜选项。
   - Formatter 情感映射兜底：若 Formatter 输出的 `choose_id` 经 final validator 判定为非法（不是 `task.choose` 的现有选项 ID），尝试基于 `sentiment.primary` 与选项文本的关键词匹配（将 `primary` 标签拆分为单字或双字关键词，与每个选项文本计算字符级重叠度，选择重叠度最高的选项 ID）。若关键词匹配仍失败，`choose_id = “”` 并记录 `sentiment_mapping_failed` 错误类别。

失败记录至少包含：

- `idx`
- reasoner 是否重试
- formatter 是否调用
- final validator 错误类别
- 最终是否使用 fallback

## 需要人工确认的问题

- 官方正式数据字段名是否与 `docs/spec/data-schema.md` 的 `idx/title/author/content/qa_words/qa_sents/choose` 完全一致。
  A：一致
- 目标词或目标句重复出现时，最终 JSON key 是保留一次还是按出现位置区分。
  A：按位置区分
- 原题没有情感选项或选项数量不是四个时，`choose_id` 应输出空字符串、`null`，还是跳过该字段。
  A：均为4个，输出空即可
- 词义和句子翻译的长度阈值是否采用本文建议的 40 / 80 中文字符。
  A：是
- fallback 缺项时是否允许空字符串；若评测器不接受空字符串，需要统一最短占位策略。
  A：接受
- Formatter 根据 sentiment 生成 `choose_id` 时，如果 `sentiment.primary` 与多个选项语义接近，或者 sentiment 与选项整体情感存在偏差，如何决定选用哪个选项 ID。
  A：Formatter 的职责本身就是从 sentiment 生成 choose_id，不是修改 draft 中的 choose_id。优先使用 `sentiment.primary` 的语义匹配；若仍有歧义，参考 `sentiment.secondary` 和 `sentiment.rationale`。若全部匹配均无法确定合理选项，`choose_id = ""` 并记录 `sentiment_mapping_failed`。
