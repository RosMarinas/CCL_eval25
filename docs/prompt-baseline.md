# P14 / P8 / FMT Prompt Baseline 实验设计

## 1. 范围与依赖

本文设计不微调模型的 baseline，用于评估：

- `P14`：14B 级 prompt-only 上限。
- `P14-fast`：14B 量化版本的速度 / 效果对照。
- `P8`：8B 级候选 reasoner sweep，用于选择 B8 / BC8 微调基座。
- `FMT`：formatter / verifier baseline，用于选择 harness 第二阶段模型。

依赖：

- 统一输入和最终输出 schema：`docs/data-schema.md`
- harness 输入输出与 validator 规则：`docs/harness.md`
- 模型候选、license、参数量和部署支持：`docs/model-research.md`

本文只定义实验协议和 prompt 模板。具体模型名称由模型调研确定；第一轮不使用 Qwen3.5-9B 作为 P8 主候选。

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

## 2. 实验组

| 实验 | 第一轮模型 | 目的 | 默认配置 |
| --- | --- | --- | --- |
| `P14` | `Qwen/Qwen3-14B` | prompt-only 上限 | bf16，vLLM，non-thinking，zero/few-shot |
| `P14-fast` | `Qwen/Qwen3-14B-AWQ` | 14B 量化速度 / 效果对照 | AWQ 4-bit，vLLM，non-thinking，zero/few-shot |
| `P8` | `Qwen/Qwen3-8B`、`Qwen/Qwen3-8B-AWQ`、`internlm/internlm3-8b-instruct` | 选择微调基座 | Qwen3 使用 non-thinking；InternLM3 normal response 为主，deep thinking 只 smoke |
| `FMT` | `google/gemma-4-E4B-it`、`Qwen/Qwen3-8B`、`Qwen/Qwen3-8B-AWQ` | 选择 harness formatter / verifier | Gemma 4 E4B 优先 Transformers；Qwen3 使用 vLLM non-thinking |

暂缓项：

- Qwen3.5-9B 暂不进入第一轮；如模型调研证明其 license、vLLM 支持、参数预算和效果明显更合适，再作为第二轮备选。
- DeepSeek-R1-Distill-Qwen、Phi-4 / Phi-4-mini、gpt-oss-20b 等暂缓；原因见 `docs/model-research.md`。
- 需要自定义推理框架、闭源权重、license 不清晰或总参数预算难以确认的模型暂缓。

## 3. 通用生成约束

P14 / P14-fast / P8 使用同一做题 prompt；只允许调整模型、shot 数、thinking 模式和解码参数。模板不使用模型专属特殊 token，具体 chat template 由推理后端按模型官方方式渲染。

生成约束：

1. 只输出最终 JSON。
2. 不输出 Markdown 代码块。
3. 不输出解释、分析、证据、草稿或多余文本。
4. `idx` 必须与输入一致。
5. `ans_qa_words` 的 key 必须覆盖所有 `qa_words` 去重后的词语。
6. `ans_qa_sents` 的 key 必须覆盖所有 `qa_sents` 去重后的句子。
7. `choose_id` 必须从 `choose` 的选项 ID 中选择；标准样本为 `A`、`B`、`C`、`D` 之一。
8. 词义和句意回答保持简洁，避免扩写成赏析。

建议解码参数按模型调研和官方建议填写到实验 ID metadata 中。主 baseline 优先使用非 thinking 模式；如评估 thinking 模式，必须单独成实验，且仍要求最终只输出 JSON。

## 4. Zero-shot Prompt 模板

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

## 5. Few-shot Prompt 模板

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
| `P14` | 5 | 上下文和能力较强，优先覆盖更多题型 |
| `P14-fast` | 3 | 控制量化模型延迟，同时保留格式示范 |
| `P8` | 3 | 平衡格式示范和延迟 |

## 6. FMT Formatter Baseline

FMT 不重新做 prompt-only 答题上限，而是评估候选 formatter / verifier 能否把 reasoner 草稿稳定整理为最终 JSON。

### 6.1 FMT 输入协议

FMT 输入复用 harness 的 formatter input：

```json
{
  "task": {
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
  },
  "reasoner_output": {
    "idx": 0,
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
  },
  "validator_report": {
    "valid_json": true,
    "missing_fields": [],
    "missing_words": [],
    "missing_sentences": [],
    "invalid_choose_id": false,
    "overlong_fields": [],
    "suspected_conflicts": []
  }
}
```

FMT 输出必须是最终 JSON，不能输出 `evidence`、`draft_answer`、`validator_report` 或解释文字。

### 6.2 Formatter Prompt 模板

```text
你是古诗词理解任务的 formatter / verifier。

输入包括 task、reasoner_output 和 validator_report。
你的任务是把 reasoner_output.draft_answer 整理成最终提交 JSON，并做轻量校验。

必须遵守：
1. 默认相信 draft_answer，不要重新做题。
2. 不要输出推理过程、解释、Markdown 或代码块，只输出一个 JSON 对象。
3. 最终 JSON 只能包含 idx、ans_qa_words、ans_qa_sents、choose_id 四个字段。
4. idx 必须使用 task.idx。
5. ans_qa_words 的 key 必须覆盖 task.qa_words 去重后的词语。
6. ans_qa_sents 的 key 必须覆盖 task.qa_sents 去重后的句子。
7. choose_id 必须来自 task.choose 的选项 ID。
8. 只在 validator_report 指出结构问题、缺项、非法选项、过长答案或明显证据冲突时，才轻微修改 draft_answer。
9. 如果缺项无法根据 draft_answer 和 evidence 补齐，使用空字符串占位，不要发明长解释。

输入：
{{formatter_input_json}}

现在只输出最终 JSON：
```

### 6.3 FMT 测试集构造

FMT baseline 至少使用三类输入：

| 子集 | 来源 | 目的 |
| --- | --- | --- |
| `fmt-clean` | 合法 draft_answer | 测 formatter 是否保持正确草稿，不改坏 |
| `fmt-format-error` | 人工或规则注入字段缺失、额外字段、非法 JSON 包裹文本 | 测格式修复能力 |
| `fmt-light-conflict` | draft 与 evidence / choose 存在轻微直接冲突 | 测轻量 verifier 能否修正明显错误 |

FMT 不评估自由重做题能力。若 formatter 需要大幅重写词义、句译或情感选择，应计入改坏风险并交给 H2/H3 harness 决策。

## 7. Few-shot 样例选择策略

样例必须只来自训练集答案，不能从 dev/test 或待预测样本泄漏答案。

选择优先级：

1. 覆盖字段形态：至少包含 1 个单词语、1 个多词语、1 个单句翻译、1 个多句翻译样例。
2. 覆盖情感选项：在固定 shot 池中尽量让 `choose_id` 分布均衡。
3. 覆盖诗歌类型：混合送别、怀古、咏物、羁旅、山水等常见主题。
4. 控制答案长度：选择答案简洁且 JSON 完全合法的样例，避免模型模仿长篇赏析。
5. 控制相似度：不得选择同一首诗、同一 `idx` 或明显来自同一原题的样例。
6. 固定 shot 池：每轮 P14 / P14-fast / P8 使用同一套固定样例，便于横向比较。

推荐准备两个 shot 池：

| shot 池 | 用途 | 规模 |
| --- | --- | ---: |
| `balanced_static` | P14 / P8 主 baseline，固定顺序，保证可复现 | 5 |
| `short_static` | P14-fast 或上下文 / 延迟受限模型 | 3 |

若后续需要相似检索 few-shot，应单独记录为新实验，不混入主 baseline。

## 8. Baseline 推理流程

### 8.1 P14 / P14-fast / P8

1. 将官方原始样本转换为统一输入 schema。
2. 按实验编号选择 zero-shot 或 few-shot 模板。
3. 渲染 prompt，保留完整输入 JSON 和固定 shot 示例。
4. 调用对应模型生成一次答案，记录开始时间、结束时间和原始输出。
5. 解析原始输出为 JSON。
6. 执行本地校验：字段集合、`idx`、目标词 key、目标句 key、`choose_id` 合法性。
7. 不调用 formatter，不做语义修正，不补写缺失答案；非法或缺字段样本计入 JSON 错误。
8. 对合法 JSON 进入评测，分别计算词义分、翻译分、情感分和总分。
9. 汇总 JSON 错误率、平均延迟和任务分数。

为保证 prompt-only baseline 可比较，主结果默认不重试。有输出但 JSON 错误时直接计错；若必须评估 retry 收益，应另起实验名，例如 `P8-retry1`。

### 8.2 FMT

1. 构造 `fmt-clean`、`fmt-format-error`、`fmt-light-conflict` 输入。
2. 调用 formatter 候选模型，只允许输出最终 JSON。
3. 用 final validator 统计 JSON 修复率、coverage 修复率、formatter 改坏率和延迟。
4. 对比 Qwen3-8B、Gemma 4 E4B 等 formatter 候选。
5. 只把 formatter 净收益高、改坏率低、延迟可接受的模型推进 H2/H3。

## 9. 结果记录表字段

主表字段固定如下：

| 实验 | 模型 | Prompt / 输入 | Shot 数 | 词义分 | 翻译分 | 情感分 | JSON 错误率 | 平均延迟 | 总分 | 备注 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P14-zero | 14B | zero-shot | 0 | | | | | | | |
| P14-few | 14B | few-shot | 5 | | | | | | | |
| P14-fast-few | 14B-AWQ | few-shot | 3 | | | | | | | |
| P8-zero | 8B sweep candidate | zero-shot | 0 | | | | | | | |
| P8-few | 8B sweep candidate | few-shot | 3 | | | | | | | |
| FMT-clean | formatter candidate | fmt-clean | 0 | | | | | | | |
| FMT-format-error | formatter candidate | fmt-format-error | 0 | | | | | | | |
| FMT-light-conflict | formatter candidate | fmt-light-conflict | 0 | | | | | | | |

字段定义：

| 字段 | 含义 |
| --- | --- |
| 词义分 | `ans_qa_words` 的官方或内部评分均值 |
| 翻译分 | `ans_qa_sents` 的官方或内部评分均值 |
| 情感分 | `choose_id` 与标准答案匹配的得分 |
| JSON 错误率 | 无法解析、字段缺失、key 不匹配或 `choose_id` 非法的样本比例 |
| 平均延迟 | 单样本从请求开始到生成结束的平均耗时 |
| 总分 | 官方总分；无官方脚本时使用内部临时权重 |

建议同时保留逐样本明细字段：`experiment_id`、`idx`、`model_name`、`param_size`、`license`、`quantization`、`backend`、`mode`、`prompt_type`、`shot_pool`、`raw_output`、`parsed_json`、`json_valid`、`json_error_type`、`latency_ms`、`word_score`、`sent_score`、`emotion_score`、`total_score`。

## 10. 实验 ID 规范

实验 ID 必须记录模型全名、参数规模、量化方式、推理后端、thinking/non-thinking 模式、prompt 类型、shot 数和主要解码参数。

推荐格式：

```text
{group}-{model_family}-{size}-{quant}-{backend}-{mode}-{prompt}
```

示例：

```text
P14-qwen3-14b-bf16-vllm-nothink-few5
P14-fast-qwen3-14b-awq4-vllm-nothink-few3
P8-qwen3-8b-bf16-vllm-nothink-few3
FMT-gemma4-e4b-bf16-transformers-nothink-jsonfix
FMT-qwen3-8b-bf16-vllm-nothink-jsonfix
```

## 11. 待确认问题

1. 官方评测脚本若发布，需要替换内部临时评分方式。
2. FMT 的 `fmt-format-error` 和 `fmt-light-conflict` 子集需要在实现时固定生成规则，避免不同 formatter 候选看到不同难度。
3. `google/gemma-4-E4B-it` 的实际加载 ID 需在实施时以 Hugging Face card 和本地 transformers 可加载结果为准；若只存在 `google/gemma-4-E4B`，实验 ID 同步改名。
