# Teacher Data 生成与过滤设计

本文定义离线教师数据生成流程，用于 BC mixed distillation 阶段构造 `short-evidence` 与 `teacher-critique` 样本。目标是让学生模型学习结构化证据、草稿答案和修正能力，而不是学习自由长 CoT。

## 1. 输入与输出

本文使用 `docs/data-schema.md` 中的统一输入 schema：

```json
{
  "idx": 0,
  "title": "诗题",
  "author": "作者",
  "content": "诗歌正文",
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

最终答案必须复用统一提交 schema：

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

## 2. Teacher Prompt

### 2.1 Short-evidence 生成 prompt

```text
你是古诗词理解任务的教师模型。请根据输入题目生成结构化短证据和草稿答案。

硬性要求：
1. 只输出一个合法 JSON 对象，不输出 Markdown、解释文字或代码块。
2. 不要输出自由长 CoT，不要写逐步推理过程。
3. evidence 只能写短证据：词义线索、句意骨架、情感线索；每条 rationale 不超过 40 个中文字符。
4. draft_answer 和 final_answer 必须使用同一套最终答案 schema。
5. ans_qa_words 必须覆盖所有 qa_words 去重后的词语，key 必须与输入完全一致。
6. ans_qa_sents 必须覆盖所有 qa_sents 去重后的句子，key 必须与输入完全一致。
7. choose_id 必须来自 choose 的选项 ID；标准样本应为 A/B/C/D 中的一个。
8. 如果题目缺少情感选项，choose_id 使用空字符串，并在 quality_flags 中加入 "missing_emotion_options"。

输入题目：
{{TASK_JSON}}

请输出：
{
  "record_type": "short_evidence",
  "idx": <idx>,
  "evidence": {
    "words": {
      "<target_word>": {
        "meaning": "<简明词义>",
        "text_clue": "<诗中依据，短语或句子片段>",
        "rationale": "<短理由，不超过40字>"
      }
    },
    "sentences": {
      "<target_sentence>": {
        "translation": "<现代汉语直译或意译>",
        "key_images": ["<意象1>", "<意象2>"],
        "rationale": "<短理由，不超过40字>"
      }
    },
    "emotion": [
      {
        "option_id": "<A|B|C|D>",
        "support": "<支持该选项的短证据>",
        "polarity": "<positive|negative|mixed|neutral>",
        "rationale": "<短理由，不超过40字>"
      }
    ]
  },
  "draft_answer": {
    "idx": <idx>,
    "ans_qa_words": {},
    "ans_qa_sents": {},
    "choose_id": "<A|B|C|D或空字符串>"
  },
  "final_answer": {
    "idx": <idx>,
    "ans_qa_words": {},
    "ans_qa_sents": {},
    "choose_id": "<A|B|C|D或空字符串>"
  },
  "quality_flags": []
}
```

### 2.2 Teacher-critique 生成 prompt

```text
你是古诗词理解任务的教师模型。请根据输入题目和一个候选错误答案，生成结构化批改意见与修正答案。

硬性要求：
1. 只输出一个合法 JSON 对象，不输出 Markdown、解释文字或代码块。
2. 不要输出自由长 CoT，不要写逐步推理过程。
3. critique 只能指出可验证的错误类型和短理由；每条 comment 不超过 50 个中文字符。
4. corrected_answer 必须使用最终答案 schema。
5. 不要改写正确且足够简洁的字段。
6. 如果无法判断某字段是否错误，将 error_type 写为 "uncertain"，并在 quality_flags 中加入 "needs_human_review"。

输入题目：
{{TASK_JSON}}

候选答案：
{{WRONG_ANSWER_JSON}}

请输出：
{
  "record_type": "teacher_critique",
  "idx": <idx>,
  "candidate_answer": {{WRONG_ANSWER_JSON}},
  "critique": {
    "word_errors": [
      {
        "target": "<target_word>",
        "error_type": "<missing|wrong_meaning|overlong|unsupported|correct|uncertain>",
        "comment": "<短批注意见>"
      }
    ],
    "sentence_errors": [
      {
        "target": "<target_sentence>",
        "error_type": "<missing|wrong_translation|overlong|unsupported|correct|uncertain>",
        "comment": "<短批注意见>"
      }
    ],
    "emotion_error": {
      "candidate_choose_id": "<A|B|C|D或空字符串>",
      "correct_choose_id": "<A|B|C|D或空字符串>",
      "error_type": "<wrong_option|invalid_option|missing|correct|uncertain>",
      "comment": "<短批注意见>"
    }
  },
  "correction_evidence": {
    "words": {},
    "sentences": {},
    "emotion": []
  },
  "corrected_answer": {
    "idx": <idx>,
    "ans_qa_words": {},
    "ans_qa_sents": {},
    "choose_id": "<A|B|C|D或空字符串>"
  },
  "quality_flags": []
}
```

## 3. Short-evidence Schema

`short_evidence` 样本用于训练 reasoner 输出结构化证据和草稿答案，也可直接抽取 `final_answer` 作为 answer-only replay 数据。

```json
{
  "record_type": "short_evidence",
  "idx": 0,
  "source": {
    "teacher_model": "model-name",
    "prompt_version": "teacher-data-v1",
    "created_at": "YYYY-MM-DD"
  },
  "task": {
    "idx": 0,
    "title": "",
    "author": "",
    "content": "",
    "qa_words": [],
    "qa_sents": [],
    "choose": {}
  },
  "evidence": {
    "words": {
      "<target_word>": {
        "meaning": "string",
        "text_clue": "string",
        "rationale": "string"
      }
    },
    "sentences": {
      "<target_sentence>": {
        "translation": "string",
        "key_images": ["string"],
        "rationale": "string"
      }
    },
    "emotion": [
      {
        "option_id": "A",
        "support": "string",
        "polarity": "positive"
      }
    ]
  },
  "draft_answer": {
    "idx": 0,
    "ans_qa_words": {},
    "ans_qa_sents": {},
    "choose_id": "D"
  },
  "final_answer": {
    "idx": 0,
    "ans_qa_words": {},
    "ans_qa_sents": {},
    "choose_id": "D"
  },
  "quality_flags": []
}
```

字段约束：

- `evidence.words` 的 key 集合必须等于 `task.qa_words` 去重后的集合。
- `evidence.sentences` 的 key 集合必须等于 `task.qa_sents` 去重后的集合。
- `meaning` 建议 6-25 个中文字符，避免百科式解释。
- `translation` 建议 10-60 个中文字符，保持现代汉语句意，不写赏析。
- `rationale` 只允许短证据说明，不允许出现“首先、然后、所以我认为”等推理链展开。
- `draft_answer` 允许与 `final_answer` 相同；如果不同，必须是 teacher 自行修正后的更优答案。
- `quality_flags` 仅使用约定枚举，如 `missing_emotion_options`、`needs_human_review`、`low_confidence_emotion`。

## 4. Teacher-critique Schema

`teacher_critique` 样本用于训练模型识别错误答案并输出修正 JSON，重点覆盖相近情感选项、词义误解、翻译过度赏析和缺项问题。

```json
{
  "record_type": "teacher_critique",
  "idx": 0,
  "source": {
    "teacher_model": "model-name",
    "prompt_version": "teacher-critique-v1",
    "created_at": "YYYY-MM-DD",
    "candidate_source": "baseline|student|synthetic"
  },
  "task": {
    "idx": 0,
    "title": "",
    "author": "",
    "content": "",
    "qa_words": [],
    "qa_sents": [],
    "choose": {}
  },
  "candidate_answer": {
    "idx": 0,
    "ans_qa_words": {},
    "ans_qa_sents": {},
    "choose_id": "A"
  },
  "critique": {
    "word_errors": [
      {
        "target": "衰草",
        "error_type": "wrong_meaning",
        "comment": "误解为茂盛草木，与衰败语境相反"
      }
    ],
    "sentence_errors": [
      {
        "target": "故关衰草遍，离别自堪悲",
        "error_type": "correct",
        "comment": "句意完整且简洁"
      }
    ],
    "emotion_error": {
      "candidate_choose_id": "A",
      "correct_choose_id": "D",
      "error_type": "wrong_option",
      "comment": "离别、衰草更贴近悲伤情绪"
    }
  },
  "correction_evidence": {
    "words": {
      "衰草": "衰败枯黄之草，指向荒凉"
    },
    "sentences": {
      "故关衰草遍，离别自堪悲": "旧关遍是衰草，离别本就令人悲伤"
    },
    "emotion": [
      {
        "option_id": "D",
        "support": "衰草、离别共同指向悲伤"
      }
    ]
  },
  "corrected_answer": {
    "idx": 0,
    "ans_qa_words": {
      "衰草": "枯黄的草，烘托荒凉离别的氛围"
    },
    "ans_qa_sents": {
      "故关衰草遍，离别自堪悲": "旧关一带长满枯草，分别本就令人悲伤"
    },
    "choose_id": "D"
  },
  "quality_flags": []
}
```

字段约束：

- `candidate_answer` 保留原始候选答案，不做静默修复。
- `critique.*_errors` 必须覆盖所有目标词和目标句。
- `error_type` 必须使用枚举值，不能自由发挥。
- `comment` 是批注意见，不是完整推理过程。
- `corrected_answer` 必须是可直接用于 answer-only 训练或最终评测的答案 JSON。

## 5. 自动过滤规则

生成后先进行本地规则过滤，再进入人工抽查池。

### 5.1 JSON 与字段合法性

- JSON 必须能被严格解析，且根节点必须是 object。
- `record_type` 只能是 `short_evidence` 或 `teacher_critique`。
- `idx` 必须与输入 `task.idx` 一致。
- 必填字段缺失时直接丢弃：`task`、`evidence` 或 `critique`、`draft_answer`、`final_answer` 或 `corrected_answer`。
- 不允许出现自由 CoT 字段名：`cot`、`chain_of_thought`、`reasoning`、`steps`、`analysis`。
- 任一字符串字段包含 Markdown 代码块标记、明显提示词残留或多段长推理时直接丢弃。

### 5.2 覆盖与选项合法性

- `ans_qa_words` 的 key 集合必须等于 `qa_words` 去重集合；缺项、多项、key 改写均过滤。
- `ans_qa_sents` 的 key 集合必须等于 `qa_sents` 去重集合；缺项、多项、key 改写均过滤。
- 当 `choose` 非空时，`choose_id` 必须属于 `choose` 的现有选项 ID；标准样本应为 `A/B/C/D`。
- 当 `choose` 为空时，`choose_id` 必须为空字符串，且必须带 `missing_emotion_options`。
- `teacher_critique.critique.emotion_error.correct_choose_id` 必须与 `corrected_answer.choose_id` 一致。

### 5.3 长度与风格

- 词义答案建议 6-35 个中文字符；超过 50 个中文字符进入人工复核，超过 80 个中文字符过滤。
- 句子翻译建议 10-80 个中文字符；超过 120 个中文字符进入人工复核，超过 180 个中文字符过滤。
- `rationale`、`comment`、`support` 单字段超过 60 个中文字符进入人工复核，超过 100 个中文字符过滤。
- 禁止输出赏析文风，如大段“表达了诗人……”堆叠；只保留解释、翻译和短证据。

### 5.4 证据与答案一致性

- `evidence.words[*].meaning` 与 `final_answer.ans_qa_words[*]` 不应语义相反；若出现明显反义词冲突，过滤。
- `evidence.sentences[*].translation` 与 `final_answer.ans_qa_sents[*]` 不应漏掉主要主语、动作或情感词；疑似漏译进入人工复核。
- `evidence.emotion[*].option_id` 至少应包含最终 `choose_id`；否则进入人工复核。
- `polarity` 与选项文本明显冲突时过滤。例如 evidence 标为 `positive`，但选择项为“悲伤、凄凉”。
- `teacher_critique` 中标记为 `correct` 的字段，不应在 `corrected_answer` 中被大幅改写；若改写超过长度或词面相似阈值，进入人工复核。
- `error_type` 为 `wrong_option` 时，`candidate_choose_id` 与 `correct_choose_id` 必须不同；否则过滤。

### 5.5 去重与采样

- 同一 `idx`、同一 `record_type`、同一 `final_answer` 或 `corrected_answer` 的重复样本只保留一条。
- 同一题可保留多条 teacher 样本，但情感选项 critique 不应被单一错误类型垄断。
- 对 `quality_flags` 含 `needs_human_review` 或 `low_confidence_emotion` 的样本降权采样，不进入高置信训练集。

## 6. 人工抽查 Checklist

人工抽查优先看自动过滤后的样本，建议每批至少抽查：

- 每个 teacher 模型输出的 5%-10%。
- 所有带 `needs_human_review`、`low_confidence_emotion` 的样本。
- 情感选项相近、教师之间答案不一致、长度接近阈值的样本。

逐条检查：

- 目标词是否全部覆盖，key 是否与原题完全一致。
- 词义是否符合诗句语境，是否误把字面义当作语境义。
- 词义答案是否简洁，是否夹带长篇赏析。
- 目标句是否全部覆盖，key 是否与原题完全一致。
- 句子翻译是否保留主要意象、动作、情感和否定关系。
- 翻译是否是现代汉语解释，而不是泛泛赏析。
- `choose_id` 是否来自合法选项，是否与诗歌整体情感一致。
- 情感短证据是否能支持所选选项，是否忽略转折、反讽或尾联变化。
- `teacher_critique` 是否准确定位候选答案错误，而不是为了修改而修改。
- 标记为 `correct` 的字段是否确实不需要修改。
- `corrected_answer` 是否可直接作为最终答案训练目标。
- 输出中是否存在自由长 CoT、分步推理、提示词残留或 Markdown。

## 7. 禁止自由长 CoT 的约束说明

本项目训练目标是 `structured evidence + draft_answer` 和最终 JSON，不训练自由长 CoT。原因：

- 自由长 CoT 会提高输出长度和格式错误率，不符合提交 JSON 的目标。
- 自由推理文本难以稳定过滤，容易混入不可验证或幻觉内容。
- BC 需要学生学习可复用的短证据字段，而不是模仿教师的冗长思考风格。
- Harness 中 formatter 只应格式化和轻量校验，不应从长推理文本重新做题。

允许输出的推理相关内容仅限以下结构化短字段：

- `meaning`
- `text_clue`
- `translation`
- `key_images`
- `support`
- `rationale`
- `comment`

这些字段必须短、可审计、可映射到原题，不能包含完整推理链。

## 8. 需要人工确认的问题

- 词义和句子翻译的长度阈值是否需要按官方评分细则进一步收紧。
  A：先不进一步收紧，保留建议阈值与过滤阈值两层规则。词义答案建议 6-35 个中文字符，超过 50 进入人工复核，超过 80 过滤；句子翻译建议 10-80 个中文字符，超过 120 进入人工复核，超过 180 过滤；`rationale/comment/support` 超过 60 复核，超过 100 过滤。
- `teacher_critique` 的候选错误答案来源比例如何分配：baseline、学生模型、合成扰动各占多少。
  A：第一版有学生模型时使用 `baseline 50% / student 30% / synthetic 20%`。若 B8 学生模型尚未产出，先使用 `baseline 70% / synthetic 30%`；B8 可用后切回 `50/30/20`。
- 含 `needs_human_review` 的样本是否完全排除训练，还是只降权用于 critique 增强。
  A：默认不进入 answer-only、short-evidence 主训练集。人工确认通过后可升级为高置信样本；未确认但只有轻微不确定的样本，可用于 critique 增强并降低采样权重，例如 `0.2x`。
- 多个教师模型答案冲突时，采用投票、强模型优先，还是人工仲裁。
  A：采用分层仲裁，不做简单投票。schema 非法直接丢弃；多教师一致则保留为高置信；词义/翻译表达不同但语义兼容时保留最简洁版本；`choose_id` 冲突时进入人工仲裁或强模型复审；复审仍冲突则标记 `low_confidence_emotion`，不进入主训练。
