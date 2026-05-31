# Teacher Data 生成与过滤设计

本文定义离线教师数据生成流程，用于 BC mixed distillation 阶段构造 `short-evidence` 与 `teacher-critique` 样本。目标是让学生模型学习结构化证据、草稿答案和修正能力，而不是学习自由长 CoT。

## 1. 输入与输出

本文使用 `docs/spec/data-schema.md` 中的统一输入 schema：

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

输出分为两种情况，取决于输入是否包含 `choose` 选项。

### 1.1 训练集样本输出（仅情感分析，无 `choose_id`）

训练集（`train-data/` 的 164 首诗）原本没有 `choose` 选项，Teacher 只生成情感分析（`sentiment`）和词义/句译，**不生成 `choose_id`**。输出匹配 Reasoner 中间输出 schema（见 `docs/spec/data-schema.md` §3）：

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
        "translation": "旧关一带长满枯草，分别本就令人悲伤",
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

**注意：** `draft_answer` 不含 `choose_id`，`evidence.emotion` 是短证据字符串数组（含情感判断依据），结构化情感分析由 `sentiment` 字段承载。这是 Reasoner 的输出格式，也是训练集 teacher 数据的输出标准。

### 1.2 评测 dev split 样本输出（情感分析 + 选项选择）

评测 dev split 的样本包含完整 `choose` 选项（A/B/C/D），Teacher 可在 `sentiment` 基础上额外生成 `choose_id`，供 Formatter 的 sentiment→option 映射训练使用。注意 `draft_answer` 与训练集一致，不包含 `choose_id`；`choose_id` 仅出现在 `final_answer` 中。

```json
{
  "idx": 0,
  "evidence": {
    "words": {
      "衰草": {
        "meaning": "枯黄的草",
        "text_clue": "故关衰草遍",
        "rationale": "烘托荒凉离别的氛围"
      }
    },
    "sentences": {
      "故关衰草遍，离别自堪悲": {
        "translation": "旧关一带长满枯草，分别本就令人悲伤",
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
  },
  "final_answer": {
    "idx": 0,
    "ans_qa_words": {
      "衰草": "枯黄的草，烘托荒凉离别的氛围"
    },
    "ans_qa_sents": {
      "故关衰草遍，离别自堪悲": "旧关一带长满枯草，分别本就令人悲伤"
    },
    "choose_id": "D"
  }
}
```

## 2. Teacher Prompt

### 2.1 Short-evidence 生成 prompt

```text
你是古诗词理解任务的教师模型。请根据输入题目生成结构化短证据、情感分析和草稿答案。

硬性要求：
1. 只输出一个合法 JSON 对象，不输出 Markdown、解释文字或代码块。
2. 不要输出自由长 CoT，不要写逐步推理过程。
3. evidence 只能写短证据：词义线索、句意骨架、情感线索；每条 rationale 不超过 40 个中文字符。
4. ans_qa_words 必须覆盖所有 qa_words 去重后的词语，key 必须与输入完全一致。
5. ans_qa_sents 必须覆盖所有 qa_sents 去重后的句子，key 必须与输入完全一致。
6. 情感分析使用 `sentiment` 字段，包含 `primary`（主要情感）、`secondary`（次要情感列表，可选）和 `rationale`（判断依据）。`sentiment.primary` 和 `sentiment.secondary` 中的每个标签必须从受控词汇表中选择（惜别感伤、送别不舍、离别愁绪、思乡怀远、羁旅思归、故园之思、忧国伤时、报国壮志、兴亡之叹、山水闲适、田园之乐、隐逸情怀、怀古伤今、历史沧桑、昔盛今衰、相思闺怨、爱情甜蜜、相思之苦、人生无常、时光易逝、仕途失意、边塞征战、将士艰辛、厌战思归、其他）。若无法归入以上标签，使用"其他"并在 quality_flags 中标记 needs_human_review。
7. evidence.emotion 是短证据字符串数组（每条 ≤ 60 字），包含情感判断依据，不包含 option_id。
8. draft_answer 仅包含词义和句译，不包含 choose_id。
9. final_answer 仅在题目包含 choose 选项（非空）时才输出 choose_id，且 choose_id 必须来自 choose 的选项 ID；若 choose 为空，不输出 choose_id 字段。
10. 若题目缺少情感选项，在 quality_flags 中加入 "missing_emotion_options"。

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
      "<情感判断依据1，不超过60字>",
      "<情感判断依据2，不超过60字>"
    ]
  },
  "sentiment": {
    "primary": "<主要情感标签，从受控词汇表选择>",
    "secondary": ["<次要情感标签1>", "<次要情感标签2>"],
    "rationale": "<情感判断依据，不超过80字，引用诗中关键词和意象>"
  },
  "draft_answer": {
    "ans_qa_words": {},
    "ans_qa_sents": {}
  }
  <if-task-has-choose>\
  ,"final_answer": {
    "ans_qa_words": {},
    "ans_qa_sents": {},
    "choose_id": "<A|B|C|D>"
  }\
  <endif>\
  ,
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
7. emotion_error 评价候选答案中 sentiment 分析的准确性，而非 choose_id 的正误。
8. sentiment.primary 和 sentiment.secondary 中的每个标签必须来自受控词汇表。

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
      "candidate_primary": "<候选答案中的 sentiment.primary>",
      "correct_primary": "<正确的 sentiment.primary 标签>",
      "candidate_secondary": ["<候选答案中的 sentiment.secondary 标签列表>"],
      "correct_secondary": ["<正确的 sentiment.secondary 标签列表>"],
      "primary_error_type": "<wrong_label|not_in_vocab|missing|correct|uncertain>",
      "secondary_error_type": "<extra_label|missing_label|wrong_label|correct|uncertain>",
      "rationale_error_type": "<no_evidence|contradicts_vocab|correct|uncertain>",
      "comment": "<短批注意见>"
    }
  },
  "correction_evidence": {
    "words": {},
    "sentences": {},
    "emotion": [
      "<情感判断依据1>",
      "<情感判断依据2>"
    ]
  },
  "corrected_sentiment": {
    "primary": "<正确的 sentiment.primary 标签>",
    "secondary": ["<正确的 sentiment.secondary 标签列表>"],
    "rationale": "<正确的情感判断依据>"
  },
  "corrected_answer": {
    "idx": <idx>,
    "ans_qa_words": {},
    "ans_qa_sents": {}
  },
  "quality_flags": []
}
```

## 3. Short-evidence Schema

`short_evidence` 样本用于训练 reasoner 输出结构化证据、情感分析和草稿答案，也可直接抽取 `draft_answer` 作为 answer-only replay 数据（仅词义和句译）。

```json
{
  “record_type”: “short_evidence”,
  “idx”: 0,
  “source”: {
    “teacher_model”: “model-name”,
    “prompt_version”: “teacher-data-v2”,
    “created_at”: “YYYY-MM-DD”
  },
  “task”: {
    “idx”: 0,
    “title”: “”,
    “author”: “”,
    “content”: “”,
    “qa_words”: [],
    “qa_sents”: [],
    “choose”: {}
  },
  “evidence”: {
    “words”: {
      “<target_word>”: {
        “meaning”: “string”,
        “text_clue”: “string”,
        “rationale”: “string”
      }
    },
    “sentences”: {
      “<target_sentence>”: {
        “translation”: “string”,
        “key_images”: [“string”],
        “rationale”: “string”
      }
    },
    “emotion”: [
      “情感短证据字符串，不超过60字”,
      “另一条情感判断依据”
    ]
  },
  “sentiment”: {
    “primary”: “惜别感伤”,
    “secondary”: [“羁旅漂泊”],
    “rationale”: “情感判断依据，不超过80字，引用诗中关键词和意象”
  },
  “draft_answer”: {
    “ans_qa_words”: {},
    “ans_qa_sents”: {}
  },
  “quality_flags”: []
}
```

字段约束：

- `evidence.words` 的 key 集合必须等于 `task.qa_words` 去重后的集合。
- `evidence.sentences` 的 key 集合必须等于 `task.qa_sents` 去重后的集合。
- `evidence.emotion` 是字符串数组，每条 ≤ 60 字，不包含 `option_id`、`polarity` 等字段，仅写情感判断的短证据。
- `meaning` 建议 6-25 个中文字符，避免百科式解释。
- `translation` 建议 10-60 个中文字符，保持现代汉语句意，不写赏析。
- `rationale` 只允许短证据说明，不允许出现”首先、然后、所以我认为”等推理链展开。
- `sentiment.primary` 必须从受控词汇表（惜别感伤、送别不舍、离别愁绪、思乡怀远、羁旅思归、故园之思、忧国伤时、报国壮志、兴亡之叹、山水闲适、田园之乐、隐逸情怀、怀古伤今、历史沧桑、昔盛今衰、相思闺怨、爱情甜蜜、相思之苦、人生无常、时光易逝、仕途失意、边塞征战、将士艰辛、厌战思归、其他）中选择。若选择”其他”，必须在 `quality_flags` 中包含 `needs_human_review`。
- `sentiment.secondary` 中的每个标签也必须从受控词汇表中选择。
- `sentiment.rationale` 不超过 80 字，必须引用诗中的具体关键词或意象作为依据。
- `draft_answer` 只包含 `ans_qa_words` 和 `ans_qa_sents`，不包含 `idx` 也不包含 `choose_id`。`idx` 由外层样本提供，`choose_id` 在训练集 teacher 数据中不存在，仅在评测 dev split 的 `final_answer` 中出现。
- `quality_flags` 仅使用约定枚举，如 `missing_emotion_options`、`needs_human_review`、`low_confidence_emotion`。

## 4. Teacher-critique Schema

`teacher_critique` 样本用于训练模型识别错误答案并输出修正 JSON，重点覆盖情感分析错误、词义误解、翻译过度赏析和缺项问题。

```json
{
  "record_type": "teacher_critique",
  "idx": 0,
  "source": {
    "teacher_model": "model-name",
    "prompt_version": "teacher-critique-v2",
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
    "evidence": {
      "words": {
        "衰草": {
          "meaning": "茂盛的草",
          "text_clue": "故关衰草遍",
          "rationale": "衰草写景"
        }
      },
      "sentences": {
        "故关衰草遍，离别自堪悲": {
          "translation": "旧关一带长满枯草，分别本就令人悲伤",
          "key_images": ["故关", "离别"],
          "rationale": "以衰败景象写离别之悲"
        }
      },
      "emotion": [
        "衰草、离别指向伤感"
      ]
    },
    "sentiment": {
      "primary": "田园之乐",
      "secondary": [],
      "rationale": "衰草体现田园风光"
    },
    "draft_answer": {
      "idx": 0,
      "ans_qa_words": {
        "衰草": "茂盛的草"
      },
      "ans_qa_sents": {
        "故关衰草遍，离别自堪悲": "旧关一带长满枯草，分别本就令人悲伤"
      }
    }
  },
  "critique": {
    "word_errors": [
      {
        "target": "衰草",
        "error_type": "wrong_meaning",
        "comment": "衰草指枯黄败草，非茂盛草木"
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
      "candidate_primary": "田园之乐",
      "correct_primary": "惜别感伤",
      "candidate_secondary": [],
      "correct_secondary": ["羁旅漂泊"],
      "primary_error_type": "wrong_label",
      "secondary_error_type": "missing_label",
      "rationale_error_type": "no_evidence",
      "comment": "诗歌通篇写离别之悲，衰草、离别、掩泪、风尘等意象均指向伤感和漂泊，并非田园之乐"
    }
  },
  "correction_evidence": {
    "words": {
      "衰草": "枯黄败草，烘托荒凉离别氛围"
    },
    "sentences": {
      "故关衰草遍，离别自堪悲": "旧关遍是衰草，离别本就令人悲伤"
    },
    "emotion": [
      "衰草、离别指向伤感",
      "掩泪、风尘何处期暗示重逢无期"
    ]
  },
  "corrected_sentiment": {
    "primary": "惜别感伤",
    "secondary": ["羁旅漂泊"],
    "rationale": "衰草、离别、悲、掩泪、风尘何处期等意象共同指向离别之伤感和前路茫茫"
  },
  "corrected_answer": {
    "idx": 0,
    "ans_qa_words": {
      "衰草": "枯黄的草，烘托荒凉离别的氛围"
    },
    "ans_qa_sents": {
      "故关衰草遍，离别自堪悲": "旧关一带长满枯草，分别本就令人悲伤"
    }
  },
  "quality_flags": []
}
```

字段约束：

- `candidate_answer` 保留原始候选答案，不做静默修复。其结构为 Reasoner 中间输出格式（含 `sentiment`，不含 `choose_id`）。
- `critique.*_errors` 必须覆盖所有目标词和目标句。
- `critique.emotion_error.primary_error_type` 枚举值：`wrong_label`（标签错误）、`not_in_vocab`（不在受控词汇表）、`missing`（缺失）、`correct`（正确）、`uncertain`（无法判断）。
- `critique.emotion_error.secondary_error_type` 枚举值：`extra_label`（多出无关标签）、`missing_label`（缺少应有标签）、`wrong_label`（标签错误）、`correct`（正确）、`uncertain`（无法判断）。
- `critique.emotion_error.rationale_error_type` 枚举值：`no_evidence`（缺乏依据）、`contradicts_vocab`（依据与标签矛盾）、`correct`（正确）、`uncertain`（无法判断）。
- `comment` 是批注意见，不是完整推理过程。
- `corrected_sentiment` 提供修正后的情感分析结果。
- `corrected_answer` 只包含词义和句译，不包含 `choose_id`（训练集 teacher-critique 的修正目标仅为词义和句译）。若任务需要修正 `choose_id`，由后续 Formatter 根据修正后的 `sentiment` 和 `task.choose` 映射生成。

## 5. 教师模型调用方式

### 5.1 模型信息

| 项目 | 值 |
| --- | --- |
| 模型名称 | DeepSeek-V4-Flash |
| API 端点 | `https://api.deepseek.com/v1/chat/completions` |
| API Key 来源 | 项目根目录 `api-key.txt`（已加入 `.gitignore`，禁止提交或打印到日志） |
| 请求格式 | OpenAI-compatible Chat Completions API |
| 推荐参数 | `temperature=0.3`, `max_tokens=4096` |

### 5.2 API Key 读取方式

API key **不得**硬编码在代码或配置文件中，必须从项目根目录的 `api-key.txt` 中读取。示例：

```python
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_ROOT, "api-key.txt"), "r") as f:
    API_KEY = f.read().strip()
```

### 5.3 执行命令

所有 teacher 数据生成脚本通过远程服务器执行：

```bash
python3 remote_run.py python src/cli/generate_teacher_data.py
```

### 5.4 速率限制与重试

- API 调用使用指数退避重试（exponential backoff），初始等待 1 秒，最大重试 5 次。
- 遇到 `429 Too Many Requests` 时等待 `(2^retry_count) + random_jitter` 秒后重试。
- 遇到 `5xx` 服务器错误时重试相同策略。
- 遇到 `4xx`（非 429）或 `401`（认证失败）时不重试，直接记入失败日志。

### 5.5 批处理

- 生成过程按 `idx` 分批处理，每批 8-16 个样本，减少 API 调用往返次数。
- 每完成一批，将结果增量保存到中间结果文件（如 `data/teacher/short_evidence_batch_*.jsonl`）。
- 所有批次完成后，合并为完整 teacher 数据文件（`data/teacher/short_evidence.jsonl`）。
- 若生成中断，下次运行时跳过已有 `idx`（以中间结果文件的 `idx` 集合为索引）。
- 每批耗时约 30-60 秒（取决于请求并发数和 API 响应速度）。

### 5.6 错误处理

- 单样本 JSON 解析失败时，单独记录错误日志，不影响同批其他样本。
- 连续 3 个样本解析失败时，触发自动降级：使用 `temperature=0.5` 重试一次；若仍失败，将样本标记为 `parse_failed` 并跳过。
- 生成完成后，统计成功率、平均 token 消耗和失败原因分布。

## 6. 自动过滤规则

生成后先进行本地规则过滤，再进入人工抽查池。

### 6.1 JSON 与字段合法性

- JSON 必须能被严格解析，且根节点必须是 object。
- `record_type` 只能是 `short_evidence` 或 `teacher_critique`。
- `idx` 必须与输入 `task.idx` 一致。
- 必填字段缺失时直接丢弃：`task`、`evidence`、`sentiment`、`draft_answer`。对 `teacher_critique`，还需校验 `critique` 和 `corrected_sentiment`。
- `short_evidence` 不再要求 `final_answer` 或 `choose_id` 字段（训练集不生成它们）。
- 不允许出现自由 CoT 字段名：`cot`、`chain_of_thought`、`reasoning`、`steps`、`analysis`。
- 任一字符串字段包含 Markdown 代码块标记、明显提示词残留或多段长推理时直接丢弃。

### 6.2 覆盖与选项合法性

- `ans_qa_words` 的 key 集合必须等于 `qa_words` 去重集合；缺项、多项、key 改写均过滤。
- `ans_qa_sents` 的 key 集合必须等于 `qa_sents` 去重集合；缺项、多项、key 改写均过滤。
- 对于训练集样本（`choose` 为空），`draft_answer` **不应**包含 `choose_id` 字段。若出现该字段，直接过滤。
- 对于评测 dev split 样本（`choose` 非空），若输出包含 `final_answer` 且其中有 `choose_id`，`choose_id` 必须属于 `choose` 的现有选项 ID（标准样本应为 `A/B/C/D`）。
- `teacher_critique` 中 `corrected_answer` 不包含 `choose_id`（修正目标为词义和句译）。
- `sentiment.primary` 必须存在于受控词汇表中（见 `docs/spec/data-schema.md` §3.2）。若非受控词汇表中的标签，直接过滤。
- `sentiment.secondary` 中的每个标签也必须存在于受控词汇表中。若有标签不在表中，触发 `not_in_vocab` 标记，进入人工复核。

### 6.3 长度与风格

- 词义答案建议 6-35 个中文字符；超过 50 个中文字符进入人工复核，超过 80 个中文字符过滤。
- 句子翻译建议 10-80 个中文字符；超过 120 个中文字符进入人工复核，超过 180 个中文字符过滤。
- `rationale`、`comment`、`support` 单字段超过 60 个中文字符进入人工复核，超过 100 个中文字符过滤。
- `sentiment.rationale` 超过 80 个中文字符进入人工复核，超过 120 个中文字符过滤。
- 禁止输出赏析文风，如大段”表达了诗人……”堆叠；只保留解释、翻译和短证据。

### 6.4 证据与答案一致性

- `evidence.words[*].meaning` 与 `draft_answer.ans_qa_words[*]` 不应语义相反；若出现明显反义词冲突，过滤。
- `evidence.sentences[*].translation` 与 `draft_answer.ans_qa_sents[*]` 不应漏掉主要主语、动作或情感词；疑似漏译进入人工复核。
- `sentiment.primary` 必须与 `evidence.emotion` 中体现的情感方向一致。例如 `sentiment.primary` 为”惜别感伤”，但 `evidence.emotion` 全指向喜悦，进入人工复核。
- `sentiment.rationale` 必须引用诗中的具体词语、意象或句子作为依据。若 `rationale` 为空或只有泛泛描述（如”诗歌表达了悲伤”），标记为 `low_confidence_emotion`。
- 若 `sentiment.primary` 为”其他”，必须在 `quality_flags` 中包含 `needs_human_review`。缺失该标记则自动添加。
- `teacher_critique` 中标记为 `correct` 的字段，不应在 `corrected_answer` 中被大幅改写；若改写超过长度或词面相似阈值，进入人工复核。
- `teacher_critique` 的 `emotion_error.primary_error_type` 与 `corrected_sentiment.primary` 必须逻辑一致：若 `primary_error_type` 为 `correct`，则 `candidate_primary` 与 `correct_primary` 必须相同；否则过滤。

### 6.5 去重与采样

- 同一 `idx`、同一 `record_type`、同一 `draft_answer`、同一 `sentiment` 的重复样本只保留一条。
- 同一题可保留多条 teacher 样本，但情感分析 critique 不应被单一错误类型垄断。
- 对 `quality_flags` 含 `needs_human_review` 或 `low_confidence_emotion` 的样本降权采样，不进入高置信训练集。

## 7. 人工抽查 Checklist

人工抽查优先看自动过滤后的样本，建议每批至少抽查：

- 每个 teacher 模型输出的 5%-10%。
- 所有带 `needs_human_review`、`low_confidence_emotion` 的样本。
- `sentiment.primary` 为"其他"的样本。
- 情感分析相近、教师之间答案不一致、长度接近阈值的样本。

逐条检查：

- 目标词是否全部覆盖，key 是否与原题完全一致。
- 词义是否符合诗句语境，是否误把字面义当作语境义。
- 词义答案是否简洁，是否夹带长篇赏析。
- 目标句是否全部覆盖，key 是否与原题完全一致。
- 句子翻译是否保留主要意象、动作、情感和否定关系。
- 翻译是否是现代汉语解释，而不是泛泛赏析。
- `sentiment.primary` 是否正确反映诗歌的主导情感。
- `sentiment.primary` 标签是否来自受控词汇表（见 `docs/spec/data-schema.md` §3.2）。
- `sentiment.secondary` 是否捕捉了情感细微差别，且不与 `primary` 矛盾（如 primary 为"惜别感伤"但 secondary 含"爱情甜蜜"则矛盾）。
- `sentiment.rationale` 是否提供有效、诗作特有的证据（引用诗中具体词语或意象），而非通用描述。
- 情感短证据（`evidence.emotion`）是否能支持 `sentiment` 分析结论，是否忽略转折、反讽或尾联变化。
- `teacher_critique` 是否准确定位候选答案的 `sentiment` 分析错误，而不是为了修改而修改。
- 标记为 `correct` 的字段是否确实不需要修改。
- `corrected_sentiment` 是否提供准确的修正。
- `draft_answer` 是否可直接作为词义和句译训练目标。
- 输出中是否存在自由长 CoT、分步推理、提示词残留或 Markdown。

## 8. 禁止自由长 CoT 的约束说明

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

情感分析相关的 `sentiment.primary`、`sentiment.secondary` 和 `sentiment.rationale` 也是允许的结构化短字段。它们不完整推理过程，而是情感分析的结果陈述。

## 9. 需要人工确认的问题

- 词义和句子翻译的长度阈值是否需要按官方评分细则进一步收紧。
  A：先不进一步收紧，保留建议阈值与过滤阈值两层规则。词义答案建议 6-35 个中文字符，超过 50 进入人工复核，超过 80 过滤；句子翻译建议 10-80 个中文字符，超过 120 进入人工复核，超过 180 过滤；`rationale/comment/support` 超过 60 复核，超过 100 过滤。
- `teacher_critique` 的候选错误答案来源比例如何分配：baseline、学生模型、合成扰动各占多少。
  A：第一版有学生模型时使用 `baseline 50% / student 30% / synthetic 20%`。若 B8 学生模型尚未产出，先使用 `baseline 70% / synthetic 30%`；B8 可用后切回 `50/30/20`。
- 含 `needs_human_review` 的样本是否完全排除训练，还是只降权用于 critique 增强。
  A：默认不进入 answer-only、short-evidence 主训练集。人工确认通过后可升级为高置信样本；未确认但只有轻微不确定的样本，可用于 critique 增强并降低采样权重，例如 `0.2x`。
- 多个教师模型答案冲突时，采用投票、强模型优先，还是人工仲裁。
  A：采用分层仲裁，不做简单投票。schema 非法直接丢弃；多教师一致则保留为高置信；词义/翻译表达不同但语义兼容时保留最简洁版本；`sentiment.primary` 冲突时进入人工仲裁或强模型复审；复审仍冲突则标记 `low_confidence_emotion`，不进入主训练。
- 教师模型使用 DeepSeek-V4-Flash API，是否需要备选模型作为降级方案。
  A：第一版仅使用 DeepSeek-V4-Flash。若 API 长时间不可用或响应质量不达标，可评估备选模型（如 DeepSeek-V3 或其他 OpenAI-compatible API）。降级方案在 `src/cli/generate_teacher_data.py` 中通过配置文件切换，不修改核心逻辑。
- `sentiment` 受控词汇表是否覆盖所有 164 首训练诗和 dev 测试诗的情感类别。
  A：需要检查。受控词汇表目前涵盖 8 大类 24 小类 + "其他"。如果在批量生成中发现某首诗无法归入任何现有标签，应按规则使用"其他"并标记 `needs_human_review`。如果"其他"占比超过 10%，说明受控词汇表覆盖不足，需要扩充。
- `sentiment.primary` 是否需要与 `evidence.emotion` 在细粒度上保持一致，还是允许一定程度的抽象。
  A：需要一致。`evidence.emotion` 是情感判断的短证据字符串，`sentiment.primary` 是从中抽象出的受控标签。如果证据明确指向"悲伤"，但标签写为"山水闲适"，则视为不合逻辑，进入人工复核。
