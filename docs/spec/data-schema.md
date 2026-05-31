# CCL25 数据 Schema 与转换规则

本文档定义 CCL25 古诗词理解与推理任务的统一数据契约。该契约供 prompt baseline、teacher-data 生成、training 数据构造、harness/formatter 和 eval 共同使用。

本文档只规定数据 schema、字段规范和转换规则，不规定训练目标、提示词策略或评测实现。

## 1. 统一输入 Schema

统一输入样本使用一个 JSON object 表示：

```json
{
  "idx": 0,
  "title": "李端公",
  "author": "卢纶",
  "content": "故关衰草遍，离别自堪悲。路出寒云外，人归暮雪时。少孤为客早，多难识君迟。掩泪空相向，风尘何处期。",
  "qa_words": ["衰草", "故关", "风尘"],
  "qa_sents": ["故关衰草遍，离别自堪悲", "掩泪空相向，风尘何处期。"],
  "choose": {
    "A": "欢快的重逢",
    "B": "仕途的无奈",
    "C": "对未来的期待",
    "D": "惜别的感伤"
  }
}
```

### 1.1 字段定义

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `idx` | integer 或 string | 是 | 样本唯一标识。若原始数据为数字，保持数字；若官方使用字符串 ID，保持字符串。下游不得重新编号。 |
| `title` | string | 否 | 诗词标题。缺失时填空字符串 `""`。 |
| `author` | string | 否 | 作者。缺失时填空字符串 `""`。 |
| `content` | string | 是 | 诗词正文。保留原始换行；只去除首尾空白。 |
| `qa_words` | array[string] | 是 | 需要解释的词语列表。顺序按原始题目给出顺序保留。 |
| `qa_sents` | array[string] | 是 | 需要翻译的句子列表。顺序按原始题目给出顺序保留。 |
| `choose` | object | 是 | 情感选项，键为选项 ID，值为选项文本。正常情况下键为 `"A"`、`"B"`、`"C"`、`"D"`。 |

#### 训练与评测输入的差异

评测数据（`eval_data.json`）和最终测试集必定包含 `choose` 字段及 A/B/C/D 四个选项。训练衍生数据（来自 `train-data/` 的 164 首诗）原本没有 `choose` 选项，因此在构造训练样本时：

- 来自训练集的样本：`choose = {}`，`choose_id` 不作为训练目标。模型只学习词义、句译和情感分析（sentiment），不学习选项选择。
- 来自评测 dev split 的样本：`choose` 保留原题的完整选项，Teacher 可生成 `choose_id` 作为训练标签，用于 Formatter 的 sentiment→option 映射训练。

这决定了本项目的两阶段情感管线：Reasoner 分析情感（不输出 `choose_id`），Formatter 将情感映射到选项。

### 1.2 规范化要求

- `idx` 是输入与输出对齐的主键，baseline、teacher-data、harness 和 eval 都必须原样传递。
- `title`、`author` 不参与输出 schema，但应保留给做题、证据生成和人工排查使用。
- `content` 不做句读改写、不合并内部换行、不转换繁简体。
- `qa_words` 和 `qa_sents` 的元素必须是字符串；转换时去除每个元素首尾空白。
- `choose` 的键必须是字符串；转换时去除键和值的首尾空白。选项文本为空时保留该键，值设为 `""`，并列入人工确认。

## 2. 最终输出 Schema

最终提交或评测输出使用一个 JSON object 表示：

```json
{
  "idx": 0,
  "ans_qa_words": {
    "衰草": "枯黄的草，烘托荒凉离别的氛围",
    "故关": "旧日关塞，也可指故乡关隘",
    "风尘": "战乱漂泊的世事"
  },
  "ans_qa_sents": {
    "故关衰草遍，离别自堪悲": "旧关一带长满枯草，分别本就令人悲伤",
    "掩泪空相向，风尘何处期。": "只能含泪相望而别，乱世漂泊中不知何时再会"
  },
  "choose_id": "D"
}
```

### 2.1 字段定义

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `idx` | 与输入 `idx` 类型一致 | 是 | 必须等于对应输入样本的 `idx`。 |
| `ans_qa_words` | object | 是 | 对每个 `qa_words` 词语给出中文释义。键必须来自统一输入的 `qa_words`。 |
| `ans_qa_sents` | object | 是 | 对每个 `qa_sents` 句子给出白话文译文。键必须来自统一输入的 `qa_sents`。 |
| `choose_id` | string | 是 | 情感选项 ID。正常情况下为 `"A"`、`"B"`、`"C"`、`"D"` 之一。 |

### 2.2 输出约束

- 最终输出不得包含 `content`、`title`、`author`、证据、草稿、解释性注释或其他额外字段。
- `ans_qa_words` 必须覆盖输入 `qa_words` 去重后的每个词语。
- `ans_qa_sents` 必须覆盖输入 `qa_sents` 去重后的每个句子。
- 输出 key 使用输入中的原始词语或句子文本，不自行改写标点、繁简体或空格。
- 答案值必须是中文字符串；缺失答案时使用 `""` 占位，并由 validator 或人工审查标记为缺项。

## 3. Reasoner 中间输出 Schema（stage1 输出）

Reasoner（BC8 训练后的学生模型或 prompt baseline 模型）在 harness 两阶段管线中的输出，不直接提交。它输出结构化证据、情感分析和草稿答案，**不包含 `choose_id`**。

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

### 3.1 字段定义

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `idx` | 与输入 `idx` 类型一致 | 是 | 必须等于对应输入样本的 `idx`。 |
| `evidence` | object | 是 | 结构化短证据，分为词义、句译、情感三类。 |
| `evidence.words` | object | 是 | key 为 `qa_words` 去重后的词语，value 包含 `meaning`（简明词义）、`text_clue`（诗中依据）和 `rationale`（不超过 40 字）。 |
| `evidence.sentences` | object | 是 | key 为 `qa_sents` 去重后的句子，value 包含 `translation`（现代汉语直译/意译）、`key_images`（意象列表）和 `rationale`（不超过 40 字）。 |
| `evidence.emotion` | array[string] | 是 | 短证据字符串数组，每条 ≤ 60 字，写出判断依据而非完整推理。 |
| `sentiment` | object | 是 | 结构化情感分析。**这是两阶段管线中 Formatter 映射到 `choose_id` 的主要依据。** |
| `sentiment.primary` | string | 是 | 主要情感标签，必须从受控词汇表（第 3.2 节）中选择。2-8 个中文字符。 |
| `sentiment.secondary` | array[string] | 否 | 次要情感标签列表，每个标签从受控词汇表中选择。 |
| `sentiment.rationale` | string | 是 | 情感判断依据，不超过 80 字。引用诗中关键词和意象。 |
| `draft_answer` | object | 是 | 草稿答案，包含词义和句译，**不包含 `choose_id`**。 |
| `draft_answer.ans_qa_words` | object | 是 | 与最终输出 schema 的 `ans_qa_words` 约束相同。 |
| `draft_answer.ans_qa_sents` | object | 是 | 与最终输出 schema 的 `ans_qa_sents` 约束相同。 |

### 3.2 情感标签受控词汇表

`sentiment.primary` 和 `sentiment.secondary` 必须从以下标签中选择，不得自由发挥。受控词汇表确保 Formatter 能在推理时将情感分析可靠映射到给定的 A/B/C/D 选项。

| 类别 | 可用标签 |
| --- | --- |
| 离别 | 惜别感伤、送别不舍、离别愁绪 |
| 思乡 | 思乡怀远、羁旅思归、故园之思 |
| 忧国 | 忧国伤时、报国壮志、兴亡之叹 |
| 山水田园 | 山水闲适、田园之乐、隐逸情怀 |
| 怀古 | 怀古伤今、历史沧桑、昔盛今衰 |
| 爱情闺怨 | 相思闺怨、爱情甜蜜、相思之苦 |
| 人生感慨 | 人生无常、时光易逝、仕途失意 |
| 边塞战争 | 边塞征战、将士艰辛、厌战思归 |

若一首诗的情感无法归入以上任一标签，`primary` 使用 `"其他"`，并在 `rationale` 中详细说明，同时在 `quality_flags` 中标记 `needs_human_review`。

### 3.3 与最终输出 Schema 的关系

- Reasoner 的 `draft_answer.ans_qa_words` 和 `draft_answer.ans_qa_sents` 可直接作为最终输出对应字段（经 Formatter 校验后）。
- Reasoner **不输出** `choose_id`。最终输出的 `choose_id` 由 Formatter（Stage2）根据 `sentiment` + `task.choose` 选项映射生成。
- Formatter 负责：JSON 格式化、字段校验、缺项补齐、以及 sentiment→choose_id 映射。

## 4. 原始样本到统一 Schema 的转换规则

官方原始字段名待确认。当前仓库已有评测样本使用 `idx/title/author/content/qa_words/qa_sents/choose`，该结构直接视为统一输入 schema。若官方发布字段名不同，按以下契约转换。

### 4.1 字段映射

| 统一字段 | 当前仓库样本字段 | 可能的官方字段名 | 转换规则 |
| --- | --- | --- | --- |
| `idx` | `idx` | `idx` / `index` / `id`，待确认 | 优先读取 `idx`；若只有 `index` 或 `id`，映射为 `idx`。不得重新编号。 |
| `title` | `title` | 待确认 | 缺失时填 `""`。 |
| `author` | `author` | 待确认 | 缺失时填 `""`。 |
| `content` | `content` | `content` / `poem` / `text`，待确认 | 转为字符串，去除首尾空白，保留内部换行和标点。 |
| `qa_words` | `qa_words` | `qa_words` / `words` / `target_words`，待确认 | 若原始值为数组，逐项转字符串；若为对象，取对象 key 作为词语列表。 |
| `qa_sents` | `qa_sents` | `qa_sents` / `sentences` / `target_sents`，待确认 | 若原始值为数组，逐项转字符串；若为对象，取对象 key 作为句子列表。 |
| `choose` | `choose` | `choose` / `options` / `emotion_options`，待确认 | 若原始值为对象，保留键值；若为数组，按顺序映射为 `A/B/C/D/...`。 |

### 4.2 带答案训练样本的转换

若原始样本是带答案的训练或蒸馏来源，可额外转换出最终输出标签，但统一输入仍只包含第 1 节字段。

| 原始答案信息 | 输出字段 | 转换规则 |
| --- | --- | --- |
| 词语释义对象，例如 `keywords` | `ans_qa_words` | 只保留 `qa_words` 中要求作答的词语；若没有显式 `qa_words`，可用 `keywords` 的 key 生成 `qa_words`，并列入人工确认。 |
| 全诗译文，例如 `trans` | 不直接等同于 `ans_qa_sents` | 全诗译文不能自动拆成目标句译文，除非原始数据提供句级对齐；否则列入人工确认或由 teacher-data 流程生成。 |
| 情感标签，例如 `emotion` | 不直接等同于 `choose_id` | 自由文本情感不能自动映射到选项 ID，除非原始数据同时提供正确选项；否则列入人工确认。 |

## 5. 边界字段处理规则

### 5.1 空字段

- `title`、`author` 缺失或为 `null`：统一填 `""`。
- `content` 缺失、为 `null` 或归一化后为空：该样本不可自动用于下游，标记为 `invalid_input_missing_content`。
- `qa_words` 缺失或为 `null`：统一为 `[]`，输出 `ans_qa_words` 必须为 `{}`，并标记为 `empty_qa_words`。
- `qa_sents` 缺失或为 `null`：统一为 `[]`，输出 `ans_qa_sents` 必须为 `{}`，并标记为 `empty_qa_sents`。
- `choose` 缺失或为 `null`：统一为 `{}`，输出 `choose_id` 必须为 `""`，并标记为 `missing_choose`。

### 5.2 重复词语

- `qa_words` 中完全相同的字符串视为重复。
- 统一输入保留原始列表顺序和重复项，便于追溯官方题面。
- 最终输出 `ans_qa_words` 是 object，无法表达重复 key；因此同一词语只输出一个答案。
- 若同一词语在不同上下文中可能有不同含义，转换器不得自行改名为 `"词语#1"`；应保留原词，并列入人工确认。

### 5.3 重复句子

- `qa_sents` 中完全相同的字符串视为重复。
- 统一输入保留原始列表顺序和重复项。
- 最终输出 `ans_qa_sents` 对同一句子只输出一个译文。
- 若重复句子在题面中实际指向不同片段或版本，需人工确认，不在 schema 中改写 key。

### 5.4 缺失选项与非法选项

- `choose` 为空对象 `{}` 时，`choose_id` 输出 `""`，该样本不应计入自动情感选择准确率。
- 若 `choose` 缺少部分 A/B/C/D，例如只有 A/B/C：保留已有选项；`choose_id` 只能从已有键中选择，无法判断时输出 `""`。
- 若 `choose` 包含非 A/B/C/D 键：保留原键，但标记为 `non_standard_choose_keys`。是否允许最终 `choose_id` 使用非标准键待人工确认。
- 若 `choose` 为数组：按数组顺序映射为 `A/B/C/D`；超过 4 个选项继续映射为 `E/F/...`，并列入人工确认。
- 若选项文本重复：保留重复选项，不自动合并；`choose_id` 仍输出选中的选项 ID。

## 6. 完整样例

### 6.1 样例一：标准四选项样本

统一输入：

```json
{
  "idx": 0,
  "title": "李端公",
  "author": "卢纶",
  "content": "故关衰草遍，离别自堪悲。路出寒云外，人归暮雪时。少孤为客早，多难识君迟。掩泪空相向，风尘何处期。",
  "qa_words": ["衰草", "故关", "风尘"],
  "qa_sents": ["故关衰草遍，离别自堪悲", "掩泪空相向，风尘何处期。"],
  "choose": {
    "A": "欢快的重逢",
    "B": "仕途的无奈",
    "C": "对未来的期待",
    "D": "惜别的感伤"
  }
}
```

最终输出：

```json
{
  "idx": 0,
  "ans_qa_words": {
    "衰草": "枯黄的草，渲染荒凉离别的氛围",
    "故关": "旧日关塞，也含故乡关隘之意",
    "风尘": "战乱漂泊的世事"
  },
  "ans_qa_sents": {
    "故关衰草遍，离别自堪悲": "旧关一带长满枯草，分别本来就令人悲伤",
    "掩泪空相向，风尘何处期。": "只能含泪相望而别，乱世漂泊中不知何时再会"
  },
  "choose_id": "D"
}
```

### 6.2 样例二：缺失作者、无词语题、缺失情感选项

原始样本字段名示例：

```json
{
  "index": "sample-002",
  "title": "登鹳雀楼",
  "poem": "白日依山尽，黄河入海流。\n欲穷千里目，更上一层楼。",
  "target_words": [],
  "target_sents": ["欲穷千里目，更上一层楼。"]
}
```

统一输入：

```json
{
  "idx": "sample-002",
  "title": "登鹳雀楼",
  "author": "",
  "content": "白日依山尽，黄河入海流。\n欲穷千里目，更上一层楼。",
  "qa_words": [],
  "qa_sents": ["欲穷千里目，更上一层楼。"],
  "choose": {}
}
```

最终输出：

```json
{
  "idx": "sample-002",
  "ans_qa_words": {},
  "ans_qa_sents": {
    "欲穷千里目，更上一层楼。": "如果想看尽更远的景色，就要再登上一层楼"
  },
  "choose_id": ""
}
```

该样本应同时标记 `empty_qa_words` 和 `missing_choose`，不计入自动情感选择准确率。

### 6.3 样例三：重复词语与重复句子

统一输入：

```json
{
  "idx": 3,
  "title": "示例诗",
  "author": "",
  "content": "春风又绿江南岸。春风又绿江南岸。",
  "qa_words": ["春风", "春风"],
  "qa_sents": ["春风又绿江南岸。", "春风又绿江南岸。"],
  "choose": {
    "A": "喜悦",
    "B": "悲伤",
    "C": "愤懑",
    "D": "恐惧"
  }
}
```

最终输出：

```json
{
  "idx": 3,
  "ans_qa_words": {
    "春风": "春天的风，也可引申为带来生机的力量"
  },
  "ans_qa_sents": {
    "春风又绿江南岸。": "春风又吹绿了江南岸边"
  },
  "choose_id": "A"
}
```

重复项只在输出 object 中保留一个 key；若官方期望重复题项分别计分，需要人工确认是否改用数组型输出。

### 6.4 样例四：Reasoner 中间输出（对样例一的 reasoner 输出）

对应样例一的统一输入，Reasoner 输出：

```json
{
  "idx": 0,
  "evidence": {
    "words": {
      "衰草": {
        "meaning": "枯黄的草",
        "text_clue": "故关衰草遍",
        "rationale": "衰草烘托荒凉离别的氛围"
      },
      "故关": {
        "meaning": "旧日关塞，也含故乡关隘之意",
        "text_clue": "故关衰草遍",
        "rationale": "故关是离别之地，与衰草共同渲染悲凉"
      },
      "风尘": {
        "meaning": "战乱漂泊的世事",
        "text_clue": "风尘何处期",
        "rationale": "风尘喻漂泊不定，暗示重逢无期"
      }
    },
    "sentences": {
      "故关衰草遍，离别自堪悲": {
        "translation": "旧关一带长满枯草，分别本就令人悲伤",
        "key_images": ["故关", "衰草", "离别"],
        "rationale": "以衰败景象写离别之悲"
      },
      "掩泪空相向，风尘何处期。": {
        "translation": "只能含泪相望而别，乱世漂泊中不知何时再会",
        "key_images": ["掩泪", "风尘"],
        "rationale": "以泪和漂泊写别后茫然"
      }
    },
    "emotion": [
      "衰草、离别、悲、掩泪、风尘等词共同指向伤感",
      "尾联'风尘何处期'暗示重逢无期，加深离别之痛"
    ]
  },
  "sentiment": {
    "primary": "惜别感伤",
    "secondary": ["羁旅漂泊"],
    "rationale": "衰草、离别、悲、掩泪、风尘何处期等意象共同指向离别之伤感和前路茫茫"
  },
  "draft_answer": {
    "ans_qa_words": {
      "衰草": "枯黄的草，烘托荒凉离别的氛围",
      "故关": "旧日关塞，也含故乡关隘之意",
      "风尘": "战乱漂泊的世事"
    },
    "ans_qa_sents": {
      "故关衰草遍，离别自堪悲": "旧关一带长满枯草，分别本就令人悲伤",
      "掩泪空相向，风尘何处期。": "只能含泪相望而别，乱世漂泊中不知何时再会"
    }
  }
}
```

Formatter 将 `sentiment.primary = "惜别感伤"` 与 `choose` 选项匹配后，输出 `choose_id = "D"`（惜别的感伤）。

## 7. 需要人工确认的问题

1. 官方正式数据的字段名是否就是 `idx/title/author/content/qa_words/qa_sents/choose`；若不是，需要确认所有官方字段名和类型。
   A：是
2. `idx` 是否始终为整数；若存在字符串 ID，下游评测是否接受字符串。
   A：是，始终为整数
3. `choose` 是否固定为 A/B/C/D 四选项；若存在少于四项、多于四项或非字母标签，最终 `choose_id` 是否允许非 A-D。
   A：是，固定为4个选项
4. `qa_words` 或 `qa_sents` 为空时，官方评测是否接受 `{}`，以及该子任务是否跳过计分。
   A：接受`{}`，跳过记分
5. 重复 `qa_words` 或重复 `qa_sents` 是否按一个 key 计分，还是需要区分题项位置。
   A：按一个key计分，区分位置
6. 句子 key 的标点是否必须与输入完全一致；例如有无句末句号是否会影响官方匹配。
   A：最好一致，可能会影响
7. 训练数据中的 `trans` 是否有官方句级对齐；若没有，不能自动作为 `ans_qa_sents` 标签。
   A：需要检查
   建议处理：默认不把全诗 `trans` 直接作为 `ans_qa_sents` 标签。只有当 `content` 与 `trans` 有稳定行级对齐，且 `qa_sents` 能严格匹配某一行或连续行时，才自动抽取对应译文。半句、截断句、跨句片段或无法严格对齐的样本，进入 teacher-data 流程生成句译；`trans` 只作为参考材料。
8. 自由文本 `emotion` 是否能映射到官方选项 ID；若没有正确选项标签，不能自动作为 `choose_id` 标签。
   A：已确认不可自动映射（见 `docs/reports/data-inspection.md`）。本项目采用两阶段方案：训练集样本不训练 `choose_id`，而是训练 Reasoner 的 `sentiment` 分析能力；推理时由 Formatter 将 `sentiment` 映射到 `choose_id`。
9. Reasoner 的 `sentiment.primary` 标签是否必须使用受控词汇表。
   A：需要检查
   建议处理：默认不把自由文本 `emotion` 自动映射为 `choose_id`。若原始数据明确提供 `choose` 与正确选项，才作为 gold `choose_id`；若只有自由文本情感，只用于 short-evidence 的情感描述或 teacher 参考，不进入 answer-only 的情感分类标签。
