# 数据构造管线

本文定义从原始数据到训练就绪数据集的完整管线。该管线供 Teacher 数据生成、训练数据组装和评测数据准备共同使用。

## 1. 数据源

| 数据源 | 路径 | 样本数 | 词义 | 句译 | 情感分析 | 情感选项 |
| --- | --- | ---: | --- | --- | --- | --- |
| 评测数据 | `data/eval_data.json` | 327 | qa_words (960) | qa_sents (665) | 无 gold label | 有（A/B/C/D） |
| 训练原始数据 | `data/train-data/*/train.json` | 164 | keywords (1304) | trans（不可直接用） | emotion（自由文本） | 无 |
| 训练集 dev split | 从训练数据划分 | ~30 | — | — | — | 无 |
| 评测集 dev split | 从 `eval_data.json` 划分 | ~50 | — | — | Teacher 生成 | 有 |

**Dev split 策略**：

- 评测 dev split：从 `eval_data.json` 随机抽取 50 首，用于 prompt baseline、harness 和最终评测。固定种子保证可复现。
- 训练 dev split：从 `data/train-data/` 的 164 首中随机抽取 30 首，用于 B8/BC8 训练 checkpoint 选择。不包含 `choose` 选项，只能评测词义和句译。
- 剩余评测样本（277 首）保留为最终测试，不得在训练任何阶段使用。

## 2. 管线总览

```text
原始数据
  ├── data/train-data/*/train.json (164首)
  │     └── Teacher 生成 (DeepSeek-V4-Flash API)
  │           ├── short-evidence 样本 (含 sentiment)
  │           ├── teacher-critique 样本 (含 sentiment critique)
  │           └── answer-only 样本 (从 keywords 构造)
  │                 │
  │                 ├── 自动过滤
  │                 ├── 人工抽查
  │                 └── 训练数据集
  │                       ├── B8 answer-only (词义 + 句译, 无情感)
  │                       ├── BC8 short-evidence (60% answer-only)
  │                       ├── BC8 short-evidence (30% evidence+draft+sentiment)
  │                       └── BC8 teacher-critique (10% critique)
  │
  └── data/eval_data.json (327首)
        ├── Dev split (~50首) → Teacher 生成 → 训练用（含 choose_id）
        ├── Few-shot 样例池（从 dev split 外选 5 首）
        └── 保留测试集 (~277首) → 仅最终评测
```

## 3. Teacher 数据生成

### 3.1 教师模型

- **模型**：DeepSeek-V4-Flash（API 调用）
- **API Endpoint**：`https://api.deepseek.com/v1/chat/completions`
- **API Key**：从项目根目录 `api-key.txt` 读取
  - `api-key.txt` 已在 `.gitignore` 中，**严禁提交、上传或打印到日志**
- **执行方式**：通过 `python3 remote_run.py python src/cli/generate_teacher_data.py` 在服务器上运行

### 3.2 生成脚本设计

脚本 `src/cli/generate_teacher_data.py` 应实现：

1. 读取输入数据（训练集 164 首或评测 dev split 50 首）
2. 调用 DeepSeek API，传入 teacher prompt（见 `docs/contracts/teacher-data.md`）
3. 速率控制：最多 10 QPS，失败重试 3 次，指数退避（1s, 2s, 4s）
4. 分批保存中间结果到 `data/teacher/` 目录
5. 记录生成日志（成功/失败/重试次数）

```bash
# 为训练集 164 首生成 short-evidence
python3 remote_run.py python src/cli/generate_teacher_data.py \
  --input data/train-data \
  --output data/teacher/train-short-evidence.jsonl \
  --type short-evidence

# 为评测 dev split 50 首生成 short-evidence + choose_id
python3 remote_run.py python src/cli/generate_teacher_data.py \
  --input data/eval_data.json \
  --output data/teacher/dev-short-evidence.jsonl \
  --type short-evidence \
  --dev-split 50 --seed 42

# 生成 teacher-critique（需要候选错误答案）
python3 remote_run.py python src/cli/generate_teacher_data.py \
  --input data/train-data \
  --output data/teacher/train-critique.jsonl \
  --type teacher-critique \
  --candidates data/baseline/e3-dev50/P8-qwen3-8b-bf16-vllm-nothink-zero.jsonl
```

### 3.3 API 调用实现要点

```python
# 密钥读取（不得硬编码）
import os
API_KEY = open("api-key.txt").read().strip()

# API 调用
import requests
response = requests.post(
    "https://api.deepseek.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"}
    },
    timeout=120
)
```

### 3.4 训练集 vs 评测集 Teacher 生成差异

| 项目 | 训练集 (164首) | 评测 dev split (50首) |
| --- | --- | --- |
| 输入 `choose` | `{}` | 原题完整选项 |
| Teacher 输出 `sentiment` | 是（参考 `emotion` 字段） | 是 |
| Teacher 输出 `draft_answer.choose_id` | 无（不生成） | 有（从原题选项中选择） |
| Teacher 输出 `final_answer.choose_id` | 无（不生成） | 有 |
| 用途 | 训练 Reasoner 的情感分析能力 | 训练/评测 sentiment→choose_id 映射 |

## 4. 自动过滤

Teacher 生成数据必须经过自动过滤规则（详见 `docs/contracts/teacher-data.md` 第 5 节），此外增加：

### 4.1 Sentiment 专项过滤

- `sentiment.primary` 不在受控词汇表中的样本 → 标记 `invalid_sentiment_primary`，进入人工复核
- `sentiment.primary = "其他"` 且 `rationale` 空泛（无具体诗句引用）→ 降低采样权重
- `sentiment.primary` 与 `evidence.emotion` 明显矛盾（如 sentiment 标"惜别感伤"但 evidence 写"全诗欢快昂扬"）→ 直接过滤
- `sentiment.secondary` 与 `sentiment.primary` 语义矛盾（如 primary="山水闲适"，secondary="边塞征战"）→ 进入人工复核

### 4.2 训练集特有过滤

- 训练集样本 `choose = {}`，Teacher 输出中若有 `choose_id` 非空 → 去除 `choose_id` 字段（Teacher 可能"幻觉"出选项）
- 训练集 Teacher 输出的 `final_answer` 若包含 `choose_id` → 去除，只保留 `final_answer.ans_qa_words` 和 `final_answer.ans_qa_sents`

## 5. 训练数据集组装

### 5.1 B8 answer-only 数据集

数据来源：训练集 `keywords` 直接构造 + Teacher `final_answer` 抽取。

```
比例：100% answer-only
词义标签：keywords（高置信）
句译标签：Teacher 生成的 final_answer.ans_qa_sents（需通过过滤）
情感标签：无（choose_id = ""，choose = {}）
```

构造脚本：

```bash
python3 remote_run.py python src/cli/build_training_data.py \
  --type answer-only \
  --keywords data/train-data \
  --teacher data/teacher/train-short-evidence.jsonl \
  --output data/training/b8-answer-only.jsonl
```

### 5.2 BC8 mixed 数据集

```
60% answer-only（同 B8）
30% short-evidence（teacher 生成的 evidence + sentiment + draft_answer）
10% teacher-critique（teacher 生成的 critique + correction）
```

构造脚本：

```bash
python3 remote_run.py python src/cli/build_training_data.py \
  --type bc8-mixed \
  --ratio 60-30-10 \
  --answer-only data/training/b8-answer-only.jsonl \
  --short-evidence data/teacher/train-short-evidence.jsonl \
  --teacher-critique data/teacher/train-critique.jsonl \
  --output data/training/bc8-mixed/
```

### 5.3 评测 dev split 的 choose_id 映射训练数据

从评测 dev split 的 Teacher 数据中抽取 sentiment→choose_id 对，用于 Formatter 微调或 few-shot：

```bash
python3 remote_run.py python src/cli/build_training_data.py \
  --type sentiment-mapping \
  --teacher data/teacher/dev-short-evidence.jsonl \
  --output data/training/sentiment-mapping.jsonl
```

### 5.4 数据版本记录

每个训练数据集必须附带 manifest，记录：
- 数据来源和版本
- Teacher 模型和 prompt 版本
- 过滤规则版本
- 样本数量和各子集比例
- 构造时间和脚本参数

## 6. Few-shot 样例池

从评测 dev split 外（即剩余的 277 首评测样本中）选取 5 首作为 few-shot 样例池：

- 1 首单词语 + 1 首多词语 + 1 首单句翻译 + 1 首多句翻译
- 情感选项 A/B/C/D 分布均衡
- 覆盖送别、怀古、咏物、羁旅、山水等常见主题
- 答案简洁，JSON 完全合法

构造脚本：

```bash
python3 remote_run.py python src/cli/build_fewshot_pool.py \
  --eval-data data/eval_data.json \
  --exclude-dev-split data/splits/dev-50-ids.txt \
  --output data/fewshot/balanced_static.json \
  --size 5
```

## 7. 输出目录结构

```text
data/
├── teacher/
│   ├── train-short-evidence.jsonl    # 训练集 164 首的 short-evidence
│   ├── train-critique.jsonl          # 训练集的 teacher-critique
│   └── dev-short-evidence.jsonl      # 评测 dev split 50 首的 short-evidence
├── training/
│   ├── b8-answer-only.jsonl          # B8 answer-only 训练数据
│   ├── bc8-mixed/                    # BC8 mixed 训练数据
│   │   ├── answer-only.jsonl
│   │   ├── short-evidence.jsonl
│   │   └── teacher-critique.jsonl
│   └── sentiment-mapping.jsonl       # sentiment→choose_id 映射训练数据
├── splits/
│   ├── dev-50-ids.txt                # 评测 dev split 的 idx 列表
│   └── train-dev-30-ids.txt          # 训练 dev split 的 idx 列表
└── fewshot/
    └── balanced_static.json          # 5-shot 样例池
```

## 8. 安全注意事项

- **`api-key.txt`**：已在 `.gitignore`，任何脚本不得硬编码 API key，不得将 key 打印到日志或输出文件
- **数据泄漏**：评测集的 277 首保留样本不得出现在任何训练数据、few-shot 样例或 Teacher 生成输入中
- **`remote_run.py`**：所有在服务器上执行的脚本必须通过 `python3 remote_run.py` 转发，不得直接 SSH
