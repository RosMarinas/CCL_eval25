# 数据检查报告

检查日期：2026-05-20

## 1. 检查范围与数据结构

本次检查只读查看了以下资料：

- `AGENTS.md`
- `docs/data-schema.md`
- `docs/teacher-data.md`
- `docs/training-plan.md`
- `data/eval_data.json`
- `data/train-data` 下 8 个 `train.json`

`data/eval_data.json` 已是统一输入 schema：共 327 题，`idx` 均唯一，全部包含 4 个情感选项；合计 960 个 `qa_words` 目标词、665 个 `qa_sents` 目标句。

`data/train-data` 的样本结构一致，字段为：

```json
{
  "title": "诗词标题",
  "content": "原文",
  "keywords": {
    "词语": "释义"
  },
  "trans": "整首现代文译文",
  "emotion": "自由文本情感描述"
}
```

训练数据共 164 首，其中 162 首有非空 `keywords`，合计 1304 条词义；2 首 `keywords` 为空。训练数据没有 `idx`、`author`、`qa_words`、`qa_sents`、`choose` 或 `choose_id`。

## 2. `content` 与 `trans` 对齐结论

结论：训练集的 `content` 与 `trans` 不具备稳定的自动行级/句级 gold 对齐条件。`trans` 是整首译文，常见情况是整段意译、重组句序或合并拆分句子；它不能直接等同于 `ans_qa_sents`。

统计结果：

| 对齐条件 | 数量 | 比例 | 说明 |
| --- | ---: | ---: | --- |
| `content` 与 `trans` 换行数一致且大于 1 | 10/164 | 6.1% | 只得到 20 个候选行级片段，范围太小 |
| 句末标点数量一致 | 106/164 | 64.6% | 最多可形成 432 个句级候选，但只是必要条件，不代表语义严格对齐 |
| 句末标点数量不一致 | 58/164 | 35.4% | 不应自动拆分成 gold 句译 |

可用判断：

- 严格行级自动构造：可行等级低。只有 10 首满足换行数一致，且一行可能包含半句、整句或联句，仍需抽查。
- 句级启发式自动构造：可行等级中低。106 首句末标点数量一致，可作为 teacher-data 候选或人工复核池，但不应直接进入高置信 answer-only。
- 对 `eval_data.json` 的 `qa_sents`：不能用训练集 `trans` 直接补齐。评测题的目标句是指定片段，必须由 teacher 或人工生成对应译文。

## 3. 自由文本 `emotion` 监督结论

结论：`emotion` 不能静默当作 gold `choose_id`，也不能直接构造可靠的情感四选一监督。

统计结果：

| 项目 | 数量 |
| --- | ---: |
| 训练样本总数 | 164 |
| `emotion` 为空或 `null` | 1 |
| 非空自由文本情感 | 163 |
| 不同非空 `emotion` 文本 | 152 |
| 形如 `A/B/C/D` 的标签 | 0 |

原因：

- `emotion` 是自由文本摘要，例如“思乡、孤独、怀念、亲情”，不是选项 ID。
- 训练数据没有 `choose` 选项集合，无法判断自由文本应映射到哪个选项。
- 多数情感是多标签或混合情绪，可能同时覆盖多个近义选项。

可靠性条件：

- 若要构造 `choose_id`，必须同时具备题面选项和显式 gold 选项，或经过 teacher/human 审核。
- 若只把 `emotion` 当作情感短证据或弱文本描述，可用于 `short_evidence.evidence.emotion` 的候选来源，但应带 `low_confidence_emotion` 或进入抽查池。
- 自动同义词规则最多只能做候选召回，不能作为最终 gold。

## 4. Answer-only 自动构造范围估计

| 子任务 | 自动构造范围 | 可行等级 | 结论 |
| --- | ---: | --- | --- |
| 词义 | 162/164 首有非空 `keywords`，共 1304 条词义 | 高 | 可由 `keywords` 构造 `qa_words` 与 `ans_qa_words`；空 `keywords` 样本不贡献词义监督 |
| 句译 | 高置信行级候选 10/164 首、20 片段；句末标点候选 106/164 首、432 片段 | 中低 | 只能构造候选，进入高置信 answer-only 前需过滤和抽查 |
| 情感 | 0/164 条可直接构造 `choose_id` | 不可行 | 自由文本 `emotion` 不能自动转四选一 gold |

推荐把自动 answer-only 拆成不同置信等级：

- 高置信词义样本：`qa_words = keywords.keys()`，`ans_qa_words = keywords`，`qa_sents = []`，`choose = {}`，`choose_id = ""`。
- 句译候选样本：只对通过严格分段检查的 `content/trans` 片段构造，默认进入复核池；复核通过后再作为 `ans_qa_sents`。
- 情感样本：不自动生成 answer-only `choose_id`。可保留原 `emotion` 作为 teacher prompt 的参考文本或弱证据，不进入主训练 gold。

## 5. Teacher-data 需要补齐的部分

对 `eval_data.json`，teacher-data 至少需要补齐：

- 327 题的最终答案 JSON。
- 960 个 `ans_qa_words` 词义答案，key 必须完全来自输入 `qa_words`。
- 665 个 `ans_qa_sents` 句译答案，key 必须完全来自输入 `qa_sents`。
- 327 个 `choose_id`，必须来自对应题目的 `choose` 选项。
- 每题的短证据：词义线索、句意骨架、情感支持证据。
- 质量标记：缺项、低置信情感、句译不确定、选项相近等。

对 `data/train-data`，若要并入训练，需要额外补齐：

- 稳定 `idx`，建议由数据源路径和样本序号生成，避免与官方 `eval_data.json` 冲突。
- `author` 缺失时填 `""`。
- 可监督的 `qa_sents/ans_qa_sents`，不能直接把整首 `trans` 当作目标句答案。
- 情感选项 `choose` 与显式 `choose_id`；若没有人工或 teacher 审核，不补 `choose_id`。
- 数据来源、转换规则版本、过滤标记和人工复核状态。

## 6. 推荐自动转换规则

可自动转换：

- `title` 原样保留；缺失作者统一填 `author = ""`。
- 内部训练样本生成稳定 `idx`，例如 `train-data/<类别>/<文件>#<offset>`。
- `content` 只去除首尾空白，保留内部换行、标点和括注。
- `keywords` 非空时：
  - `qa_words` 使用 `keywords` 的 key，保持 JSON 原顺序；
  - `ans_qa_words` 使用 `keywords` 原值；
  - 空释义进入人工确认。
- 没有情感选项时：
  - `choose = {}`；
  - `choose_id = ""`；
  - 不计入情感选择训练或评测。
- 句译只在以下条件全部满足时自动生成候选：
  - `content` 与 `trans` 分段数量一致；
  - 每个源片段和译文片段均非空；
  - 片段 key 使用 `content` 原文，不改标点；
  - 候选带来源和 `needs_human_review`，通过复核后才升为高置信。

不可自动转换：

- 不把整首 `trans` 自动拆成 `ans_qa_sents` gold。
- 不把自由文本 `emotion` 自动映射为 `choose_id`。
- 不根据 `emotion` 自动生成四个选项后再自选答案，除非该流程经过 teacher/human 审核并保留来源。
- 不改写目标词或目标句 key，不为了避免重复而添加 `#1/#2`。
- 不用句末标点数量一致作为唯一通过条件；它只能产生候选。
- 不把带空 `keywords` 的样本伪造词义答案。

## 7. 检查命令与结果摘要

本次没有运行 Python 脚本，也没有创建临时脚本；只使用 `sed`、`rg`、`find`、`jq` 做只读检查。

主要命令摘要：

```bash
sed -n '1,240p' AGENTS.md
sed -n '1,260p' docs/data-schema.md
sed -n '1,560p' docs/teacher-data.md
sed -n '1,280p' docs/training-plan.md
sed -n '1,220p' data/eval_data.json
find data/train-data -name train.json -print
jq '.[0]' data/train-data/.../train.json
jq 'length' data/eval_data.json
jq -s 'map(length) | add' data/train-data/.../train.json
```

关键结果：

- `eval_data.json`：327 题，960 个目标词，665 个目标句，327 题全部有 4 个选项。
- `train-data`：164 首，统一字段为 `title/content/keywords/trans/emotion`。
- 词义：162 首有非空 `keywords`，共 1304 条词义。
- 句译：10 首满足换行数一致；106 首满足句末标点数量一致。
- 情感：1 条 `emotion` 为空或 `null`，152 个不同非空自由文本情感，0 条可直接作为 `A/B/C/D` 标签。

## 8. 待确认问题

- 训练集中自动生成的内部 `idx` 格式是否采用路径加序号，还是另建数据 manifest 管理。
- 句译候选是否允许先以 `needs_human_review` 低权重进入 `short_evidence`，还是必须人工确认后才能进入任何训练集。
- 情感自由文本是否需要先归一到项目级情感 taxonomy；即使归一，也仍不能替代 `choose_id`。
- Teacher 生成 `choose_id` 时，是否采用单教师、强模型复审，还是多教师冲突仲裁。
