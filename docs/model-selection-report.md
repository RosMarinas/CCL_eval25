# E2a Candidate Gate Report

本文基于 `docs/baseline-smoke-results.md` 和 `data/baseline/smoke-summary.json` 做候选进入正式 dev baseline 的 gate 判断。本文不做最终模型选择，也不比较任务得分；最终选择必须等 E3 在同一 dev split 上完成词义、翻译、情感和 JSON 稳定性评测后再定。

## 1. Smoke 结论

第一轮 smoke 覆盖了 `P14`、`P14-fast`、`P8` 多个 8B 候选和 `FMT` formatter 候选。所有候选均可运行，无 load 或 generate 失败，`vLLM` 环境可用于下一阶段 baseline。

关键观察：

| experiment_id | sample_count | JSON 错误率 | 平均延迟 ms | P95 延迟 ms | gate 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `P14-qwen3-14b-bf16-vllm-nothink-zero` | 1 | 0.0 | 5081.6 | 5081.6 | 进入 E3 主线 |
| `P14-fast-qwen3-14b-awq4-vllm-nothink-zero` | 1 | 0.0 | 17989.17 | 17989.17 | 进入 E3 对照，重点复核延迟 |
| `P8-qwen3-8b-bf16-vllm-nothink-zero` | 2 | 0.5 | 1499.22 | 1739.56 | 进入 E3 主线 |
| `P8-qwen3-8b-awq4-vllm-nothink-zero` | 2 | 0.5 | 6051.45 | 6574.04 | 进入 E3 主线，验证速度/质量权衡 |
| `P8-internlm3-8b-instruct-bf16-vllm-normal-zero` | 2 | 1.0 | 2478.39 | 2626.74 | 进入 E3 对照，重点复核 extra text |
| `FMT-qwen3-8b-bf16-vllm-nothink-jsonfix` | 2 | 0.0 | 1925.1 | 2137.33 | 进入 E3 formatter 主线 |
| `FMT-gemma-4-e4b-it-bf16-vllm-direct-json-jsonfix` | 2 | 0.5 | 1528.77 | 2316.79 | 进入 E3 formatter 对照，重点复核缺项修复 |

本轮样本量很小，JSON 错误率和延迟只作为 gate 信号，不作为稳定排序依据。

## 2. 候选分层

### Tier 1 主线

这些配置进入正式 dev baseline 的主表，用于建立 prompt-only 上限、8B reasoner 主线和 formatter 主线。

| experiment_id | 进入原因 | E3 关注点 |
| --- | --- | --- |
| `P14-qwen3-14b-bf16-vllm-nothink-zero` | 14B prompt-only 上限候选；smoke 无 JSON 错误；中文任务适配预期最强。 | dev 上的任务分、JSON 稳定性、是否需要 few-shot。 |
| `P8-qwen3-8b-bf16-vllm-nothink-zero` | 后续 B8 / BC8 最主要基座；延迟最低；smoke 可运行。 | 句子 key 标点不匹配是否高频出现。 |
| `P8-qwen3-8b-awq4-vllm-nothink-zero` | 与 Qwen3-8B 同轴的量化对照；可评估部署成本。 | AWQ 的延迟和质量是否在 dev 上反常；句子 key 问题是否同样存在。 |
| `FMT-qwen3-8b-bf16-vllm-nothink-jsonfix` | formatter smoke JSON 错误率为 0；与 Qwen reasoner 同族，中文和 JSON 指令稳定性较好。 | 是否会重做题、是否改坏 clean draft、缺项修复是否可靠。 |

### Tier 2 对照

这些配置进入 E3 对照表，不作为第一优先主线，但必须保留用于解释量化、模型族和 formatter 替代方案。

| experiment_id | 对照价值 | 风险 |
| --- | --- | --- |
| `P14-fast-qwen3-14b-awq4-vllm-nothink-zero` | 与 P14 bf16 同模型轴，衡量 14B AWQ 的速度 / 质量权衡。 | smoke 延迟高于 bf16 P14，需要 dev 样本确认是否为首轮或 AWQ 后端开销。 |
| `P8-internlm3-8b-instruct-bf16-vllm-normal-zero` | 非 Qwen 中文 8B 对照，可判断 P8 选择是否过度绑定 Qwen。 | Markdown fence / extra text 明显；JSON-only 约束需增强。 |
| `FMT-gemma-4-e4b-it-bf16-vllm-direct-json-jsonfix` | 小参数 formatter 对照；若质量足够，可降低 harness 总参数量。 | 对缺项 draft 修复不足，可能输出空答案或非法 `choose_id`。 |

### Tier 3 暂缓

以下模型暂不进入 E3 主 baseline。它们仍可作为后续补充，但不应挤占 dev-50 / dev-100 的第一轮预算。

| 模型 | 暂缓原因 |
| --- | --- |
| `Qwen/Qwen3.5-9B` | 项目约束中第一轮暂缓；可在 Qwen3-8B dev 结果不足时作为第二轮 9B 备选。 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-*` | reasoning 输出倾向强，可能增加 JSON-only 风险；适合后续专项验证，不进入主 baseline。 |
| Phi 系列 | 中文古诗词任务针对性不足；优先级低于 Qwen / InternLM / Gemma formatter 轴。 |
| `openai/gpt-oss-20b` | 参数预算贴近 20B 上限，且不适合与 formatter 组合进小于 20B 的 harness。 |

## 3. 风险标签

| 风险标签 | 触发候选 | smoke 证据 | E3 处理 |
| --- | --- | --- | --- |
| `sentence_key_punctuation_mismatch` | `P8-qwen3-8b-bf16-vllm-nothink-zero`、`P8-qwen3-8b-awq4-vllm-nothink-zero` | idx 0 的第二个句子输出去掉了句末 `。`，导致 `missing_sentence_key` 和 `extra_sentence_key`。 | prompt 强化 key 必须逐字复制；validator 增加可观测的标点 normalize 实验标记。 |
| `extra_text_markdown_fence` | `P8-internlm3-8b-instruct-bf16-vllm-normal-zero` | 两条 smoke 均输出 fenced JSON；其中一条还伴随句子 key 标点不匹配。 | parser 可剥离 fenced JSON，但继续记录 `extra_text`，不要静默当作无错。 |
| `formatter_incomplete_repair` | `FMT-gemma-4-e4b-it-bf16-vllm-direct-json-jsonfix` | `fmt-format-error` 中输出空词义、空句译和空 `choose_id`，触发 `empty_required_answer`、`invalid_choose_id`。 | E3 单独统计 clean draft 改坏率、缺项修复率和非法选项率。 |
| `awq_latency_regression` | `P14-fast-qwen3-14b-awq4-vllm-nothink-zero` | smoke 平均延迟约 17.99s，高于 P14 bf16 的约 5.08s。 | dev-50 复核是否为 AWQ 首轮编译/加载后效应、后端配置或真实生成慢。 |

## 4. E3 实验矩阵

E3 建议先跑 `dev-50`；若预算允许，主线升到 `dev-100`，Tier 2 至少保留 `dev-50`。所有实验使用同一 dev split、同一 prompt 版本、同一 decode 参数和同一 JSON/schema 统计脚本。

### E3-dev-50 必跑

| 优先级 | experiment_id | 样本量 | 目的 |
| --- | --- | ---: | --- |
| 主线 | `P14-qwen3-14b-bf16-vllm-nothink-zero` | 50 | 建立 14B zero-shot prompt-only 上限。 |
| 主线 | `P8-qwen3-8b-bf16-vllm-nothink-zero` | 50 | 验证 B8 / BC8 基座候选的任务分和 JSON 稳定性。 |
| 主线 | `P8-qwen3-8b-awq4-vllm-nothink-zero` | 50 | 验证 8B AWQ 速度 / 质量权衡。 |
| 主线 | `FMT-qwen3-8b-bf16-vllm-nothink-jsonfix` | 50 | 验证 formatter 修复能力和改坏率。 |
| 对照 | `P14-fast-qwen3-14b-awq4-vllm-nothink-zero` | 50 | 复核 14B AWQ 延迟异常和量化质量。 |
| 对照 | `P8-internlm3-8b-instruct-bf16-vllm-normal-zero` | 50 | 验证非 Qwen 中文 8B 对照的 JSON-only 风险。 |
| 对照 | `FMT-gemma-4-e4b-it-bf16-vllm-direct-json-jsonfix` | 50 | 验证小参数 formatter 是否能稳定修复缺项。 |

### E3-dev-100 扩展

若 `dev-50` 中 Tier 1 没有严重失败，扩展以下实验到 `dev-100`：

| experiment_id | 样本量 | 扩展条件 |
| --- | ---: | --- |
| `P14-qwen3-14b-bf16-vllm-nothink-zero` | 100 | dev-50 JSON 错误率可控，作为上限曲线。 |
| `P8-qwen3-8b-bf16-vllm-nothink-zero` | 100 | 作为训练前主基座对照。 |
| `P8-qwen3-8b-awq4-vllm-nothink-zero` | 100 | 若 dev-50 延迟或质量不明显劣化。 |
| `FMT-qwen3-8b-bf16-vllm-nothink-jsonfix` | 100 | 若 clean draft 改坏率可控。 |

Tier 2 是否扩展到 `dev-100` 由 E3-dev-50 决定：`P14-fast` 只有在延迟回落或成本收益明确时扩展；`InternLM3` 只有在 parser + prompt 修复后 JSON 错误率显著下降时扩展；`Gemma FMT` 只有在缺项修复率和非法选项率达标时扩展。

## 5. E3 前最小修复

1. Prompt 强化 key 复制约束：
   - 对 `ans_qa_words` 和 `ans_qa_sents` 明确写入：“key 必须逐字复制输入数组中的原始字符串，包括标点、空格和全半角字符；不能删改句末标点。”
   - 在 few-shot 中至少保留一个含句末 `。` 的句子 key，示范原样复制。

2. Parser 支持 fenced JSON 剥离：
   - 可从 Markdown code fence 或 JSON 前后解释中提取第一个 JSON 对象。
   - 继续记录 `extra_text`，并在结果表中单独统计；不能因为成功解析就把它当成完全合法输出。

3. Validator 标点 normalize 只作为显式实验：
   - 不在主 baseline 中静默 normalize 句末标点。
   - 若要测试宽松规则，必须用独立标记，例如 `validator=punct-normalized`，并同时保留 strict schema 结果。
   - E3 报告中分别展示 strict JSON/schema 错误率和 normalized-after-parse 错误率。

4. Formatter 缺项修复压力测试：
   - `FMT` dev 输入至少保留 `fmt-clean`、`fmt-format-error`、`fmt-light-conflict` 三类。
   - 单独统计 clean draft 改坏率、缺项修复率、非法 `choose_id` 修复率和空答案率。

## 6. Gate 结论

进入 E3 的主线不是最终选型，只是“值得正式 dev baseline 验证”的候选集合。当前 gate 结论为：

- `Qwen3-14B bf16` 作为 P14 上限主线。
- `Qwen3-8B bf16` 和 `Qwen3-8B-AWQ` 作为 P8 主线。
- `Qwen3-8B` 作为 FMT 主线。
- `Qwen3-14B-AWQ`、`InternLM3-8B`、`Gemma 4 E4B` 保留为 E3 对照。
- 其他模型维持暂缓，等 E3 暴露出的能力或成本缺口再决定是否补跑。
