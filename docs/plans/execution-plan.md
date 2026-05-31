# 执行计划

本文定义从零到最终提交的完整执行步骤。每步包含：输入、输出、执行命令、验证方式。

## 前置条件

- [ ] 服务器可通过 `python3 remote_run.py` 访问
- [ ] `api-key.txt` 存在于项目根目录（已在 `.gitignore`，不得提交）
- [ ] 远程环境已安装基础依赖（`uv pip install -e .`）
- [ ] GPU 依赖已通过 `requirements-remote.txt` 安装
- [ ] 文档集（`docs/*.md`）已完善且通过一致性检查

## Phase 1: 数据准备

**目标**：准备好所有原始数据、dev split、few-shot 样例池。

### Step 1.1: 验证原始数据

```bash
python3 remote_run.py python src/cli/validate_data.py --input data/eval_data.json
python3 remote_run.py python src/cli/validate_data.py --input data/train-data
```

**验证**：所有 327 + 164 = 491 条样本通过 schema 校验；记录 warnings（空字段、重复项等）。

### Step 1.2: 构造 Dev Split

```bash
python3 remote_run.py python src/cli/create_splits.py \
  --eval-data data/eval_data.json \
  --train-data data/train-data \
  --eval-dev-size 50 \
  --train-dev-size 30 \
  --seed 42 \
  --output-dir data/splits/
```

**输出**：
- `data/splits/dev-50-ids.txt` — 评测 dev split 的 50 个 idx
- `data/splits/train-dev-30-ids.txt` — 训练 dev split 的 30 个 idx
- `data/splits/eval-dev-50.json` — 评测 dev split 完整样本
- `data/splits/train-dev-30.json` — 训练 dev split 完整样本

**验证**：
- Dev split 不与训练集重叠
- 评测 dev split 的 `choose` 选项 A/B/C/D 分布均衡（每个选项至少 8 条）
- 覆盖多词语、多句子、无词语、无句子等边界形态

### Step 1.3: 构造 Few-shot 样例池

```bash
python3 remote_run.py python src/cli/build_fewshot_pool.py \
  --eval-data data/eval_data.json \
  --exclude-ids data/splits/dev-50-ids.txt \
  --output data/fewshot/balanced_static.json \
  --size 5
```

**验证**：5 个样例覆盖了单/多词语、单/多句翻译、不同情感选项、不同诗歌类型。所有样例来自 dev split 之外。

## Phase 2: Teacher 数据生成

**目标**：使用 DeepSeek-V4-Flash API 生成训练所需的全部 teacher 数据。

### Step 2.1: 为训练集生成 Short-evidence

```bash
python3 remote_run.py python src/cli/generate_teacher_data.py \
  --input data/train-data \
  --type short-evidence \
  --model deepseek-v4-flash \
  --api-key-file api-key.txt \
  --output data/teacher/train-short-evidence.jsonl \
  --batch-size 10 \
  --rate-limit 5
```

**预计耗时**：164 首 × ~3s/首 ≈ 8-10 分钟（含重试）
**输出**：`data/teacher/train-short-evidence.jsonl`（~164 条，每条含 evidence + sentiment + draft_answer；训练集无 `choose` 选项，不包含 `choose_id`）

### Step 2.2: 为评测 Dev Split 生成 Short-evidence

```bash
python3 remote_run.py python src/cli/generate_teacher_data.py \
  --input data/splits/eval-dev-50.json \
  --type short-evidence \
  --model deepseek-v4-flash \
  --api-key-file api-key.txt \
  --output data/teacher/dev-short-evidence.jsonl \
  --batch-size 10 \
  --rate-limit 5
```

**输出**：`data/teacher/dev-short-evidence.jsonl`（~50 条，含 evidence + sentiment + draft_answer + choose_id + final_answer）

### Step 2.3: 生成 Teacher-critique

Teacher-critique 需要候选错误答案。**注意：此步骤依赖 Phase 4 的 baseline 输出。** 若 baseline 尚未执行，首轮应使用合成扰动（synthetic perturbations）作为候选，或推迟此步骤到 Phase 4 之后。

```bash
# 方案 A：使用 baseline 输出（需先执行 Phase 4 Step 4.1）
python3 remote_run.py python src/cli/extract_errors.py \
  --baseline data/baseline/e3-dev50/P8-qwen3-8b-bf16-vllm-nothink-zero.jsonl \
  --output data/teacher/candidates.jsonl

# 方案 B：使用合成扰动（无需 baseline，适用于首轮）
python3 remote_run.py python src/cli/generate_candidates.py \
  --input data/train-data \
  --teacher data/teacher/train-short-evidence-filtered.jsonl \
  --output data/teacher/candidates-synthetic.jsonl \
  --perturb-types word_swap,sentence_omit,emotion_flip

# 生成 teacher-critique
python3 remote_run.py python src/cli/generate_teacher_data.py \
  --input data/teacher/candidates.jsonl \
  --type teacher-critique \
  --model deepseek-v4-flash \
  --api-key-file api-key.txt \
  --output data/teacher/train-critique.jsonl \
  --batch-size 10 \
  --rate-limit 5
```

**输出**：`data/teacher/train-critique.jsonl`（~80-120 条，取决于候选错误样本数量）
**依赖**：Phase 4 Step 4.1（若方案 A）或无（若方案 B）

### Step 2.4: Teacher 数据过滤

```bash
python3 remote_run.py python src/cli/filter_teacher_data.py \
  --input data/teacher/train-short-evidence.jsonl \
  --output data/teacher/train-short-evidence-filtered.jsonl \
  --strict

python3 remote_run.py python src/cli/filter_teacher_data.py \
  --input data/teacher/dev-short-evidence.jsonl \
  --output data/teacher/dev-short-evidence-filtered.jsonl \
  --strict

python3 remote_run.py python src/cli/filter_teacher_data.py \
  --input data/teacher/train-critique.jsonl \
  --output data/teacher/train-critique-filtered.jsonl \
  --strict
```

**验证**：
- JSON 解析错误率 = 0（过滤后）
- `sentiment.primary` 100% 在受控词汇表中
- `ans_qa_words` key 覆盖 100% 目标词
- `ans_qa_sents` key 覆盖 100% 目标句
- 人工抽查 5-10% 样本（按 `docs/contract/teacher-data.md` 第 6 节 checklist）

## Phase 3: 训练数据组装

**目标**：将过滤后的 teacher 数据 + 原始 keywords 组装成训练数据集。

### Step 3.1: 组装 B8 Answer-only 数据

```bash
python3 remote_run.py python src/cli/build_training_data.py \
  --type answer-only \
  --keywords data/train-data \
  --teacher data/teacher/train-short-evidence-filtered.jsonl \
  --output data/training/b8-answer-only.jsonl
```

**验证**：样本数 > 150；`choose_id` 全部为 `""`（训练集无选项）。

### Step 3.2: 组装 BC8 Mixed 数据

```bash
python3 remote_run.py python src/cli/build_training_data.py \
  --type bc8-mixed \
  --ratio 50-25-25 \
  --answer-only data/training/b8-answer-only.jsonl \
  --short-evidence data/teacher/train-short-evidence-filtered.jsonl \
  --teacher-critique data/teacher/train-critique-filtered.jsonl \
  --output-dir data/training/bc8-mixed/
```

**验证**：
- 三类数据比例符合 60/30/10（误差 ±5%）
- `short-evidence` 子集每条都含 `sentiment` 字段
- `teacher-critique` 子集每条都含 `critique.emotion_error`
- 训练集数据不含来自评测 dev split 的 idx

### Step 3.3: 组装 Sentiment Mapping 数据（Formatter 用）

```bash
python3 remote_run.py python src/cli/build_training_data.py \
  --type sentiment-mapping \
  --teacher data/teacher/dev-short-evidence-filtered.jsonl \
  --output data/training/sentiment-mapping.jsonl
```

**验证**：每条含 `sentiment` + `choose_id` pair；来自评测 dev split 50 首。

## Phase 4: Prompt Baseline

**目标**：建立不微调的 prompt-only baseline，验证基座模型能力。

### Step 4.1: 运行 P14 / P8 Baseline

```bash
# 14B prompt baseline (上限)
python3 remote_run.py python src/cli/run_baseline_matrix.py \
  --experiment P14-qwen3-14b-bf16-vllm-nothink-zero \
  --model Qwen/Qwen3-14B \
  --prompt zero \
  --input data/splits/eval-dev-50.json \
  --output data/baseline/e3-dev50/

# 8B prompt baseline (训练基座)
python3 remote_run.py python src/cli/run_baseline_matrix.py \
  --experiment P8-qwen3-8b-bf16-vllm-nothink-zero \
  --model Qwen/Qwen3-8B \
  --prompt zero \
  --input data/splits/eval-dev-50.json \
  --output data/baseline/e3-dev50/
```

**验证**：JSON 错误率 < 10%（8B）、< 5%（14B）；记录词义分、翻译分、情感分（choose_id 准确率）。

### Step 4.2: 运行 Thinking Ablation

```bash
python3 remote_run.py python src/cli/run_baseline_matrix.py \
  --experiment P14-qwen3-14b-bf16-vllm-think-zero \
  --model Qwen/Qwen3-14B \
  --mode think \
  --prompt zero \
  --input data/splits/eval-dev-50.json \
  --output data/baseline/e3-dev50/

python3 remote_run.py python src/cli/run_baseline_matrix.py \
  --experiment P8-qwen3-8b-bf16-vllm-think-zero \
  --model Qwen/Qwen3-8B \
  --mode think \
  --prompt zero \
  --input data/splits/eval-dev-50.json \
  --output data/baseline/e3-dev50/
```

### Step 4.3: 运行 FMT Formatter Baseline

```bash
python3 remote_run.py python src/cli/run_baseline_matrix.py \
  --experiment FMT-qwen3-8b-bf16-vllm-nothink-jsonfix \
  --model Qwen/Qwen3-8B \
  --task formatter \
  --input data/baseline/e3-dev50/P8-qwen3-8b-bf16-vllm-nothink-zero.jsonl \
  --output data/baseline/e3-dev50/
```

**Gate 判断**（按 `docs/reports/model-selection-report.md` 标准）：
- P8 JSON 错误率 > 15% → 必须执行完整 B8
- P8 JSON 错误率 5-15% → 执行完整 B8
- P8 JSON 错误率 < 5% → B8 缩短为格式确认

## Phase 5: B8 Answer-only QLoRA

**目标**：将 8B 基座适配到最终 JSON 格式。

### Step 5.1: 训练 B8

```bash
python3 remote_run.py python src/cli/train_b8.py \
  --base-model Qwen/Qwen3-8B \
  --train-data data/training/b8-answer-only.jsonl \
  --dev-data data/splits/train-dev-30.json \
  --output-dir checkpoints/B8/ \
  --lora-r 16 --lora-alpha 32 \
  --lr 1e-4 \
  --epochs 2 \
  --batch-size 4 --grad-accum 16 \
  --seq-length 2048
```

**验证**：
- JSON 错误率低于 P8 baseline
- `ans_qa_words` 和 `ans_qa_sents` key 覆盖 100%
- 对训练 dev split 评测词义分和句译分

## Phase 6: BC8 Mixed Distillation

**目标**：在 B8 基础上通过混合蒸馏提升古诗词理解能力。

### Step 6.1: 训练 BC8

```bash
python3 remote_run.py python src/cli/train_bc8.py \
  --base-model Qwen/Qwen3-8B \
  --lora-checkpoint checkpoints/B8/best/ \
  --train-data data/training/bc8-mixed/ \
  --dev-data data/splits/train-dev-30.json \
  --output-dir checkpoints/BC8/ \
  --lr 5e-5 \
  --epochs 1 \
  --batch-size 4 --grad-accum 16
```

**验证**：
- 词义分和句译分不低于 B8
- `sentiment.primary` 在受控词汇表中的比例 > 95%
- JSON 格式错误率不高于 B8

### Step 6.2: BC8-final Answer-only Replay

```bash
python3 remote_run.py python src/cli/train_replay.py \
  --base-model Qwen/Qwen3-8B \
  --lora-checkpoint checkpoints/BC8/best/ \
  --train-data data/training/b8-answer-only.jsonl \
  --dev-data data/splits/train-dev-30.json \
  --output-dir checkpoints/BC8-final/ \
  --lr 2e-5 \
  --epochs 0.5
```

**验证**：
- JSON 格式错误率降至最低
- 任务分无明显回退（总分下降 < 0.02）

## Phase 7: Harness 评测

**目标**：在评测 dev split 上分别评测 direct-final submission runner 和 true harness。

术语：

- **Submission runner**：模型直接生成最终提交 JSON，用于最终答案质量、JSON 错误率和延迟评测。当前 `src/cli/run_submission_eval.py` 属于此路径。
- **True harness**：Reasoner 输出 `evidence + sentiment + draft_answer`，其中 `draft_answer` 不含 `choose_id`；Formatter 或本地 mapper 根据 `sentiment + task.choose` 生成最终 `choose_id`。当前 `src/cli/run_harness.py` 属于此路径。

当前阶段保留 submission runner 作为强 baseline，并单独评测 true harness。不要把 direct-final 结果误解为已经验证了 sentiment→choose_id 的两阶段设计。

### Step 7.1: 运行 direct-final submission baseline

```bash
python3 remote_run.py python src/cli/run_submission_eval.py \
  --reasoner-checkpoint checkpoints/BC8-v1-lr5e5-ep1-10steps/adapter/ \
  --dev-data data/splits/eval50.json \
  --mode h1 \
  --output-dir data/harness/
```

### Step 7.2: 运行 direct-final H2（BC8-final direct output + Formatter）

```bash
python3 remote_run.py python src/cli/run_submission_eval.py \
  --reasoner-checkpoint checkpoints/BC8-v1-lr5e5-ep1-10steps/adapter/ \
  --base-model Qwen/Qwen3-8B \
  --dev-data data/splits/eval50.json \
  --mode h2 \
  --output-dir data/harness/
```

**验证**：
- Formatter sentiment→choose_id 映射准确率
- Formatter 改坏率（包括情感映射改坏）
- JSON 错误率、formatter 调用率、延迟

### Step 7.3: 补 true harness

```bash
python3 remote_run.py python src/cli/run_harness.py \
  --reasoner-checkpoint checkpoints/BC8-v1-lr5e5-ep1-10steps/adapter/ \
  --dev-data data/splits/eval50.json \
  --mode h1 \
  --output-dir data/harness/

python3 remote_run.py python src/cli/run_harness.py \
  --reasoner-checkpoint checkpoints/BC8-v1-lr5e5-ep1-10steps/adapter/ \
  --base-model Qwen/Qwen3-8B \
  --dev-data data/splits/eval50.json \
  --mode h2 \
  --output-dir data/harness/
```

true harness 入口已存在，要求 reasoner prompt 输出 `evidence + sentiment + draft_answer` 而不是最终 JSON。`src/cli/run_submission_eval.py` 结果仍只代表 submission runner 指标。

### Step 7.4: Harness Gate 判断

| 条件 | 决策 |
| --- | --- |
| H2 总分 > H1 且 formatter 改坏率 ≤ 2% | H2 进入主候选 |
| H2 只降低 JSON 错误率但 net_gain ≤ 0 | 优先 BC8-final + 规则 postprocess |
| H2 情感映射改坏率 > 5% | 检查 Formatter prompt，考虑单独训练 sentiment mapper |

## Phase 8: 最终评测

**目标**：在完整评测集上运行最终候选方案。

### Step 8.1: 最终候选评测

```bash
python3 remote_run.py python src/run_final_eval.py \
  --experiment FINAL \
  --checkpoint <最终候选 checkpoint> \
  --mode <direct|harness> \
  --input data/eval_data.json \
  --output data/final/
```

**输出**：最终提交 JSON + 评测指标。

### Step 8.2: 生成最终选择记录

按 `docs/contract/eval-plan.md` 第 8 节格式填写最终选择摘要。

## Phase 9: BCD 消融（条件执行）

**前置条件**（全部满足才启动）：

- [ ] BC8-final 在 dev_main 上结果稳定
- [ ] H1 / H2 harness 结果已记录
- [ ] BCD 启动前 checkpoint 已保存

### Step 9.1: BCD0 快速风险检测

按 `docs/plans/bcd-plan.md` 第 3 节执行。若 JSON 崩坏或输出重复，直接放弃 BCD。

### Step 9.2: BCD1 循环 + 继续 QLoRA

按 `docs/plans/bcd-plan.md` 第 4 节执行。

### Step 9.3: BCD-H

按 `docs/plans/bcd-plan.md` 第 6 节执行。

## 脚本清单

| 脚本 | Phase | 用途 |
| --- | --- | --- |
| `src/cli/validate_data.py` | 1 | 数据 schema 校验（计划中，当前仓库未实现） |
| `src/cli/create_splits.py` | 1 | 构造 dev split（计划中，当前仓库未实现） |
| `src/cli/build_fewshot_pool.py` | 1 | 构造 few-shot 样例池（计划中，当前仓库未实现） |
| `src/cli/generate_teacher_data.py` | 2 | Teacher 数据生成（DeepSeek API） |
| `src/cli/extract_errors.py` | 2 | 从 baseline 提取错误样本（计划中，当前仓库未实现） |
| `src/cli/generate_candidates.py` | 2 | 使用合成扰动生成候选错误样本 |
| `src/cli/filter_teacher_data.py` | 2 | Teacher 数据自动过滤 |
| `src/cli/build_training_data.py` | 3 | 训练数据集组装 |
| `src/cli/train_b8.py` | 5 | B8 answer-only QLoRA |
| `src/cli/train_bc8.py` | 6 | BC8 mixed distillation |
| `src/cli/train_replay.py` | 6 | BC8-final answer-only replay |
| `src/cli/run_baseline_matrix.py` | 4 | Prompt baseline 评测 |
| `src/cli/run_harness.py` | 7 | True harness 推理 |
| `src/cli/run_submission_eval.py` | 8 | 最终评测 (Submission Runner) |
| `src/cli/generate_submit.py` | 8 | 生成最终提交数据格式 |

所有脚本均通过 `python3 remote_run.py` 在服务器上执行。
