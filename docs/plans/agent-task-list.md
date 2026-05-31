# CCL25 Agent Task List

This file contains issue-ready task cards for agents. Labels: `ready-for-agent`.

## Dependency Order

```text
1. Data schema (incl. Reasoner intermediate schema with sentiment)
2. Prompt / formatter baseline   blocked by 1
3. Teacher data                  blocked by 1
4. Data pipeline                 blocked by 1, 3
5. Harness / formatter           blocked by 1
6. QLoRA training                blocked by 1, 3, 4
7. Eval and ablation             blocked by 1, 2, 5
8. BCD ablation                  blocked by 6, 7
```

Note: Sentiment analysis is a two-stage pipeline. Reasoner (Stage1) outputs `sentiment` (NOT `choose_id`). Formatter (Stage2) maps `sentiment` → `choose_id`. Training data from `train-data/` (164 poems) has no `choose` options. Teacher model is DeepSeek-V4-Flash (API), key in `api-key.txt`.

---

## 1. Define unified data schema and conversion rules for CCL25 tasks

Type: AFK

## What to build

Define the conversion contract from the official CCL25 raw samples into the unified input/output JSON used by baseline, teacher-data generation, training, harness, and evaluation. Includes the Reasoner intermediate output schema for the two-stage sentiment pipeline.

## Acceptance criteria

- [ ] Define the unified input schema, including `idx`, poem text, target words, target sentences, and emotion options.
- [ ] Define the unified final output schema, including `ans_qa_words`, `ans_qa_sents`, and `choose_id`.
- [ ] Define the Reasoner intermediate output schema, including `evidence`, `sentiment` (with `primary`, `secondary`, `rationale`), and `draft_answer` (without `choose_id`).
- [ ] Define the sentiment label controlled vocabulary (8 categories, ~24 labels).
- [ ] Distinguish training input (may lack `choose`) from evaluation input (must have `choose`).
- [ ] Specify handling rules for empty fields, duplicate words, duplicate sentences, and missing emotion options.
- [ ] Provide complete examples for each schema type (input, final output, intermediate output).
- [ ] List boundary cases that need human confirmation.

## Blocked by

None - can start immediately.

---

## 2. Design P14/P8/FMT baseline experiments

Type: AFK

## What to build

Design prompt-only baselines for Qwen3-14B and multiple 8B-level reasoner candidates, plus formatter baselines for Qwen3-8B and Gemma 4 E4B. These experiments measure the non-finetuned upper bound, choose the reasoner finetuning base, and choose the formatter candidate for harness work.

## Acceptance criteria

- [ ] Provide a zero-shot prompt template.
- [ ] Provide a few-shot prompt template.
- [ ] Require the model to output only the final JSON.
- [ ] Include `P14` for Qwen3-14B and `P14-fast` for Qwen3-14B-AWQ.
- [ ] Include `P8` as an 8B-level candidate sweep; do not include Qwen3.5-9B in the first round.
- [ ] Include `FMT` formatter baselines for Qwen3-8B and Gemma 4 E4B.
- [ ] Keep the prompt template compatible with Qwen3-14B, 8B-level reasoner candidates, Qwen3-8B formatter, and Gemma 4 E4B formatter.
- [ ] Define experiment ID fields: model full name, parameter size, quantization, backend, thinking mode, prompt type, shot count, and decoding parameters.
- [ ] Define result table fields, including task scores, JSON error rate, and latency.

## Blocked by

- Task 1: Define unified data schema and conversion rules for CCL25 tasks.

---

## 3. Design teacher-data generation and filtering

Type: AFK

## What to build

Design the offline teacher-data workflow using DeepSeek-V4-Flash API. Generate `short-evidence` (with `sentiment` analysis) and `teacher-critique` (with sentiment critique) samples used by BC training.

## Acceptance criteria

- [ ] Provide teacher prompts that require structured JSON output with `sentiment` field.
- [ ] Define the `short-evidence` schema including `evidence`, `sentiment` (primary/secondary/rationale), and `draft_answer` (without `choose_id`).
- [ ] Define the `teacher-critique` schema with emotion critique targeting sentiment analysis accuracy.
- [ ] Explicitly prohibit free-form long CoT.
- [ ] Specify DeepSeek-V4-Flash API calling method: endpoint, key management (from `api-key.txt`), rate limiting, retry.
- [ ] Define sentiment label controlled vocabulary and normalization rules.
- [ ] Define automatic filtering rules for JSON validity, sentiment vocabulary compliance, field coverage, and evidence-sentiment consistency.
- [ ] Distinguish training poem generation (no `choose`, sentiment only) from eval dev poem generation (has `choose`, includes `choose_id`).
- [ ] Define a human audit checklist for word meaning, sentence translation, sentiment analysis, and critique quality.

## Blocked by

- Task 1: Define unified data schema and conversion rules for CCL25 tasks.

---

## 4. Design reasoner-to-formatter harness (two-stage sentiment)

Type: AFK

## What to build

Design the minimal two-stage inference harness: `reasoner -> local validator -> formatter -> final validator`. Reasoner outputs sentiment analysis (NOT choose_id); Formatter additionally maps sentiment to choose_id.

## Acceptance criteria

- [ ] Define the reasoner output schema: `evidence + sentiment + draft_answer` (no `choose_id` in draft_answer).
- [ ] Define the formatter input schema: original task + evidence + sentiment + draft answer + validator report.
- [ ] Provide a formatter prompt that (a) formats and validates JSON, and (b) maps sentiment → choose_id based on sentiment.primary matching against task.choose options.
- [ ] Define local validator rules for invalid JSON, missing sentiment fields, sentiment.primary vocabulary compliance, missing target words/sentences.
- [ ] Define conditions for skipping the formatter (only when task.choose is empty — no options to map).
- [ ] Define retry policy: when to retry reasoner, when to call formatter, and sentiment mapping fallback (keyword matching between sentiment.primary and option text).

## Blocked by

- Task 1: Define unified data schema and conversion rules for CCL25 tasks.

---

## 5. Design end-to-end data construction pipeline

Type: AFK

## What to build

Design the complete data pipeline from raw data sources to training-ready datasets, including teacher data generation (DeepSeek API), filtering, and training data assembly.

## Acceptance criteria

- [ ] Document all data sources and their characteristics (train-data 164 poems, eval_data 327 poems).
- [ ] Define dev split strategy (eval dev 50, train dev 30) and few-shot pool construction.
- [ ] Define DeepSeek-V4-Flash API calling procedure: endpoint, key management, rate limiting, batch processing, retry logic.
- [ ] Specify teacher data generation scripts and commands (via `python3 remote_run.py`).
- [ ] Specify auto-filtering pipeline for teacher outputs.
- [ ] Specify training dataset assembly: B8 answer-only, BC8 mixed (60/30/10), sentiment mapping data.
- [ ] Document security constraints: API key in api-key.txt (never committed), eval data leak prevention.
- [ ] Define output directory structure and data versioning.

## Blocked by

- Task 1: Define unified data schema and conversion rules for CCL25 tasks.
- Task 3: Design teacher-data generation and filtering.

---

## 6. Design B8/BC8 QLoRA training plan

Type: AFK

## What to build

Design the QLoRA training plan for the Qwen 8B reasoner, covering B8 answer-only training (without choose_id), BC8 mixed distillation (with sentiment analysis training), answer-only replay, and experiment naming.

## Acceptance criteria

- [ ] Provide recommended B8 answer-only QLoRA settings. B8 output is final JSON without `choose_id` (training poems have no `choose` options).
- [ ] Provide recommended BC8 mixed-distillation settings. Sentiment analysis is a core training target.
- [ ] Use the initial data mix: `answer-only 50% / short-evidence 25% / teacher-critique 25%`.
- [ ] Define training output contracts: `final_json` (no choose_id for training poems), `evidence_draft` (with sentiment, without choose_id in draft), `critique_correction` (emotion critique targets sentiment analysis).
- [ ] Define when to adjust the data mix (e.g., sentiment vocabulary compliance drops).
- [ ] Define the answer-only replay procedure.
- [ ] Define checkpoint naming and experiment IDs.

## Blocked by

- Task 1: Define unified data schema and conversion rules for CCL25 tasks.
- Task 3: Design teacher-data generation and filtering.
- Task 5: Design end-to-end data construction pipeline.

---

## 7. Design evaluation and ablation tracking

Type: AFK

## What to build

Design the evaluation and ablation tracking scheme with two-stage sentiment scoring (sentiment analysis accuracy + mapping accuracy).

## Acceptance criteria

- [ ] Define experiment result fields: word score, translation score, sentiment primary accuracy, sentiment mapping accuracy, emotion (choose_id) score, JSON error rate, latency.
- [ ] Define JSON error categories (separate Reasoner intermediate errors from final output errors).
- [ ] Define formatter regression rate: split into format regression and sentiment mapping regression.
- [ ] Define per-task error analysis templates with sentiment-specific error types (sentiment_misanalysis, sentiment_mapping_error, sentiment_vocab_mismatch).
- [ ] Define ablation tables for baseline, BC, harness (with sentiment accuracy columns), and BCD.
- [ ] Define minimum dev split or sample-size requirements for each experiment.

## Blocked by

- Task 1: Define unified data schema and conversion rules for CCL25 tasks.
- Task 2: Design P14/P8/FMT baseline experiments.
- Task 4: Design reasoner-to-formatter harness.

---

## 8. Design BCD loop-block ablation

Type: AFK

## What to build

Design BCD0, BCD1, and BCD2 loop-block ablations. This task should start only after BC8-final and the evaluation protocol are stable.

## Acceptance criteria

- [ ] Define the strategy for looping the middle third of Transformer blocks.
- [ ] Define BCD0: direct inference-time loop for quick risk detection.
- [ ] Define BCD1: loop structure plus continued QLoRA.
- [ ] Define BCD2: gated loop plus continued QLoRA.
- [ ] Define abandonment conditions: higher JSON error rate, translation regression, vLLM deployment complexity, or unstable gains.
- [ ] Define comparison against `BC8-final` and `BC8-final + harness`.

## Blocked by

- Task 6: Design B8/BC8 QLoRA training plan.
- Task 7: Design evaluation and ablation tracking.
