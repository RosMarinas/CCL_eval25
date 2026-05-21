# CCL25 Agent Task List

This file contains issue-ready task cards for agents. Labels: `ready-for-agent`.

## Dependency Order

```text
1. Data schema
2. Prompt / formatter baseline blocked by 1
3. Teacher data           blocked by 1
4. Harness / formatter    blocked by 1
5. QLoRA training         blocked by 1, 3
6. Eval and ablation      blocked by 1, 2, 4
7. BCD ablation           blocked by 5, 6
```

---

## 1. Define unified data schema and conversion rules for CCL25 tasks

Type: AFK

## What to build

Define the conversion contract from the official CCL25 raw samples into the unified input/output JSON used by baseline, teacher-data generation, training, harness, and evaluation.

## Acceptance criteria

- [ ] Define the unified input schema, including `idx`, poem text, target words, target sentences, and emotion options.
- [ ] Define the unified final output schema, including `ans_qa_words`, `ans_qa_sents`, and `choose_id`.
- [ ] Specify handling rules for empty fields, duplicate words, duplicate sentences, and missing emotion options.
- [ ] Provide at least two complete examples.
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

Design the offline teacher-data workflow for generating `short-evidence` and `teacher-critique` samples used by BC training.

## Acceptance criteria

- [ ] Provide teacher prompts that require structured JSON output.
- [ ] Define the `short-evidence` schema.
- [ ] Define the `teacher-critique` schema.
- [ ] Explicitly prohibit free-form long CoT.
- [ ] Define automatic filtering rules for JSON validity, missing fields, invalid options, and option/reason contradictions.
- [ ] Define a human audit checklist for word meaning, sentence translation, emotion choice, and critique quality.

## Blocked by

- Task 1: Define unified data schema and conversion rules for CCL25 tasks.

---

## 4. Design reasoner-to-formatter harness

Type: AFK

## What to build

Design the minimal inference harness: `reasoner -> local validator -> formatter -> final validator`. The harness separates reasoning quality from final JSON formatting.

## Acceptance criteria

- [ ] Define the reasoner output schema: `structured evidence + draft_answer`.
- [ ] Define the formatter input schema: original task + evidence + draft answer.
- [ ] Provide a formatter prompt that says not to redo the task by default, only format and lightly verify.
- [ ] Define local validator rules for invalid JSON, missing fields, invalid `choose_id`, overlong answers, and missing target words/sentences.
- [ ] Define conditions for skipping the formatter.
- [ ] Define retry policy: when to retry the reasoner, when to call the formatter, and when to fall back to rule-based repair.

## Blocked by

- Task 1: Define unified data schema and conversion rules for CCL25 tasks.

---

## 5. Design B8/BC8 QLoRA training plan

Type: AFK

## What to build

Design the QLoRA training plan for the Qwen 8B/9B reasoner, covering B8 answer-only training, BC8 mixed distillation, answer-only replay, and experiment naming.

## Acceptance criteria

- [ ] Provide recommended B8 answer-only QLoRA settings.
- [ ] Provide recommended BC8 mixed-distillation settings.
- [ ] Use the initial data mix: `answer-only 60% / short-evidence 30% / teacher-critique 10%`.
- [ ] Define when to adjust the data mix.
- [ ] Define the answer-only replay procedure.
- [ ] Define checkpoint naming and experiment IDs.
- [ ] State whether each training target emits final JSON or `structured evidence + draft_answer`.

## Blocked by

- Task 1: Define unified data schema and conversion rules for CCL25 tasks.
- Task 3: Design teacher-data generation and filtering.

---

## 6. Design evaluation and ablation tracking

Type: AFK

## What to build

Design the evaluation and ablation tracking scheme so P14/P14-fast/P8/FMT, B8, BC8, H1/H2/H3, and BCD1 remain comparable.

## Acceptance criteria

- [ ] Define experiment result fields: word score, translation score, emotion score, JSON error rate, average latency, and total score.
- [ ] Define JSON error categories.
- [ ] Define formatter regression rate: how often formatter changes a correct draft into a wrong final answer.
- [ ] Define per-task error analysis templates.
- [ ] Define ablation tables for baseline, BC, harness, and BCD.
- [ ] Define minimum dev split or sample-size requirements for each experiment.

## Blocked by

- Task 1: Define unified data schema and conversion rules for CCL25 tasks.
- Task 2: Design P14/P8/FMT baseline experiments.
- Task 4: Design reasoner-to-formatter harness.

---

## 7. Design BCD loop-block ablation

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

- Task 5: Design B8/BC8 QLoRA training plan.
- Task 6: Design evaluation and ablation tracking.
