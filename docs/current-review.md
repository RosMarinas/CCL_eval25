# Current Review

## Design Review

The project should keep two separate execution concepts:

- **Submission runner**: direct final JSON generation. This is the practical path for final-answer quality, JSON error rate, and latency.
- **True harness**: `reasoner -> validator -> formatter/local mapper -> final validator`. The reasoner emits `evidence`, `sentiment`, and `draft_answer` without `choose_id`; the formatter or mapper emits final `choose_id`.

This split is necessary because the training poems do not contain gold `choose_id`. Training the reasoner to emit `choose_id` from those poems would create false supervision. The better design is to train sentiment analysis and separately test sentiment-to-option mapping on eval-style samples.

## Implementation Review

Stable core modules are already reasonably shallow:

- `src/schema.py`: input/output normalization and validation.
- `src/eval.py`: JSON parsing, error categories, experiment records, formatter regression metrics.
- `src/baseline.py`: prompt rendering and prompt-baseline records.
- `src/harness.py`: harness validator and fallback utilities.

The main implementation drift is naming and responsibility:

- `src/cli/run_submission_eval.py` evaluates direct-final outputs.
- `src/cli/run_harness.py` evaluates true reasoner outputs with `evidence + sentiment + draft_answer`.
- Runtime entrypoints are now consolidated under `src/cli/`, but several evaluation helpers still need a keep-vs-archive decision.
- Docs are now categorized into `contracts`, `plans`, `reports`, and `dcr`, while `workspace-state.md` and `current-review.md` remain top-level control documents during cleanup.

The immediate code-level harness contract is improved: `src/harness.py` no longer requires `draft_answer.choose_id` for reasoner drafts and routes missing choice mapping to formatter when `task.choose` exists.

## Result Review

Remote E3 dev100 baseline:

| Experiment | JSON err | Hard err | Format err | Avg latency |
| --- | ---: | ---: | ---: | ---: |
| P14 Qwen3-14B bf16 nothink | 0.00 | 0.00 | 0.00 | 4507 ms |
| P14 Qwen3-14B bf16 think | 0.00 | 0.00 | 1.00 | 17633 ms |
| P8 Qwen3-8B bf16 nothink | 0.03 | 0.03 | 0.03 | 1862 ms |
| P8 Qwen3-8B bf16 think | 0.03 | 0.01 | 0.99 | 10889 ms |
| P8 Qwen3-8B AWQ nothink | 0.00 | 0.00 | 0.01 | 7259 ms |
| InternLM3-8B think | 0.07 | 0.00 | 1.00 | 3046 ms |
| P14-fast Qwen3-14B AWQ nothink | 0.00 | 0.00 | 0.00 | 17126 ms |
| InternLM3-8B normal | 0.07 | 0.00 | 1.00 | 3046 ms |

Teacher/training data:

- Train teacher filtering: 153 / 164 passed after strict filtering.
- BC8 mixed data: `67-33-0`, with 164 answer-only, 80 short-evidence, 0 teacher-critique, 219 train and 25 validation samples.

BC8-v1 remote eval:

- JSON error rate: 0.02.
- Error count: 1 / 50.
- The remaining error is a malformed final JSON around idx 7, not a measured true-harness sentiment mapping failure.

## Risks

- The current direct-final path may look good while true sentiment-to-option mapping remains untested.
- The BC8 mixed dataset has no teacher-critique samples despite docs still describing a 60-30-10 target.
- Moving scripts before updating remote commands could break checkpoint/eval workflows.
- Moving docs before link updates could make future agents follow stale paths.
- Local sync must keep protecting remote data/checkpoint artifacts; otherwise local cleanup can overwrite useful remote state.

## Recommended Next Experiments

1. **Submission runner sanity**: preserve the current direct-final BC8-v1/BC8-final evaluation as the fast submission baseline.
2. **True harness smoke**: add a small true-harness reasoner prompt that emits `evidence + sentiment + draft_answer` on 5-10 eval samples, then use `src/harness.py` to map through formatter/local mapper.
3. **Local mapper baseline**: implement a deterministic sentiment-to-choice mapper before using an LLM formatter for every sample. Use it as H0 for mapping.
4. **Formatter gate**: only promote H2 if it improves final score or fixes JSON errors without introducing choice-mapping regressions.
5. **Data mix review**: explicitly decide whether to keep `67-33-0` or generate teacher-critique data before more BC8 training.

## Stop Conditions

- Stop directory moves if any remote unit test fails after a move batch.
- Stop true-harness expansion if reasoner outputs omit `sentiment` or produce invalid controlled vocabulary labels above a small threshold.
- Stop formatter promotion if it fixes formatting but increases wrong `choose_id` rate.
- Stop BC8 training expansion if direct-final JSON errors are already low but semantic/choice metrics are not being measured.
- Stop sync changes if they would upload local `data/splits`, `data/teacher`, `data/training`, `data/harness`, `data/fewshot`, `data/baseline/e3-dev*`, or `checkpoints`.
