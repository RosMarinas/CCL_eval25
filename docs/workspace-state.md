# Workspace State

## Source Of Truth

Temporary priority order for the cleanup phase:

1. Latest user instruction: review first, protect remote artifacts, then organize docs/code/results, then review design/progress/results.
2. Contract docs: `docs/spec/data-schema.md`, `docs/spec/harness.md`, `docs/contract/eval-plan.md`, `docs/plans/training-plan.md`.
3. Current code and tests under `src/` and `tests/`.
4. Remote runtime artifacts under `data/` and `checkpoints/`.

Older chat and old plans are evidence, not authority.

## Local Control-Plane Files

Core library files currently live directly under `src/`:

- `src/schema.py`: unified input/output schema normalization and validation.
- `src/eval.py`: result records, JSON parsing, error categories, and regression metrics.
- `src/baseline.py`: prompt rendering and prompt-baseline record generation.
- `src/cli/run_baseline_matrix.py`: remote vLLM prompt-baseline matrix runner.
- `src/harness.py`: local reasoner/formatter validation and fallback utilities.

Tests currently live directly under `tests/`:

- Stable unit tests: `test_baseline.py`, `test_eval.py`, `test_harness.py`, `test_schema.py`.
- New guard tests: `test_filter_teacher_data.py`, `test_sync_protection.py`.

Runtime entrypoints currently live under `src/cli/`:

- Data construction: `generate_teacher_data.py`, `filter_teacher_data.py`, `build_training_data.py`.
- Training/evaluation: `train_b8.py`, `train_bc8.py`, `train_replay.py`, `run_submission_eval.py`, `run_harness.py`, `run_baseline_matrix.py`.
- Runtime/evaluation helpers: `run_baseline_smoke.py`, `eval_b8.py`, `eval_bc8.py`, `eval_bc8_final.py`, `inspect_b8_results.py`.
- Blocked artifact-management helper: `organize_checkpoints.py` remains in `src/cli/` but should not be run without explicit approval.

## Remote Runtime Artifacts

Remote artifacts are the authoritative runtime state. They are not all mirrored locally.

Known remote data artifacts:

- `data/splits/eval50.json`
- `data/splits/eval-dev-50.json`
- `data/teacher/train-short-evidence.jsonl`
- `data/teacher/train-short-evidence-filtered.jsonl`
- `data/teacher/train-short-evidence-filtered.report.json`
- `data/teacher/eval50-short-evidence.jsonl`
- `data/teacher/eval50-short-evidence-filtered.jsonl`
- `data/teacher/eval50-short-evidence-filtered.report.json`
- `data/training/b8-answer-only.jsonl`
- `data/training/bc8-mixed/train.jsonl`
- `data/training/bc8-mixed/val.jsonl`
- `data/training/bc8-mixed/metadata.json`
- `data/training/sentiment-mapping.jsonl`
- `data/baseline/e3-dev50/summary.json`
- `data/baseline/e3-dev100/summary.json`

Known remote checkpoint artifacts:

- `checkpoints/B8-v2-lr5e5-ep2/`
- `checkpoints/BC8-v2-lr5e5-ep2/`
- `checkpoints/B8-v1-lr2e5-ep2-10steps/`
- `checkpoints/BC8-v1-lr5e5-ep1-10steps/`
- `checkpoints/BC8-final/`
- `checkpoints/BC8-final-v2/`
- `checkpoints/BC8-final-v3/`
- `checkpoints/_archive/`
- `checkpoints/VERSIONS.md` (Outdated, does not reflect v2/v3)

## Known Design Decisions

- Training data has no gold `choose_id`; do not train reasoner to infer `choose_id` from training poems alone.
- True harness means: reasoner emits `idx`, `evidence`, `sentiment`, and `draft_answer`; formatter or local mapper emits final `choose_id`.
- Direct-final generation remains useful as a submission runner/baseline, but it is not the true reasoner-to-formatter harness.
- `src/harness.py` now accepts reasoner drafts without `draft_answer.choose_id` and routes them to formatter when `task.choose` exists.
- Remote artifact directories are protected in `sync.sh` and ignored locally to avoid overwriting server state.

## Known Drift

- `src/cli/run_submission_eval.py` currently prompts BC8-v1 to produce final JSON directly, then wraps that final JSON as `draft_answer`; this is a direct-final evaluation path, not true harness evaluation.
- `src/cli/run_harness.py` now targets the true harness path with `evidence + sentiment + draft_answer`.
- Some docs describe from-zero execution, while remote state is already past B8-v2, BC8-v2, and BC8-final-v3 experiments. `BC8-final-v3` with the H2 harness currently achieves a 4% final JSON error rate, indicating excellent format adherence.
- The runtime entrypoints are now consolidated under `src/cli/`, but several helpers still need a keep-vs-archive decision.
- Docs are categorized now, but some planning and state files intentionally remain at the top level during cleanup.

## Do Not Sync Or Move Yet

Do not move or overwrite these remote-owned artifact paths without a separate approval gate:

- `data/splits/`
- `data/teacher/`
- `data/training/`
- `data/harness/`
- `data/fewshot/`
- `data/baseline/e3-dev*/`
- `checkpoints/`

Do not move script paths referenced by remote commands until the move map and link updates are approved.

## Proposed Docs Categories

Docs category move has been applied. Current categories:

- `docs/spec/`: schema and harness contracts.
- `docs/contract/`: evaluation, prompt, and teacher-data contracts.
- `docs/plans/`: execution, training, and model/route planning.
- `docs/reports/`: baseline, gate, data inspection, model selection, and result reports.
- `docs/agents/`: agent metadata and workflow configuration.

## Docs Move Map Applied

| Previous path | Current path | Reason | Status |
| --- | --- | --- | --- |
| `docs/spec/data-schema.md` | `docs/spec/data-schema.md` | Core schema contract | Applied; references updated |
| `docs/spec/harness.md` | `docs/spec/harness.md` | True harness contract | Applied; references updated |
| `docs/contract/eval-plan.md` | `docs/contract/eval-plan.md` | Evaluation contract | Applied; references updated |
| `docs/contract/prompt-baseline.md` | `docs/contract/prompt-baseline.md` | Prompt contract and baseline protocol | Applied; references updated |
| `docs/contract/teacher-data.md` | `docs/contract/teacher-data.md` | Teacher-data contract | Applied; references updated |
| `docs/execution-plan.md` | `docs/plans/execution-plan.md` | From-zero and phase execution plan | Applied; references updated |
| `docs/training-plan.md` | `docs/plans/training-plan.md` | B8/BC8 training plan | Applied; references updated |
| `docs/bcd-plan.md` | `docs/plans/bcd-plan.md` | Later-stage BCD plan | Applied; references updated |
| `docs/data-pipeline.md` | `docs/plans/data-pipeline.md` | Data construction plan | Applied; references updated |
| `docs/agent-task-list.md` | `docs/plans/agent-task-list.md` | Issue-ready task decomposition | Applied; references updated |
| `docs/data-inspection.md` | `docs/reports/data-inspection.md` | Data schema inspection report | Applied; references updated |
| `docs/model-research.md` | `docs/reports/model-research.md` | Model research notes | Applied; references updated |
| `docs/model-selection-report.md` | `docs/reports/model-selection-report.md` | Model gate/report artifact | Applied; references updated |
| `docs/baseline-smoke-results.md` | `docs/reports/baseline-smoke-results.md` | Baseline smoke report | Applied; references updated |
| `docs/e3-dev50-results.md` | `docs/reports/e3-dev50-results.md` | E3 dev50 result report | Applied; references updated |
| `docs/e4-gate-report.md` | `docs/reports/e4-gate-report.md` | E4 gate decision report | Applied; references updated |
| `docs/dcr/harness-contract.md` | `docs/spec/harness.md` | Harness design-change record | Absorbed into harness spec and deleted |
| `docs/workspace-state.md` | `docs/workspace-state.md` | Temporary cleanup control document | Keep at top level until cleanup closes |
| `docs/current-review.md` | `docs/current-review.md` | Temporary current review document | Keep at top level until cleanup closes |
| `docs/agents/*` | `docs/agents/*` | Agent metadata already categorized | No move |

The next remaining move risk is behavioral, not structural: command consumers and remote habits must be updated before more renames.

## Source And Script Classification

Depth check:

- Command: `find src tests -mindepth 4 -type f`
- Result: no files reported. Current `src/` and `tests/` satisfy the shallow-depth constraint.

| Current path | Proposed path | Type | Risk | Validation command |
| --- | --- | --- | --- | --- |
| `src/schema.py` | `src/schema.py` | library | Low; core imports already use this path | `python3 remote_run.py python -m unittest tests.test_schema` |
| `src/eval.py` | `src/eval.py` | library | Low; shared error taxonomy | `python3 remote_run.py python -m unittest tests.test_eval` |
| `src/baseline.py` | `src/baseline.py` | library | Low; prompt utilities and records | `python3 remote_run.py python -m unittest tests.test_baseline` |
| `src/harness.py` | `src/harness.py` | library | Medium; true harness contract is still being corrected | `python3 remote_run.py python -m unittest tests.test_harness` |
| `src/cli/run_baseline_matrix.py` | `src/cli/run_baseline_matrix.py` | runtime entrypoint moved out of `src/` | Medium; remote docs and habits needed path updates | `python3 remote_run.py python -m unittest tests.test_baseline tests.test_eval` |
| `src/cli/generate_teacher_data.py` | `src/cli/generate_teacher_data.py` | runtime entrypoint | Medium; uses API key and remote-only API calls | dry-run/help plus teacher-data tests |
| `src/cli/filter_teacher_data.py` | `src/cli/filter_teacher_data.py` | runtime entrypoint with test coverage | Low; imported by `tests/test_filter_teacher_data.py` | `python3 remote_run.py python -m unittest tests.test_filter_teacher_data` |
| `src/cli/build_training_data.py` | `src/cli/build_training_data.py` | runtime entrypoint | Medium; produces remote training artifacts | add focused unit tests before moving |
| `src/cli/train_b8.py` | `src/cli/train_b8.py` | runtime entrypoint | High; remote checkpoint-producing path | do not move before command references are updated |
| `src/cli/train_bc8.py` | `src/cli/train_bc8.py` | runtime entrypoint | High; remote checkpoint-producing path | do not move before command references are updated |
| `src/cli/train_replay.py` | `src/cli/train_replay.py` | runtime entrypoint | High; remote checkpoint-producing path | do not move before command references are updated |
| `src/cli/run_submission_eval.py` | `src/cli/run_submission_eval.py` | runtime entrypoint; direct-final submission evaluation | Medium; keep as baseline, but do not confuse with true harness | remote smoke plus harness tests |
| `src/cli/run_harness.py` | `src/cli/run_harness.py` | runtime entrypoint; true reasoner-to-formatter harness | Medium; new path should get a focused remote smoke run | `python3 remote_run.py python -m unittest tests.test_harness` plus one remote smoke command |
| `src/cli/generate_submit.py` | `src/cli/generate_submit.py` | runtime entrypoint; generates output format | Low | |
| `src/cli/generate_candidates.py` | `src/cli/generate_candidates.py` | runtime entrypoint; synthetic candidate generation | Low | |
| `tests/test_baseline.py` | `tests/test_baseline.py` | test | Low | included in full unit command |
| `tests/test_eval.py` | `tests/test_eval.py` | test | Low | included in full unit command |
| `tests/test_harness.py` | `tests/test_harness.py` | test | Low | included in full unit command |
| `tests/test_schema.py` | `tests/test_schema.py` | test | Low | included in full unit command |
| `tests/test_filter_teacher_data.py` | `tests/test_filter_teacher_data.py` | test | Low | included in full unit command |
| `tests/test_sync_protection.py` | `tests/test_sync_protection.py` | test | Low | included in full unit command |
| `src/cli/run_baseline_smoke.py` | `src/cli/run_baseline_smoke.py` | runtime/helper script moved out of tests | Medium; may be referenced by docs/model-selection report | remote smoke command or unit harness |
| `src/cli/eval_b8.py` | `src/cli/eval_b8.py` or archive after current review | one-off analysis/runtime evaluation | Medium; may encode useful checkpoint eval behavior | remote eval smoke after path update |
| `src/cli/eval_bc8.py` | `src/cli/eval_bc8.py` or archive after current review | one-off analysis/runtime evaluation | Medium; may encode useful checkpoint eval behavior | remote eval smoke after path update |
| `src/cli/eval_bc8_final.py` | `src/cli/eval_bc8_final.py` or archive after current review | one-off analysis/runtime evaluation | Medium; may encode useful checkpoint eval behavior | remote eval smoke after path update |
| `src/cli/eval_p14.py` | `src/cli/eval_p14.py` or archive after current review | one-off analysis/runtime evaluation | Medium; may encode useful checkpoint eval behavior | remote eval smoke after path update |
| `src/cli/inspect_b8_results.py` | `src/cli/inspect_b8_results.py` or archive after current review | one-off analysis | Low; read-only artifact inspection | help/read smoke |
| `src/cli/organize_checkpoints.py` | keep blocked until explicit approval | one-off artifact management | High; checkpoint movement can affect remote state | manual approval and dry run required |

Do not apply this map until remote command references and artifact risks are reviewed.

## Verification Commands

Primary remote unit verification:

```bash
python3 remote_run.py python -m unittest tests.test_harness tests.test_schema tests.test_eval tests.test_baseline tests.test_filter_teacher_data tests.test_sync_protection
```

Remote artifact inventory:

```bash
python3 remote_run.py find data -maxdepth 3 -type f
python3 remote_run.py find checkpoints -maxdepth 3 -type f
```
