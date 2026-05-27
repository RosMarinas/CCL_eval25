# DCR: Harness Contract Split

## Trigger

The repository had two competing harness meanings. The docs define a two-stage
reasoner-to-formatter pipeline, while the older direct-final path generated
final JSON directly and wrapped that final JSON as a synthetic `draft_answer`.

## Current Approved Design

`docs/contracts/data-schema.md` and `docs/contracts/harness.md` define:

- Reasoner output: `idx`, `evidence`, `sentiment`, and `draft_answer`.
- `draft_answer` contains `ans_qa_words` and `ans_qa_sents`.
- Reasoner does not output `choose_id`.
- Formatter maps `sentiment` plus `task.choose` to final `choose_id`.

## Disputed Point

The implementation currently treats direct final JSON generation as harness
evaluation. It also validates `draft_answer.choose_id` as if it were required
when `task.choose` exists.

## Evidence Checked

- `docs/contracts/data-schema.md` says Reasoner does not output `choose_id`.
- `docs/contracts/harness.md` says Formatter generates `choose_id`.
- `src/cli/run_submission_eval.py` prompts BC8-v1 to output final JSON directly.
- `src/cli/run_harness.py` now targets the true harness path.
- `src/harness.py` now accepts missing `draft_answer.choose_id` and maps the
  final choice through formatter or local sentiment mapping.
- Remote artifacts show direct-final BC8-v1 is useful: dev-50 has one core JSON
  error, but this does not validate the two-stage contract.

## Finding

The docs direction is better for the long-term task design because the training
set has no gold `choose_id`; the model should learn sentiment analysis first,
then map sentiment to options at evaluation time. The current implementation is
still useful, but the direct-final path should be treated as a submission runner
or direct-final baseline, not as the true reasoner-to-formatter harness.

## Options

1. Keep current implementation as the only harness.
   - Simple and already produces useful final JSON metrics.
   - Does not test the intended two-stage sentiment mapping.

2. Replace current implementation with docs-only two-stage harness.
   - Aligns with the intended contract.
   - Risks losing a strong direct-final baseline and makes debugging harder.

3. Split the concepts.
   - Keep direct-final generation as `submission runner` / baseline.
   - Implement true harness separately: reasoner emits evidence, sentiment, and
     draft; formatter or local mapper emits `choose_id`.

## Recommendation

Use option 3. This recommendation has now been applied:

- `src/harness.py` accepts the documented reasoner contract.
- `src/cli/run_submission_eval.py` remains the direct-final baseline.
- `src/cli/run_harness.py` is the true harness entrypoint.

## Required Document Updates

- Keep `docs/contracts/harness.md` as the true two-stage contract.
- Keep `src/cli/run_submission_eval.py` documented as the direct-final baseline.
- Keep `src/cli/run_harness.py` documented as the true harness entrypoint.
- Add a current-state report that records remote E3/B8/BC8 artifacts as remote
  runtime state, not local source files.

## Execution Impact

The contract fix is complete. Remaining work is organizational: trim or archive
one-off helpers under `src/cli/` and keep remote command references consistent.

## Validation Plan

- Keep a regression test for reasoner output with `sentiment` and no
  `draft_answer.choose_id`.
- Verify that formatter failure falls back to local sentiment mapping.
- Run remote unit tests and `py_compile` through `python3 remote_run.py ...`.
