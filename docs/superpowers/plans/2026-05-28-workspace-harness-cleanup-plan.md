# Workspace Harness Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the CCL25 workspace by first reviewing source truth, then organizing code/docs/results without disturbing remote runtime artifacts, then reviewing design/progress/results for the next research step.

**Architecture:** Treat local files as source/control-plane files and remote `data/*` plus `checkpoints/*` as runtime artifacts. Keep `src/` and `tests/` shallow, preserve the direct-final submission runner separately from the true reasoner-to-formatter harness, and update docs only after each verified fact is checked.

**Tech Stack:** Python 3.11, `uv`, remote execution through `python3 remote_run.py`, `unittest`, rsync-based `sync.sh`, Markdown docs.

---

### Task 1: Freeze Source Of Truth And Inventory

**Files:**
- Read: `docs/dcr/harness-contract.md`
- Read: `docs/contracts/data-schema.md`
- Read: `docs/contracts/harness.md`
- Read: `docs/plans/execution-plan.md`
- Read: `src/harness.py`
- Read: `src/cli/run_harness.py`
- Read: remote `data/` and `checkpoints/` listings
- Modify: `docs/workspace-state.md`

- [ ] **Step 1: Capture local status**

Run:

```bash
git status --short
find src tests docs scripts -maxdepth 3 -type f | sort
```

Expected: a local inventory that separates tracked edits, untracked scripts, docs, and tests.

- [ ] **Step 2: Capture remote artifact status**

Run:

```bash
python3 remote_run.py find data -maxdepth 3 -type f
python3 remote_run.py find checkpoints -maxdepth 3 -type f
```

Expected: a remote inventory including `data/splits/`, `data/teacher/`, `data/training/`, `data/baseline/e3-dev50`, `data/baseline/e3-dev100`, and B8/BC8 checkpoints.

- [ ] **Step 3: Write the state document**

Create or update `docs/workspace-state.md` with these sections:

```markdown
# Workspace State

## Source Of Truth
## Local Control-Plane Files
## Remote Runtime Artifacts
## Known Design Decisions
## Known Drift
## Do Not Sync Or Move Yet
## Verification Commands
```

- [ ] **Step 4: Verify state doc against facts**

Run:

```bash
python3 remote_run.py python -m unittest tests.test_harness tests.test_schema tests.test_eval tests.test_baseline tests.test_filter_teacher_data tests.test_sync_protection
```

Expected: all tests pass. If a remote artifact path in the doc is not present in the remote listing, fix the doc before continuing.

### Task 2: Split Harness Concepts Without Moving Directories

**Files:**
- Modify: `docs/contracts/harness.md`
- Modify: `docs/plans/execution-plan.md`
- Modify: `src/cli/run_harness.py` comments/docstring only
- Modify: `src/harness.py` only if tests reveal a contract gap
- Test: `tests/test_harness.py`

- [ ] **Step 1: Add terminology to docs**

Update `docs/contracts/harness.md` to define:

```markdown
- Submission runner: direct final JSON generation used for final-answer metrics.
- True harness: reasoner emits evidence, sentiment, and draft_answer; formatter or local mapper emits choose_id.
```

- [ ] **Step 2: Mark current script accurately**

Update only the docstring/comment in `src/cli/run_harness.py` so it says it currently evaluates the direct-final/submission path unless explicitly run with a true reasoner-output prompt.

- [ ] **Step 3: Add or keep contract tests**

Ensure `tests/test_harness.py` contains a test where a reasoner output has `sentiment` and no `draft_answer.choose_id`, and the next action is `call_formatter`.

- [ ] **Step 4: Verify harness contract**

Run:

```bash
python3 remote_run.py python -m unittest tests.test_harness
```

Expected: all harness tests pass.

### Task 3: Organize Docs By Category

**Files:**
- Modify: docs only
- Do not move code or data in this task

- [ ] **Step 1: Propose doc categories in `docs/workspace-state.md`**

Use these categories:

```text
docs/contracts/      schema, harness contracts
docs/plans/          execution and training plans
docs/reports/        baseline/gate/result reports
docs/agents/         agent metadata
docs/dcr/            decision-change records
```

- [ ] **Step 2: Prepare a move map**

Add a table with columns:

```markdown
| Current path | Proposed path | Reason | Link updates needed |
```

- [ ] **Step 3: Do not move yet**

Stop after the move map. Wait for approval before moving docs because link updates can be noisy.

### Task 4: Organize Code And Scripts By Responsibility

**Files:**
- Read: `src/*.py`
- Read: `src/cli/*.py`
- Read: `tests/*.py`
- Modify: `docs/workspace-state.md`

- [ ] **Step 1: Classify Python files**

Classify each file as one of:

```text
library: belongs in src/
test: belongs in tests/
runtime entrypoint: belongs in `src/cli/`
one-off analysis: should be archived or converted to tests
```

- [ ] **Step 2: Check depth constraint**

Run:

```bash
find src tests -mindepth 4 -type f
```

Expected: no source/test file deeper than the two-level rule. If output appears, list it in `docs/workspace-state.md`.

- [ ] **Step 3: Prepare a move map**

Add a table with columns:

```markdown
| Current path | Proposed path | Type | Risk | Validation command |
```

- [ ] **Step 4: Do not move yet**

Stop after the move map. Wait for approval before moving scripts because remote commands may reference current paths.

### Task 5: Review Design, Progress, Implementation, And Results

**Files:**
- Create: `docs/current-review.md`
- Read: `docs/workspace-state.md`
- Read: `docs/dcr/harness-contract.md`
- Read: remote result summaries

- [ ] **Step 1: Pull current remote metrics**

Run:

```bash
python3 remote_run.py python -m json.tool data/baseline/e3-dev100/summary.json
python3 remote_run.py python -m json.tool data/training/bc8-mixed/metadata.json
python3 remote_run.py python -m json.tool checkpoints/BC8-v1-lr5e5-ep1-10steps/eval_result.json
```

Expected: E3 dev100 summary, BC8 data mix metadata, and BC8-v1 eval result are readable.

- [ ] **Step 2: Write `docs/current-review.md`**

Use this structure:

```markdown
# Current Review

## Design Review
## Implementation Review
## Result Review
## Risks
## Recommended Next Experiments
## Stop Conditions
```

- [ ] **Step 3: Verify before recommendation**

Run:

```bash
python3 remote_run.py python -m unittest tests.test_harness tests.test_schema tests.test_eval tests.test_baseline tests.test_filter_teacher_data tests.test_sync_protection
```

Expected: all tests pass before claiming the review is ready.

### Task 6: Execution Gate

**Files:**
- Read: `docs/workspace-state.md`
- Read: `docs/current-review.md`

- [ ] **Step 1: Ask for approval before moving files**

Before any move, present:

```markdown
## Proposed Moves
## Sync Risk
## Remote Paths At Risk
## Validation Commands
```

- [ ] **Step 2: Move only after approval**

Use `git mv` for tracked files and normal `mv` only for untracked local-only files.

- [ ] **Step 3: Verify after each batch**

Run:

```bash
python3 remote_run.py python -m unittest tests.test_harness tests.test_schema tests.test_eval tests.test_baseline tests.test_filter_teacher_data tests.test_sync_protection
git status --short
```

Expected: tests pass, and `git status --short` contains only intentional moves/edits.
