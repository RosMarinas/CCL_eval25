# CCL25 Eval Documentation

This directory contains all the architectural decisions, plans, contracts, and reports for the CCL25 Eval project. The documentation is organised systematically to ensure clarity and traceability of the evaluation pipeline and model training process.

## Documentation Structure

### `spec/`
Contains the strict definitions for schemas and harness architecture.
- [data-schema.md](spec/data-schema.md): Core input/output JSON schema definition.
- [harness.md](spec/harness.md): The true harness interface contract (reasoner -> formatter).

### `contract/`
Contains evaluation procedures and data formats.
- [eval-plan.md](contract/eval-plan.md): The evaluation metrics, ablation tracking, and error categories.
- [prompt-baseline.md](contract/prompt-baseline.md): Prompt configurations and baseline protocol.
- [teacher-data.md](contract/teacher-data.md): Rules for the teacher model data generation.

### `plans/`
Contains the execution roadmaps and training strategies.
- [execution-plan.md](plans/execution-plan.md): From-zero data preparation and command execution steps.
- [training-plan.md](plans/training-plan.md): QLoRA training steps, configurations, and phase definitions (B8, BC8, BC8-final).
- [bcd-plan.md](plans/bcd-plan.md): Block-Coordinate Descent plans (future steps).
- [data-pipeline.md](plans/data-pipeline.md): Data pipeline structure and toolchain descriptions.
- [agent-task-list.md](plans/agent-task-list.md): Detailed task breakdown for AI agents.

### `reports/`
Contains empirical results, experiment evaluations, and milestone reports.
- [model-selection-report.md](reports/model-selection-report.md): Baseline evaluation and model gate decision log.
- [baseline-smoke-results.md](reports/baseline-smoke-results.md): Output from quick smoke tests.
- [data-inspection.md](reports/data-inspection.md): Inspection report on raw evaluation data schema.
- [e3-dev50-results.md](reports/e3-dev50-results.md), [e4-gate-report.md](reports/e4-gate-report.md), etc.

### `agents/`
Contains agent metadata and workflow configuration.

### Top-Level State
- [workspace-state.md](workspace-state.md): Tracks current drift, latest checkpoint progress, and known divergences between implementation and planning. Always check this file for the latest runtime context!
