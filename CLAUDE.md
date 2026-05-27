# CLAUDE.md

## Core Philosophy

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 项目说明

1. 你的修改应该使得项目更易于理和维护，而不是更复杂。需要进行改进时请直接在源代码中修改，而不是新建文件;
2. 你可以编写脚本临时测试功能，但是请将脚本放在tests/目录下，并在测试完成后及时清理;
3. 用户可能会修改代码，请你在覆盖用户修改前，先检查用户的改动内容，确保不会覆盖用户的重要修改.

## 环境
- 使用 uv 作为包管理器，通过修改 pyproject.toml 来添加新的依赖，使用 `uv run` 来运行代码。
- 本地环境为 macOS，不要在本地直接运行代码。项目代码会通过 `sync.sh` 自动同步到 Linux 服务器，除非出现问题，否则你不需要手动同步
  你只需要在本地修改代码。
- 依赖管理：
  - 轻量级/跨平台依赖：本地 `uv add <包名>`，然后执行 `python3 remote_run.py uv pip install -e .` 同步到服务器。
  - GPU 依赖（vllm, torch, transformers 等）：这些包没有 macOS wheel，不能通过 `uv add` 添加。它们写在 `requirements-remote.txt` 中。服务器初始环境已通过这些安装，**不需要也不应该重复安装**。如需新增 GPU 依赖，编辑 `requirements-remote.txt` 后执行 `python3 remote_run.py uv pip install -r requirements-remote.txt`。
  - **禁止** `remote_run.py uv sync`：`uv sync` 会根据 pyproject.toml 删除远程未声明的包（会毁掉 vllm/torch 等 GPU 环境）。
  - `uv.lock` 已被 sync.sh 排除同步（平台相关，macOS vs Linux 不能共用），远程会在 `uv pip install` 时自行解析。
- 运行代码：使用转发器 `python3 remote_run.py <command>`。例如 `python3 remote_run.py python train.py`。服务器输出会自动返回。

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues at `github.com:RosMarinas/CCL_eval25`. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the canonical label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs` at the repo root. See `docs/agents/domain.md`.
