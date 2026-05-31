# P14 Baseline: 14B Prompt-Only vs BC8-v2 QLoRA

## Goal

Establish a 14B prompt-only baseline (no fine-tuning) on the full 327-sample eval set, to quantify the gain from BC8 QLoRA training. Compare JSON error rate, format quality, and official task scores.

## Approach

| Item | Choice | Rationale |
| --- | --- | --- |
| Model | `Qwen/Qwen3-14B-AWQ` | Fits single 48GB GPU (~8GB), fast inference |
| Prompt | `render_prompt_text` (same as BC8 training) | Fair comparison, same input format |
| Harness | H1 rule-only postprocess (no formatter) | Consistent with BC8 evaluation |
| Data | `data/eval_data.json` (327 samples) | Same as BC8-v2 full eval |

## Steps

### 1. Script: `src/cli/eval_p14.py`

Adapt from `tests/full_eval.py`:
- Remove LoRA loading (`PeftModel`)
- Load Qwen3-14B-AWQ directly via `AutoModelForCausalLM.from_pretrained`
- Keep `render_prompt_text` prompt, error classification, output format
- Output to `data/baseline/P14/full_eval.json`

### 2. Run eval

```bash
uv run python src/cli/eval_p14.py \
  --base-model Qwen/Qwen3-14B-AWQ \
  --device cuda:0 \
  --output data/baseline/P14/full_eval.json
```

### 3. Generate submit.json for official scoring

Adapt `generate_submit.py` to support no-LoRA mode (`--no-lora --base-model Qwen/Qwen3-14B-AWQ`).

### 4. Compare

| Metric | P14 (prompt-only) | BC8-v2 (QLoRA) |
| --- | --- | --- |
| JSON error rate | ? | 1.83% |
| Format error rate | ? | 2.14% |
| Task A (word) | ? | 0.813 |
| Task B (translation) | ? | 0.569 |
| Emotion accuracy | ? | 0.813 |
| Total score | ? | 0.6915 |

## Files

- **New**: `src/cli/eval_p14.py` — prompt-only 14B evaluation script
- **Modify**: `src/cli/generate_submit.py` — add `--no-lora` mode for 14B
