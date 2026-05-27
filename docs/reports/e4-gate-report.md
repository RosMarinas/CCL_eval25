# E4 Gate Report: B8 Base Model & Execution Scope

Date: 2026-05-27
Based on: E3 dev-50 baseline results (`data/baseline/e3-dev50/`)

## 1. E3 dev-50 Results Recap

| experiment_id | json_err | hard_err | fmt_err | avg_lat_ms |
|---|---:|---:|---:|---:|
| P14-qwen3-14b-bf16-vllm-nothink-zero | 0.00 | 0.00 | 0.00 | 4564 |
| P14-qwen3-14b-bf16-vllm-think-zero | 0.00 | 0.00 | 1.00 | 18089 |
| P8-qwen3-8b-bf16-vllm-nothink-zero | 0.04 | 0.04 | 0.04 | 1879 |
| P8-qwen3-8b-bf16-vllm-think-zero | 0.04 | 0.02 | 0.98 | 11376 |
| P8-qwen3-8b-awq4-vllm-nothink-zero | 0.00 | 0.00 | 0.02 | 7303 |
| P8-internlm3-8b-instruct-bf16-vllm-think-zero | 0.06 | 0.00 | 1.00 | 3021 |
| P8-internlm3-8b-instruct-bf16-vllm-normal-zero | 0.06 | 0.00 | 1.00 | 3021 |
| P14-fast-qwen3-14b-awq4-vllm-nothink-zero | 0.00 | 0.00 | 0.00 | 17423 |

## 2. Per-Sample Error Analysis

### P8 Qwen3-8B bf16 nothink (2 errors, hard_err = 4%)

| idx | Error | Root Cause |
|--:|---|---|
| 30 | `missing_top_field`, `wrong_field_type`, `missing_word_key`, `missing_sentence_key` | **JSON typo**: `)` instead of `}` to close `ans_qa_words` object. Raw: `"烽火":"战争中的信号火"),` |
| 49 | `missing_top_field`, `wrong_field_type`, `missing_word_key`, `missing_sentence_key` | **JSON typo**: Missing closing `"` on last `ans_qa_words` value. Raw: `"砧杵":"捣衣声，表现秋夜的寂静与生活气息}` |

Both are **character-level JSON syntax errors** — punctuation mistakes, not content or comprehension failures. The model understands the schema but makes typographical errors during token-by-token JSON generation.

### P8 Qwen3-8B bf16 think (2 errors)

| idx | Error | Root Cause |
|--:|---|---|
| 0 | `missing_sentence_key`, `extra_sentence_key` | **Sentence key punctuation**: Stripped trailing `。` from sentence key |
| 14 | `parse_error` | **Thinking trace overflow**: JSON unparseable from `<think>` output |

### P8 Qwen3-8B AWQ nothink (0 errors)

Clean across all 50 samples. However, AWQ quantization makes it less suitable as a QLoRA base (training on already-quantized weights adds complexity).

### P8 InternLM3-8B (6 errors each mode, hard_err = 0%)

All 6 errors are `missing_sentence_key` + `extra_sentence_key` (sentence key punctuation mismatch). Outputs are wrapped in ````json` fence, which the parser strips — the content is valid, hence 0% hard error.

## 3. Gate Decision

### 3.1 Base Model Selection: Qwen3-8B bf16

| Candidate | hard_err | Latency | QLoRA Suitable | Verdict |
|---|---|---|---|---|
| Qwen3-8B bf16 | 4% | 1879ms | Yes (full precision) | **SELECTED** |
| Qwen3-8B AWQ | 0% | 7303ms | No (pre-quantized) | Deployment fallback |
| InternLM3-8B | 0% | 3021ms | Would work but diff family | Rejected (sentence key issues + fence wrapping) |

**Reasoning:**
- Qwen3-8B bf16 has full-precision weights, ideal as QLoRA base
- Fastest inference (1879ms) — headroom for training overhead
- Errors are purely mechanical (JSON punctuation typos), not conceptual
- Same model family as the planned Qwen3-8B formatter — consistent tokenizer and chat template
- AWQ is pre-quantized: training on AWQ weights with QLoRA would nest quantization (quantized base + 4-bit QLoRA), increasing complexity without clear benefit

### 3.2 B8 Execution Scope: Shortened

Per `docs/plans/training-plan.md` Section 2 rules:

| P8 Core Error Rate | B8 Strategy |
|---|---|
| < 5% | B8 shortened to format confirmation (~0.3 epoch), or skip directly to BC8 |
| 5%–15% | Full B8 |
| > 15% | Full B8 with increased epochs |

**P8 Qwen3-8B bf16 hard_err = 4% → < 5% → Shortened B8**

**Decision: Shortened B8**, not skipped.

Rationale for not skipping:
- B8 at 0.3 epoch is extremely cheap (a few minutes of training)
- Provides a clean format checkpoint that serves as fallback if BC8 degrades formatting
- The 2 errors are typographical — even minimal answer-only training should eliminate them
- De-risks the pipeline: if BC8 introduces format regression, B8 is the recovery point

Shortened B8 parameters:
- 0.3 epoch (or fewer steps — early-stop when dev JSON error rate hits 0)
- Standard QLoRA config: lora_r=16, lora_alpha=32, lr=1e-4
- Only answer-only data (final JSON input → final JSON output)
- Accept if dev JSON error rate = 0%; if errors persist, extend to full B8

### 3.3 Fallback Plan

If shortened B8 does not resolve the typos:
1. Extend to full B8 (2 epochs) with increased answer-only data
2. If Qwen3-8B bf16 continues to produce typos after full B8, switch base to Qwen3-8B AWQ and skip B8 (go directly to BC8 with AWQ base)

## 4. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| B8 shortened doesn't fix typos | Low | Medium | Extend to full B8; AWQ fallback |
| BC8 introduces format regression | Medium | Low | B8 checkpoint as recovery point; answer-only replay |
| AWQ latency too high for harness | Medium | Low | Acceptable for final submission; 7.3s is within budget |
| Sentence key punctuation issue (think mode) | High | Medium | Already handled by parser in nothink; train with strict key-copy examples |

## 5. Next Steps

1. **Phase 2**: Generate teacher data (short-evidence + sentiment) for 164 training poems via DeepSeek-V4-Flash API
2. **Phase 3**: Assemble B8 answer-only training data
3. **Phase 5**: Run shortened B8 training (0.3 epoch)
4. Verify B8 eliminates JSON typos on dev-50, then proceed to BC8
