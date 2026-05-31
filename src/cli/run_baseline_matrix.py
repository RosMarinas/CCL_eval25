"""
E3 dev-100 prompt-only baseline runner.

Runs all 8 experiments sequentially via vLLM on the remote GPU server.
Each model generates answers for the first 100 eval samples.
Per-sample errors are isolated; consecutive failures trigger early abort.
Completed models are checkpointed so a restart skips finished work.

Usage:
    python3 remote_run.py python src/cli/run_baseline_matrix.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.baseline import ModelConfig, PromptConfig, build_experiment_id, run_prompt_baseline
from src.eval import CORE_JSON_ERRORS, classify_json_errors, compute_json_error_rates
from src.inference import (
    VLLM_PORT, API_URL,
    start_vllm, wait_for_vllm, is_vllm_ready, cleanup_vllm,
    kill_orphan_vllm, is_port_open, served_model_name, run_text, inspect_runtime,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "eval_data.json"
E3_DIR = ROOT / "data" / "baseline" / "e3-dev100"
DOC_PATH = ROOT / "docs" / "e3-dev100-results.md"
MASTER_DETAIL_PATH = E3_DIR / "results.jsonl"
MASTER_SUMMARY_PATH = E3_DIR / "summary.json"
FAILURE_PATH = E3_DIR / "failures.jsonl"
DEV_SPLIT_SIZE = 100
GENERATE_TIMEOUT = int(os.environ.get("E3_GENERATE_TIMEOUT", "240"))
VLLM_LOAD_TIMEOUT = int(os.environ.get("E3_VLLM_LOAD_TIMEOUT", "3600"))
MAX_CONSECUTIVE_FAILURES = 5
MAX_TOKENS_THINK = 2048
MAX_TOKENS_NOTHINK = 768

BASE_DECODE_PARAMS = {"temperature": 0, "top_p": 0.8}


def _decode_params(thinking_mode: str) -> dict[str, Any]:
    return {
        **BASE_DECODE_PARAMS,
        "max_tokens": MAX_TOKENS_THINK if thinking_mode == "thinking" else MAX_TOKENS_NOTHINK,
    }

EXPERIMENTS: list[dict[str, Any]] = [
    # ---- Tier 1 main line ----
    {
        "group": "P14",
        "model": "Qwen/Qwen3-14B",
        "parameter_scale": "14.8B",
        "quantization": "bf16",
        "thinking_mode": "non-thinking",
    },
    {
        "group": "P14",
        "model": "Qwen/Qwen3-14B",
        "parameter_scale": "14.8B",
        "quantization": "bf16",
        "thinking_mode": "thinking",
    },
    {
        "group": "P8",
        "model": "Qwen/Qwen3-8B",
        "parameter_scale": "8.2B",
        "quantization": "bf16",
        "thinking_mode": "non-thinking",
    },
    {
        "group": "P8",
        "model": "Qwen/Qwen3-8B",
        "parameter_scale": "8.2B",
        "quantization": "bf16",
        "thinking_mode": "thinking",
    },
    {
        "group": "P8",
        "model": "Qwen/Qwen3-8B-AWQ",
        "parameter_scale": "8.2B",
        "quantization": "awq4",
        "thinking_mode": "non-thinking",
    },
    {
        "group": "P8",
        "model": "internlm/internlm3-8b-instruct",
        "parameter_scale": "8B",
        "quantization": "bf16",
        "thinking_mode": "thinking",
    },
    # ---- Tier 2 control ----
    {
        "group": "P14-fast",
        "model": "Qwen/Qwen3-14B-AWQ",
        "parameter_scale": "14.8B",
        "quantization": "awq4",
        "thinking_mode": "non-thinking",
    },
    {
        "group": "P8",
        "model": "internlm/internlm3-8b-instruct",
        "parameter_scale": "8B",
        "quantization": "bf16",
        "thinking_mode": "normal",
    },
]


# ============================================================
# Entry point
# ============================================================


def baseline_runner() -> int:
    """Run the full E3 dev-100 baseline matrix and write all outputs."""
    E3_DIR.mkdir(parents=True, exist_ok=True)
    (E3_DIR / "logs").mkdir(parents=True, exist_ok=True)

    all_tasks = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    dev_tasks = all_tasks[:DEV_SPLIT_SIZE]

    all_summaries: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    all_cleanup: list[dict[str, Any]] = []
    all_detail_paths: list[Path] = []

    try:
        for i, spec in enumerate(EXPERIMENTS):
            experiment_id, model_config = _build_configs(spec)
            print(f"\n{'=' * 60}")
            print(f"[{i + 1}/{len(EXPERIMENTS)}] {experiment_id}")
            print(f"{'=' * 60}")

            if _model_already_done(experiment_id):
                print(f"[SKIP] {experiment_id} already has {DEV_SPLIT_SIZE} results — skipping.")
                summary = _load_existing_summary(experiment_id)
                if summary is not None:
                    all_summaries.append(summary)
                all_detail_paths.append(_per_model_detail_path(experiment_id))
                continue

            before = inspect_runtime()
            if before["port_open"]:
                print("[WARN] Port 8000 in use before startup — attempting cleanup.")
                kill_orphan_vllm()

            proc = None
            stage = "load"
            try:
                proc = start_vllm(spec, log_dir=E3_DIR / "logs")
                wait_for_vllm(proc, timeout_s=VLLM_LOAD_TIMEOUT)
                stage = "generate"
                detail_rows = run_experiment(spec, dev_tasks)
                summary = make_summary(spec, detail_rows, "ok", "")
                all_summaries.append(summary)
                _write_per_model(experiment_id, detail_rows, summary)
                all_detail_paths.append(_per_model_detail_path(experiment_id))
                print(f"[OK] {experiment_id}: json_error_rate={summary['json_error_rate']:.3f}, "
                      f"avg_latency={summary['avg_latency_ms']:.0f}ms")
            except Exception as exc:
                failure = {
                    "experiment_id": experiment_id,
                    "model": spec["model"],
                    "stage": stage,
                    "error_summary": f"{type(exc).__name__}: {exc}"[:2000],
                    "traceback": traceback.format_exc()[-2000:],
                    "next_step": _next_step_for(str(exc)),
                }
                all_failures.append(failure)
                all_summaries.append(summary_for_failure(spec, str(exc)))
                print(f"[FAIL] {experiment_id} ({stage}): {exc}")
            finally:
                all_cleanup.append(cleanup_vllm(proc, spec))

    except KeyboardInterrupt:
        print("\n[ABORT] Interrupted — progress on completed models is saved.")

    # Aggregate master outputs
    _write_master_detail(all_detail_paths)
    _write_master_summary(all_summaries)
    _write_failures(all_failures)
    _write_report(all_summaries, all_failures, all_cleanup, all_detail_paths)
    return 0 if not all_failures else 1


# ============================================================
# Per-model logic
# ============================================================


def _build_configs(spec: dict[str, Any]) -> tuple[str, tuple[ModelConfig, PromptConfig]]:
    model_config = ModelConfig(
        group=spec["group"],
        model_name=spec["model"],
        parameter_scale=spec["parameter_scale"],
        quantization=spec["quantization"],
        backend="vllm",
        thinking_mode=spec["thinking_mode"],
    )
    prompt_config = PromptConfig(
        prompt_type="zero-shot",
        shot_count=0,
        decoding_params=_decode_params(spec["thinking_mode"]),
    )
    experiment_id = build_experiment_id(model_config, prompt_config)
    return experiment_id, (model_config, prompt_config)


def run_experiment(spec: dict[str, Any], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _, (model_config, prompt_config) = _build_configs(spec)
    generate_fn = _make_safe_generate(spec)
    results = run_prompt_baseline(tasks, model_config, prompt_config, generate_fn)
    rows: list[dict[str, Any]] = []
    for result, task in zip(results, tasks):
        categories = classify_json_errors(
            result.raw_output, task, result.parsed_json, result.parse_errors
        )
        row = result.to_record()
        row["json_error_categories"] = categories
        # json_valid 只依赖核心错误（输出不可用才算错）
        core_categories = [c for c in categories if c in CORE_JSON_ERRORS]
        row["core_valid"] = not core_categories
        rows.append(row)
    return rows


def _make_safe_generate(spec: dict[str, Any]):
    failures = [0]

    def generate(prompt: str, metadata: dict[str, Any]) -> str:
        try:
            result = _call_chat_completion(prompt, spec, metadata)
            failures[0] = 0
            return result
        except Exception as exc:
            failures[0] += 1
            metadata["generate_error"] = f"{type(exc).__name__}: {exc}"
            if failures[0] >= MAX_CONSECUTIVE_FAILURES:
                name = spec["model"]
                raise RuntimeError(
                    f"Aborting {name}: {MAX_CONSECUTIVE_FAILURES} consecutive generation failures"
                ) from exc
            return ""

    return generate


def _call_chat_completion(prompt: str, spec: dict[str, Any], metadata: dict[str, Any]) -> str:
    """Send one chat-completion request with thinking-mode-specific payload."""
    served_name = served_model_name(spec)
    model_name = spec["model"]
    thinking_mode = spec["thinking_mode"]

    # Build messages
    content = prompt
    if model_name.startswith("Qwen/Qwen3"):
        if thinking_mode in ("non-thinking", "normal"):
            content = f"{prompt}\n/no_think"
            metadata["nothink_handling"] = "enable_thinking=False + /no_think suffix"
        else:
            metadata["think_handling"] = "enable_thinking=True"
    elif "internlm3" in model_name.lower():
        if thinking_mode == "thinking":
            metadata["think_handling"] = "chat_template_kwargs enable_thinking=True (experimental)"

    messages = [{"role": "user", "content": content}]
    payload: dict[str, Any] = {
        "model": served_name,
        "messages": messages,
        **_decode_params(thinking_mode),
    }

    # Thinking-mode chat_template_kwargs
    if model_name.startswith("Qwen/Qwen3"):
        enable = thinking_mode == "thinking"
        payload["chat_template_kwargs"] = {"enable_thinking": enable}
    elif "internlm3" in model_name.lower() and thinking_mode == "thinking":
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=GENERATE_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {text[:1000]}") from exc
    return body["choices"][0]["message"]["content"]


def _next_step_for(error: str) -> str:
    lowered = error.lower()
    if "timeout" in lowered or "did not become ready" in lowered:
        return "Increase VLLM_LOAD_TIMEOUT or pre-download weights."
    if "not found" in lowered or "gated" in lowered or "401" in lowered or "403" in lowered:
        return "Verify HF model ID, license gate, and remote credentials."
    if "out of memory" in lowered or "cuda" in lowered or "oom" in lowered:
        return "Retry AWQ or lower max-model-len / gpu-memory-utilization."
    if "consecutive generation failures" in lowered:
        return "Inspect vLLM log for server crash; may need to restart model."
    if "port 8000" in lowered:
        return "Stop or move the existing service before retrying."
    return "Inspect the per-model vLLM log under data/baseline/e3-dev100/logs/."


# ============================================================
# Metrics
# ============================================================


def make_summary(
    spec: dict[str, Any], rows: list[dict[str, Any]], status: str, notes: str
) -> dict[str, Any]:
    experiment_id, _ = _build_configs(spec)
    latencies = [row["latency_ms"] for row in rows if row.get("latency_ms") is not None]
    categories = [row.get("json_error_categories", []) for row in rows]
    rates = compute_json_error_rates(categories)
    return {
        "experiment_id": experiment_id,
        "group": spec["group"],
        "model": spec["model"],
        "thinking_mode": spec["thinking_mode"],
        "backend": "vllm",
        "quantization": spec["quantization"],
        "prompt_type": "zero-shot",
        "shot_count": 0,
        "sample_count": len(rows),
        "json_error_rate": rates["json_error_rate"],
        "hard_json_error_rate": rates["hard_json_error_rate"],
        "format_style_error_rate": rates["format_style_error_rate"],
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p95_latency_ms": _p95(latencies),
        "status": status,
        "notes": notes,
    }


def summary_for_failure(spec: dict[str, Any], notes: str) -> dict[str, Any]:
    experiment_id, _ = _build_configs(spec)
    return {
        "experiment_id": experiment_id,
        "group": spec["group"],
        "model": spec["model"],
        "thinking_mode": spec["thinking_mode"],
        "backend": "vllm",
        "quantization": spec["quantization"],
        "prompt_type": "zero-shot",
        "shot_count": 0,
        "sample_count": 0,
        "json_error_rate": None,
        "hard_json_error_rate": None,
        "format_style_error_rate": None,
        "avg_latency_ms": None,
        "p95_latency_ms": None,
        "status": "failed",
        "notes": notes[:300],
    }


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    index = max(0, math.ceil(0.95 * len(values)) - 1)
    return round(sorted(values)[index], 2)


# ============================================================
# Output & checkpointing
# ============================================================


def _per_model_detail_path(experiment_id: str) -> Path:
    return E3_DIR / f"{experiment_id}.jsonl"


def _per_model_summary_path(experiment_id: str) -> Path:
    return E3_DIR / f"{experiment_id}.summary.json"


def _model_already_done(experiment_id: str) -> bool:
    path = _per_model_detail_path(experiment_id)
    if not path.exists():
        return False
    try:
        lines = [line for line in path.read_text(encoding="utf-8").strip().split("\n") if line]
        return len(lines) >= DEV_SPLIT_SIZE
    except Exception:
        return False


def _load_existing_summary(experiment_id: str) -> dict[str, Any] | None:
    path = _per_model_summary_path(experiment_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_per_model(
    experiment_id: str, rows: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    _per_model_detail_path(experiment_id).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    _per_model_summary_path(experiment_id).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_master_detail(paths: list[Path]) -> None:
    lines: list[str] = []
    for path in paths:
        if path.exists():
            lines.extend(
                line for line in path.read_text(encoding="utf-8").strip().split("\n") if line
            )
    MASTER_DETAIL_PATH.write_text(
        "".join(line + "\n" for line in lines),
        encoding="utf-8",
    )


def _write_master_summary(summaries: list[dict[str, Any]]) -> None:
    MASTER_SUMMARY_PATH.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "dev_split": f"first-{DEV_SPLIT_SIZE}",
                "decode_params": BASE_DECODE_PARAMS,
                "max_tokens": {"think": MAX_TOKENS_THINK, "nothink": MAX_TOKENS_NOTHINK},
                "summaries": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_failures(failures: list[dict[str, Any]]) -> None:
    if not failures:
        return
    FAILURE_PATH.write_text(
        "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in failures),
        encoding="utf-8",
    )


# ============================================================
# Report
# ============================================================


def _write_report(
    summaries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    cleanup_records: list[dict[str, Any]],
    detail_paths: list[Path],
) -> None:
    DOC_PATH.write_text(
        _render_report(summaries, failures, cleanup_records, detail_paths),
        encoding="utf-8",
    )


def _render_report(
    summaries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    cleanup_records: list[dict[str, Any]],
    detail_paths: list[Path],
) -> str:
    lines = [
        "# E3 dev-100 Baseline Results",
        "",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        "",
        f"Scope: {DEV_SPLIT_SIZE} samples (first {DEV_SPLIT_SIZE} from eval_data.json), "
        "8 prompt-only experiments. No LLM judge scores — JSON stability and latency only.",
        "",
        f"Decode params: `temperature={BASE_DECODE_PARAMS['temperature']}`, "
        f"`top_p={BASE_DECODE_PARAMS['top_p']}`, "
        f"max_tokens=**{MAX_TOKENS_THINK}** (think) / **{MAX_TOKENS_NOTHINK}** (nothink).",
        "",
        "## Result Table",
        "",
        "| # | experiment_id | thinking | sample_count | json_error_rate | hard_err "
        "| format_err | avg_lat_ms | p95_lat_ms | status |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for i, row in enumerate(summaries, 1):
        lines.append(
            f"| {i} | {_md(row['experiment_id'])} | {_md(row.get('thinking_mode',''))} "
            f"| {_md(row['sample_count'])} "
            f"| {_md(_fmt_rate(row.get('json_error_rate')))} "
            f"| {_md(_fmt_rate(row.get('hard_json_error_rate')))} "
            f"| {_md(_fmt_rate(row.get('format_style_error_rate')))} "
            f"| {_md(row.get('avg_latency_ms'))} "
            f"| {_md(row.get('p95_latency_ms'))} "
            f"| {_md(row['status'])} |"
        )

    # Thinking vs nothinking comparison
    _append_think_vs_nothink(lines, summaries)

    # JSON error samples
    _append_error_samples(lines, detail_paths)

    # Latency table
    _append_latency_table(lines, summaries)

    # Failures
    _append_failures(lines, failures)

    # Cleanup
    cleaned = all(r.get("terminated") or r.get("pid") is None for r in cleanup_records)
    lines.extend([
        "",
        "## vLLM Cleanup",
        "",
        f"All vLLM processes cleaned up: `{cleaned}`.",
        "",
        f"Per-model details: `data/baseline/e3-dev100/<experiment_id>.jsonl`",
        f"Machine-readable summary: `{MASTER_SUMMARY_PATH.relative_to(ROOT)}`",
        f"Failures: `{FAILURE_PATH.relative_to(ROOT)}`",
        "",
    ])
    return "\n".join(lines)


def _append_think_vs_nothink(lines: list[str], summaries: list[dict[str, Any]]) -> None:
    """Add thinking-ablation comparison table for paired experiments."""
    lines.extend([
        "",
        "## Thinking vs Nothink Ablation",
        "",
    ])
    pairs = [
        ("P14 Qwen3-14B bf16", "P14", "Qwen/Qwen3-14B", "bf16"),
        ("P8 Qwen3-8B bf16", "P8", "Qwen/Qwen3-8B", "bf16"),
    ]
    lines.append(
        "| pair | mode | json_error_rate | hard_err | format_err "
        "| avg_lat_ms | p95_lat_ms |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"
    )
    for label, group, model, quant in pairs:
        think = _find_summary(summaries, group, model, quant, "thinking")
        nothink = _find_summary(summaries, group, model, quant, "non-thinking")
        for mode_label, s in [("think", think), ("nothink", nothink)]:
            if s is None:
                lines.append(f"| {label} | {mode_label} | — | — | — | — | — |")
            else:
                lines.append(
                    f"| {label} | {mode_label} "
                    f"| {_md(_fmt_rate(s.get('json_error_rate')))} "
                    f"| {_md(_fmt_rate(s.get('hard_json_error_rate')))} "
                    f"| {_md(_fmt_rate(s.get('format_style_error_rate')))} "
                    f"| {_md(s.get('avg_latency_ms'))} "
                    f"| {_md(s.get('p95_latency_ms'))} |"
                )

    # InternLM3 think vs normal
    lines.extend([
        "",
        "InternLM3-8B:",
        "",
        "| mode | json_error_rate | hard_err | format_err "
        "| avg_lat_ms | p95_lat_ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for mode in ("thinking", "normal"):
        s = _find_summary(
            summaries, "P8", "internlm/internlm3-8b-instruct", "bf16", mode
        )
        if s is None:
            lines.append(f"| {mode} | — | — | — | — | — |")
        else:
            lines.append(
                f"| {mode} "
                f"| {_md(_fmt_rate(s.get('json_error_rate')))} "
                f"| {_md(_fmt_rate(s.get('hard_json_error_rate')))} "
                f"| {_md(_fmt_rate(s.get('format_style_error_rate')))} "
                f"| {_md(s.get('avg_latency_ms'))} "
                f"| {_md(s.get('p95_latency_ms'))} |"
            )


def _append_error_samples(lines: list[str], detail_paths: list[Path]) -> None:
    lines.extend(["", "## JSON Error Samples", ""])
    collected: list[dict[str, Any]] = []
    for path in detail_paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if not line:
                continue
            row = json.loads(line)
            if row.get("json_error_categories"):
                collected.append(row)
    if not collected:
        lines.append("No JSON/schema errors observed.")
        return
    lines.extend([
        "| idx | experiment_id | error categories | raw_output excerpt |",
        "| ---: | --- | --- | --- |",
    ])
    for row in collected[:20]:
        lines.append(
            f"| {_md(row.get('idx'))} | {_md(row.get('experiment_id'))} "
            f"| {_md(', '.join(row.get('json_error_categories', [])))} "
            f"| {_md(_excerpt(row.get('raw_output',''), 200))} |"
        )
    if len(collected) > 20:
        lines.append(f"| ... | ({len(collected) - 20} more error records) | | |")


def _append_latency_table(lines: list[str], summaries: list[dict[str, Any]]) -> None:
    lines.extend([
        "",
        "## Latency Records",
        "",
        "| experiment_id | avg_latency_ms | p95_latency_ms | sample_count |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in summaries:
        lines.append(
            f"| {_md(row['experiment_id'])} "
            f"| {_md(row.get('avg_latency_ms'))} "
            f"| {_md(row.get('p95_latency_ms'))} "
            f"| {_md(row['sample_count'])} |"
        )


def _append_failures(lines: list[str], failures: list[dict[str, Any]]) -> None:
    lines.extend(["", "## Run Failures", ""])
    if not failures:
        lines.append("No failures recorded.")
        return
    lines.extend([
        "| experiment_id | stage | error summary | next step |",
        "| --- | --- | --- | --- |",
    ])
    for f in failures:
        lines.append(
            f"| {_md(f.get('experiment_id',''))} | {_md(f.get('stage',''))} "
            f"| {_md(_excerpt(f.get('error_summary',''), 250))} "
            f"| {_md(f.get('next_step',''))} |"
        )


# ============================================================
# Report helpers
# ============================================================


def _find_summary(
    summaries: list[dict[str, Any]],
    group: str,
    model: str,
    quantization: str,
    thinking_mode: str,
) -> dict[str, Any] | None:
    for s in summaries:
        if (
            s.get("group") == group
            and s.get("model") == model
            and s.get("quantization") == quantization
            and s.get("thinking_mode") == thinking_mode
        ):
            return s
    return None


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def _md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")[:500]


def _excerpt(value: Any, limit: int = 160) -> str:
    text = "" if value is None else str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":
    raise SystemExit(baseline_runner())
