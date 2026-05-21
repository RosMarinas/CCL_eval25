from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.baseline import ModelConfig, PromptConfig, run_prompt_baseline
from src.eval import classify_json_errors, compute_json_error_rates


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "eval_data.json"
OUT_DIR = ROOT / "data" / "baseline"
DOC_PATH = ROOT / "docs" / "baseline-smoke-results.md"
DETAIL_PATH = OUT_DIR / "smoke-results.jsonl"
SUMMARY_PATH = OUT_DIR / "smoke-summary.json"
FAILURE_PATH = OUT_DIR / "smoke-failures.jsonl"
VLLM_PORT = 8000
API_URL = f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions"
LOG_DIR = OUT_DIR / "logs"

DECODE_PARAMS = {
    "temperature": 0,
    "top_p": 0.8,
    "max_tokens": 768,
}


EXPERIMENTS = [
    {
        "group": "P14",
        "model": "Qwen/Qwen3-14B",
        "parameter_scale": "14.8B",
        "quantization": "bf16",
        "backend": "vllm",
        "prompt_type": "zero-shot",
        "shot_count": 0,
        "sample_count": 1,
    },
    {
        "group": "P14-fast",
        "model": "Qwen/Qwen3-14B-AWQ",
        "parameter_scale": "14.8B",
        "quantization": "awq4",
        "backend": "vllm",
        "prompt_type": "zero-shot",
        "shot_count": 0,
        "sample_count": 1,
    },
    {
        "group": "P8",
        "model": "Qwen/Qwen3-8B",
        "parameter_scale": "8.2B",
        "quantization": "bf16",
        "backend": "vllm",
        "prompt_type": "zero-shot",
        "shot_count": 0,
        "sample_count": 2,
    },
    {
        "group": "P8",
        "model": "Qwen/Qwen3-8B-AWQ",
        "parameter_scale": "8.2B",
        "quantization": "awq4",
        "backend": "vllm",
        "prompt_type": "zero-shot",
        "shot_count": 0,
        "sample_count": 2,
    },
    {
        "group": "P8",
        "model": "internlm/internlm3-8b-instruct",
        "parameter_scale": "8B",
        "quantization": "bf16",
        "backend": "vllm",
        "prompt_type": "zero-shot",
        "shot_count": 0,
        "sample_count": 2,
        "thinking_mode": "normal",
    },
    {
        "group": "FMT",
        "model": "Qwen/Qwen3-8B",
        "parameter_scale": "8.2B",
        "quantization": "bf16",
        "backend": "vllm",
        "prompt_type": "fmt",
        "shot_count": 0,
        "sample_count": 2,
    },
    {
        "group": "FMT",
        "model": "google/gemma-4-E4B-it",
        "parameter_scale": "4.5B effective / 8B embeddings",
        "quantization": "bf16",
        "backend": "vllm",
        "prompt_type": "fmt",
        "shot_count": 0,
        "sample_count": 2,
        "thinking_mode": "direct-json",
    },
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cleanup_records: list[dict[str, Any]] = []

    for spec in EXPERIMENTS:
        before = inspect_runtime()
        if before["port_open"]:
            failure = failure_record(spec, "load", "port 8000 already in use before startup")
            failure["precheck"] = before
            failures.append(failure)
            summaries.append(summary_for_failure(spec, failure["error_summary"]))
            continue

        proc = None
        try:
            proc = start_vllm(spec)
            wait_for_server(proc, timeout_s=int(os.environ.get("SMOKE_VLLM_LOAD_TIMEOUT", "3600")))
            run_details = run_experiment(spec, tasks)
            details.extend(run_details)
            summaries.append(make_summary(spec, run_details, "ok", ""))
        except Exception as exc:  # noqa: BLE001 - smoke must record environment failures.
            stage = "load" if proc is not None and not is_server_ready() else "generate"
            failures.append(failure_record(spec, stage, str(exc)))
            summaries.append(summary_for_failure(spec, str(exc)))
        finally:
            cleanup_records.append(cleanup_vllm(proc, spec))

    write_jsonl(DETAIL_PATH, details)
    write_jsonl(FAILURE_PATH, failures)
    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "decode_params": DECODE_PARAMS,
                "summaries": summaries,
                "failures": failures,
                "cleanup": cleanup_records,
                "detail_path": str(DETAIL_PATH.relative_to(ROOT)),
                "failure_path": str(FAILURE_PATH.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    DOC_PATH.write_text(render_report(summaries, details, failures, cleanup_records), encoding="utf-8")
    return 0


def inspect_runtime() -> dict[str, Any]:
    return {
        "port_open": is_port_open("127.0.0.1", VLLM_PORT),
        "vllm_processes": run_text(["pgrep", "-af", "vllm"], check=False).splitlines(),
    }


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def start_vllm(spec: dict[str, Any]) -> subprocess.Popen:
    served_name = served_model_name(spec)
    log_path = LOG_DIR / f"{served_name}.log"
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        spec["model"],
        "--served-model-name",
        served_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "float16" if spec["quantization"] == "awq4" else "bfloat16",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.85",
        "--trust-remote-code",
    ]
    if spec["quantization"] == "awq4":
        cmd.extend(["--quantization", "awq"])
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    proc._smoke_log_file = log_file  # type: ignore[attr-defined]
    proc._smoke_log_path = log_path  # type: ignore[attr-defined]
    return proc


def wait_for_server(proc: subprocess.Popen, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}; see {proc._smoke_log_path}")  # type: ignore[attr-defined]
        if is_server_ready():
            return
        time.sleep(5)
    raise TimeoutError(f"vLLM server did not become ready within {timeout_s}s; see {proc._smoke_log_path}: {last_error}")  # type: ignore[attr-defined]


def is_server_ready() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/v1/models", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def run_experiment(spec: dict[str, Any], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_config = ModelConfig(
        group=spec["group"],
        model_name=spec["model"],
        parameter_scale=spec["parameter_scale"],
        quantization=spec["quantization"],
        backend=spec["backend"],
        thinking_mode=spec.get("thinking_mode", "non-thinking"),
    )
    prompt_config = PromptConfig(
        prompt_type=spec["prompt_type"],
        shot_count=spec["shot_count"],
        decoding_params={**DECODE_PARAMS},
    )
    input_items = make_formatter_inputs(tasks) if spec["group"] == "FMT" else tasks[: spec["sample_count"]]

    def generate(prompt: str, metadata: dict[str, Any]) -> str:
        return call_chat_completion(prompt, spec, metadata)

    results = run_prompt_baseline(input_items, model_config, prompt_config, generate)
    rows = []
    for result, item in zip(results, input_items):
        task = item["task"] if spec["group"] == "FMT" else item
        categories = classify_json_errors(result.raw_output, task, result.parsed_json)
        row = result.to_record()
        row["json_error_categories"] = categories
        row["json_valid"] = not categories
        row["metadata"]["fmt_case"] = item.get("fmt_case") if isinstance(item, dict) else None
        rows.append(row)
    return rows


def call_chat_completion(prompt: str, spec: dict[str, Any], metadata: dict[str, Any]) -> str:
    served_name = served_model_name(spec)
    messages = [{"role": "user", "content": prompt}]
    if spec["model"].startswith("Qwen/Qwen3"):
        messages[0]["content"] = f"{prompt}\n/no_think"
        metadata["nothink_handling"] = "sent chat_template_kwargs enable_thinking=false; appended /no_think fallback"
    payload = {
        "model": served_name,
        "messages": messages,
        **DECODE_PARAMS,
    }
    if spec["model"].startswith("Qwen/Qwen3"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=int(os.environ.get("SMOKE_GENERATE_TIMEOUT", "240"))) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {text[:1000]}") from exc
    return body["choices"][0]["message"]["content"]


def make_formatter_inputs(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_task = tasks[0]
    error_task = tasks[1]
    return [
        {
            "fmt_case": "fmt-clean",
            "task": clean_task,
            "reasoner_output": {
                "idx": clean_task["idx"],
                "evidence": {"words": {}, "sentences": {}, "emotion": []},
                "draft_answer": {
                    "ans_qa_words": {word: f"{word}在诗中的意思" for word in clean_task["qa_words"]},
                    "ans_qa_sents": {sent: "该句的现代汉语翻译" for sent in clean_task["qa_sents"]},
                    "choose_id": next(iter(clean_task["choose"])),
                },
            },
            "validator_report": {
                "valid_json": True,
                "missing_fields": [],
                "missing_words": [],
                "missing_sentences": [],
                "invalid_choose_id": False,
                "overlong_fields": [],
                "suspected_conflicts": [],
            },
        },
        {
            "fmt_case": "fmt-format-error",
            "task": error_task,
            "reasoner_output": {
                "idx": error_task["idx"],
                "evidence": {"words": {}, "sentences": {}, "emotion": []},
                "draft_answer": {
                    "ans_qa_words": {error_task["qa_words"][0]: "偷看，窥探"},
                    "choose_id": "Z",
                    "analysis": "extra field should be removed",
                },
            },
            "validator_report": {
                "valid_json": True,
                "missing_fields": ["ans_qa_sents"],
                "missing_words": error_task["qa_words"][1:],
                "missing_sentences": error_task["qa_sents"],
                "invalid_choose_id": True,
                "overlong_fields": [],
                "suspected_conflicts": [],
            },
        },
    ]


def make_summary(spec: dict[str, Any], rows: list[dict[str, Any]], status: str, notes: str) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in rows]
    categories = [row["json_error_categories"] for row in rows]
    rates = compute_json_error_rates(categories)
    return {
        "experiment_id": rows[0]["experiment_id"] if rows else expected_experiment_id(spec),
        "group": spec["group"],
        "model": spec["model"],
        "backend": spec["backend"],
        "quantization": spec["quantization"],
        "prompt_type": spec["prompt_type"],
        "shot_count": spec["shot_count"],
        "sample_count": len(rows),
        "json_error_rate": rates["json_error_rate"],
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p95_latency_ms": p95_latency_ms(latencies),
        "status": status,
        "notes": notes,
    }


def summary_for_failure(spec: dict[str, Any], notes: str) -> dict[str, Any]:
    return {
        "experiment_id": expected_experiment_id(spec),
        "group": spec["group"],
        "model": spec["model"],
        "backend": spec["backend"],
        "quantization": spec["quantization"],
        "prompt_type": spec["prompt_type"],
        "shot_count": spec["shot_count"],
        "sample_count": 0,
        "json_error_rate": None,
        "avg_latency_ms": None,
        "p95_latency_ms": None,
        "status": "failed",
        "notes": notes[:300],
    }


def p95_latency_ms(latencies: list[float]) -> float | None:
    if not latencies:
        return None
    index = max(0, math.ceil(0.95 * len(latencies)) - 1)
    return round(sorted(latencies)[index], 2)


def expected_experiment_id(spec: dict[str, Any]) -> str:
    model = spec["model"].split("/")[-1].lower().replace(".", "-")
    return f"{spec['group']}-{model}-{spec['quantization']}-vllm-nothink-{spec['prompt_type']}"


def failure_record(spec: dict[str, Any], stage: str, error: str) -> dict[str, Any]:
    return {
        "model": spec["model"],
        "experiment_id": expected_experiment_id(spec),
        "stage": stage,
        "error_summary": error[:2000],
        "next_step": next_step_for(error),
    }


def next_step_for(error: str) -> str:
    lowered = error.lower()
    if "timeout" in lowered or "did not become ready" in lowered:
        return "Increase load timeout or pre-download weights, then retry the same smoke."
    if "not found" in lowered or "gated" in lowered or "401" in lowered or "403" in lowered:
        return "Verify Hugging Face model ID, license gate, and remote credentials."
    if "out of memory" in lowered or "cuda" in lowered:
        return "Retry AWQ or lower max model length / GPU memory utilization."
    if "port 8000" in lowered:
        return "Stop or move the existing service before retrying."
    return "Inspect the per-model vLLM log under data/baseline/logs/."


def cleanup_vllm(proc: subprocess.Popen | None, spec: dict[str, Any]) -> dict[str, Any]:
    record = {"model": spec["model"], "pid": None, "terminated": False, "postcheck": None}
    if proc is None:
        record["postcheck"] = inspect_runtime()
        return record
    record["pid"] = proc.pid
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        record["terminated"] = True
    log_file = getattr(proc, "_smoke_log_file", None)
    if log_file is not None:
        log_file.close()
    time.sleep(3)
    record["postcheck"] = inspect_runtime()
    return record


def served_model_name(spec: dict[str, Any]) -> str:
    return spec["model"].split("/")[-1].lower().replace(".", "-")


def run_text(cmd: list[str], check: bool = True) -> str:
    result = subprocess.run(cmd, check=check, capture_output=True, text=True)
    return result.stdout.strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def render_report(
    summaries: list[dict[str, Any]],
    details: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    cleanup_records: list[dict[str, Any]],
) -> str:
    lines = [
        "# Baseline Smoke Results",
        "",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        "",
        "Scope: 1-2 sample smoke for prompt baseline and formatter links. No LLM judge scores were computed.",
        "",
        "Decode params: `temperature=0`, `top_p=0.8`, `max_tokens=768`. Qwen3 requests sent `chat_template_kwargs.enable_thinking=false` and appended `/no_think` as a fallback.",
        "",
        "## Result Table",
        "",
        "| experiment_id | group | model | backend | quantization | prompt_type | shot_count | sample_count | json_error_rate | avg_latency_ms | p95_latency_ms | status | notes |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summaries:
        lines.append(
            "| {experiment_id} | {group} | {model} | {backend} | {quantization} | {prompt_type} | {shot_count} | {sample_count} | {json_error_rate} | {avg_latency_ms} | {p95_latency_ms} | {status} | {notes} |".format(
                **{key: md_cell(value) for key, value in row.items()}
            )
        )

    lines.extend(["", "## JSON Error Samples", ""])
    error_rows = [row for row in details if row["json_error_categories"]]
    if not error_rows:
        lines.append("No JSON/schema errors were observed in completed runs.")
    else:
        lines.extend([
            "| idx | experiment_id | raw_output excerpt | error categories | validation_error |",
            "| --- | --- | --- | --- | --- |",
        ])
        for row in error_rows[:10]:
            lines.append(
                f"| {md_cell(row['idx'])} | {md_cell(row['experiment_id'])} | {md_cell(excerpt(row['raw_output']))} | {md_cell(', '.join(row['json_error_categories']))} | {md_cell(row.get('validation_error', ''))} |"
            )

    lines.extend([
        "",
        "## Latency Records",
        "",
        f"Per-sample details: `{DETAIL_PATH.relative_to(ROOT)}`.",
        "",
        "| experiment_id | avg_latency_ms | p95_latency_ms | sample_count |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in summaries:
        lines.append(
            f"| {md_cell(row['experiment_id'])} | {md_cell(row['avg_latency_ms'])} | {md_cell(row['p95_latency_ms'])} | {md_cell(row['sample_count'])} |"
        )

    lines.extend(["", "## Run Failures", ""])
    if not failures:
        lines.append("No model startup or generation failures were recorded.")
    else:
        lines.extend([
            "| model | stage | error summary | next step |",
            "| --- | --- | --- | --- |",
        ])
        for failure in failures:
            lines.append(
                f"| {md_cell(failure['model'])} | {md_cell(failure['stage'])} | {md_cell(excerpt(failure['error_summary'], 220))} | {md_cell(failure['next_step'])} |"
            )

    cleaned = all(record.get("terminated") or record.get("pid") is None for record in cleanup_records)
    lines.extend([
        "",
        "## vLLM Cleanup",
        "",
        f"This run cleaned up all vLLM processes it started: `{cleaned}`.",
        "",
        f"Machine-readable summary: `{SUMMARY_PATH.relative_to(ROOT)}`.",
        f"Failure details: `{FAILURE_PATH.relative_to(ROOT)}`.",
        "",
    ])
    return "\n".join(lines)


def md_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")[:500]


def excerpt(value: Any, limit: int = 160) -> str:
    text = "" if value is None else str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
