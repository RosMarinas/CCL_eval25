"""Unified model loading and vLLM server management.

Consolidates:
  - 4 separate model-loading implementations into load_model() and
    load_model_with_lora()
  - 2 copies of vLLM management code from run_baseline_matrix and
    run_baseline_smoke
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared vLLM constants
# ---------------------------------------------------------------------------

VLLM_PORT = 8000
API_URL = f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model(
    base_model: str,
    device: str = "auto",
    for_training: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model in 4-bit NF4 with double quant and bf16 compute dtype.

    Args:
        base_model: HuggingFace model ID.
        device: device_map value (\"auto\", \"cuda:0\", or 0-indexed).
        for_training: if True, calls prepare_model_for_kbit_training and
            disables cache for gradient checkpointing compatibility.
    """
    bnb_config = _build_bnb_config()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if for_training:
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
    else:
        model.eval()
        model.config.use_cache = True

    return model, tokenizer


def load_model_with_lora(
    base_model: str,
    checkpoint: str | None = None,
    device: str = "auto",
    for_training: bool = False,
) -> tuple[PeftModel | AutoModelForCausalLM, AutoTokenizer]:
    """Load model in 4-bit NF4, optionally applying a LoRA adapter.

    Args:
        base_model: HuggingFace model ID.
        checkpoint: path to LoRA adapter directory. If None, returns the
            base model directly (prompt-only mode).
        device: device_map value.
        for_training: passed through to load_model.
    """
    model, tokenizer = load_model(base_model, device=device,
                                   for_training=for_training)

    if checkpoint is not None:
        model = PeftModel.from_pretrained(model, checkpoint)
        if for_training:
            model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

# Attribute names attached to the vLLM subprocess object for log tracking.
_LOG_FILE_ATTR = "_vllm_log_file"
_LOG_PATH_ATTR = "_vllm_log_path"


def start_vllm(spec: dict[str, Any], log_dir: Path,
               cwd: Path | None = None) -> subprocess.Popen:
    """Launch a vLLM OpenAI-compatible API server.

    Args:
        spec: dict with keys \"model\", \"quantization\" (\"awq4\" or other).
        log_dir: directory for per-model server logs.
        cwd: working directory for the subprocess (defaults to CWD).
    """
    served_name = served_model_name(spec)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{served_name}.log"

    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", spec["model"],
        "--served-model-name", served_name,
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        "--dtype", "float16" if spec.get("quantization") == "awq4" else "bfloat16",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
        "--trust-remote-code",
    ]
    if spec.get("quantization") == "awq4":
        cmd.extend(["--quantization", "awq"])

    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    # Attach log handles for cleanup and error reporting
    setattr(proc, _LOG_FILE_ATTR, log_file)
    setattr(proc, _LOG_PATH_ATTR, log_path)
    return proc


def _proc_log_path(proc: subprocess.Popen) -> Path:
    return getattr(proc, _LOG_PATH_ATTR, Path("unknown"))


def wait_for_vllm(proc: subprocess.Popen, timeout_s: int) -> None:
    """Block until the vLLM server is ready or the process exits."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM exited early with code {proc.returncode}; "
                f"see {_proc_log_path(proc)}"
            )
        if is_vllm_ready():
            return
        time.sleep(5)
    raise TimeoutError(
        f"vLLM not ready within {timeout_s}s; "
        f"see {_proc_log_path(proc)}"
    )


def is_vllm_ready() -> bool:
    """Check whether the vLLM server responds to /v1/models."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{VLLM_PORT}/v1/models", timeout=2,
        ) as response:
            return response.status == 200
    except Exception:
        return False


def cleanup_vllm(
    proc: subprocess.Popen | None, spec: dict[str, Any],
) -> dict[str, Any]:
    """Terminate a vLLM server and return a post-check record."""
    record: dict[str, Any] = {
        "model": spec["model"],
        "pid": None,
        "terminated": False,
        "postcheck": None,
    }
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

    log_file = getattr(proc, _LOG_FILE_ATTR, None)
    if log_file is not None:
        log_file.close()
    time.sleep(3)
    record["postcheck"] = inspect_runtime()
    return record


def kill_orphan_vllm() -> None:
    """Kill any lingering vLLM processes from previous runs."""
    try:
        subprocess.run(
            ["pkill", "-f", "vllm.entrypoints"],
            timeout=30, check=False,
        )
        time.sleep(5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def served_model_name(spec: dict[str, Any]) -> str:
    return spec["model"].split("/")[-1].lower().replace(".", "-")


def run_text(cmd: list[str], check: bool = True) -> str:
    result = subprocess.run(cmd, check=check, capture_output=True, text=True)
    return result.stdout.strip()


def inspect_runtime() -> dict[str, Any]:
    return {
        "port_open": is_port_open("127.0.0.1", VLLM_PORT),
        "vllm_processes": run_text(
            ["pgrep", "-af", "vllm"], check=False,
        ).splitlines(),
    }
