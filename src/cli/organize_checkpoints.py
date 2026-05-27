#!/usr/bin/env python3
"""Organize checkpoint directories with clean versioned names.

Does NOT delete anything — only renames and creates a VERSIONS.md tracker.
"""
import os
import json
import shutil
from pathlib import Path
from datetime import datetime

CKPT = Path("/home/swh/AI/Experiment/CCL25_eval/checkpoints")

# ── Mapping: current_name -> new_name ──────────────────────────
# Empty dirs get a .empty suffix for clarity
RENAME_MAP = {
    # B8 series
    "B8-old": "_archive/B8-v0-failed-1step-lr0",
    "B8": "B8-v1-lr2e5-ep2-10steps",
    "B8-v2": "_archive/B8-v2-empty",

    # BC8 series
    "BC8": "_archive/BC8-v0-first-attempt",
    "BC8-v2": "_archive/BC8-v2-empty",
    "BC8-v3": "BC8-v1-lr5e5-ep1-10steps",  # BEST: 2% error

    # BC8-final replay series
    "BC8-final-v1": "_archive/BC8-final-v0-overfit-lr2e5-10steps",
    # BC8-final: current replay, keep name for now (eval in progress)
}

# ── Build version history ───────────────────────────────────────
VERSIONS = []

def scan_checkpoint(name: str, path: Path) -> dict:
    """Extract metadata from a checkpoint directory."""
    info = {"name": name, "path": str(path.relative_to(CKPT))}

    # Check for eval results (avoid /dev in filename via glob)
    for f in sorted(path.rglob("*_eval*.json")):
        try:
            data = json.loads(f.read_text())
            info["eval"] = {
                "file": str(f.relative_to(path)),
                "error_rate": data.get("json_error_rate"),
                "error_count": data.get("error_count"),
                "total": data.get("total"),
            }
        except Exception:
            pass

    # Check trainer_state.json for training metrics
    for f in sorted(path.rglob("trainer_state.json")):
        try:
            state = json.loads(f.read_text())
            hist = state.get("log_history", [])
            if hist:
                last = hist[-1]
                info["training"] = {
                    "global_step": state.get("global_step"),
                    "final_loss": last.get("loss"),
                    "final_lr": last.get("learning_rate"),
                }
        except Exception:
            pass

    # Check for adapter
    adapter = path / "adapter" / "adapter_model.safetensors"
    if adapter.exists():
        info["adapter"] = {
            "size_mb": round(adapter.stat().st_size / (1024 * 1024), 1)
        }

    return info


def main():
    CKPT.mkdir(parents=True, exist_ok=True)
    archive = CKPT / "_archive"
    archive.mkdir(parents=True, exist_ok=True)

    # Ensure archive subdirs exist
    for new_name in RENAME_MAP.values():
        new_path = CKPT / new_name
        if "/" in new_name:
            new_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Execute renames ─────────────────────────────────────────
    for old_name, new_name in RENAME_MAP.items():
        old_path = CKPT / old_name
        new_path = CKPT / new_name

        if not old_path.exists():
            print(f"SKIP (not found): {old_name}")
            continue

        if new_path.exists():
            print(f"SKIP (target exists): {old_name} -> {new_name}")
            continue

        print(f"RENAME: {old_name} -> {new_name}")
        shutil.move(str(old_path), str(new_path))

    # ── Scan all checkpoints for VERSIONS.md ────────────────────
    print("\n=== Scanning all checkpoints ===\n")
    for entry in sorted(CKPT.iterdir()):
        if entry.name.startswith("_") or not entry.is_dir():
            continue
        info = scan_checkpoint(entry.name, entry)
        VERSIONS.append(info)
        print(f"  {entry.name}")
        if "eval" in info:
            e = info["eval"]
            print(f"    Eval: {e['error_rate']:.0%} error ({e['error_count']}/{e['total']})")
        if "training" in info:
            t = info["training"]
            print(f"    Training: step={t['global_step']}, loss={t['final_loss']:.4f}, lr={t['final_lr']:.2e}")
        if "adapter" in info:
            print(f"    Adapter: {info['adapter']['size_mb']} MB")

    # Also scan archive
    for entry in sorted(archive.iterdir()):
        if not entry.is_dir():
            continue
        info = scan_checkpoint(entry.name, entry)
        VERSIONS.append(info)

    # ── Write VERSIONS.md ───────────────────────────────────────
    lines = [
        "# Checkpoint Versions",
        f"\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Active Checkpoints\n",
    ]

    for v in VERSIONS:
        name = v["name"]
        if "_archive" in v.get("path", ""):
            continue
        lines.append(f"### {name}")
        if "training" in v:
            t = v["training"]
            lines.append(f"- **Training**: {t['global_step']} steps, final loss = {t['final_loss']:.4f}")
        if "eval" in v:
            e = v["eval"]
            lines.append(f"- **Eval**: {e['error_rate']:.1%} JSON error ({e['error_count']}/{e['total']})")
        if "adapter" in v:
            lines.append(f"- **Adapter**: {v['adapter']['size_mb']} MB")
        lines.append("")

    lines.append("## Archive\n")
    for v in VERSIONS:
        if "_archive" not in v.get("path", ""):
            continue
        lines.append(f"- **{v['name']}**")
        if "eval" in v:
            lines.append(f"  - Eval: {v['eval']['error_rate']:.1%} error")
        if "training" in v:
            lines.append(f"  - Training: {v['training']['global_step']} steps")
        lines.append("")

    readme = CKPT / "VERSIONS.md"
    readme.write_text("\n".join(lines))
    print(f"\nVERSIONS.md written to {readme}")


if __name__ == "__main__":
    main()
