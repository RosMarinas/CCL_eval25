#!/usr/bin/env python3
"""Inspect B8 training results on remote server."""
import json
import os
from pathlib import Path

BASE = Path("/home/swh/AI/Experiment/CCL25_eval/checkpoints")

for ckpt_dir in sorted(BASE.iterdir()):
    print(f"\n{'='*60}")
    print(f"Directory: {ckpt_dir}")
    print(f"{'='*60}")
    for root, dirs, files in os.walk(ckpt_dir):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            print(f"  {fpath} ({size} bytes)")
            if f.endswith(".json") and "eval" in f.lower():
                with open(fpath) as fh:
                    data = json.load(fh)
                    print(f"    CONTENT: {json.dumps(data, indent=4)}")
            if f == "trainer_state.json":
                with open(fpath) as fh:
                    state = json.load(fh)
                    hist = state.get("log_history", [])
                    if hist:
                        last = hist[-1]
                        print(f"    last_log: step={last.get('step')}, loss={last.get('loss')}, lr={last.get('learning_rate')}")
                    print(f"    best_metric: {state.get('best_metric')}")
                    print(f"    best_model_checkpoint: {state.get('best_model_checkpoint')}")
                    print(f"    global_step: {state.get('global_step')}")
