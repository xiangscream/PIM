#!/usr/bin/env python3
"""Inspect one real typed-decisions case through Laya's inference internals.

This is an M0-A anatomy tool, not production code.
It intentionally accesses Laya internals so the learning trace is visible.
"""

import argparse
import json
import platform
import sys

import numpy as np
import torch
from datasets import load_dataset

import laya
from laya.common import (
    QTYPES,
    build_sequence,
    collate_items,
    confidence_from_probs,
    render_options,
    temp_bucket,
)


def softmax_np(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="convaiinnovations/laya")
    p.add_argument("--device", default="cuda")
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--question", default=None, help="question id; defaults to first gold-backed question")
    return p.parse_args()


def main():
    args = parse_args()

    print("=== ENVIRONMENT ===")
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_runtime:", torch.version.cuda)
        print("gpu:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print("vram_gib:", round(props.total_memory / (1024 ** 3), 2))
    print("laya:", getattr(laya, "__version__", "unknown"))

    print("\n=== LOAD MODEL ===")
    agent = laya.load(args.model, device=args.device)
    print("resolved_device:", agent.device)
    print("dtype:", agent.dtype)
    print("encoder:", agent.cfg.get("encoder"))
    print("head_layers:", agent.cfg.get("head_layers"))
    print("max_len:", agent.cfg.get("max_len"))
    print("head_max_len:", agent.cfg.get("head_max_len"))

    print("\n=== LOAD ONE REAL CASE ===")
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    row = ds[args.row]
    state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
    questions = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
    gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]

    if args.question:
        qid = args.question
    else:
        qid = next(q for q in questions if q in gold)

    qdef = questions[qid]
    internal = agent._to_internal(qdef)
    options = render_options(internal)

    print("row:", args.row)
    print("question_id:", qid)
    print("question_definition:")
    print(json.dumps(qdef, ensure_ascii=False, indent=2))
    print("gold:")
    print(json.dumps(gold.get(qid), ensure_ascii=False, indent=2))
    print("state:")
    print(json.dumps(state, ensure_ascii=False, indent=2)[:4000])

    print("\n=== SEQUENCE CONSTRUCTION ===")
    seq, markers = build_sequence(
        agent.tok,
        state,
        internal,
        agent.cfg.get("max_len", 512),
        agent.cfg.get("head_max_len", 192),
    )
    print("decision_type:", internal["t"])
    print("rendered_options:")
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    print("sequence_length:", len(seq))
    print("marker_positions:", markers)

    tokens = agent.tok.convert_ids_to_tokens(seq)
    print("marker_token_windows:")
    for i, m in enumerate(markers):
        lo = max(0, m - 3)
        hi = min(len(tokens), m + 10)
        print(f"  option[{i}] @ {m}: {tokens[lo:hi]}")

    item = {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES[internal["t"]],
    }
    batch = collate_items([[item]], agent.tok.pad_token_id)

    print("\n=== MODEL FORWARD ===")
    use_amp = agent.device.type == "cuda"
    with torch.no_grad():
        with torch.autocast(
            device_type=agent.device.type,
            dtype=agent.dtype,
            enabled=use_amp,
        ):
            logits_t, act_t = agent.model(
                batch["input_ids"].to(agent.device),
                batch["attention_mask"].to(agent.device),
                batch["marker_pos"].to(agent.device),
                batch["marker_mask"].to(agent.device),
                batch["qtype"].to(agent.device),
            )

    k = len(markers)
    logits = logits_t[0, :k].float().cpu().numpy()
    raw_p = softmax_np(logits)

    qt = QTYPES[internal["t"]]
    bucket = temp_bucket(qt, k)
    temp = float(agent.temperature_by_options.get(bucket, agent.temperature[qt]))
    cal_p = softmax_np(logits / max(1e-3, temp))

    print("logits_shape:", tuple(logits_t.shape))
    print("act_logits_shape:", tuple(act_t.shape))
    print("raw_logits:", logits.tolist())
    print("raw_probabilities:", raw_p.tolist())
    print("temperature_bucket:", bucket)
    print("temperature:", temp)
    print("calibrated_probabilities:", cal_p.tolist())
    print("entropy_confidence:", confidence_from_probs(cal_p, k))

    print("\n=== PUBLIC API CROSS-CHECK ===")
    public = agent.predict(state, {qid: qdef})
    print(json.dumps(public, ensure_ascii=False, indent=2))

    print("\n=== INTERPRETATION ===")
    winner = int(cal_p.argmax())
    print("winning_option_index:", winner)
    print("winning_option:", options[winner])
    print("No output tokens were generated; the decision came from option-marker logits.")


if __name__ == "__main__":
    main()
