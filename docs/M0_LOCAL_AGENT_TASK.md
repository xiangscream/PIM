# M0-A Task for Local OpenCode Agent

Role: execution and evidence collection.

Model/runtime: OpenCode local agent (DeepSeek V4.1 Flash).

## Objective

Complete **M0-A — Upstream Anatomy** only.

Do not start full fine-tuning yet.

## Fixed upstream

- repository: `https://github.com/NandhaKishorM/laya`
- commit: `42626c348753fbb17572a813127df2278a1ec527`
- package baseline: `0.3.4`

## Work

1. Clone/fetch Laya in a separate external working directory. Do not vendor the upstream source into PIM.
2. Checkout the exact commit above and record `git status`.
3. Record local environment:
   - OS
   - Python
   - GPU
   - VRAM
   - NVIDIA driver
   - CUDA
   - torch
   - transformers
   - laya
4. Read the implementation paths behind:
   - `build_sequence`
   - `render_options`
   - `build_model`
   - `proper_reward`
   - `Agent.predict`
   - calibration temperatures
5. Use exactly one `LocalLLaMA/typed-decisions` training case and trace one decision end to end.
6. Create a small inspect script in the PIM branch that prints, without secrets or raw private data:
   - decision type;
   - option labels;
   - token sequence length;
   - marker positions;
   - output tensor shapes;
   - raw logits;
   - calibrated probabilities;
   - predicted decision.
7. Write `docs/m0/LAYA_ANATOMY.md` in clean explanatory language.

## Required explanation

The report must answer in plain language:

- Why Laya can answer multiple typed alternatives without generating text.
- What the option markers represent.
- What differs between choice / score / noul.
- What the decision head consumes from the encoder.
- Where calibration is applied.
- What `proper_reward` is rewarding.
- Which parts are generic enough for PIM and which are Laya-specific.

## Discipline

- Do not modify upstream Laya.
- Do not change model architecture.
- Do not start Tail2 integration.
- Do not create photography training data yet.
- Do not claim a result was reproduced unless it actually ran.
- If the 421M model cannot be loaded locally, finish the static anatomy and report the exact blocker instead of inventing output.
- Keep model weights, caches and large datasets out of Git.
- Do not commit credentials or machine-private paths.

## Handoff

Return:

```text
PIM commit:
Laya upstream commit:
Environment:
Commands run:
What worked:
What failed:
Key findings:
Files changed:
Open questions:
```

Stop after M0-A and wait for review before M0-B.
