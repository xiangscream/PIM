# PIM — Photography Intuition Model

PIM is an experimental project for bounded, probabilistic photography decisions inside the Camera Intelligence Stack.

The long-term target is a model/runtime that maps visual context, device-independent photography state, task constraints, and short history into calibrated typed decisions such as `WAIT`, `REFRAME`, `CAPTURE`, or `ESCALATE`.

## Current phase

**M0 — Laya Reproduction**

Before training on photography data, reproduce and understand the Laya System-One decision workflow end to end:

`dataset -> typed decision encoding -> fine-tuning -> evaluation -> probability calibration -> inference`

Laya is the first experimental backend, not the definition of PIM.

## Related project

Tail2 provides the first real device/runtime environment from which a canonical photography state and replay dataset can later be built.

- Tail2: https://github.com/xiangscream/tail2
- Laya upstream: https://github.com/NandhaKishorM/laya
