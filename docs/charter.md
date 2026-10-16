# Project charter

## Purpose

Build a local runtime in which GLM-5.3 EXLR8 `up1` remains resident and supports long, growing agent sessions predictably. The motivating report is that the current `kiojuvr/mlx-lm` GLM-5.3 path can stop when KV caching is introduced and may retain memory afterward, while GLM-5.2 is substantially more stable. This investigation found plausible allocation/lifetime mechanisms, **not a reproduced root cause**.

The first major success criterion is generation across growing context and repeated turns without full-cache relocation, runaway temporary allocations, swap pressure, stall/deadlock, or abandoned live-state growth. Legitimate KV growth is expected and explicitly budgeted.

## Scope and hardware

- One M3 Ultra, 512 GB unified memory; MLX with narrowly selected Metal acceleration.
- GLM-5.3 backbone, 78 layers; EXLR8 `up1` main experts K4, with dense/vocabulary formats honored individually.
- Single session and one in-flight state transaction initially; exact token-prefix continuation.
- Runtime-owned fixed-block MLA latent, floating RoPE channel, and independent full-indexer caches.
- Optional latent-only int8 after the floating-cache baseline; DSA selection before dequantization.
- Ordinary sampling and a small local generation interface before any server.

## First usable release

A local command-line/library runtime that loads the exact `up1` profile, generates ordinary text, appends turns without replay, handles cancel/close, reports capacity and memory, and survives a declared long-context soak. The initial supported context cap is measured, not inferred from the model's 1,048,576-position configuration. Aim first at 128 Ki tokens if actual admission permits; do not advertise it before measurement. Preserve the checkpoint tokenizer, template, EOS IDs, and generation settings rather than inventing a chat dialect.

OpenAI-compatible local serving is a later delivery milestone. MTP is not required for this first usable core.

## Non-goals

Generic model/manifest support, training/conversion, distributed execution, tensor parallelism, multimodal support, multi-tenant batching, automatic disk offload, transparent sliding-window eviction, broad prompt-cache persistence, speculative trees, benchmark leadership, or a release-certification framework.

No continuous re-derivation of GLM math against an official BF16 implementation is planned. Working GLM and EXLR8 implementations already establish the major components; validation concentrates on interfaces we change.

## Why a new runtime

The new boundary is **execution-state ownership**, not a new model architecture. Retrofitting another general MLX-LM fork would inherit a growing set of generation, batching, prompt-cache, environment-switch, and lazy-state conventions. A small owned layer loop and session/cache API can keep the proven equations and packed operators while giving cache growth, commit, cancellation, and memory admission one authority.

Reuse MLX and tokenizer tooling as dependencies. Adapt a narrow set of GLM/EXLR8 code with attribution. Neither MLXLR8's optimized resident defaults nor ds41f's qualification machinery should become the runtime's implicit contract.
