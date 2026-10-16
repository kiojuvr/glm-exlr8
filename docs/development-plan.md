# Development plan

Each milestone has one outcome gate and a small comparison/measurement record. No mandatory evidence bundles, release certificates, ever-growing test matrices, or upstream proof campaign. Keep a runnable ordinary baseline throughout. Do not wait for the full download to perform model-free work.

| Milestone | Objective | Exit condition | Dependencies | Principal risk |
|---|---|---|---|---|
| **1. Freeze interfaces; resolve storage feasibility** | Record supported checkpoint/config contracts; adopt small exact EXLR8 reference operators with notices; experiment with MLX page ownership. | K4 side1/side2 + representative dense format outputs match the existing reference; a model-free append/eval/gather/close run demonstrates sealed-page stability and bounded tail/scratch lifetime, or identifies the minimal native storage requirement. | Current source snapshots; synthetic packed fixtures only. | Confusing array identity with physical storage; accidentally importing approximate defaults. |
| **2. Exact resident up1, ordinary short generation** | Sequential packed load without random/full expert matrices; owned simple GLM forward loop and sampler. MTP off. | Completed up1 metadata/files admitted; exact-operator reference and adopted path agree on fixed short tokens, including first sparse selection; ordinary text works. Resident/load peak inventory leaves declared page/scratch/OS reserve. | M1; completed required main weight files/tokenizer/manifest. | Weight/load headroom, typed handoffs, format/layout mismatch. |
| **3. Integrate owned fixed-block cache and transactions** | Replace temporary short-reference cache with block MLA/RoPE/indexer storage; bounded score and selected-row paths; cancellation/close. | Same-weight floating contiguous-vs-block comparisons pass below/above K and across a page boundary; growing decode/prefill append never relocates sealed prefix pages; all-state completion/frontiers and release are observable. | M1 storage decision, M2 ordinary generation. | Lazy update chains, hidden dense fallback, selected gather correctness. |
| **4. Growing-session stability — first major success / usable core** | Repeated suffix prefill, decode, turn boundaries, cancellation, capacity rejection and teardown at measured long context. | A declared long run (initial target 128 Ki tokens if admitted) with multiple turns and a page-growing decode segment shows expected linear cache growth, bounded scratch/graph history, no new swap pressure, stall or abandoned-memory drift. Cancel/resume and close/recreate preserve correct frontiers and return live state to its budget. | M3; resident hardware run. | Slow index scan versus actual hang, allocator retention, non-extension prompts. |
| **5. Optional q8 cache and measured native performance** | Qualify latent-only int8 separately; enable proven exact native arithmetic where profiling justifies it. | Floating remains a usable fallback. Q8 passes selected-row/error comparisons and repeats the relevant long-session run. Each promoted native route passes its boundary comparison and retains memory/lifecycle properties. | M4; native build pinned if used. | Quantized trajectory drift; compact native ABI; speed regressions hiding memory costs. |
| **6. MTP, optional** | Add predictor and verification state without compromising ordinary sessions. | Forced accept/reject/partial accept, EOS/cancel, and continuation use explicit private tails and commit; target ordinary baseline still works; adopted paths do not silently re-quantize weights. | M4; stable cache contracts, useful M5 paths. | Multiplied pending states, reference iteration-index sharing, hidden four-row MXFP4 dispatch. |
| **7. OpenAI-compatible local serving and packaging** | Expose the stable runtime with minimal request admission, streaming and shutdown semantics. | Local client can generate/cancel/continue within declared limits; disconnect does not orphan state; licenses/resources included. | M4 core; MTP remains optional, not a server dependency. | Lookahead/disconnect ownership, broad compatibility distracting from core. |

M2 may use the existing contiguous cache **only as a short diagnostic scaffold**, with a strict context limit and explicit deletion path. It cannot graduate to the usable runtime. If M1 requires native storage work, do it as part of M3, not as an excuse to adopt an unsafe growing cache. Existing exact weight kernels can be reused from M1; “native performance later” does not mean reimplementing the codec in slow Python.

M5 contains independent workstreams: q8 is not a prerequisite for exact native speedups, nor vice versa. M7 may follow M5 while M6 remains deferred; there is no feature-count gate.

## First implementation task after this phase

**Build a small, model-free MLX fixed-block append/lifetime experiment, not a production runtime package.**

1. Use 256-row synthetic latent/RoPE/indexer blocks, one stream, and a small admitted pool.
2. Append through many boundaries, gather selected rows across blocks, explicitly evaluate all writes/outputs, and release scratch every step.
3. Compare private-tail updates with retained-view/pending-output variants. Check values, evaluated memory peaks and post-close memory after synchronization; use allocation instrumentation for backing identity if available. Python `id()` alone is insufficient.
4. Distinguish active/live bytes, allocator cache, RSS and peak; run close/recreate repeatedly.
5. Decide between private evaluated MLX tails, bounded tail-page copying, and a minimal native storage boundary. Record the decision in a short note and stop; do not build a general cache framework first.

This is a bounded answer to the highest-risk interface. Exact operator adaptation and normal model loading follow it within M1/M2. It needs no model download.

## Explicit deferrals

Whole-expert FP8 expansion/LRU caches, affine4 or MXFP4 replacements, scale folding/re-quantization, indexer compression, RoPE quantization, dynamic cache precision conversion, page-native sparse fusion, aggressive `mx.compile` capture of mutable sessions, multistream overlap, token lookahead, broad prompt-prefix sharing/persistence, MTP, speculative trees, general server/batching, and benchmark/release formalism.

Small exact acceleration may be promoted later only with a concrete profile and stable memory. No approximate weight profile is needed to call the core useful.

## Dependencies and blockers

**No blocker to planning or M1.** The checkpoint directory was partial at inspection: metadata and some dense content were available, but main expert/manifest/tokenizer availability was not established. M2 full-model runs require the relevant download, not all alternate K2/K3 files.

Two later decisions need measurement rather than more architecture prose: physical MLX page-update/reuse behavior, and actual resident/loading/scratch headroom on the target machine. Native extension compatibility is also unverified; build/smoke it only when selecting that route. No sibling repository should be edited to resolve these issues.
