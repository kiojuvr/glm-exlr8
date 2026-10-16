# Recommended architecture

## Decision

Own a small Python/MLX execution core, with checked Metal operator boundaries. Adapt GLM equations and exact EXLR8 operators; do not fork an entire reference runtime. Start **one resident model, one session, one in-flight transaction, floating MLA pages**. Add latent-only int8 as an independent representation choice later.

Use fixed-capacity, independently allocated token blocks with a host page directory. Sealed blocks never change. An exclusive active tail accepts appends; growth adds blocks rather than replacing the prefix. This is logical paging, not a requirement for a virtual-memory subsystem or a GPU pointer-table attention kernel on day one.

A monolithic preallocated maximum-context slab is rejected as the default: it reserves too much capacity upfront and an MLX copy-on-write update can duplicate an enormous array. A rotating cache is rejected because silently dropping historical rows changes DSA semantics. Geometrically growing contiguous caches reduce frequency but retain full-relocation spikes. A fixed-block design limits any tail-copy fallback to one block, and can later use native page-table addressing without changing session semantics.

**Important:** MLX slice assignment is not proof of in-place physical mutation. The first storage experiment must verify private-tail update behavior, aliases, completion, and allocator reuse. If storage donation is unreliable, a bounded copy of the active page is initially acceptable; sealed pages must never relocate. If even that fails the memory/lifetime budget, introduce a minimal owned native append/gather boundary before performance work. Do not mutate externally owned Metal buffers behind MLX's dependency model.

## Components and data flow

```text
checkpoint metadata + up1 manifest + exact file records
  -> admission/inventory -> resident packed weight owner
  -> tokenizer/template -> token suffix
  -> session transaction (reserved capacity, committed frontier)
  -> embedding
  -> 78 GLM layers:
       norm -> Q and KV projections -> normalized latent + RoPE
       -> append private MLA/RoPE tail blocks
       full indexer: append indexer keys -> causal scores/top-k over blocks
       shared indexer: reuse most recent full layer indices (not its KV)
       -> map absolute positions to block/row
       -> gather selected latent (+ scales/bias if q8) and floating RoPE
       -> dequantize selected latent only -> absorbed MLA -> output projection
       -> residual/norm -> routed top-8 + shared expert or dense MLP
  -> final norm/vocabulary projection -> sampler
  -> evaluate ALL changed state + output -> completion barrier
  -> publish frontier and token; release step scratch
```

The owned layer executor carries top-k indices only within the current forward call until replaced by the next full indexer. No cross-turn hidden selection authority. Exact dtype, scales, normalization, routing correction, RoPE interleaving, and early causal rows come from the GLM reference and checkpoint records, not a rewritten generic attention abstraction.

### Ownership

| Lifetime | Owner | Contents / release rule |
|---|---|---|
| Process/model | Resident model | Packed experts, dense/vocab records, signed scales, any explicitly approved decoded projections, kernel handles. Sequential load/evaluate/release per layer; no full BF16 model construction. |
| Session | Session state | Token history, committed consumed-token frontier, page directories, committed pages, configuration/weight identity, capacity reservation. Model contains no session caches. |
| Transaction | Executor | Active-tail write ownership, append reservation, pending frontier, current layer activations/DSA indices. Complete or invalidate before another transaction. |
| Layer/chunk | Bounded scratch owner | Score tiles, top-k merges, selected tuples, dequantized selected rows, attention buffers. No unbounded lists of lazy chunk outputs. |
| GPU in-flight | Execution stream | Explicit completion/fence determines when scratch or pages may be reused. Python deletion is not a fence. |

Begin with one stream and conservative completion per decode transaction/prefill chunk; additional layer/score-tile barriers keep memory bounded where needed. Evaluate all changed MLA, RoPE, and indexer state even when logits do not depend on it. Remove temporary tensor handles after evaluation. Logging retains scalars/IDs, not tensors, closures over activations, or exception tracebacks indefinitely.

At transaction start, check input/prefix and capacity before any mutation. All layers target the same new frontier. Publish only after state/output completion. GPU failure after partial mutation invalidates the session; no silent replay or fictitious rollback. Validation/admission rejection before writes leaves the old state valid. Cancellation is observed at completed transaction/chunk boundaries. An unrecoverable GPU stall may require process termination; Python cannot promise to cancel a hung Metal command safely.

Track **consumed tokens versus sampled-but-not-yet-consumed token** explicitly. Ordinary generation often has a one-token difference. Turn boundaries, EOS, output length, and cancellation must record which token is actually cached; do not claim cache length equals visible history by convention. Initially avoid speculative lookahead. Close drains known in-flight work, drops all page/scratch owners, and leaves only model residency plus an explicitly bounded allocator reserve. Trim adjusts valid rows and releases unused tail pages after completion; it is not just an offset change.

## Cache layout and algorithms

### Layout

Proposed starting block length **256 tokens** (a measurement parameter, not a semantic constant). Allocate blocks lazily within an admitted session ceiling; do not eagerly create thousands of empty page tensors at maximum context.

- Each of 78 layers: latent `[1,1,P,512]`, RoPE `[1,1,P,64]` in BF16 initially.
- Only the 21 full-indexer layers in the observed configuration: independent keys `[1,1,P,128]`, BF16; no meaningful values tensor.
- Optional latent q8: packed uint32 width128, scales width8, biases width8, group64, with BF16 metadata if matching the baseline. This is MLX affine int8, not an invented signed symmetric-byte format.
- Host metadata: absolute block start, valid row count, representation, owner/reference count, pending/completed state. Never confuse a valid-row prefix with allocated capacity.

Common token ranges across layers simplify transaction accounting; allocations remain independent so a single alias cannot pin a model-wide slab. A page directory owns references, not concatenated tensor views. Scratch pools have a fixed byte ceiling and cannot silently grow with context.

### Append and decode

Reserve pages for all affected layers before forward. Append only into exclusively owned tails, including indexer state on full layers even below top-k length. Seal full blocks. On decode, compute DSA scores in block groups, retain local candidates, and merge exact global top-k with absolute row IDs. Score accumulation and head weights remain FP32. Evaluate/retire each group so looping over pages does not merely create a giant lazy graph. Use a fixed-size block group for efficiency, not a full-indexer-cache concatenation.

Top-k over block candidates is mathematically sufficient: no discarded candidate can beat a block's own K better candidates. Preserve causal masking and reference tie handling where practical; document/evaluate numerical ties rather than requiring a new global proof. The indexer still scans history: sparse attention is not sublinear indexer work.

Map selected absolute IDs to page/row. Group gathers per touched page, write results into bounded compact slots, and restore selection order. No broadcast of the entire cache across query/head axes. Under q8, gather packed rows **and matching scales/biases** before dequantization. Gather RoPE separately. The attention adapter sees compact floating latent/RoPE and compact remapped indices; it never requires full floating history.

When available causal rows are fewer than K=2048, attend to the entire valid prefix with explicit valid lengths. This bounded early prefix can be compacted directly. Do not give duplicated/padded top-k slots attention mass. An unsupported path at long context must fail clearly or use a budgeted block/selected-row fallback, never densify the full history.

### Long prefill

Chunk input by admitted scratch budget, not a fixed maximum-context mask. Within a chunk, tile queries and keys for DSA score/top-k; use absolute causal positions, including the transition across the first K rows. Complete groups as they are consumed. Query-by-history dense masks and per-head expanded MLA K/V are prohibited at large context.

Compact sparse scratch is roughly `Q_tile * K * (latent + RoPE) * dtype_size`, before scores/output. At BF16 and Q_tile=16 this alone is 36 MiB; q8 tuple gathers, head intermediates, and output projection require additional room. Start with small measured tiles and an explicit total scratch cap (candidate 256 MiB per operator, not a guarantee that the entire transaction fits in 256 MiB).

### Continuation and reuse

Keep the live pages across turns. The newly encoded prompt must exactly extend the canonical consumed/history state under the tokenizer/template contract, accounting for any pending token; append only the suffix. Reject divergent input rather than silently replaying or discarding cache. Keep tokenizer/template/weight/cache identity in session metadata.

First support same-session reuse only. Later, bounded immutable full-page prefix sharing can use reference counts; a shared partial tail must be copied once before append. Prefix pages are released only after all owners and in-flight work finish. No prompt-cache deep copies, automatic longest-suffix matching, or persistence in the first release.

### Optional compression and later MTP

Choose floating versus q8 at session creation. Quantize each incoming latent chunk directly; never convert a long live cache wholesale at a token threshold. RoPE/indexer stay floating. Compare both cache types using identical weights and consumed tokens; q8 is lossy and its effect on hidden states can change later routing/DSA selection.

MTP later needs private proposal/verification tails, accepted-prefix commit, discard of rejected pages/rows after completion, and explicit predictor/indexer ownership. Honor checkpoint `index_share_for_mtp_iteration` only in its established GLM semantics. Do not simply share mutable target tail storage or add `trim()` to ordinary decode and assume transactions are solved.

## Native boundaries

- **Already required and reusable:** MLXLR8 exact packed trellis Metal decode/GEMV, with K/side/layout checks; these are existing weight execution, not a new speed project.
- **Unchanged native candidates:** versioned FP32 DSA score operators on contiguous indexer block tiles; decode score symbol supports one query and 32 heads ×128 dimensions. Apply absolute page-aware causality externally when local offsets do not fit the ABI. General score output is query-by-key, so tile explicitly.
- **Sparse MLA:** mlx-lm `glm_dsa_sparse_mla_attention` supports floating matching dtypes FP16/BF16, 64 query heads, singleton latent head, latent512/RoPE64, uint32 selections, last-axis contiguous layout. The inspected native fast path is multi-query (`L>1`), not ordinary one-token decode. Compact gathered rows plus local indices/valid lengths can reuse it for prefill; apply original causality before setting compact `causal=False`. Ordinary decode can use MLX SDPA over its selected compact rows.
- **Not drop-in:** current sparse MLA accepts arrays, not page tables or q8 latent pages. Its C++/Metal arithmetic can remain unchanged behind compact adapters, but zero-copy page-native access needs later kernel work. q4/q8 V-up and affine-MoE kernels expect MLX affine weight layouts; EXLR8 K4 bitstreams are not that ABI.
- Pin a compatible MLX/native build together. mlx-lm native documentation specifies MLX 0.32.2 and nanobind 2.15.0; MLXLR8 also contains 0.32.3-derived arithmetic. Do not presume binary compatibility or combine arbitrary installed wheels. Preserve `omlx_glm_kernels.metallib` lookup unless deliberately changing loader and packaging together.

## Capacity model

Observed config: 78 layers, 21 full indexers, latent512, RoPE64, indexer128. Packed main expert payload alone is approximately **338.82 GiB** (`75*257*3*6144*2048*4/8`). This excludes side vectors, dense/vocab, decoded KV projections, allocator cache, scratch, and OS. Publisher total up1 is approximately 378 **GB** including MTP; do not confuse decimal GB with GiB or equate disk bytes with live residency.

Raw cache payload estimates (one session, no page slack/scratch):

| Context | Floating cache | Latent q8 + floating RoPE/indexer |
|---|---:|---:|
| 32 Ki | 2.91 GiB | 1.76 GiB |
| 128 Ki | 11.63 GiB | 7.05 GiB |
| 256 Ki | 23.25 GiB | 14.11 GiB |
| 1 Mi | 93.00 GiB | 56.44 GiB |

Floating bytes/token = `78*(512+64)*2 + 21*128*2 = 95,232`. Q8 estimate includes two BF16 metadata values per group: `78*(512+8*2*2+64*2) + 21*128*2 = 57,792`.

Inference: a resident packed floating-cache baseline at useful context is plausible on 512 GB; max configured context is **not** yet admitted. Full expert FP8 expansion would roughly double the 338.82 GiB payload and is not viable as a default. Inventory actual decoded/deduplicated arrays, per-layer load peak, process/Metal memory, OS reserve and swap before selecting a cap. Admission must reserve weights + pages to cap + scratch + loading/allocator margin + OS headroom. Never use a high MLX memory limit as proof of available physical capacity.
