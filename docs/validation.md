# Minimum useful validation ladder

## Principle

Validate changed interfaces with fixed consumed tokens before free generation. Reuse upstream semantics and existing compact fixtures; retain only a short result summary, source/profile identity and key measurements. A failure expands investigation only around the failing seam. An upstream README or a benchmark speed claim is not a correctness gate.

### 1. Small exact operator contract

Compare adopted EXLR8 operators against MLXLR8 `trellis.py` CPU/reference operators and existing small tests. Cover K4 side1/side2, mandatory E4M3 rounding, signs/scales and 128-channel transforms. Inspect one representative record per dense format actually used by the checkpoint. Compare dequantized small matrices and operator outputs with declared dtype/reduction tolerance; exact codebook/packing checks can be byte-exact.

Do not compare against ordinary EXL3 without EXLR8 rounding. Do not use current MLXLR8 optimized resident output as the exact-weight oracle: it may use JoinedMXFP4. Its `optimize=False` dense/expert paths are the integrated reference. Lossless expansion and changed reduction ordering require numerical comparison, not an assumption of bitwise logits.

### 2. GLM semantics with weights controlled

Use tiny deterministic compatible layers/configurations and identical decoded small weights to compare mlx-lm GLM and the adopted layer loop: normalized Q/KV, RoPE positions, indexer scores/selection, routing, layer outputs/logits. Check short causal context, first sparse transition near K=2048, and full-to-shared propagation. Preserve FP32 indexer weights/accumulation and configured epsilon.

For full-checkpoint checks, compare the new runtime to exact MLXLR8 with identical up1 weights and supplied token sequence. A separately quantized mlx-lm checkpoint cannot be an equality oracle for up1. If desired, load only a small layer's decoded EXLR8 matrices into a diagnostic mlx-lm layer; never instantiate a 744B BF16 model just for this comparison.

Use short teacher-forced logits/hidden checkpoints first. Then fixed-seed ordinary generation is a smoke test, not the sole oracle: minor reductions can change expert routing or sampling. Compare selected sets with awareness of numerical ties. Investigate material deviations at the first affected boundary rather than repeatedly reopening all GLM semantics.

### 3. Storage and lifecycle contract

Against a small floating contiguous reference, compare block append, absolute position lookup, causal lengths, gather order, sparse output and consumed-token frontier. Include page crossing, full/shared layers, last partial page, and lengths below/at/above K. Exercise prefill chunking versus one-shot short input, suffix continuation, capacity rejection before writes, EOS/length stop, cancellation, and close/recreate.

Confirm sealed blocks are not replaced during growth. Instrument physical allocations where possible; otherwise allocation peaks/copy traces must support the claim, and stable Python handles alone must not be reported as proof. In M1 compare output-only evaluation, all-state synchronous evaluation, and ds41f-style all-state/output `async_eval`, input/control `eval`, then stream synchronization before reuse. Repeat with private tails and intentionally retained lazy outputs/views, then drop retained references to check recovery. Correct values alone do not pass this experiment: identify the completion/release boundary that permits bounded backing/scratch reuse, and distinguish retained aliases from pending GPU work.

Use the same bounded storage experiment plus small exact-weight/native score/sparse-MLA smoke comparisons to select between candidate MLX 0.32.2 and 0.32.3 builds in M1; rebuild extensions for each candidate rather than mixing binaries. Record the chosen baseline and any disabled optional native routes in one short decision note, confirmed at M2 entry. This checks changed storage/native interfaces, not upstream model semantics.

Before M2 loading, validate the completed required up1 subset's manifest selection, file completeness/sizes/hashes and publisher-defined set identity as described in the development plan. Match loaded representations to that inventory at exit; the in-progress download is no dependency for M1.

Once q8 is introduced, compare exact same weights/tokens against floating pages: latent reconstruction error, layer/logit effects and selected attention output. Establish a short calibration tolerance before promotion; do not invent universal error thresholds now. Separate loss from quantization from errors in page/scales/bias association. A short quality match does not exempt q8 from a long-context continuation run.

### 4. Full-model growing-session soak

Run one declared representative long workload rather than a combinatorial matrix: chunked prefill, sustained decode across page growth, several exact-prefix agent turns, cancel/resume, rejection at capacity, close, and a second session. Record hardware/MLX/native build, weight profile, context, actual token frontiers and phase timings.

Sample low-overhead memory at load, prefill chunks, page growth, turn boundaries and teardown:

- computed live weight/page/scratch bytes and page counts;
- MLX active, cached and peak allocator memory (available APIs for the pinned build);
- process RSS, OS memory pressure and swap delta, not just MLX counters;
- time-to-progress and stream completion latency, especially around relocation-sized spikes.

Expected behavior is **weights + linear admitted page growth + bounded transient envelope**, not flat RSS while the context grows. After close and synchronization, active state should return near the model-only baseline; allocator-cached memory may remain up to the chosen pool policy, and RSS need not immediately fall. Repeat lifecycle cycles to distinguish a pool plateau from abandoned live allocations. Reserve an OS margin and stop before swap pressure, not after the system becomes unresponsive.

A slow scan over a longer indexer cache is not a deadlock. Use scalar progress markers for scheduling, layer/phase completion, state commit, and sampled-token publication. If a reference GLM-5.3 stall can be reproduced safely, use a separate bounded process and the same markers to identify the first non-completing phase. Do not hold the new runtime hostage to conclusively diagnosing every old stall.

**M4 pass:** measured run stays within admission/scratch bounds, sealed-prefix relocation count is zero, progress continues, OS pressure remains acceptable, frontiers remain coherent, and repeated teardown shows no accumulating live state. Record actual workload and supported limit; do not claim infinite sessions or max model context from one soak.

### 5. Local promotion checks

For a native or compressed path, rerun the corresponding operator comparison and the affected lifecycle/long-context segment. Assert native dtype/layout/symbol constraints before dispatch, including the sparse-MLA fast path's selected-row dimension `K >= 16` and multi-query requirement. Smaller selections use a bounded MLX fallback, not an unsupported native call. Keep a stable ordinary fallback; unsupported paths must not secretly allocate a full floating cache. MTP later gets a small separate acceptance/rejection/cancellation ladder because it changes transaction semantics.

## What is not continuously re-proven

- Official GLM BF16 equations, training intent, EXLR8 encoder quality, or every unrelated upstream model.
- All possible manifests, hardware generations, prompt formats, native tiles and precisions.
- Every ds41f proof/release certificate, source/module hash closure, or historical reference boundary.
- All model layers against an official implementation after a change confined to the page directory or CLI.
- Full-model long soaks after documentation-only edits or an unaffected pure helper change.

Retest according to the changed contract: codec, GLM wiring, cache storage/precision, native arithmetic, session lifecycle, or serving. Keep one small regression set for previously fixed concrete failures; do not grow a permanent qualification program around speculative risks.
