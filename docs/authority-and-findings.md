# Authorities, provenance, and repository findings

## Inspection baseline

Source snapshots inspected (local working trees were clean):

| Repository | HEAD | Role |
|---|---|---|
| `../mlx-lm` | `bab20d9` | GLM-5.3 behavioral authority; includes earlier GLM-5.2 cache/native work |
| `../MLXLR8` | `09e4398` | EXLR8 codec, loader, operators, exact and approximate acceleration |
| `../ds41f-mlx` | `e0df90d` | Session ownership, explicit completion, native/lifetime lessons |

The `mlx-lm` HEAD change adds GLM-5.3 support atop existing DSA/cache machinery; its history does not establish that a newly invented cache alone caused the reported regression. No model execution or stall reproduction was performed here. `glm53-flash-mlx` was not needed: the selected GLM DSA code answers the immediate questions; Flash-specific machinery is not a semantics authority for this target.

The partially downloaded checkpoint already supplied `config.json`, `README.md`, and `LICENSE` under `/Volumes/KIOXIA-PRO-1/models/kindlingai/glm-5.3-exlr8-k2-k3-k4`. Only these metadata files were used; download completion was not required. Publisher metadata describes v0.5 and weight-set ID `2011c0cedde962d19fc43e74c1d409279e398d40d27a6fdb4bcb7b49ea2ea6fb`; verify the actual completed manifest/files before loading.

All source paths below are relative to the named reference repository, not proposed production modules.

## Authority/adaptation map

| Subsystem | Source files | Authority / limit | Recommended action and provenance |
|---|---|---|---|
| GLM configuration and layer semantics | mlx-lm `mlx_lm/models/glm_moe_dsa.py`: `ModelArgs`, attention, decoder/model; `deepseek_v32.py`; `mla.py`; `rope_utils.py` | Full/shared scheduling, MLA absorption/projection, routing, norm/RoPE behavior. Not an authority for our cache lifetime or default memory policy. | Adapt minimal layer/config math; retain Apple MIT notices and dependency provenance. Read inherited base methods when adapting, not just the GLM subclass. |
| Full/shared DSA | Same GLM file: `_indexer_topk`, `_update_indexer_k_cache`, model loop; DeepSeek `Indexer` | Full layers compute indices; shared layers reuse preceding full layer's indices. Shared does **not** mean shared MLA storage. FP32 head weights/score accumulation matter. | Preserve schedule, formula, causal/short-prefix behavior. Reimplement storage traversal only; compare this seam once. |
| EXLR8 physical format | MLXLR8 `exlr8/trellis.py`, `weights.py`, `experts.py`, `dense.py`; publisher manifest/model card | E4M3-rounded MCG codebook, tiles, signs/scales, transforms, seven dense formats, up1 selection. MLXLR8 is an executable reference, not permission to redefine the format. | Adapt reader/reference operators and required Metal sources. Apache-2.0 runtime; inherited exllamav3 MIT, PonyExl3 Apache notices apply. Preserve `ATTRIBUTIONS.md`, `NOTICE`, and relevant third-party texts when importing. |
| Compact integrated EXLR8 model | MLXLR8 `exlr8/model.py` | Useful packed-callable wiring that avoids random/full expert initialization. Not the final authority for all configurable GLM semantics or cache ownership. | Adapt wiring selectively; reconcile normalization/config precision against mlx-lm. For example MLXLR8 hardcodes attention/indexer eps 1e-6 while mlx-lm exposes `indexer_norm_eps`. Avoid a wholesale fork. Apple MIT + oMLX Apache lineage is acknowledged by MLXLR8. |
| MLA cache semantics | mlx-lm `mlx_lm/models/cache.py`: `GlmMlaKVCache`, `QuantizedGlmMlaKVCache`; GLM `_gather_cached_latent` | Correct latent-only affine int8 representation and selected-row dequantization; not a stable-storage design. | Own replacement; adapt quantization/gather contract, not growing backing arrays. MIT source notices if copied. |
| GLM native arithmetic | mlx-lm `mlx_lm/custom_kernels/glm_moe_dsa/{fast.py,csrc/*,LICENSE,README.md}` | Proven supported ABI/shapes and arithmetic, not generic paged-cache access. Vendored from oMLX `eca4a7c`. | Initially wrap unchanged score/sparse-attention operators behind checked adapters. Apache-2.0 plus retained MLX/Steel file notices; keep metallib resource lookup/build identity. |
| EXLR8 native arithmetic | MLXLR8 `exlr8/{trellis,dense_trellis,fast_decode,fast_prefill,kv_lookup,indexer_scores,vocab_projection}.py`, `metal/*` | Packed/expanded operators with explicit dispatch constraints. Not all optimized dispatch is exact. | Reuse exact packed trellis kernel first. Adopt other exact routes only after boundary comparison; keep upstream source/change notices. |
| Session/frontier/commit | ds41f `ds41f_mlx/runtime/{target_forward,continuation_session}.py`, `ds41f_mlx/prefill_fp8_mlx/arena.py` | Design lessons: single live owner, exact prefix extension, evaluate all mutated state, clear transient tensor handles. DeepSeek seven-slot state and 40-layer logic are **not** GLM semantics. | Reimplement a small GLM-specific state machine; do not copy model-specific transactions or admission identity machinery. `target_forward.py` has MIT/DeepSeek/oMLX notices. |
| Admission/native ownership | ds41f `runtime/resource_admission.py`, `serving/capacity.py`, `runtime/kv_persistence.py`; docs `milestone-8-long-session-architecture.md` | Explicit resource ownership, allocator-policy retirement, idle serialization boundary. Not a requirement for module hashing/certificates. | Reuse principles, not subsystem. No repository-root LICENSE was found in ds41f: inspect component permissions before copying any original code; use independent implementation where unclear. |
| Tokenizer/chat and model license | Completed checkpoint config/template/tokenizer/generation files and `LICENSE` | Exact token boundaries, EOS handling, model/quantization terms. Not MLX execution policy. | Load externally; do not vendor weights or captured user prompts. Use existing tokenizer dependencies with their licenses. |

## Established cache behavior

1. **Allocation/growth:** mlx-lm `KVCache.step=256` reserves a rounded append region and concatenates old storage with new zeros when capacity is exceeded. Slice assignment then writes the append. `GlmMlaKVCache` inherits this. Quantized GLM cache repeats growth for all three packed/scales/bias arrays and floating RoPE. Capacity grows in increments, not by a page table. `ChunkedKVCache` also concatenates: its name is not evidence of physical paging.
2. **Precision:** `QuantizedGlmMlaKVCache` supports only 8 bits, default group 64. Latent uses `mx.quantize`'s affine `(uint32 packed, scales, biases)` tuple; RoPE remains original floating dtype. DSA indexer uses separate ordinary `KVCache` with width 128 and zero-width values. Quantizing/changing group size on an existing cache can materialize an entire latent tensor and a second representation.
3. **Gather:** GLM `_gather_cached_latent` gathers each quantized tuple member first, then dequantizes selected rows. Sparse decode already avoids full float latent materialization. Sparse prefill has bounded Python paths and a native compact-selected path; unsupported/fallback routes can still dequantize full latent and create dense sparse masks or expanded K/V. These fallbacks are not acceptable at arbitrary context in the new runtime.
4. **Scheduling/laziness:** full indexers update keys even when context is shorter than top-k and indices are unused. `mx.depends` ties indexer state into latent state so evaluating the latter also advances the former. This is intentional graph control, not evidence of a bug by itself. Growing arrays, functional slice updates, unevaluated dependency chains, and retained aliases can nevertheless extend old-buffer lifetimes.
5. **Evaluation/release:** mlx-lm generation evaluates cache states between prefill chunks, then clears allocator cache. Ordinary decode pipelines lookahead and evaluates token/logprob dependencies, rather than using an explicit all-state commit transaction. A suspended generator can retain lookahead outputs and locals. `trim()` changes offsets, not physical capacity. Dropping Python cache owners does not free memory still referenced by graphs, stream work, aliases, or a prompt-cache container. `mx.clear_cache()` releases reusable allocator storage, not live referenced arrays.
6. **Independent confirmation:** MLXLR8 `fast_cache.py` explicitly evaluates pending outputs before replacing expert slots and warns unrelated lazy outputs can force buffer copies. ds41f `target_forward.execute` evaluates every mutated state and synchronizes before publishing a frontier; `arena.AllocationState` drops tensor handles after last use. These support bounded-lifetime design, not a diagnosis of the GLM-5.3 stall.

## Exact EXLR8 versus changed values

“Exact” here means preserving the publisher's decoded weights and intended typed handoffs, **not** bitwise identical logits under every reduction schedule.

| Route | Weight-value classification | Baseline policy |
|---|---|---|
| `trellis.mcg_codebook`, packed `trellis_linear` / `expert_linear` | Mandatory finite E4M3 round-to-nearest-even **before** side scales; this rounding is the format itself, not extra quantization. Signed vectors, 128-channel normalized Hadamards, side1/side2 layout must remain. | Reference/default. |
| `dense_trellis.expand_trellis`, `expanded_linear` | Lossless inner E4M3 expansion to MXFP8 with unit E8M0 exponents (127); preserves F32 side vectors/transforms. Changes storage size and potentially reduction rounding. | Optional exact acceleration with budget. |
| `static_cache` / `fast_cache` | Exact E4M3 expert expansion; not a change of source values. Adds residency and potentially alias/COW lifetime costs. | Defer expert expansion; not needed for stable baseline. |
| Format2 int4, format3 MXFP8, format4 block-FP8, format5 row-FP8, format6 BF16, format7 FP32 | These are **stored checkpoint formats**, not runtime substitutes. F32 scale calculations and prescribed output casts matter. | Preserve individually; do not replace with a common generic quantizer. |
| `kv_lookup` and dense K4 weight-load kernels | Representation/lookup changes intended to retain decoded BF16 projections and MLX arithmetic. Strong exactness candidates, constrained to tested records/shapes. | Compare locally before enabling. |
| `native_affine.NativeAffine` | Experimental implementation exercised by tests at `09e4398`, not wired into the production runtime path. Re-encodes inner K4 matrices to affine4, group128; changes weights. | Keep disabled. |
| `joined_mxfp4.JoinedMXFP4` | Folds gate/up input scales into inner matrices then re-quantizes to MXFP4. At `09e4398`, `ExpertLayer` installs it only when `store.resident` and `store.optimize` are true, `self.k == 4`, and `(self.hidden, self.intermediate) == (6144, 2048)`. Under those conditions it deletes packed weights and adjusts scales; resident mode alone does not enable it. Not merely a faster EXLR8 decoder. | Off; separate explicit approximate profile only if ever justified. |
| `dense_trellis.encode_mxfp4` / `shared_fp4_linear` | Re-quantizes expanded FP8 inner weights. `DenseLinear` can dispatch side2 output projection to this for four-row decode/verification. | Off, including later MTP unless separately qualified. |
| Fused norm/router, work ordering, half/split-K prefill | Not automatically a weight transformation; can change arithmetic/rounding or masking behavior. | Defer and validate only the route adopted. |

A global `optimize=True` is therefore not a meaningful exactness policy. Separate **storage encoding, arithmetic dispatch, and approximate transformations**. Starting MLXLR8-style `optimize=False` for both dense and experts is the simplest baseline; then enable individual known-exact routes.

## Provenance decisions

No production code was imported in this phase. Do not assign a blanket project license that purports to cover upstream components. Before copying code, record source revision/path, retain licenses and NOTICE/copyright headers, mark modifications, and include necessary license texts in distribution. Apache-2.0 is a reasonable proposed license for new runtime code; adoption can happen with the first source import.

The downloaded model license is **custom GLM-5.3**, not plain MIT: it includes notice obligations and a security-review condition for commercial use by qualifying Model-as-a-Service businesses exceeding aggregate $10B revenue over consecutive 12 months. Quantization additions are separately CC BY 4.0; that does not relax GLM conditions. Keep model/quantization attribution and license distinct from runtime code. Review applicability before third-party API distribution; this is not a legal opinion.
