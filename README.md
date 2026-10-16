# GLM-EXLR8

Building toward a focused **GLM-5.3 + EXLR8 `up1` runtime for Apple Silicon**, initially the M3 Ultra with 512 GB unified memory.

The primary goal is reliable resident, growing agent sessions—not another general-purpose MLX-LM fork. The first major success is ordinary generation with stable cache ownership, no full-cache relocation, bounded scratch/graph lifetimes, and no progressive stall or abandoned-memory growth.

**Status: M1 complete; M2 blocked at checkpoint admission. No runtime implementation exists.** The checkpoint download is incomplete; see the [M2 inventory/blocker](docs/m2-status.md) and CPU-only preflight in `tools/m2_preflight.py`. The [M1 storage decision](docs/m1-storage-decision.md) and MLX 0.32.3 baseline are unchanged. M3 must not begin.

## Initial design

- Adopt GLM-5.3 behavior from the existing `mlx-lm` GLM DSA implementation.
- Adapt MLXLR8's exact EXLR8 operators; do **not** silently adopt its resident MXFP4 re-quantization.
- Own session transactions, admission, fixed-block MLA/RoPE/indexer storage, selected-row gathering, and release.
- Begin single-session, single-flight, floating cache; qualify optional latent-only int8 separately.
- Reuse existing Metal arithmetic where compatible; defer speculative decoding and HTTP serving.

## Documents

- [Charter](docs/charter.md)
- [Architecture and cache design](docs/architecture.md)
- [Repository findings, authorities, and provenance](docs/authority-and-findings.md)
- [Development milestones](docs/development-plan.md)
- [Bounded validation strategy](docs/validation.md)
- [Risk register](docs/risks.md)

The model-free experiment lives in `experiments/`, separate from future production code. It found stable sealed buffers, bounded active-tail COW when aliases are retained, and release after reference retirement plus completion. A native storage boundary is deferred. Subsequent model work still requires the M2 admission gates in the development plan.
