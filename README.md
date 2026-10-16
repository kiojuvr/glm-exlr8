# GLM-EXLR8

Planning a focused **GLM-5.3 + EXLR8 `up1` runtime for Apple Silicon**, initially the M3 Ultra with 512 GB unified memory.

The primary goal is reliable resident, growing agent sessions—not another general-purpose MLX-LM fork. The first major success is ordinary generation with stable cache ownership, no full-cache relocation, bounded scratch/graph lifetimes, and no progressive stall or abandoned-memory growth.

**Status: architecture and planning only. No runtime implementation exists.** The checkpoint download is independent of this phase.

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

The next implementation task is a **model-free fixed-block MLX storage/lifetime experiment**, as specified in the development plan. It must establish whether private evaluated MLX pages can be updated without replacing sealed storage, before choosing a native append boundary.
