# M2 — blocked at checkpoint admission

**M2 is incomplete. No checkpoint is admitted; M3 must not begin.**

The requested up1 baseline cannot safely run against the checkpoint currently
present. The download is still in progress. No partial-weight execution, runtime
compensation, approximate weight transformation, or full-model allocation was
attempted. The [M1 storage decision](m1-storage-decision.md) is unchanged.

## Observed inputs

Inventory at **2026-10-16 16:27 UTC**, root:
`/Volumes/KIOXIA-PRO-1/models/kindlingai/glm-5.3-exlr8-k2-k3-k4`.
Counts are a point-in-time observation, not an admission certificate; the external
`hf download` process is still writing its download cache and was left untouched.

| Required main up1 input | Present | Missing |
|---|---:|---:|
| Dense layers 000–077 + vocabulary | 79 | 0 |
| Expert K4 layers 003–077 | 2 | 73 |
| Expert index layers 003–077 | 69 | 6 |

The two complete K4 paths are layers **028 and 070**. Missing index layers are
**072–077**. The 153 present required files total **21,133,745,751 bytes**
(19.68 GiB); all present required safetensors have matching declared file extents.
That is a structural check, **not** a checksum or record-semantics check.

Required metadata missing:

- `manifests/exlr8-manifest-up1.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `hashes/` publisher size/hash metadata (directory absent)

`config.json`, `chat_template.jinja`, and `LICENSE` are present. Basic config
identity matches GLM-5.3 DSA: 78 layers, hidden6144, 256 routed + one shared expert,
top-8, latent512/RoPE64/indexer128. Tokenizer/template compatibility cannot be
validated without the tokenizer assets. Layer 078/MTP, alternate K2/K3 payloads,
and TP index slices are **not** required by the main up1 preflight.

67 `.incomplete` files occupied 42,518,769,791 logical bytes in the download cache
at observation time. These bytes are neither completed weights nor live model
residency and are excluded from the required-file inventory.

Expected set ID remains
`2011c0cedde962d19fc43e74c1d409279e398d40d27a6fdb4bcb7b49ea2ea6fb`.
**Verified set ID: unknown.** The manifest/hash metadata is absent. Per the local
publisher model card, the ID hashes sorted `<path> <size> <sha256>\n` lines for
all expert/dense weight files. When metadata arrives, reconstruct that ID from
the full published hash catalog, but hash/size-check only the required main up1
payload subset; do not demand downloads of alternate profiles or MTP.

## Implemented and tested

`tools/m2_preflight.py` is a small **CPU-only, non-admitting first stage**, not a
model loader. It checks main up1 path presence, basic target config identity,
bounded safetensors header/file extents, manifest's claimed set ID when present,
and publisher hash-metadata presence. It reports partial downloads separately.
No source was imported from a reference repository.

It deliberately **never** returns `admitted=true`: complete paths alone cannot
substitute for manifest selection, index/record interpretation, published payload
hash/size verification, and recomputed set identity. Those admission checks and
all runtime/operator integration remain unfinished. Exit 2 means missing/invalid
inputs; exit 3 means structural readiness still requiring verification.

```sh
.venv-0323/bin/python -m unittest discover -s tests -v
.venv-0323/bin/python tools/m2_preflight.py > /tmp/m2-preflight.json
# Current result: exit 2, BLOCKED_INCOMPLETE_OR_INVALID, admitted=false.
```

Four bounded regression tests passed, including truncated/oversized headers,
wrong identity, missing inputs, exclusion of optional profiles, and confirmation
that even structurally complete fixtures cannot grant admission.

Installed distribution metadata confirms MLX/Metal **0.32.3**, nanobind **3.1.0**.
M1's successful native arithmetic and exact trellis smoke results remain the
baseline; no version decision was reopened. No newly adopted GLM/EXLR8 execution
seam exists yet, so no new reference comparison or generation result is claimed.

## Memory and next gate

Model-only MLX active residency, load peak, short-generation peak, allocator cache,
process model RSS, and practical resident headroom are **unmeasured**. There is no
model to profile yet. Whole-system `memory_pressure -Q` reported 98% free;
`vm.swapusage` was 61.62 MiB used before and after preflight (no change at displayed
resolution). These idle/preflight observations do not establish model headroom.
No MLX memory limit was raised and no GPU allocation was needed for preflight.

Resume M2 after the downloader completes the required inputs. First implement
and pass full manifest/record/hash admission, then the exact packed-weight and
owned short GLM execution seams, reference comparisons, deterministic generation
plus continuation/close, and resident/load/generation memory gates. Do not build
an untested runtime around missing inputs or call this blocked attempt M2-complete.
No production paging, q8, MTP, serving, or M3 work was started.
