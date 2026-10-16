# M1 — bounded storage/lifetime decision

## Decision and scope

**PASS for the requested model-free M1 storage gate. M2 may begin; M2 is not implemented here.**

Select **private evaluated MLX active tails**, with independently allocated immutable
sealed pages. No mandatory tail copy was observed under exclusive ownership.
Allow **bounded copy-on-write of the active page only** when a transaction really
must retain its old tail; release that reference at completion. Do not promise
in-place updates in the presence of aliases. **Defer a native storage boundary.**

Select **MLX 0.32.3 + mlx-metal 0.32.3**, with **nanobind 3.1.0** for rebuilt
extensions. Storage behavior was equivalent to 0.32.2. The selected build aligns
with MLXLR8's tested exact EXLR8 path, and the unchanged GLM native sources also
rebuilt and passed the bounded arithmetic smoke below. Do not reuse 0.32.2 binaries.

This implements only the user's bounded M1 experiment, not the broader historical
M1 operator-adoption work: no codec/runtime source was imported, no loader/model,
cache framework, serving, or MTP was implemented. Exact-operator adoption and the
normal checkpoint admission/residency gates remain work for M2, not claims made by
this experiment. The read-only upstream exact-kernel smoke is not adoption or a
full-model equivalence test.

## Required ownership/completion rule for M3

1. Keep pages private; never concatenate the history. Gather only selected rows
   into bounded compact outputs, consuming/retiring them within the transaction.
2. Evaluate **all changed latent, RoPE, and indexer state plus outputs**. Either
   `mx.eval(*state, output)` or `mx.async_eval(*state, output)` was bounded here.
3. Before publishing progress or reusing scratch, **`mx.synchronize(stream)`**;
   drop temporary/old-tail/output handles at this boundary. An independent control
   `mx.eval(control)` is not a substitute for all-state submission and completion.
4. Seal completed pages. Close drains known work, drops owners, and synchronizes.
   Abandoned *unevaluated* scratch/output graphs can simply be dropped; submitted
   work still requires completion. Do not execute abandoned graphs just to free them.

Use the conservative ds41f-style all-state async submission, control evaluation,
then stream synchronization initially. Ordinary eval alone kept values/backings
correct, but immediate post-eval counters sometimes included up to ~598 KB of
transients. Synchronization plus handle retirement gave the exact page budget.
This observation does not establish undocumented allocator callback timing.
Allocator-cache clearing is **not** a completion or ownership boundary.

## Workload and physical observation

Original experiment code is in `experiments/`; sibling repositories were read-only.
Machine: Apple M3 Ultra, 512 GiB unified memory; macOS Darwin 25.5.0 arm64;
Python 3.13.14. Both candidates use their matching Metal wheel.

Each independently allocated BF16 page has `[1,1,256,512]` latent,
`[1,1,256,64]` RoPE and `[1,1,256,128]` indexer components: **360,448 bytes**.
Each cycle appends 2,121 rows in 133 updates, seals eight pages, and leaves a
73-row tail (nine allocated pages). Five disciplines × four close/recreate cycles
per version = 160 seals, 2,660 append boundaries, 42,420 synthetic rows/version.
Only one default GPU stream and one transaction at a time are used per run.

Disciplines: ordinary lazy assignment + all-state `eval`; all-state `async_eval`
+ synchronization; ds41f-style all-state async + independent control eval +
synchronization; output-only eval control; and intentionally forced, transaction-
bounded tail COW. The output-only control does not evaluate indexer until seal or
final drain; its indexer backing-replacement field is `null`, not a false zero.
It is not an acceptable publication discipline despite eventual value correctness.

`probe.cpp` is a **read-only diagnostic**, separately built for each wheel. After
evaluation it reads `array.buffer().ptr()`, allocator size and array offset, never
`data()`, a NumPy/buffer export, or Python `id()`. It retains only integers.
Every subsequent append checks all sealed component handles/sizes/offsets. Thus
"stable" means the same live MLX/Metal allocator buffer, not fixed physical DRAM
addresses or a guarantee against OS/driver memory migration. No evaluated sealed
array is rewritten; the probe does not itself force evaluation or copy storage.

Values are checked separately against a NumPy position/column/component formula:
all written rows, zero unused capacity, and ordered selected-row gathers spanning
pages at each seal and the final partial tail. Gather assembles compact selected
rows only. Retained snapshots and materialized lazy outputs have independent
expected-value checks; the lazy-abandonment control intentionally never evaluates
its abandoned outputs. Assertions also bound memory and verify release/replacement
counts. All checks passed on both versions.

## Measurements

Raw scalar samples: `experiments/results-0.32.2.json` and
`experiments/results-0.32.3.json`. No tensors are retained by telemetry.

| Observation | 0.32.2 | 0.32.3 |
|---|---:|---:|
| Private-tail backing replacements (all-state variants) | 0 | 0 |
| Sealed component backing replacements (all variants) | 0 | 0 |
| Forced COW component replacements per cycle | 133 each | 133 each |
| Completed nine-page live/active bytes | 3,244,032 | 3,244,032 |
| Ordinary/all-state/control peak bytes | 3,939,456 | 3,939,456 |
| Bounded-COW peak bytes | 4,299,900 | 4,299,900 |
| Active bytes after each of 20 closes | 0 | 0 |
| Allocator cache after private/control closes | 4.12–4.29 MB | 4.12–4.29 MB |
| Allocator cache after COW closes | 4.64 MB | 4.64 MB |
| Current RSS range during page-cycle samples | 68.6–84.2 MB | 68.7–84.3 MB |

MB in this table is decimal. Peak above final live payload was ~0.66 MiB without
forced COW and ~1.01 MiB with COW. Completed growth followed page count; close
active-memory drift was **zero**. Allocator cache plateaued rather than accumulating
abandoned state. RSS rose during warmup; late COW-cycle RSS drift was only ~0.13 MB
per four-cycle series, and is not evidence of a live Metal leak. RSS is not a
complete inventory of Metal memory.

Four-cycle completion-call totals were ~149–166 ms per discipline on 0.32.3;
scalar progress continued through every boundary. Total experiment was ~2.72 s
per version. These tiny runs, with polling/host validation overhead, are not a
performance benchmark or proof of long-duration stability.

### Reproduced unwanted lifetime extension (identical on both builds)

Each adverse case retains 32 old latent tails, replacing their backing on all
32 updates. Each old latent allocation is 256 KiB; retained views are only one
row, but alias and pin the whole allocation. Aliasing is checked via the probe.

| After owner drop + synchronization + allocator-cache clear | Active bytes |
|---|---:|
| 32 retained evaluated views | 8,388,608 (8 MiB) |
| 32 retained lazy outputs, each depending on old tail + 1 MiB evaluated scratch | 41,943,040 (40 MiB) |

Neither synchronization nor `clear_cache()` fixes these reachable live allocations.
Evaluating retained lazy reductions released their input graphs/scratch, leaving
128 bytes of scalar results; keeping the views evaluated did **not** release their
backing. Dropping all retained handles and synchronizing returned active memory
to **zero**, both after evaluation and when abandoning lazy outputs without ever
evaluating them. Cache could then hold ~8 MiB or ~16–16.25 MiB (whole-allocation
allocator granularity), and clearing it returned cached bytes to zero. Adverse
peak was 42,729,472 bytes (~40.75 MiB), not unbounded allocation.

These bounded holds demonstrate linear lifetime extension if such references
were accumulated indefinitely; they are not a claim to have run an unbounded
lazy graph. Per-page evaluation and transaction retirement prevent that accumulation
in the selected design. A rolling old-tail alias control instead forced only
one active-page copy per update and stayed within the measured bounded envelope.

Safety: hard MLX memory limit 128 MiB, allocator cache target 16 MiB; fail on
observed peak >128 MiB or RSS growth >256 MiB. Positive cycles additionally assert
peak <8 MiB; adverse cases <64 MiB; closed active bytes must be zero. Allocations
are fixed/small; no model, long history, or machine-scale memory stress was used.
No claim about whole-machine swap activity from other processes is made.

## Bounded native/version check

The first 0.32.3 diagnostic build using nanobind 2.15.0 compiled but rejected
`mlx.core.array` arguments. Rebuilding with nanobind 3.1.0 fixed the type boundary;
0.32.2 used nanobind 2.15.0. This is a measured ABI difference, not presumed binary
compatibility. The tiny probe itself is a native-extension smoke for both builds.

Also rebuilt the unchanged GLM native CMake sources from `../mlx-lm` `bab20d9`
in this repository's ignored build directories, preserving upstream metallib
naming/lookup. `native_smoke.py` directly calls the rebuilt extension (no fallback):

- BF16 decode score: 32 heads ×128, one query, 4,096 keys, FP32 weights/output;
  max absolute error **2.98e-8** against independent NumPy arithmetic.
- BF16 sparse MLA: 64 heads, two queries, latent512/RoPE64, uint32 K=16, no causal
  mask; max absolute error **2.44e-4**, tolerance atol=2e-3/rtol=1e-2.

Both passed identically. The score guard rejects fewer than 4,096 keys (the
initial 64-key smoke correctly failed admission). Thus a 256-row page is **not**
a drop-in native score invocation. Native arithmetic is available, but page-aware
score dispatch remains disabled until a checked bounded adapter exists; use the
planned bounded MLX score fallback initially. Sparse MLA remains restricted to
its supported multi-query/K>=16 ABI; no one-query promotion is implied.

Read-only MLXLR8 `09e4398` existing
`TrellisMetalTests::test_dense_and_decode` also passed on both builds: six tiny
K2/K3/K4 side1/side2 subtests, including required K4, comparing decoded matrices
and packed linear outputs to its exact CPU reference. No approximate profile was
used. No upstream source/licenses were copied into this repository.

## Reproduction

From this repository; `uv`, CMake, and the local Xcode Metal toolchain are required.
Create isolated environments, for example:

```sh
uv venv --python python3.13 .venv-0322
uv pip install --python .venv-0322/bin/python mlx==0.32.2 mlx-metal==0.32.2 nanobind==2.15.0 numpy==2.5.4 pytest==9.1.1
uv venv --python python3.13 .venv-0323
uv pip install --python .venv-0323/bin/python mlx==0.32.3 mlx-metal==0.32.3 nanobind==3.1.0 numpy==2.5.4 pytest==9.1.1
for v in 0322 0323; do
  cmake -S experiments -B .build-$v -DPython_EXECUTABLE="$PWD/.venv-$v/bin/python" -DCMAKE_BUILD_TYPE=Release
  cmake --build .build-$v -j 4
  PYTHONPATH=.build-$v .venv-$v/bin/python experiments/storage_lifetime.py --output /tmp/m1-$v.json
  cmake -S ../mlx-lm/mlx_lm/custom_kernels/glm_moe_dsa/csrc -B .build-glm-$v \
    -DPython_EXECUTABLE="$PWD/.venv-$v/bin/python" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD/.build-glm-$v" -DBUILD_SHARED_LIBS=ON
  cmake --build .build-glm-$v -j 4
  PYTHONPATH=.build-glm-$v .venv-$v/bin/python experiments/native_smoke.py
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../MLXLR8 EXLR8_TEST_METAL=1 \
    .venv-$v/bin/python -m pytest -q -p no:cacheprovider \
    ../MLXLR8/tests/test_trellis.py::TrellisMetalTests::test_dense_and_decode
done
```

## Still unknown

Full up1 residency/loading headroom, weight/cache allocator interaction,
78-layer transaction peaks, real score/gather/attention scratch, long-duration
behavior under memory pressure, throughput, and compiled/multistream execution
remain unmeasured. This finite allocation-level observation is not a guarantee
for different graphs, dtypes, builds, escaped references, or full-model residency.
Confirm the pinned build/native paths at M2 entry and recheck these lifetime rules
at the actual M3 seams; no new native storage implementation is justified now.
