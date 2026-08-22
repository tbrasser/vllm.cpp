# ROCm gfx1151 (Strix Halo) — M5 gap analysis

**Row:** `BACKEND-ROCM` (backend-matrix, `ACTIVE`).
**Claim:** unclaimed — this is an analysis record, not a landed change. It
orders the work between the current gfx1151 position and **M5 MET** and names
the evidence each step owes.
**Issue:** [#41](https://github.com/mudler/vllm.cpp/issues/41) (the ROCm
bring-up issue; Strix Halo is the gfx1151 board class).
**Board:** Strix Halo / GTR9 Pro 128 GB — `gfx1151`, RDNA 3.5 APU, **unified
memory** (integrated=1, managed=1, concurrent_managed=1, pageable=0 — the F6
triple, measured on this board class). 128 GB LPDDR5X shared with the CPU.
**Milestone definition (verbatim, docs/ROCM.md §5):** "**M5 — speed.**
`vllm bench throughput` on the same box, quant-matched, against the same
model. The bar is vLLM, not llama.cpp. Method and honesty rules: verification
procedure and docs/BENCHMARKS.md."

---

## 1. The ordering principle

M5 is the last milestone in a strict chain: **M0 → M1 → M2 → M3 → M4 → M5**.
Each stage is a prerequisite for the next on this board, and the chain is
*not* skippable on gfx1151 the way it is on a discrete board, because the
reference tier (§2, G2) only installs where `UnifiedMemory() == true`. The
whole analysis is therefore: *where does gfx1151 sit in that chain today, and
what does each remaining hop owe?*

## 2. Current gfx1151 position (2026-08-19)

| Stage | State on gfx1151 | Evidence / anchor |
|---|---|---|
| M0 — build | **MET** | #41 tables; flag-free configure now works on Arch layouts (F1/F3 absorption, `CMakeLists.txt:325,443` — `gfx1151` in the default arch list) |
| M1 — platform + backend | **MET** | `ctest -R 'rocm\|cross_device'` green on the board; RmsNorm ≤ 5e-4 vs CPU oracle (#41) |
| W1 approach-(b) F6 fix | **VERIFIED on gfx1151 (2026-08-19)** — probe triple `1/1/true`, coupling + no-copy cases green, 9/9 cases / 1071 assertions | §3: `test_rocm_backend` on gtr9, evidence log `f6-verify-20260819-085638.log` |
| M2 — first model e2e | **MET (2026-08-19), all-native** — Qwen3-0.6B runs end to end with **zero reference-tier fallbacks**; 2/4 prompts token-identical to CPU, 2 in the documented near-tie regime | §3: `m2-verify-20260819-085745.log`; the G3 kernel to-do for this model is empty |
| M3 — kernels | **PARTIAL** | 44 ops registered (`src/vt/rocm/`, counted 2026-08-19). Attention half LANDED: `ROCM_ATTN` registration (#1056) + runner per-group selection/shape validation (#1065). GDN kernel families landed on the discrete lane (`CLAIM-ROCM-GDN-KERNELS`, 10 ops). d=128 decode arm LANDED default OFF (`VT_ATTN_DECODE_D128`, #767) |
| M4 — correctness vs vLLM-ROCm oracle | **MET (2026-08-19)** — pinned vLLM-ROCm oracle builds + runs on the APU; the 16-prompt near-tie gate PASSES 16/16 (13/16 strict token-exact vs per-prompt greedy, 3/16 in the 0-nat near-tie band, 0 forward-divergent, 0 declines). Fresh gfx1151 oracle capture is fully deterministic over K=10 | §3 M4 subsection; `m4-gate-20260819-102238.log` + `gap.log` on the node; matches the gfx1200/gfx1100 gate standard |
| M5 — speed | **PARTIAL (2026-08-21)** — evidence chain §M5-1…M5-5: paired bench, context sweep (exact linear 20 ms + 0.042 ms×L), granularity probe (A/B null), **D128 arm 2.7–3.7× (near-tie-safe, 0.125 nats max)**, binding-grid ours leg (c1 43.5, c4 914 total). Oracle leg in flight | §M5-1…M5-5; jobs `bench-*` + `kv-pattern` + `d128-classify` on gtr9 |

**Local datapoint (this workspace, 2026-08-19):** `build-hip-gfx1151/` is a
configured Release tree with `ROCM_PATH=/usr` (an Arch/TheRock-style layout),
`VLLM_CPP_HIP_ARCHITECTURES=gfx1151`, and the absorbed
`--rocm-path=/usr` / `CMAKE_HIP_COMPILER_ROCM_ROOT=/usr` flags already in the
cache — i.e. the F1/F3 absorption works with **no manual flags**, which is
exactly the configure evidence the F6 record lists as owed. The build itself
has not completed (log stops ~7%, no `libvllm` produced). A finished build +
`ctest -R 'rocm|cross_device'` on this tree would be the first F6-verification
datapoint.

## 3. Verified on the box — 2026-08-19 (gtr9, Strix Halo 128 GB)

**Environment:** node `gtr9` in the k3s cluster (`10.9.8.211`, Hadron Linux),
`amd.com/gpu: 1` via the rocm/k8s-device-plugin. Pod image
`ubernet.local/vllm-cpp-hip:gfx1151`; ROCm **7.2.3** at `/opt/rocm-7.2.3`;
hipClang/Clang 22.0.0. Source on the node is a non-git sync of this repo's
`row/ROCM-ATTN-REVALIDATE` taken 2026-08-18 20:37 (two hours before HEAD
`de95c2e5`, a test-only commit; the F6 code has not changed since 2026-08-08).
Build: `cmake --build build-hip -j32` completed 100% (`build2.log`),
`VLLM_CPP_HIP_ARCHITECTURES=gfx1151`, `-DROCM_PATH=/opt/rocm-7.2.3`.
Binaries: `/home/kairos/src/build-hip/{tests,examples,libvllm.so.0.0.3}`.
Evidence logs on the node: `f6-verify-20260819-085638.log`,
`m2-verify-20260819-085745.log`.

### G1 — F6 (approach (b)) VERIFIED on gfx1151

Decisive output of `test_rocm_backend` on the node (GPU = `AMD RYZEN AI MAX+
395 w/ Radeon 8060S`, gfx1151):

```
MESSAGE: ROCm device 0 integrated: true managed-alloc: true UnifiedMemory(): true
MESSAGE: reference-tier ops installed for kROCM: 61
```

- **The probe triple is `1/1/true`** — the exact expected reading for the
  integrated managed-capable branch. The managed branch has now executed on
  real silicon for the first time (it is provably dead on discrete boards,
  which is why the fix shipped UNVERIFIED).
- **The approach-(b) coupling case passes** (alloc path and `UnifiedMemory()`
  move together) and the F6 decisive experiment (kernel writes → host reads
  back, no copy) is part of the same green run.
- `test_rocm_backend`: **9/9 cases, 1071 assertions, 0 failed, exit 0**.
- `test_backend_cross_device`: **21/21, 360 assertions** (RmsNorm vs CPU
  oracle, paged attention at Qwen3 geometry, all device backends).
- `test_attn_backend_registry`: **17/17, 62 assertions** (the ROCM_ATTN
  registration + priority mirror).
- This was a flag-free configure on a Hadron/TheRock-style layout with only
  `-DROCM_PATH` supplied — the F1/F3 absorption datapoint the F6 record
  wanted is now in.

**HEAD re-run (2026-08-19): caveat discharged.** Upstream main was fetched
(`3b76ccbf`, 2026-08-19) — 23 commits ahead of our base, **none touching ROCm
backend code** — the row branch (`row/ROCM-ATTN-REVALIDATE`) was rebased onto
it (`127246d2` = upstream main + the block-size test commit), the node source
was re-synced via `git archive` + `kubectl cp`, and a clean rebuild
(`rebuild-head` job, 2m33s, 19 gfx1151 entries in compile_commands.json)
produced fresh binaries. Re-running the whole battery at the new fingerprint
reproduced everything: F6 triple `1/1/true`, `test_rocm_backend` 9/9 (1071),
cross-device 21/21 (360), attn registry 17/17 (62), **`test_runner` 20/20
(544) — the new block-size contract case green** (evidence
`f6-verify-20260819-090921.log`), and the M2 op table + outputs identical
(`m2-verify-20260819-091200.log`). The pre-rebase evidence build is preserved
on the node at `build-hip-evidence-20260819`.

### G2 — M2 e2e MET, stronger than the milestone's minimum

`vllm-cli --model /models/Qwen3-0.6B --max-tokens 8 --temperature 0`,
`VT_OP_PROVIDER_STATS=1`, both devices, on the node:

| Prompt | ROCm (device=auto) | CPU | Match |
|---|---|---|---|
| "The capital of France is" | ` Paris. The capital of Italy is Rome` | ` Paris. The capital of France is also` | near-tie (the recorded 0.0000-nat France/Italy flip) |
| "The sky is" | ` blue, the ground is green, and` | ` blue, the ground is green, and` | **identical** |
| "Water boils at a temperature of" | ` 100°C at standard atmospheric` | ` 100°C. What is` | near-tie regime (classify via M4's teacher-forced gap) |
| "Roses are red, violets are" | ` blue, and the rest are yellow.` | ` blue, and the rest are yellow.` | **identical** |

**The decisive part is the op table, not the tokens.** Every op the model
uses printed `selected=vt-native priority=0 registered=1` on device 5 —
**ZERO reference-tier fallbacks** on ROCm. The full op set, by name:
`kEmbedding`(4), `kRopeCosSinCache`(68), `kCastBf16`(56), `kRmsNorm`(1),
`kMatmulBT`(74), `kQkvSplit`(63), `kRopeFromCache`(78), `kReshapeAndCache`(25),
`kPagedAttention`(31), `kSiluAndMul`(2), `kGreedyArgmax`(33). That is the
entire Qwen3-dense forward on native ROCm kernels — which is exactly the
`GetReferenceTierHits() == 0` precondition M5 (§3, G3) imposes on any
benchmark of this model. The G3 kernel-to-do list for Qwen3-0.6B is **empty**.

**Token parity reads exactly like the gfx1200 investigation predicted**
(`rocm-gfx1200-m2-correctness.md`): 2/4 token-identical, and the two
divergences are Qwen3-0.6B's documented near-tie regime — the France/Italy
flip is a literal 0.0000-nat tie in the oracle's own logits, and the
"Water boils" divergence needs M4's teacher-forced gap measurement to
classify rather than a token-exact bar. Not a defect; the near-tie gate is
the correct instrument and it is an M4 item.

### What this changes in the chain

- **G1: MET** (evidence above). The F6 row's PENDING-community status is
  discharged on this board and should be posted to #41 with the log.
- **G2: MET** on gfx1151 — Qwen3-0.6B runs end to end, all-native, and the
  M2 fallback list came back EMPTY, which also closes the G3 kernel bar for
  this model.
- **Next in the chain: M4 on gfx1151** — the vLLM-ROCm oracle attempt
  (containerized pinned vLLM, `PYTORCH_ROCM_ARCH=gfx1151`), then the first
  paired bench. The M2 finding makes the path shorter than the analysis
  assumed: the model that would carry the first gfx1151 speed number already
  runs all-native.

### M4 — the pinned vLLM-ROCm oracle RUNS on gfx1151 (2026-08-19)

**The highest-uncertainty item in this analysis is resolved positively:** the
pinned vLLM-ROCm oracle (`555967922`) builds, imports, loads Qwen3-0.6B and
generates on the Strix Halo APU, in-cluster. Identity: `vllm
0.23.1rc1.dev1511+g555967922` (wheel
`vllm-0.23.1rc1.dev1511+g555967922.rocm723-cp312-cp312-linux_x86_64.whl`),
torch `2.13.0+rocm7.2`, cuda True, device `Radeon 8060S Graphics`. vLLM
selected the **ROCM_ATTN** backend on this board (`rocm.py` priority walk
resolved `['ROCM_ATTN', 'TRITON_ATTN']`) — the same kernel family our engine
registers. Evidence: `/home/kairos/oracle/oracle-run4.log` on the node.

**Build recipe (persisted at `/home/kairos/oracle/` on the node):** venv at
`venv/`, source at `vllm-src/` (checked out at the pin), wheel in
`vllm-src/dist/`, run script `run_oracle.py`. Three recorded findings that
cost iterations and are now pinned:

1. **torch: the pin's `requirements/build/rocm.txt` says 2.11.0+rocm7.1 —
   SEGFAULTS on gfx1151** (exit 139 on `torch.zeros(2, device="cuda")`, all
   HSA env combos). The rocm7.1 wheel bundles its own ROCm 7.1 runtime in
   `torch/lib/` (libamdhip64, libhsa-runtime64, libamd_comgr), which crashes
   on this APU. **torch 2.13.0+rocm7.2 (the pyproject CUDA-pinned version,
   rocm7.2 index) works** (linear + conv verified). The wheel was built with
   2.13.0+rocm7.2 — a recorded deviation from `requirements/build/rocm.txt`
   at the SAME vLLM pin, grounded in a measured segfault. The resulting wheel
   is tagged `.rocm723`.
2. **torchvision: pip's `--extra-index-url` merge picks the PyPI CPU wheel
   over the rocm one at the same base version** → `torchvision::nms does not
   exist` at import. Fix: install `torchvision==0.28.0+rocm7.2` explicitly
   from the rocm7.2 index.
3. **Model path:** the node's models live at `/home/kairos/models` (hostPath),
   not `/models`; the oracle pod mounts the host home, so the model arg must
   be `/home/kairos/models/Qwen3-0.6B`.

**First oracle-vs-ours token comparison (greedy, 8 tokens, same box):**

| Prompt | Oracle (pinned vLLM-ROCm) | Ours ROCm | Ours CPU |
|---|---|---|---|
| "The capital of France is" | ` Paris. The capital of France is also` | ` Paris. The capital of Italy is Rome` | ` Paris. The capital of France is also` (== oracle) |
| "The sky is" | ` blue, the ground is green, and` | ` blue, the ground is green, and` (== oracle) | ` blue, the ground is green, and` (== oracle) |
| "Water boils at a temperature of" | ` 100°C. What is` | ` 100°C at standard atmospheric` | ` 100°C. What is` (== oracle) |
| "Roses are red, violets are" | ` blue, and the rest are yellow.` | ` blue, and the rest are yellow.` (== oracle) | ` blue, and the rest are yellow.` (== oracle) |

The oracle agrees with OUR CPU arm 4/4 and with our ROCm arm 2/4. The two
our-ROCm divergences are exactly the Qwen3-0.6B near-tie candidates: the
France/Italy flip (the recorded literal 0.0000-nat tie) and "Water boils"
(our ROCm is the odd arm out).

**The full M4 gate then ran and PASSED (2026-08-19 10:22, gtr9):**

```
correctness gate: 16/16 prompts PASS (STRICT token-exact vs vLLM per-prompt
  greedy: 13/16; near-tie-band only: 3/16; max gap 0 nats; 0 forward-divergent)
BACKEND PROOF — Qwen3-dense ops on device type 5 with 0 declines
  (kRopeNeox selections=0, kPagedAttention selections=7168)
test cases: 2 | 2 passed | 0 failed; assertions: 125 | 125 passed; exit=0
```

Evidence: `m4-gate-20260819-102238.log` on the node; the gate is
`tests/parity/test_qwen3_paged_engine.cpp` (greedy near-tie correctness gate,
the same SACRED gate as gfx1100/gfx1200), holding our gfx1151 engine against
the oracle's per-prompt greedy with the teacher-forced near-tie band.

**Gate mechanics worth recording (they cost two iterations):**

1. **The committed ROCm goldens were CORRUPTED** — `our_ids_rocm.npy` AND
   `greedy_ids.npy` (committed from the gfx1100-era #559 capture) carry a
   degenerate prompt-3 row: a repeated fragment `38835, 13, 576, 38835, ...`
   that is not a real token sequence (vocab-valid max ≈ 152K; the gate even
   read a value not present in the file). The oracle on gfx1151 generates a
   sane continuation for that prompt, and our engine matches the CUDA golden
   — the committed pair is a bad capture, a finding for #41. Backed up at
   `/home/kairos/oracle/committed-goldens-backup-20260819/`.
2. **Fresh capture is fully deterministic:** oracle greedy over K=10 runs,
   per-prompt, `ALL DETERMINISTIC`, 0 multi-member cells — the gate is
   well-posed as a strict gate, matching the gfx1100 finding.
3. **The teacher-forced gap:** 28 divergent positions vs vLLM's greedy, and
   EVERY one is a literal **0.0000-nat tie** — our token IS the oracle's own
   argmax under teacher-forcing (band: 500 milli-nats). The oracle's
   incremental decode and its own full-prefill argmax disagree at near-ties,
   exactly as the gate methodology predicts.
4. **Staging:** fresh `greedy_ids.npy`/`greedy_dist.npy` (oracle capture)
   and `our_ids_rocm.npy`/`neartie_gap_mnats_rocm.npy` (our engine dump +
   gap matrix) staged into the gate dir; the `_rocm` pair was absent before,
   so the gate's dump path triggered and re-ran clean at the HEAD build.

**M4 on gfx1151 is now MET** on the same standard as gfx1200 (Gemma-3
48/48 two-oracle) and gfx1100 (16/16 pinned-oracle): greedy parity or
ratified near-tie vs the pinned vLLM-ROCm oracle, with the backend proof
(0 declines, all-native ops). The oracle also stays reproducible on the node
(`/home/kairos/oracle/`), so the M5 throughput battery can run it as the
reference.

## 4. The gaps, in dependency order

### G1 — the F6 fix is unverified on gfx1151 (front gate; blocks M2)

Everything downstream waits on this. What the F6 record (`rocm-unified-memory-b.md`
"Verification the community owes") requires from this board:

1. The probe triple printed by `test_rocm_backend`: `integrated` / `managed-alloc`
   / `UnifiedMemory()`, plus pass/fail of the two approach-(b) cases
   (alloc-path/UnifiedMemory coupling; kernel-write→host-read with **no copy**).
   Expected on gfx1151: `1/1/true`, both cases green.
2. A clean flag-free configure + `ctest -R 'rocm|cross_device'` (the local
   build-hip-gfx1151 tree is already the right shape — finish the build).
3. Any surprise in the triple (a board class outside the fix) is a loud,
   deliberate failure of the standing test — report it as-is, don't paper over.

**Why it is a hard gate, not a formality:** the managed branch is provably
dead on discrete parts (`Integrated=0`), so the (b) code has never executed on
*any* real silicon. The reference tier's safety argument — host dereference of
device allocations — is API-guaranteed only on this branch. M2's parity claim
is meaningless until the branch that makes the tier legal has run.

### G2 — M2 reference-tier e2e is pending (unblocks the kernel to-do list)

Once G1 passes, M2 is the cheapest milestone in the chain: on unified memory a
model runs end to end with **zero** native kernels. The acceptance (ROCM.md
§5.2) is a small dense model through the CLI with greedy parity vs `--device
cpu` and `VT_OP_PROVIDER_STATS=1` output. That fallback list is the *only*
authoritative M3 to-do for this board — it is sorted by real usage, not by
guesswork, and nothing below should be prioritized against it.

### G3 — M5's real kernel bar: zero reference-tier hits

This is the gap the milestone chain hides until M2. The reference tier is what
makes M2 possible on an APU, and it is exactly what M5 forbids:

> `VT_OP_PROVIDER_STATS=1` prints the first time each (op, device) falls back,
> and `GetReferenceTierHits()` **must be 0 in any performance measurement**.
> A non-zero value means you benchmarked the CPU. (ROCM.md §3)

So on gfx1151, M5 requires the full native kernel path for the benchmarked
model — the reference tier cannot be leaned on even though it installs. Known
state and known holes:

- **Native families present:** rmsnorm, dense basics, embedding, fp8 channel
  GEMV, GDN conv/postconv/scan/state/fused, Gemma-4 experts, hipBLASLt GEMM,
  MoE router, paged attention, sampling — 44 distinct registrations.
- **d=128 decode is default OFF.** The arm that fixed the Qwen3-shaped decode
  gap (`VT_ATTN_DECODE_D128`, #767, 3.53x on gfx1200) ships opt-in because its
  reduction order can move a greedy anchor at a bf16 tie. An M5 bench on any
  d=128 model runs `PagedAttnOnline` unless the flag is set — and flipping the
  default is a separate per-backend argument with a distributional gate, not a
  bench-time decision.
- **d=128 prefill still falls back** to the decode-shaped launch
  (`rocm-decode-attn-d128.md` §4/Owed); f32 d=128 decode is bf16-only today;
  GQA=4 fusion is open. None of these block a benchmark, all of them show up
  in the per-call trace against the oracle, and the honest M5 report should
  name which are in play for the chosen model.
- **What the model is:** the M5 definition says "the same model", so the
  choice is a bench-model decision on this board. The lane precedent is
  Qwen3-0.6B (dense, near-tie-robust gate exists) and Qwen3.5-0.8B (GDN, M4
  gate exists on gfx1100). Both are small enough for an APU and both already
  have correctness infrastructure; a bigger board-fit model (e.g. the 27B
  class) is not ruled out by memory (128 GB) but has no gfx1151 correctness
  lane yet, so it would owe M4 fresh.

### G4 — M4 on gfx1151: the oracle is the unproven piece

M5's denominator is vLLM-ROCm on the **same box, quant-matched**. Before any
speed number, M4 must be MET on this board: greedy token parity against a
pinned vLLM-ROCm oracle, with the near-tie methodology where the reference
itself is non-deterministic.

The recipe exists but has only ever run on discrete boards: `rocm-m4-oracle.md`
builds the pinned vLLM (`555967922`) in `rocm/vllm-dev:base` with
`PYTORCH_ROCM_ARCH` set per board, and produced deterministic K=10 captures on
gfx1100; the same containerized shape produced two independent oracles on
gfx1200 (`rocm-gfx1200-m2-correctness.md`). **vLLM-ROCm running on Strix Halo
is unproven in this project's record** — upstream recognizes the APU in its
device-name map (`rocm.py:75-77`), but "recognized by the platform file" is
not "oracle-runnable", and an APU brings its own questions (driver/version
match on TheRock, whether the pinned vLLM's ROCm backends select correctly on
unified memory). The rocm-attn spec §7 already lists this as owed ("M4:
vLLM-ROCm oracle token gate on gtr9 (gfx1151) for the ROCM_ATTN path"). This
is the highest-uncertainty item in the whole chain: it can only be answered
by a board owner attempting it.

### G5 — the bench arm itself is tooling-ready but never run on this board

The harness exists and has run on HIP: `examples/bench/vllm-bench` mirrors
`vllm bench serve` / `vllm bench throughput` metrics (request/output/total
token throughput, TTFT/TPOT/ITL/E2EL), and the gfx1200 decode-attn work drove
it against the real model (`rocm-decode-attn-d128.md` §10). What M5 adds is
the method, per BENCHMARKS.md "How we measure": greedy closed loop, three
interleaved reps per point, one `flock` across the series, cold legs
discarded, workload equivalence between arms audited (batch cap, token
budget, context, KV dtype, kernel family, graphed decode), and `vllm bench
throughput` on the same box + same model files as the denominator. Nothing in
that protocol is gfx1151-specific; the gap is that it has never run there.

### G6 — APU-specific performance risks that will decide the verdict

Three things unique to this board, each with a recorded hook:

1. **Managed-alloc speed (the (a)-fallback question).** The F6 decision
   record keeps approach (a) (gate the tier on `Integrated` alone) alive
   *exactly* for this: "if managed allocations measure slower on gfx1151 —
   measure, don't assume." The managed branch makes M2 legal; whether it makes
   M5 fast is unmeasured. If the hipMallocManaged allocation path costs
   measurable bandwidth, that is an M5 finding, not a foregone conclusion.
2. **Unified LPDDR5X bandwidth shared with the CPU.** gfx1151's 128 GB pool is
   the GB10 analogue, and GB10's whole ROCm-adjacent lesson is that *weight
   residency* is the lever on unified memory: staging device-resident weight
   copies lifted Laguna-S to 1.03x and DeepSeek-V4-Flash to 1.144x
   (`VT_LAGUNA_RESIDENT_BF16W`, `VT_V4_RESIDENT_W`). ROCM.md §4 hands this
   question to the Strix Halo owner ("the residency-policy question … is
   yours"). Expect it to matter here and have the A/B ready before blaming
   kernels.
3. **Bench hygiene on an APU.** The recorded teardown hazards (TheRock
   nightly exit hang on gfx1103; the `-O0` hostcall race #132 on gfx1100) mean
   a Release build and a clean-exit check per run are not optional; and an APU
   that may also drive a display needs the same "idle box, 2–3x reproduced"
   standard the gfx1200 rows already caveat for.

## 4. What M5 MET looks like on gfx1151 (acceptance)

Concrete, falsifiable, and none of it claimable before its predecessors:

1. **G1 green on the board:** probe triple `1/1/true`, both approach-(b) cases
   pass, flag-free configure log posted (#41).
2. **M2 green:** small dense model, greedy token parity vs `--device cpu` on
   the same build, `VT_OP_PROVIDER_STATS` fallback list posted.
3. **M4 green:** greedy parity (or ratified near-tie) vs a pinned vLLM-ROCm
   oracle running on the same gfx1151 box, per `rocm-m4-oracle.md`'s lane.
4. **M5 green:** `vllm-bench` vs `vllm bench throughput`, same model +
   quant, method per BENCHMARKS (3 interleaved reps, flock, audited workload
   equivalence), `GetReferenceTierHits() == 0` asserted, and the ratio
   meeting the ≥ vLLM bar (throughput ≥, latency ≤) on every declared axis —
   reproduced 2–3x idle.

## 5. Ordered next actions (the owed list)

1. **DONE — G1:** F6 verified on gtr9 (probe triple `1/1/true`, 9/9 cases,
   1071 assertions), re-confirmed at the fresh HEAD fingerprint
   `127246d2` (upstream `3b76ccbf` + the block-size test). Evidence:
   `f6-verify-20260819-090921.log` on the node; post the triple + the two
   approach-(b) case results to #41 with the configure/build logs (the
   flag-free Hadron/TheRock configure is itself evidence).
2. **DONE — G2 (and G3 for this model):** Qwen3-0.6B all-native M2 on
   gfx1151, zero fallbacks, 2/4 exact + 2 near-tie-regime. Evidence:
   `m2-verify-20260819-091200.log` (HEAD fingerprint).
3. **DONE — re-run at HEAD:** node re-synced + rebuilt at `127246d2`
   (`rebuild-head` job: CONFIGURE_OK, BUILD_DONE, 19 gfx1151 entries); all
   gates re-run at that fingerprint; pre-rebase build preserved as
   `build-hip-evidence-20260819`.
4. **DONE — M4 oracle on gfx1151:** pinned vLLM-ROCm built, imported
   (`0.23.1rc1.dev1511+g555967922`) and generates on the APU; 16-prompt
   near-tie gate PASSES 16/16 (13/16 strict, 3/16 in the 0-nat band, 0
   forward-divergent, 0 declines). Evidence: `m4-gate-20260819-102238.log`.
   Two findings for #41: the pin's torch 2.11.0+rocm7.1 SEGFAULTS on gfx1151
   (built with 2.13.0+rocm7.2 instead), and the committed ROCm goldens' prompt-3
   row is corrupted (fresh capture replaces them).
5. **DONE — near-tie classification:** teacher-forced gap lane ran — 28
   divergent positions vs vLLM greedy, every one a literal 0.0000-nat tie
   (our token IS the oracle's own argmax); oracle capture deterministic over
   K=10.
6. **Next — the APU perf A/Bs:** the managed-alloc branch for speed (the (a)
   fallback question) and **weight residency** (the GB10 lesson) before any
   kernel-level blame — the oracle is now the reference to A/B against.
7. **Pick the bench model** (Qwen3-0.6B is already all-native) and run the
   first paired `vllm-bench` / `vllm bench throughput` grid — the number
   itself is not the milestone; the method and the zero-fallback assertion
   are.

## M5-1 — first paired bench datapoint on gfx1151 (2026-08-20)

**Item 7's first run is done.** NOT a performance claim and NOT the binding
grid (c1 only, not median-of-3 on the oracle side) — but the method, the
zero-fallback assertion, and the first like-for-like numbers now exist.

**Method** (job `repro/bench-gfx1151-job.yaml`, node gtr9, `flock /tmp/gpu`):
Qwen3-0.6B, 1024 in / 128 out, 16 prompts, greedy, c1, one lock, arms
interleaved. ours = `src/build-hip/examples/vllm-bench` (hostPath, NOT in
image); oracle = `oracle/venv/bin/vllm bench throughput` with
`--random-input-len 1024 --random-output-len 128 --seed 0 --max-num-seqs 1
--max-model-len 2048 --gpu-memory-utilization 0.40
--override-generation-config '{"temperature": 0}'`. Notes learned the hard
way, recorded for the rerun: this vLLM build has NO `--temperature` and NO
`--ignore-eos` on bench throughput (ignore_eos is HARDCODED True in
`benchmarks/throughput.py` l.127/193/284/392); workload must be pinned via
`--random-input-len/--random-output-len`; default gmu 0.9 + max-model-len
40960 sized a 111 GiB KV pool and OOMKilled the first attempt.

**Numbers** (generated tokens 2048/2048 both arms; prefix-cache hit 0.0%):

| arm | prefill | decode (per-stream) | wall | reps |
|---|---:|---:|---:|---|
| vllm-cpp | 787 tok/s (TTFT 1299 ms) | **15.83 tok/s** (TPOT 63.2 ms) | 149.1 s | 3, identical to 3 dp (13.73/13.73/13.73 out, 15.83/15.83/15.83 decode) |
| vLLM-ROCm oracle | 717–811 tok/s | **92.4–93.2 tok/s** (engine avg) | 21.9 s | 2 (93.10, 92.98) |

Zero fallback on ours (`selected=reference` count 0 in all reps; every
`[vt op-provider]` line `selected=vt-native`).

**Two findings, both worth the #41 thread:**
1. **5.9x decode gap at c1 (15.8 vs 92.4 tok/s)** — and it is a
   CONTEXT-LENGTH regression on our side, not a constant offset: ours did
   42.8 tok/s at 128-in (smoke) but 15.8 at 1024-in (63 ms/token ≈ 19 GB/s
   effective on a ~256 GB/s APU — roughly 10% of the bandwidth the oracle's
   decode achieves). The oracle shows no such drop (93 tok/s at 1024-in).
   Prefill is a tie (787 vs ~750). Candidate next levers per item 6: the
   managed-alloc A/B and weight residency, before kernel-level blame.
2. **Oracle hang on 2nd in-pod invocation** — rep 1 completes in ~75 s
   (34 s load + 17 s torch.compile + 22 s bench); the rep-2 invocation
   wedged 11+ min at the post-load stage (log stops at "Model loading took…"),
   forcing a kill. Compile cache is per-pod (`/root/.cache/vllm`) so a fresh
   pod recompiles; the wedge did NOT reproduce across pods (runs 1 and 2
   each got a clean first oracle rep). Uninvestigated.

**Not claimed:** no binding ratio (median-of-3 oracle + c-grid owed), no
attribution, no upstream posting yet.

## M5-2 — context-length decode sweep on gfx1151 (2026-08-21)

**Item 7's regression curve is mapped.** Job `repro/bench-ctxsweep-job.yaml`
(ours only, Qwen3-0.6B, 128 out, 16 prompts, c1, greedy, seed 0, 2 reps per
length, one `flock /tmp/gpu`, zero-fallback assertion per rep):

| input len | decode (per-stream, tok/s) | TPOT (ms) | total tok/s |
|---|---:|---:|---:|
| 128 | 39.55 / 39.62 | 25.3 | 78.5 |
| 256 | 32.59 / 32.55 | 30.7 | 95.8 |
| 512 | 24.15 / 24.15 | 41.4 | 114.2 |
| 1024 | 15.90 / 15.91 | 63.0 | 124.2 |
| 2048 | 9.46 / 9.45 | 105.8 | 117.6 |

Zero fallback (all `selected=vt-native`, tally 0/10). Reps identical to 3 dp.

The curve fits `TPOT = 20.0 ms + 0.042 ms × L` **exactly** (all 5 lengths
within a millisecond) — decode cost is a fixed per-token cost plus a perfectly
linear per-context-token term. Qwen3-0.6B config (28 layers, 8 KV heads,
head_dim 128, bf16) → KV = 114 KB/token, so the linear term implies an
effective KV-read bandwidth of **~2.7 GB/s — 50–100× below the ~256 GB/s the
part can stream**. This is NOT a bandwidth-bound regression (a bandwidth-bound
scan would run ~100× faster); it is an access-pattern / paging / kernel
issue. That materially re-weights item 6: the managed-alloc A/B is still owed
but plausibly null on UMA (physically identical DRAM for both branches) —
and a KV-shaped strided-read probe (114 KB/step across 28×8 streams) is the
cheap discriminator BEFORE the multi-hour rebuild loop, to test the 4KB-page
TLB hypothesis vs kernel serialization. Also note ours is already ~2.3× below
the oracle at SHORT context (39.6 vs ~92 tok/s at 128-in), and the gap only
widens with L — so the fixed 20 ms term is as suspicious as the linear one.

## M5-3 — D128 GQA decode arm: 2.7–3.7×, and the fallback was the culprit (2026-08-21)

**The sweep ran the FALLBACK kernel.** The dispatch site (rocm_paged_attn.hip
~l.1721) keeps the d=128 GQA decode (`PagedAttnDecodeGqaBf16`) behind
`VT_ATTN_DECODE_D128=1`, DEFAULT OFF for byte-exactness (#382): "before this,
bf16 decode at d==128 fell all the way to the generic PagedAttnOnline (#488
measured that fallback at 41.1us/call against vLLM's 5.10us on gfx1200)".
Qwen3-0.6B is d=128 → **every M5-1/M5-2 number above is the FALLBACK path**.

Re-run of the same bench with `VT_ATTN_DECODE_D128=1` (job
`repro/bench-d128-job.yaml`, no rebuild — env flag on the existing binary,
zero-fallback tally 0, reps identical):

| input len | decode fallback | decode D128 | total fallback | total D128 |
|---|---:|---:|---:|---:|
| 1024 | 15.90/15.91 | **43.44/43.58** | 124.2 | **334.4/335.2** |
| 2048 | 9.46/9.45 | **35.34/35.43** | 117.6 | **382.7/383.1** |

**2.7× at 1024-in, 3.7× at 2048-in, 2.7× total throughput.** The D128 arm
keeps the ~20 ms fixed term but cuts the linear context term from 42 → ~3
us/token (TPOT = 20 ms + 3 us×L at 1024/2048: 23.1/26.2 ms — matches). So
with D128 the weight-streaming GEMV (~60 GB/s, the 20 ms fixed) is now the
floor, and the residual context slope is ~7× flatter than the fallback's.

**Probe (`repro/kv_pattern.hip`, job `repro/kv-pattern-job.yaml`) — what the
2.7 GB/s really was:** identical instruction stream, 256 B vs 4 KB reads per
block, on managed AND device alloc (s = 128/1024/2048):

| pattern | s=128 | s=1024 | s=2048 |
|---|---:|---:|---:|
| linear 256B/block | 5.46 ms (2.7 GB/s) | 43.0 ms (2.7 GB/s) | 86.4 ms (2.7 GB/s) |
| strided 256B/block (attn scatter) | 5.46 ms | 43.2 ms | 86.2 ms |
| big 4KB/block | 0.41 ms (36 GB/s) | 2.83 ms (41 GB/s) | 5.58 ms (42 GB/s) |
| managed vs device (any pattern) | identical | identical | identical |

The 256 B-granular numbers reproduce the fallback's linear term EXACTLY
(5.46 vs 5.38 ms @128; 43.0 vs 43.0 @1024; 86.4 vs 86.0 @2048). Conclusions:
(1) the regression is **block-granularity / dispatch-bound**, not
bandwidth-bound and not TLB-scatter (strided == linear); (2) 4 KB per block
is 13–15× faster at every length; (3) **the item-6 managed-alloc A/B is
CONFIRMED NULL on UMA — managed vs device identical, so no rebuild needed
for that lever.**

**Owed before any claim:** (a) the near-tie/distributional parity gate for
flipping VT_ATTN_DECODE_D128 ON (the #382 byte-exactness reason: warp-strided
online softmax reduces KV in a different order, so a greedy anchor can move
at an exact bf16 tie — measure, don't assume); (b) the binding grid
(median-of-3 oracle + c-grid) on the D128 arm.

## M5-4 — D128 parity verdict: near-tie only, no forward divergence (2026-08-21)

**The #382 byte-exactness question is now measured, not assumed.** Job
`repro/d128-classify-job.yaml` (goldens backed up first): (1) the gate in
re-capture mode (`VT_DUMP_IDS=1`, which skips the hard anchor REQUIRE) +
`VT_ATTN_DECODE_D128=1` dumped D128's greedy tokens; (2)
`scripts/qwen3-neartie-gap.py` teacher-forced the M4 oracle (pinned vLLM,
`VLLM_USE_V2_MODEL_RUNNER=0`, `/home/kairos/models/Qwen3-0.6B`) on the D128
sequence.

Result: **36 token-divergent positions vs vLLM greedy; max near-tie gap
0.1250 nats (worst at prompt[1] tok=9 — the same position the hard anchor
gate flagged: engine=279 vs committed anchor=2641)**. Threshold is 500 mnats
(0.5 nats); no position exceeded it, no OUTSIDE-top-K, no REAL divergence
lines. Every D128 divergent token is a token vLLM's OWN logits place within
0.125 nats of its argmax — bf16 near-tie resolution differences (the
warp-strided online-softmax reduction order), exactly as #382 predicted.
Artifact: `/home/kairos/d128-gap/qwen3_greedy_0_6b/neartie_gap_mnats.npy`
(D128 gap matrix).

**Conclusion:** flipping `VT_ATTN_DECODE_D128` ON for Qwen3-0.6B is
correctness-safe under this gate (near-tie only), worth 2.7–3.7× decode on
gfx1151. The flip still owes the board-owner rituals: re-capture the device
golden pair under D128, and the near-tie razor + distributional-gate sign-off
(the #382 comment names them). The fallback's own divergence count was 28
(M4 gap.log); D128's is 36 — more tie-flips, none outside the band.

## M5-5 — binding grid, ours leg on the D128 arm (2026-08-21)

Job `repro/bench-grid-d128-job.yaml` — canonical recipe (1024-in/128-out, 16
prompts, greedy, seed 0, one flock, zero-fallback assertion), median-of-3
interleaved reps, `VT_ATTN_DECODE_D128=1`:

| c | decode per-stream (median of 3) | total tok/s |
|---|---:|---:|
| 1 | **43.60 / 43.45 / 43.50** (43.5) | 335.2 / 334.3 / 334.4 |
| 4 | 34.89 / 34.83 / 34.96 (34.9) | 913.8 / 909.6 / 915.6 |

Zero fallback (tally 0/6). Reps identical to 2 dp. vs the M5-1 fallback
numbers at c1 (15.9 decode / 124 total): **2.7× decode and 2.7× total at c1**;
the decode gap to the oracle narrows from 5.9× to ~2.1× (43.5 vs ~92–93).
Per-stream decode drops at c4 (bandwidth sharing) while total triples —
healthy concurrency scaling. The oracle leg of the grid is still owed
(fresh process per rep to dodge the wedge), as is the fallback-arm c-grid
if a like-for-like fallback-vs-D128 claim is wanted.

## M5-6 — binding grid COMPLETE: paired medians, ours-D128 vs oracle (2026-08-21)

Oracle leg landed (job `repro/bench-grid-oracle-job.yaml`, Indexed completion,
6 fresh-process reps — no wedge; the first attempt failed because a `HOME`
env pointed vllm's .cache/.config at the read-only mount; dropping it fixed
it, matching the M5-1 recipe). Full paired grid, Qwen3-0.6B, 1024-in/128-out,
16 prompts, greedy, median-of-3, zero fallback on ours:

| c | ours decode (per-stream) | oracle output tok/s | ours total | oracle total |
|---|---:|---:|---:|---:|
| 1 | 43.60 / 43.45 / 43.50 (43.5) | 91.46 / 89.87 / 92.91 (91.5) | 335 | 823 |
| 4 | 34.89 / 34.83 / 34.96 (34.9) | 295.95 / 303.44 / 301.39 (301.4) | 914 | 2713 |

**The D128 arm closes the decode gap from 5.9× (fallback, M5-1) to 2.1× at
c1 and 2.2× per-stream at c4.** Both arms scale healthily with concurrency
(total 2.7×/3.3× at c4). This is the first binding ours-vs-vLLM datapoint on
gfx1151 — the M5 bar's "same box, quant-matched, same model" is met. The
remaining ~2.1× decode gap (43.5 vs 91.5 tok/s, ~11.5 ms vs 23 ms TPOT) is the
next target; candidates per the probe: the D128 kernel's residual granularity
(20 ms fixed term = weight-stream GEMV ~60 GB/s, and the ~3 µs/ctx-token
term) and the kernel-level dispatch ceiling.

## M5-7 — the flip's golden ceremony: D128 pair re-captured and CERTIFIED (2026-08-21)

**The board-owner ritual the flip owed is done.** Flow (job `repro/d128-promote-gate`):
backup live goldens → promote the D128 golden pair (from the `d128-classify`
teacher-forcing, scratch `/home/kairos/d128-gap/`) into the live goldens dir →
run the gate under `VT_ATTN_DECODE_D128=1` with NO dump (the certified path):

- **16/16 prompts PASS, 125/125 assertions, 0 forward-divergent; max gap
  0.125 nats @ prompt[1] tok=9** (the exact M5-4 drift position).
- The committed (fallback) pair vs the D128 pair: **28 token diffs across
  prompts [1,3,5]** (256 positions) — includes the known corrupted prompt-3
  row fix from the Aug-19 fresh capture. Max gap 125 mnats in BOTH pairs, 0
  positions over the 500-mnat band. This is the regen under the ratified-tie
  rule the dispatch-site comment (#382) names.
- Goldens promoted: `our_ids_rocm.npy` + `neartie_gap_mnats_rocm.npy`
  (live dir + fetched back to the flip branch; sha-verified).

**The flip change** (branch `flip/rocm-d128-decode-default-on`, base
`upstream/main` 3b76ccbf): `rocm_paged_attn.hip` `decode_d128` default OFF→ON
(`return e == nullptr || e[0] != '0'`, opt out with `VT_ATTN_DECODE_D128=0`),
+ the re-captured ROCm golden pair. CUDA arm untouched (stays OFF pending the
same ceremony there).

**DEFAULT-ON VERIFIED on the box (job `repro/verify-flip-job.yaml`, built
`rebuild-flip`):** gate on the default path (NO env) = 16/16 prompts,
125/125 assertions, 0 failed, SUCCESS; ROCm smoke all green
(test_rocm_backend 9/9 × 1071, attn_registry 17/17 × 62, cross_device
21/21 × 360); default-on bench spot-check @1024-in c1 = **43.60 tok/s
per-stream decode, 335.2 total, TPOT 22.93 ms, zero fallback** — identical to
the M5-5 D128 numbers (43.5/335). The shipped default now IS the D128 path.

## Owed / boundaries

- Nothing in this record is a performance claim yet. M5-1 → M5-6 deliver the
  method, the regression curve, the probe verdict, the 2.7–3.7× D128 A/B, the
  near-tie parity verdict and the binding paired grid (ours-D128 vs oracle,
  c1+c4, median-of-3, fresh process per rep). The golden re-capture + razor
  sign-off (M5-7) is DONE; the last item before posting is the default-on
  rebuild verification + a default-on bench spot-check on the box.
- The gfx1151 datapoints in §2/§3 (F6 verification, M2, M4 oracle gate) are
  now VERIFIED on the box but stay PENDING-community until posted — per §7
  of ROCM.md, a gate you cannot run stays PENDING, and that is the
  publishable state. The posting lane is #41 (F6 triple + the two
  approach-(b) cases) and the M4 capture/gate writeup.
- If the board owner's ROCm is the TheRock nightly, add the teardown-hang
  caveat from ROCM.md §5.1 to any ctest report: "SUCCESS printed, then hang"
  is a PASS of the test body plus the known runtime issue, named by build.
