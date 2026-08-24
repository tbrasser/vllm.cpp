## Row

`BACKEND-ROCM` — issue #41. Spec: [`.agents/specs/rocm-attn-backend.md`](.agents/specs/rocm-attn-backend.md) (draft, open for co-iteration).

## What changed

Adds the engine-level attention-backend registration for ROCm, closing the
seam where `SelectAttentionBackendName` threw for `kROCM` even though the
runtime kernel path worked:

- New `RocmAttentionBackend` (name `ROCM_ATTN`) registered for `kROCM` in
  the attention registry, on the same name-only footing as the Metal,
  Vulkan, and Tenstorrent rows.
- `RocmPlatform::get_attn_backend_priority` mirrors `rocm.py:407-441`
  `_get_backend_priorities` VERBATIM at pin `555967922` (verified by
  fetching the file at that SHA): dense `[ROCM_ATTN, ROCM_AITER_FA,
  ROCM_AITER_UNIFIED_ATTN, TRITON_ATTN, TURBOQUANT]`, MLA `[TRITON_MLA]`,
  sparse `[ROCM_AITER_MLA_SPARSE]`. Unregistered names are skipped by the
  walk, so dense requests resolve to the first registered name, `ROCM_ATTN`.
- The backend reports the NHD KV layout `(num_blocks, 2, ...)` that this
  runtime actually allocates and the stride-driven ROCm kernel reads —
  deliberately not upstream's K/V-outermost `(2, num_blocks, ...)`
  (`rocm_attn.py:247-256`, `rocm.py:521-522`). This is recorded as **one
  exact tracked exception** (§3 of the spec): the name identifies the kernel
  family that runs, and the deviation is pinned here and in the header.
- Upstream's `use_kv_connector` gate (`rocm.py:432-433`) does not apply to
  this registration because our shape is the shared symmetric NHD layout
  (the same one `FLASH_ATTN` allocates), not the asymmetric views that guard
  protects — reasoning recorded in the header + spec §4.
- Registry test asserts the `kROCM` registration; `test_rocm_backend.cpp`
  flips from "priority is empty" to checking the verbatim list,
  registration, selection, and NHD shape.

## Reachability (staged-slice contract)

Nothing in `src/`/`include/` consumes the selection yet — this PR is the
additive registration alone, exercised by tests. The runtime consumption
lands in the sibling PR `row/BACKEND-ROCM-ATTN-RUNNER` (#1065), which the
row contract specifies as a separate concern (see the `## Owed` section of
the spec: the runner work is proposed to move to its own row,
`BACKEND-ATTN-SELECTION-RUNNER`).

## Evidence

- CPU tier (this branch): `test_attn_backend_registry` passes (57
  assertions incl. the kROCM registration + walk checks), `test_platform`
  passes; clean HIP build (`test_rocm_backend` compiles).
- gfx1151 (GTR9 Pro 128 GB), contributor hardware: the `test_rocm_backend`
  case "the ROCm platform self-registers and is selected over CPU" passed
  with registration/selection/shape verified on real silicon.
- Honest note on gating, per the review: `test_rocm_backend.cpp` is
  `VLLM_CPP_HIP`-gated and there is no HIP job in `.github/workflows/ci.yml`,
  so the substantive assertions are contributor-hardware evidence, not CI
  gate evidence. The only CI-covered assertion from this PR is the single
  `HasAttentionBackend(kROCM, "ROCM_ATTN")` line.

## How to verify

```
ctest --test-dir build-hip -R attn_backend_registry
ctest --test-dir build-hip -R rocm_backend   # on a ROCm host
```

## Honest gaps

- No upstream AITER/Triton integration: the lists mirror `rocm.py` at the
  pinned revision; the AITER names are registered nowhere and are skipped
  by the walk.
- Layout deviation is a deliberate tracked exception (§3 of the spec) — if
  a real upstream-layout ROCm attention kernel lands, this registration
  flips shape with it and stops being an exception.

FOLLOWING_AGENTS_PROTOCOL

Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: AGENT:deepseek-v4 [Freebuff]
