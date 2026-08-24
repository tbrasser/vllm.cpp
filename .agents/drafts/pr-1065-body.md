## Row

`BACKEND-ROCM` — issue #41. Depends on #1056 (registration). Spec:
[`.agents/specs/rocm-attn-backend.md`](.agents/specs/rocm-attn-backend.md).

## What changed

Wires the attention-backend selection into the engine's runtime path, which
previously never called it (`SelectAttentionBackendName` had zero production
callers on main — the engine-level registry was dead code). Second pass
addressing the review's five findings:

1. **Resolution is per GROUP, not per runner**, and lives INSIDE the
   `full_attn_group_id_ >= 0` region — a pure-GDN / pooling model pays no
   selection (the stale runner.h comment is now true). Dense groups resolve
   loudly (a platform with no registered dense backend fails at init, only
   for models that need one). MLA groups resolve `TRITON_MLA` on CUDA —
   whose 3-dim `get_kv_cache_shape` is exactly the fused cache
   `deepseek_v2.cpp` views — and stay op-driven on devices with no
   registered MLA backend (CPU, ROCm today), because `TritonMLAImpl` is not
   registry-gated.
2. **The shape check is real and tested.** It moved to
   `vllm::v1::CheckKvCacheShape` (registry.h/cpp) with the per-group
   expected view — NHD 5-dim for dense groups, fused MLA 3-dim for MLA
   groups — and a new registry test registers a deliberately mis-shaped
   scratch backend (upstream's K/V-outermost shape) and asserts the throw,
   plus positive controls proving the comparison is not an echo of the
   engine's own numbers.
3. **`test_kimi_linear_paged` fixed**: fixture block size 8 → 16 (vLLM gate
   block size), the `%16` contract being now reachable.
4. **`test_bench` fixed + the `%16` contract made deliberate at the entry
   points**: `bench_core.h` rounds the synthetic unified block up to a
   multiple of 16; `server_main --block-size` validates (positive, %16)
   with a clear error instead of a bare `stoi`. Upstream enforces the
   constraint too; this makes it an announced change at the two shipped
   paths rather than a side effect.
5. `VT_ATTN_SELECT_LOG=1` prints one line per attention layer
   (`kind=dense|mla backend=... device=... shape=[...]`).

## Evidence

- CPU tier (this branch): `test_attn_backend_registry` **17/17** (61
  assertions, incl. the new mis-shaped case), `test_runner` **19/19** (543),
  `test_kimi_linear_paged` **8/8** (206), `test_bench` **11/11** (80),
  `test_llm_engine` **24/24** (493), plus `test_prepare_inputs` /
  `test_mla_attention_block` green — clean `-Werror` build, 0 warnings.
- On gfx1151 (GTR9 Pro 128 GB), Gemma-3-1B-it through the full engine with
  `VT_ATTN_SELECT_LOG=1` (first pass, pre-review):

  ```
  [attn-select] backend=ROCM_ATTN device=5 shape=[256,2,32,1,256]   (every layer)
  vllm-cli: run=1/1 ... completion_tokens=8
   Paris.
  ```

  Selection resolves to `ROCM_ATTN` on the real device, the NHD geometry
  validates, and the model still generates identically.

## How to verify

```
VT_ATTN_SELECT_LOG=1 vllm-cli --device auto --model <dir> --prompt "The capital of France is"
ctest --test-dir build-hip -R 'runner|attn_backend_registry|kimi_linear_paged|bench|llm_engine'
```

## Honest gaps

- **Depends on the registration PR** (#1056): without it the ROCm priority
  walk resolves to an empty list and a dense request throws at init. Merge
  order matters: registration first, then this PR. Both branch from
  `0f8580e26`, touch disjoint files, and rebase cleanly; the dependency is
  semantic, not textual.
- **MLA on CPU/ROCm stays op-driven** (no registered MLA backend): recorded
  in the runner, not an error — the engine's MLA execution
  (`TritonMLAImpl`) is not registry-gated.
- **Scoping**: this is a different concern from the ROCm row (the row's
  contract was zero runner edits) and is proposed to move to its own row /
  issue / spec (`BACKEND-ATTN-SELECTION-RUNNER`) — see the spec's `## Owed`.
- The `%16` enforcement becoming reachable is a deliberate, announced
  change (entry-point fixes above); if the maintainer prefers it as a
  standalone PR, we can split it out.

FOLLOWING_AGENTS_PROTOCOL

Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: AGENT:deepseek-v4 [Freebuff]
