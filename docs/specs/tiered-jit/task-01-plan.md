# Task 1 implementation plan

Status: steps 1–5 are complete, including the available-host proof and
exit-criterion review. The tested fork remains pinned; native x86-64 and
emulated AArch64 pass. Native AArch64 hardware is still unavailable: this is an
explicit validation gap, not a native Arm conformance claim. The next
implementation task is Task 2's publication/lifetime foundation; production
frontend cutover and final native cross-target validation remain later tasks.

This is a working checklist for [Task 1](spec.md#task-1-freeze-and-prove-the-executable-abi-and-state-contracts),
not another specification. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md).
The task's scope and exit criteria remain in `spec.md`. Update this file in
place, without session logs. Agree architectural changes with the maintainer
and update the affected specification section; keep local implementation
details in code. This plan may be removed when the task is complete.

## Steps

- [x] **Inspect the existing implementation and backend.** Locate Nixe's
  lowering, architectural state, helpers, FP handling and code-output path.
  Inspect the local Wasmtime fork on branch `nixe`, using the
  [specified release base](spec.md#wasmtime-fork-and-working-branch).
  Identify reusable code and the backend changes needed for both host ISAs.
  Check available native test hosts and the current instruction-coverage
  baseline. Record concrete blockers, not speculative infrastructure needs.

- [x] **Prove the risky backend mechanisms first.** Introduce the minimum
  shared ABI definitions needed to exercise reserved registers,
  NativeFrame-relative spills and prologue-free entries. Generate small
  fragments for both targets and check the emitted instructions and spill
  extent. Also establish how physical entry constraints, final state maps and
  selected labels can be represented before committing to the full integration.
  Execute both external entries of one optimized body, including loops, without
  relying on computations or allocator edits in a skipped analysis root.
  If this requires replacing the allocator or machine backend, stop and discuss
  the architectural change rather than implementing a fallback.

- [x] **Implement the shared state contracts.** Complete the identities,
  counters, NativeFrame, dispatch payload and entry/exit contracts named by
  Task 1. Implement or reuse one use/def, partial-write and liveness analysis
  for both tiers, accounting for helpers, faults and architectural observations.
  Add focused tests for preservation and safe omission of guest values.

- [x] **Complete the backend and native boundary.** Implement the remaining
  fork hooks for physical state maps, selected labels, patchpoints, relocations,
  fault offsets, caller-owned output and required landing instructions. Connect
  the gateway, canonical/fast entries, canonical exit and cycle-safe bridges
  to these real outputs. Preserve caller and guest FP state and lazy flags.
  Keep production epoch/reclamation machinery in Task 2; any temporary support
  for the isolated proof stays test-only.

  Completed: final entry/exit maps now drive canonical loads, generic bridges,
  dirty writeback and lazy NZCV materialization through the actual gateway.
  Tests compile both units independently with every source/target allocator
  combination and execute empty and integer/SIMD-cycle edges. Allocation-chosen
  entry/exit spills also execute under pressure. Instruction-attached prefault
  maps translate to the shared contracts at the actual trap offsets; the fork's
  native x86-64 fault tests cover capture, including compound atomics. Shared FP
  ownership completes pending FPSR after software writeback and restores the
  caller. Both encoders pass; execution is native x86-64 and QEMU AArch64.
  Protected dispatch and epoch publication remain in Task 2; the test-owned
  signal capture is not the production fault/retry path.

- [x] **Run the native proof and finish integration.** Exercise empty and
  register-cycle bridges, randomized register pressure, the fixed spill
  partition and exact state reconstruction. Inspect both targets' output for
  the Task 1 exit criteria, and execute on available native hosts. Record any
  missing native execution explicitly; encoding or cross-compilation is not
  execution evidence. Use conventional tests, not a new testing framework or
  benchmark runner. Verify the final tested fork pin and lockfile (initial
  integration pin is already in place). Remove superseded task-local scaffolding;
  identify any test-only proof support that must remain until the later production
  cutover.

  Completed on available hosts: one invocation crosses both link shapes;
  seeded allocation pressure reconstructs every mapped value and checks pins,
  SP and spill bounds. Actual final maps feed arithmetic and logical lazy
  recipes. Fork regressions and the exact Git pin are verified. Obsolete
  inspection notes and duplicate test setup were consolidated; retained
  test-only support and the missing native AArch64 run are listed below.

Test each step as it becomes executable; the final step consolidates evidence,
not the first opportunity to discover integration failures. Full code-cache
management, production linking, LCQ cutover, HCQ and benchmarks are outside this
task.

## Integration handoff

- Fork: `/home/pladaria/projects/wasmtime`, branch `nixe`, commit
  `e2a984d96678207094c0fc50057c8b6bcfd68715`, based on stable Wasmtime
  `v48.0.1` (`7bac2c2775808aaec5d4aa5627a5e447b51102cf`).
  All five Cranelift dependencies and the lockfile resolve to that exact Git
  revision (`0.135.1`). There is no local-path override or floating branch.
- Shared contracts are in [abi.rs](../../../crates/cpu-jit/src/abi.rs);
  instruction effects and liveness are in
  [analysis.rs](../../../crates/cpu-jit/src/analysis.rs). Reuse the existing A64
  decoder, typed instruction representation and semantics; do not introduce a
  second IR or liveness classifier.
- `native/backend.rs::AllocatedBoundary` translates final typed operands into
  shared physical bindings. Operand indices are the original CLIF boundary
  argument/result order. Eliminated entry inputs must be explicitly omitted;
  required inputs, invalid banks, reserved registers and out-of-extent spills
  are rejected at compilation time.
- Lazy flags share `LazyFlags<Value>`: SSA operands in the current frontend,
  physical locations at native boundaries. The frontend's CLIF emitter and
  `native/flags.rs` serve different emission stages; they are not one emitter.
  The fork preserves operand locations, not guest NZCV recipes or persistent
  host flags. Never infer `NzcvLocation::Host` from the preceding machine
  instruction. Full frontend-to-boundary migration remains in Task 3.
- The fork accepts `Any` or fixed-register entry constraints. Allocation-chosen
  entry spills work; forced spills and caller-selected spill offsets do NOT.
  The earlier forced-stack-definition experiment produced corrupt input with
  `single_pass` even though the allocator checker passed. Do not restore that
  option without fixing and executing its semantics.
- `nixe::set_entries` exports selected native labels, including landing
  instructions and allocator edits. Its analysis root and outgoing critical
  edges are not executable. `nixe_entry`, `nixe_state` and `nixe_exit`
  export typed final maps; exit patches are 8-byte x86-64 or 4-byte AArch64
  units. `patch_exit` modifies owner-supplied bytes; it supplies no W^X,
  publication, maintenance rendezvous or far-branch island.
- Prefault spans retain PRE-instruction operands at actual native memory PCs,
  including each trapping instruction in compound atomics. An ordinary state
  marker does not describe a later memory instruction. Fault maps do not yet
  implement Nixe's production retry/commit protocol (Tasks 3 and 8).
- `Context::take_compiled_code` transfers bytes, relocations and metadata to the
  owner. Task 2 supplies production allocation/publication. The Nixe ABI rejects
  ordinary CLIF system arguments, calls/returns and dynamic frames; this is not
  permission to fall back to the old context-tail ABI.
- `native/gateway.rs` saves host nonvolatiles once and restores FP before Rust
  result processing. The caller must publish its epoch before executable-address
  lookup and revalidate admission/reachability. Tests use invocation-local
  protection with strongly owned code, not the Task 2 concurrency protocol.
  `fp_env.rs` is shared with the old compiler: software FPSR writeback precedes
  host-status collection, and quiescence follows complete canonical state.

## Exit-criterion evidence

These are conventional regression tests, not a separate harness or benchmark
system. Paths under `native/` are relative to `crates/cpu-jit/src/`; fork
paths are relative to the Wasmtime checkout.

| Criterion | Evidence |
| --- | --- |
| One gateway, empty and cyclic edges in one invocation | `native/tests/backend.rs::one_gateway_crosses_empty_then_cyclic_edges_between_two_compiled_units`: A:first → B → A:second; actual final maps, both encoders, all four allocator pairings, one canonical ingress and exit. |
| Reserved registers, SP and bounded spills | `randomized_final_maps_reconstruct_live_state_with_bounded_spills`: eight seeded cases per encoder/allocator, 3–31 GPR/vector pairs, shuffled operand order and 12 KiB explicit reservations. Execution checks frame/arena/poll pins, unchanged aligned SP, a reserved-transfer canary and the tail beyond the reported frame extent. Every mapped vector is reconstructed, not left canonical to hide bad spills. |
| Exact frame limit and instruction shapes | Fork `cranelift/codegen/src/nixe.rs`: actual byte disassembly on both encoders, register pressure, exact 16 KiB acceptance and overflow rejection. Nixe additionally checks compiled VCode and executes the resulting boundaries. |
| NZCV and FP completion | `native/tests/backend/arithmetic.rs` resolves Add/Subtract/AddCarry/SubtractCarry operands from actual final maps at 32/64 bits, with low/high pressure and shared CPU arithmetic as the oracle. Integrated logical recipes, FP division-by-zero, sticky FPSR, caller restoration and whole-state comparisons cover the canonical exit. |
| Selective flags and transfers | `native/tests/flags.rs`, `canonical.rs` and the transfer tests cover partial masks, nested conditional recipes, host carry conventions, constants, aliases, cross-bank/overlapping spills and seeded register cycles. |
| Exact entry, patch and fault offsets | Nixe integration consumes selected labels/landings and patched exits, and compares prefault PCs with trap records. Fork `boundary_tests.rs` covers ranges/alignment; `multi_entry.rs` covers optimized loops and independent entries; `cranelift/jit/tests/nixe_faults.rs` captures real x86-64 SIGSEGV state, including compound atomics. |
| One semantic/state-analysis path | Shared contracts/analysis are consumed by the existing compiler and the new boundary. Tests cover partial writes, observations, CFG joins and flag liveness; no new decoder or test-only instruction classifier was introduced. |

## Validation

Use Rust `1.97.1`. Final commands and results are recorded here, not in a
generated baseline or session log.

Nixe, native Linux x86-64: 156 unit, 2 dependency-boundary and 2 differential
tests pass (160 total). Strict Clippy passes without test-warning exceptions;
formatting and whitespace checks also pass.

```sh
cargo test --offline --locked -p nixe-cpu-jit --quiet
cargo clippy --offline --locked -p nixe-cpu-jit --lib --tests --no-deps -- -D warnings
cargo fmt --all --check
git diff --check
```

AArch64 under QEMU: 46 tests pass, including ABI, analysis, FP and the 29
native-boundary tests (the existing production executor's tests are excluded):

```sh
cargo test --offline --locked -p nixe-cpu-jit --lib \
  --target aarch64-unknown-linux-gnu \
  --config 'target.aarch64-unknown-linux-gnu.linker="aarch64-linux-gnu-gcc"' \
  --config 'target.aarch64-unknown-linux-gnu.runner=["qemu-aarch64", "-L", "/usr/aarch64-linux-gnu"]' \
  --quiet -- --skip direct::
```

Fork, native Linux x86-64:

```sh
cargo +1.97.1 test --offline --locked -p cranelift-codegen --no-default-features --features std,unwind,all-native-arch,disas --lib --quiet
cargo +1.97.1 test --offline --locked -p cranelift-reader --lib --quiet
cargo +1.97.1 test --offline --locked -p cranelift-jit --tests --quiet
```

The fork checks pass 247 codegen, 43 reader and 10 JIT tests. No fork source
changes or new revision were required for step 5.

The x86-64 host-requirement test also passes under `qemu-x86_64 -cpu
max,lahf-lm=off`, exercising rejection when CPUID does not expose LAHF/SAHF.

## Retained support and remaining scope

- Nixe's test-owned JITModule byte storage is an isolated W^X owner, not a
  production allocator. Retain it until tests can use Task 2's owner without
  introducing a production compatibility layer.
- Independent assembly input/output capture is still needed to verify native
  emitters and preservation of registers not written canonically. Fork-local
  byte owners and shuffles keep backend regression tests independent of Nixe.
  These are test-only; none is a callable production fallback.
- The old production compiler and its typed FP veneers remain required until
  Task 3's vertical cutover. Remove superseded code during that cutover; Task 9
  is the final residue audit, not permission to retain obsolete paths.
- Native AArch64 hardware is still unavailable. QEMU execution and dual-encoder
  inspection are recorded as such, never as native Arm validation. Execute the
  focused tests on the future native host; native cross-target conformance
  remains mandatory for Task 10. Task 1 does not prove production publication,
  reclamation, fault retry, complete frontend migration or performance.
