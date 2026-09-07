# Task 3 implementation plan

Status: planned; implementation has not started. Tasks 1–2 supply the native
contracts, fork output and publication/lifetime foundation. Homebrew execution
still uses the old `direct` executor. Available validation is native x86-64 and
AArch64 QEMU 11.1.1; native Arm hardware validation remains pending.

This is a working checklist for
[Task 3](spec.md#task-3-cut-synchronous-lcq-over-as-one-vertical-slice),
not another specification. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md).
Update progress and concrete decisions here in place, without session logs or
a separate decision register. Agree architectural changes with the maintainer
and update the affected spec section; keep implementation details in code.
This plan may be removed when the task is complete.

## Scope

Connect demanded guest instruction capture and synchronous LCQ compilation to
the real NativeFrame ABI, executable cache, protected dispatch and runtime.
Preserve currently supported guest instruction semantics while replacing their
old execution boundary. Reuse the decoder, normalized instructions, analysis,
typed helper semantics and FP owner; do not introduce another semantic IR.

The completed task has one callable JIT path and no old HCQ promotion/workers.
External edges use source-local canonical fallbacks until Task 4 implements
native links, PICs and the RSB. Functional sampling/background admission remain
disabled until Task 5; new HCQ construction remains Task 6. An LCQ-only interval
is intentional, not a reason to retain the old optimizer as a fallback.

No benchmark runner, homebrew suite, testing framework, runtime tuners or
compatibility executor is part of this task. Test each step as it becomes
executable; do not route homebrews to an incomplete implementation.

## Steps

- [ ] **Inspect the production cutover boundaries.** Trace runtime dispatch,
  current instruction coverage, helpers/SVCs, FP-mode changes, exclusive state,
  control requests and progress reporting. Inspect instruction fetch, executable
  snapshots, invalidation notifications and mapping/write synchronization in
  the memory authority. Determine which interfaces already satisfy the required
  ordering and which must change; a cursor check alone is not a coherent byte
  snapshot or a pre-mutation rendezvous. Identify the signal/landing machinery
  reusable without the old JIT fault registry. Confirm supported process-wide
  memory backend selection without adding per-access fallback.

  Record the concrete ownership/API decisions and any real blocker below before
  dependent edits. Use existing semantic tests as the coverage baseline, not a
  new generated inventory or measurement tool.

  **Exit:** the route from demanded PC to canonical runtime return and the
  ordering of capture, mutation and fault retry are explicit. Replacement
  boundaries are identified without changing runtime behavior in this step.

- [ ] **Implement demanded capture and exact-key compile ownership.** Capture
  one coherent owned instruction image with exact physical/mapping dependencies
  and its publication cursor. Decode from the demanded BlockKey only, stopping
  at the first specified terminator/architectural boundary or 512 instructions.
  Do not fetch a successor, scan a function or discover a region. Preserve
  precise fetch-failure attribution; an indirect entry into an existing
  fragment may create another independently indexed fragment.

  Add the specified same-key/current-generation compile claim to the reusable
  cold dispatch state. The winner uses vCPU-local compiler state; same-key
  contenders may wait only after leaving their reader epoch and restoring FP.
  Do not hold process, cache or memory locks during compilation or waiting.
  Closure, failure and shutdown wake waiters; stale completion releases only
  its exact claim. Reserve the emission identity before version-bearing native
  emission and carry its admission epoch through publication.

  **Exit:** tests cover terminators, the emergency cut, page boundaries, overlap,
  stale capture and cancellation. Same-key contention has one winner per claim
  generation, while different-key compilers can make progress concurrently.
  Claims and empty dispatch slots do not accumulate after abandoned work.

- [ ] **Lower LCQ into the real native ABI and cache.** Port the existing
  instruction lowering to one demanded fragment, with `opt_level=none` and
  `single_pass`. Use shared effects/liveness and the fork's entry/exit/fault
  boundaries; obtain physical bindings, labels and spill extent from final
  allocation. Keep lazy NZCV as shared SSA recipes until its consumers require
  materialization. Never assume the backend leaves a persistent host flag
  producer available. Polls and address confinement must preserve live flags.

  Emit canonical ingress and source-local exits with exact dirty/live state,
  destination PC, edge kind and the reserved CodeVersion. Consume owned output
  once, relocate at final RX addresses and publish complete CodeUnits through
  Task 2. Supported integer/SIMD operations remain native; architectural helpers
  have typed effects and correct FP suspension/completion. Unsupported behavior
  is an attributed failure, never interpreter or old-JIT fallback. Complete
  memory/fault lowering in step 4 before production cutover.

  **Exit:** non-memory guest fragments execute through protected lookup and the
  gateway with correct state, branches, flags and FP. Final maps agree with the
  emitted exits; stale output never becomes reachable. No JITModule or legacy
  NativeContext is required by these fragments.

- [ ] **Connect direct memory and real fault dispatch.** Lower supported scalar,
  SIMD, pair, atomic and exclusive accesses with their existing architectural
  semantics. Normal RAM uses the pinned arena, required confinement and native
  operations, without generated permission/backing lookups or eager canonical
  checkpoints. Account for compound-access ordering, subaccesses and commit
  stages; preserve unaffected state on every fault.

  Connect WorkerFaultContext capture/landing support to the fixed native-PC
  directory and the active Invocation epoch. Keep that protection through
  normal-stack resolution and retry, including flags, FP and physical spills.
  A recoverable RAM fault resumes the identical native instruction; a valid
  non-RAM access performs one typed cold operation; invalid guest access reports
  its precise architectural fault. Unattributed/nested/impossible faults and
  repeated unchanged tracking faults fail precisely. Do not reuse an earlier
  instruction checkpoint or keep the old JIT registry as a second lookup path.

  **Exit:** delivered-fault tests on both available execution hosts cover retry,
  nonretry reconstruction and compound-access side effects. Ordinary RAM has
  the specified native shape. Preserve the interpreter's distinct fixed-stub
  use of shared fault support without retaining the legacy JIT executor.

- [ ] **Integrate guest-memory invalidation with the coordinator.** Register
  exact affected units/compile claims before acknowledging notifications.
  Find all fragments covering changed executable bytes, including overlapping
  roots and physical aliases. Serialize code writes, mapping/permission changes
  and tracking transitions with Closing/Closed: exclude new admission and
  publication, drain active execution/fault dispatch, remove roots, then expose
  the memory mutation and reopen with a fresh epoch. A writing vCPU must leave
  its own epoch before waiting for this rendezvous.

  Revalidate captured memory state on publication and handle invalidation-stream
  overrun explicitly, never by assuming missed records affected nothing.
  Reuse Task 2 retirement, strong references, directory grace periods and
  reclamation; extend exact maintenance records for memory work rather than
  introducing a second coordinator. No cache/memory lock is nested under JIT
  state while applying mutations or waiting.

  **Exit:** controlled races cover compilation versus writes/remapping, physical
  aliases, overlapping LCQ roots, executable writes and reader-held fault data.
  No old code runs after the mutation becomes visible, no stale compilation
  publishes, and no operation waits for its own active reader.

- [ ] **Switch the runtime, implement control budgets and remove old execution.**
  Make runtime JitProcess/JitThread use the new owner and vCPU-local compiler.
  A miss leaves native mode, compiles only the demanded key, then restarts
  admission; unresolved branches/calls/returns use the canonical resolver.
  Preserve guest call/LR behavior, SVCs, loader return, stop requests and
  scheduler-visible faults through the new boundary.

  Integrate PollBudget with runtime slices: a nonpositive slice never enters
  native code; completed straight-line work is charged at block boundaries and
  required backedges, not through per-instruction progress stores. Honor the
  independent block/backedge maintenance check, reconcile spent work once on
  exit and preserve NZCV/FP ordering. Keep the shared budget representation but
  do not emit functional samples, maintain hotness tables or enqueue HCQ work.
  Update callers/tests that incorrectly require exact instruction stepping.

  Wire cold capacity reclamation and terminal shutdown, including outstanding
  compile claims. Remove the superseded production region discovery/executor,
  PublishedRegion ownership, lookup chains, context-tail gateways, JITModules,
  per-entry promotion and old HCQ workers/configuration that only served them.
  Move reusable semantics instead of retaining adapters to the old context.
  Update exports and boundary tests to the new implementation; do not defer
  callable legacy removal to Task 9. Linked Task 1 proof fixtures remain test-only
  until Task 4 replaces their manual link ownership.

  **Exit:** configured JIT execution uses only the new LCQ path, with no silent
  fallback or legacy HCQ startup. Zero/small slices and stop requests remain
  responsive with bounded overshoot. Shutdown drains compilation and releases
  real code/metadata; tests assert the new contracts rather than legacy shapes.

- [ ] **Validate the vertical slice and close the handoff.** Run the existing
  instruction differential tests against the new JIT, preserving coverage of
  previously supported NormalizedA64 variants. Compare architectural state and
  precise failures at actual observation boundaries, not exact slice counts.
  Consolidate focused capture/deduplication, fault, invalidation, capacity,
  shutdown and emitted-shape evidence from the preceding steps.

  Run affected CPU, memory, direct-memory and runtime tests as well as formatting
  and Clippy. Execute on native x86-64 and AArch64 QEMU; run native Arm when
  available and label that evidence separately. Smoke-test an already available
  legal homebrew through the normal runtime if its required guest services are
  supported; do not create a benchmark suite or conceal unrelated blockers.
  Inspect remaining legacy references and record the concrete Task 4 handoff.

  **Exit:** every Task 3 criterion has code/test evidence, supported guest
  coverage is retained, and there is one callable JIT route. Record commands,
  results and actual gaps here; do not claim final linked-JIT performance or
  native Arm conformance from this LCQ/QEMU milestone.

## Starting points and decisions

- New foundation: `cpu-jit/src/lifetime{,/unit}.rs`, `executable.rs`, `abi.rs`
  and `native/`. Task 2's `native/tests/published.rs` and unlinked backend proofs
  demonstrate the real publication/gateway path, not guest decoding or signal
  delivery. Reuse that code, not a parallel runtime.
- Current production: `cpu-jit/src/direct/{mod,compiler,region,lookup}.rs` and
  the instruction-specific compiler files; runtime binding is in
  `runtime/src/process/execution.rs`. Shared memory contracts/authority are in
  `cpu/src/memory/`; invalidation publication is in `memory/src/invalidation.rs`;
  signal capture/landing support is in `cpu-direct-memory`.
- Keep the tested Wasmtime revision
  `e2a984d96678207094c0fc50057c8b6bcfd68715`. Inspect
  `/home/pladaria/projects/wasmtime`, branch `nixe`, if integration exposes a
  concrete backend gap. Do not assume a fork change is needed; if one is needed,
  test it locally and pin the agreed committed revision for reproducibility.
- Step 1 records any additional decisions here. No compile-claim layout,
  memory API or signal adaptation has been implemented by this plan.

## Specification reading by step

- Steps 1–2: [LCQ](spec.md#lcq-baseline-compiler),
  [keys](spec.md#keys-and-dispatch-publication),
  [direct memory](spec.md#direct-memory-and-fault-authority) and
  [publication](spec.md#compilation-and-publication-pipeline).
- Steps 3–4: [native ABI](spec.md#native-fast-chain-abi),
  [helpers](spec.md#helpers-and-architectural-boundaries) and
  [fault retry](spec.md#native-fault-retry).
- Steps 5–7: [invalidation](spec.md#code-and-mapping-invalidation),
  [coordinator](spec.md#maintenance-coordinator),
  [control budget](spec.md#control-budget-and-functional-sampling) (sampling
  remains disabled), [reclamation](spec.md#epoch-reclamation) and
  [migration](spec.md#migration-map).
