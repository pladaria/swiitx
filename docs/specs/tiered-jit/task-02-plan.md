# Task 2 implementation plan

Status: Task 2 is complete (steps 1–6), validated on native x86-64 and AArch64 QEMU 11.1.1.
Task 1 supplies the native contracts, fork output and executable boundary. Its native
AArch64 hardware validation remains pending; QEMU results are not native Arm
evidence.

This is a working checklist for
[Task 2](spec.md#task-2-build-the-bounded-publication-and-lifetime-foundation),
not another specification. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md).
Update progress and concrete decisions here in place, without session logs or
a separate decision register. Agree architectural changes with the maintainer
and update the affected specification section; keep implementation details in
code. This plan may be removed when the task is complete.

## Scope

Build the production publication/lifetime foundation and prove it with
synthetic native units containing no inter-unit links. These units are test
inputs to the real allocator, registries, publication and gateway, not a
parallel runtime. The new foundation must not use JITModule as its code owner.

Guest LCQ lowering/cutover belongs to Task 3. Static links, PICs, RSBs, islands
and linked cutover belong to Task 4; sampling, workers and HCQ construction
remain later tasks. Reserve the specified island capacity without implementing
linking here. Do not add benchmarks, a testing framework, runtime tuners or
production compatibility adapters to make the isolated proof easier.

## Steps

- [x] **Inspect ownership and settle the publication protocol.** Trace the
  current executable, dispatch and fault owners and the Task 1 gateway's caller
  obligations. Identify reusable host-memory/fault operations without copying
  the old append-only registries. Choose the concrete representation of the
  coordinator, immutable payload publication, reader epochs and strong-reference
  ownership. Explain the ordering in nearby code comments, including the
  reader-publication/store-to-lookup race; naming acquire/release operations
  alone is not a proof. Resolve reservation geometry, including the 2047 MiB
  reservation's non-full final segment, against the specified bounds before
  implementing address arithmetic. Record any required spec clarification here
  and resolve it before dependent work.

  **Exit:** every address/metadata reader has a named lifetime protection;
  publication, admission closure and reclamation have explicit linearization
  points. The implementation order and replacement boundaries are clear without
  adding another architecture document.

  Completed: ownership and protocol decisions are recorded below. The spec now
  clarifies the final segment, alias/backing accounting and writes to unpublished
  spans. Its soft-limit table label now agrees with the code-plus-metadata rule.
  No runtime or fork behavior changed during this inspection.

- [x] **Implement admission, protected readers and coherent dispatch first.**
  Build Open/Closing/Closed coordination with checked admission epochs and
  request sequences, reusable vCPU reader registration, and the generational
  dispatch/unit/family registry foundation. Publish complete immutable
  DispatchPayloads, retaining replaced payloads and removed slots until safe.
  Start with nonexecuted synthetic records to test this protocol before adding
  executable storage. Save caller FP before lookup, publish the reader epoch
  before reading an executable address, check Open, resolve the payload and
  revalidate admission/reachability before entry. Clear the epoch only after
  canonical/FP completion, including unsuccessful admission and error paths.
  Closing excludes new entry/publication; wait without holding JIT/cache/memory
  locks. Requests arriving during Closing/Closed must be drained or retained
  according to the spec before reopening at a fresh epoch.

  **Exit:** controlled reader/publisher/closer interleavings cannot expose a
  torn payload, admit an old epoch, reuse a protected slot or lose a maintenance
  request. The protocol does not rely on test sleeps or x86-only ordering.

  Completed in `crates/cpu-jit/src/lifetime.rs` and `lifetime/registry.rs`:
  checked Open/Closing/Closed coordination, exclusive transition ownership,
  sequence-aware acknowledgements, terminal shutdown, reusable registered
  readers, protected FP/epoch guards, coherent payload replacement and
  epoch-delayed dispatch-slot reuse. Registry/index growth is prepared outside
  state; payload destruction also happens outside it. The typed slab is shared
  infrastructure for dispatch/readers and the unit/family owners assembled in
  step 4, not placeholder production CodeUnits. Poison or counter exhaustion
  closes admission with an error instead of resuming inconsistent state.

  The coordinator coalesces reason notifications, not target records. Later
  code/mapping operations register and apply their exact records before batch
  acknowledgement. Optional link installation can retain its request across
  reopen; new safety work must still drain while Closed. The link installer and
  its 4096-record limit remain Task 4 work. Storage and complete-unit publication
  are covered by steps 3–4 below; actual CodeUnit/fault reclamation remains step 5.
  The legacy production executor is unchanged until Task 3's cutover.

- [x] **Implement staged output and bounded W^X storage.** Consume owned
  backend bytes, relocations, labels and state/fault maps without recompiling
  or passing ownership to JITModule. Implement the specified virtual reservation,
  on-demand segments, RW/RX alias permissions and instruction-cache coherence
  for both Linux hosts. Relocations use eventual executable addresses, not RW
  alias addresses; reject unsupported/out-of-range relocations precisely.
  Implement smallest-fitting-span/lowest-address allocation, bump fallback and
  immediate coalescing of aborted unpublished spans. Charge committed storage,
  metadata and retired-but-live allocations to the specified budgets, including
  the LCQ reserve; account for metadata replacement overlap. Keep allocation and
  relocation outside the JIT-state mutex. No callable bytes may be overwritten
  outside the maintenance protocol.

  **Exit:** executable synthetic units come from the new owner; actual mappings
  contain no RWX region and the RW alias is inaccessible outside allowed writes.
  Aborted output returns its real span and budget. Alignment, relocation,
  fragmentation, reserve and hard-limit cases have focused tests.

  Implemented in `crates/cpu-jit/src/executable.rs`, `executable/output.rs`
  and `executable/linux.rs`: owned final backend bytes/symbols/labels/maps;
  checked relocations against RX addresses; sparse memfd reservation with
  separate RX/RW views; segment commitment and hole-punch decommit; coalescing
  best-fit spans, alignment and island space; coupled storage/metadata charges.
  AArch64 cleans the RW alias, invalidates the RX alias using CTR_EL0 line sizes,
  and uses the pinned Wasmtime pipeline-synchronization implementation.
  No fork changes or extra compilation pass were needed.

  `Lifetime` now receives the same cache owner. Reader records, payloads, slab
  capacities and the cold key index all use its budget, including old/new
  replacement overlap. The key index uses hashbrown's actual allocation size,
  not an estimate of std::HashMap's private layout. Destruction returns the
  charge only after freeing storage, outside the JIT-state mutex.

  The executable reuse tests pass under QEMU 11.1.1 (see validation below).
  No QEMU-specific runtime workaround or ignored test was added. Step 4 supplies
  complete CodeUnits and the native-PC directory; epoch/reference-driven unit
  release remains step 5. Isolated storage tests retain their leases explicitly.

- [x] **Publish complete units and the fixed native-PC directory.** Assemble
  immutable CodeUnits with coupled allocation, entries, dependencies, state
  maps and fault records. Implement one fixed directory slot per possible
  segment and immutable sorted tables tagged by segment generation. Publish
  directory/metadata before dispatch makes the unit reachable. Preserve old
  table snapshots while protected readers can use them; do not free a retired
  unit's fault/dependency records early. Signal-time lookup must be bounded,
  lock-free and allocation-free; establish the protection carried through
  normal-stack fault dispatch rather than acquiring a potentially stale owner
  after lookup. Revalidate the captured admission epoch before the first
  DispatchPayload store; no fallible work may leave partial reachable output.

  **Exit:** every reachable synthetic entry and native fault PC resolves to its
  exact live owner and version. Unpublished, padding and non-fault addresses
  yield no record. Publication racing Closing either completes coherently or
  releases only its own unpublished resources.

  Completed in `lifetime/unit.rs` and `lifetime/directory.rs`: immutable
  CodeUnits couple the installed span, ABI/identities, exact instruction image,
  selected entries/contracts, dependencies, semantic state maps and fault
  intervals. Mutable lifecycle/publication epochs live in the registry record,
  under JIT state; signal-visible CodeUnit contents never change. Initial HCQ
  families pin LCQ images and preserve baseline dispatch. Family reshape and
  optimized instruction discovery/lowering remain later tasks.

  Preparation builds charged payloads/tables outside state. Publication
  revalidates admission, each slot/reachability, the captured memory cursor,
  family overlap and the segment-table snapshot before any reachable store.
  Concurrent changes to that table discard the candidate without touching old
  roots; independent segments need not compete for a predicted registry slot.
  The reusable dependency hash index adds associations once per unit/page;
  its storage grows outside state, rather than copying the index per publication.

  The 128-slot directory release-publishes immutable, generation-tagged sorted
  intervals before dispatch. Lookup performs bounds checks, one acquire load
  and a binary search, with no lock, allocation or reference counting. The
  returned fault/state data borrow the already-active Invocation epoch through
  normal-stack use. Replaced tables have their own retirement epochs and are
  collected outside state after reader quiescence, with both old/new storage
  charged while owned. The metadata-only publisher is now test-only.

  Seventeen focused tests cover multi-entry ownership, exact fault attribution,
  staged invisibility, old/new readers, publication/Closing races, changed
  cursors, competing candidates, malformed metadata, capacity/exhaustion,
  registry/index growth, baseline pins and coupled destruction. HCQ identity
  exhaustion disables optimizer publication without closing installed LCQ;
  required LCQ/lifecycle identity failures still close admission. Native signal
  transport/gateway integration remains the integrated proof/cutover work in
  step 6 and Task 3. The legacy production executor and Wasmtime are unchanged.

- [x] **Complete unlink, retirement and real reuse.** For these unlinked units,
  remove dispatch reachability under the coordinator and follow the specified
  CodeUnit lifecycle. Reclaim only when execution/fault readers are quiescent
  and compiler/link snapshot references have been released. Implement the
  snapshot ownership now; do not create worker queues or native link machinery
  merely to exercise references. Return executable spans and dispatch/unit/
  family slots, coalesce free space, and detach coupled metadata only when safe.
  Clear a wholly free segment's directory entry safely, decommit its backing
  storage and republish only with a fresh generation. Enforce the stated
  pressure policy for resident unlinked synthetic units, capacity errors and
  shutdown. Counter exhaustion and stale handles must never affect a new owner.

  **Exit:** retained readers/references delay actual reuse, not just accounting;
  releasing the final protection permits reuse of the same span and metadata
  slots. Segment decommit/republication cannot return a stale native-PC result.
  Repeated churn stays bounded and shutdown leaves no foundation-owned mapping
  or metadata allocation.

  Completed in `lifetime/unit/reclaim.rs`: exact per-unit maintenance targets,
  strong compiler/link snapshots, HCQ baseline pins and Closed dispatch removal.
  LCQ replacement schedules the old owner for cutover without clearing newer
  roots. Batch acknowledgement cannot discard undrained target records.

  Reclamation waits for both the unit's execution epoch and the separate grace
  period of its detached fault table. Code destruction returns the actual span
  outside JIT state before releasing held unit/family slots; dispatch slots
  also retain their CodeUnit users. Empty segments lose their directory entry
  before decommit, and recommit receives a fresh generation. Closed, exclusively
  owned fault tables can be withdrawn and compacted without allocation so a
  full metadata budget cannot prevent reclamation.

  Pressure first collects reclaimable storage and abandoned empty dispatch
  reservations, then retires oldest HCQ before unpinned LCQ. Pending whole-segment
  release stops unnecessary further eviction, but never refunds live bytes.
  HCQ reports insufficient capacity without waiting on snapshots; LCQ may use its
  hard-limit headroom, otherwise receives an explicit capacity error.
  `try_finish_shutdown` is nonblocking: retained snapshots or staged outputs
  postpone completion. Once released, it unmaps both cache aliases and frees
  the foundation's indexes outside JIT state. The terminal process object and
  any externally retained inactive reader remain charged until actually dropped.

  Seventeen focused reclamation tests plus a held-slot regression cover real
  reuse, two reader grace periods, baseline pins, stale/cross-process handles,
  exhaustion, cutover, requests arriving during Closed, full-budget progress,
  pressure order, shutdown and 128 LCQ/HCQ churn cycles with stable storage.
  No workers, linking machinery, test framework or Wasmtime changes were added.

- [x] **Run the integrated lifetime proof and close the handoff.** Execute
  unlinked synthetic units through the real protected lookup, new code owner,
  Task 1 adapters and gateway. Replace their temporary test-owned storage and
  invocation protection where the real foundation now supplies it. Exercise
  concurrent replacement, admission closure, fault-table readers, held compiler
  references, stale publication, pressure, slot/segment reuse and shutdown.
  Inspect mappings as well as accounting. Run focused regressions throughout
  the steps, then consolidate the Task 2 exit evidence here. Inspect both
  target implementations and execute on available hosts; explicitly distinguish
  native execution from QEMU and any still-missing hardware validation.

  **Exit:** every Task 2 criterion has code and test evidence; no append-only
  code, node, fault-table or retired-unit owner remains in the new foundation.
  Document only the old production owners still needed for Task 3's cutover
  and any linked test fixtures still awaiting Task 4. No fallback makes those
  old owners part of the new lifetime protocol.

  Completed: the unlinked backend reconstruction/arithmetic proofs now consume
  owned final output through the real cache, CodeUnit publication, admission
  guard, adapters and gateway. Dynamic-PC and invalid-budget gateway tests also
  use real reader epochs. Isolated register/adapter fixtures use direct cache
  leases instead of JITModule; they make no dispatch/epoch claim.

  Integration exposed an ordering defect: assigning CodeVersion during
  publication preparation could relabel metadata without changing native exits.
  `begin_unit` now reserves a single-use emission identity and admission epoch
  before version-bearing emission. Preparation validates both instead of
  rewriting state maps. Tests reject wrong-version maps and stale identities,
  and check exhaustion. Lifetime setup checks the native host requirement once.

  `native/tests/published.rs` executes a held old entry while another thread
  publishes its replacement and waits for Closed. It then executes the new
  entry, retains fault metadata across the second reader grace period, reclaims
  under hard-budget pressure, republishes at the same address with a fresh
  segment generation, executes again and drains shutdown with a held snapshot.
  Full-state ingress/exit also checks deferred flags, FPSR, pinned registers and
  completion before reader quiescence. Step 3's mapping inspection and step 5's
  exact backing-identity/unmapping tests supply the W^X and physical-release
  evidence; no counter-only stand-in is used for executable reuse.

  Remaining boundaries are explicit: the old `direct` executor/JITModules,
  lookup and NativeFaultRegistry belong to Task 3's production cutover, including
  wiring signal capture/landing/retry to this directory. The current tests prove
  exact protected attribution of a real native load, not delivered-signal retry.
  Only the linked `gateway` proof and the two linked `backend` proofs retain
  JITModule/manual invocation fixtures until Task 4 supplies native link lifetime.
  No legacy owner participates in the new foundation. Wasmtime is unchanged.

## Ownership and protocol decisions (step 1)

### Replacement boundaries

| Current owner | New foundation / reuse |
| --- | --- |
| `direct/compiler.rs`: DirectCompiler and CompilerRuntime own JITModules | Owned staging output and the executable cache own new units. The old compiler/runtime remain only until Task 3's cutover. |
| `direct/mod.rs`: ProcessState.regions, retired and dependency sets | Generational unit/family slabs and versioned dependency handles; no permanent retired vector. |
| `direct/lookup.rs`: RegionLookup.native_keys retains Arc nodes and generated bucket chains | Cold BlockKey index plus generational DispatchSlots and immutable DispatchPayloads. No raw slot address escapes into generated code. |
| `cpu-direct-memory/src/lib.rs`: NativeFaultRegistry owns an append-only region vector/page hash | Fixed 128-slot segment directory and reclaimable immutable sorted tables. Reuse WorkerFaultContext capture/landing stacks, not its registry lifetime assumptions. |
| `abi.rs`, `native/backend.rs`, `native/gateway.rs`, `fp_env.rs` | Reuse contracts, checked counters, final maps, machine gateway and FP owner. NativeFrame's scalar epochs are invocation-local values, not the shared reader announcement. |

The direct interpreter's fixed-access batches also use WorkerFaultContext.
Its existing registry must not be silently reinterpreted as the new JIT
directory. Task 3 will wire the new JIT attribution while preserving that
distinct interpreter use, without retaining a legacy JIT fallback.

### State and readers

Use one short JIT-state mutex for coordinator state, registry/index mutation,
gateway admission/key lookup and reclamation eligibility scans. Coordinator
phase, checked admission/execution epochs, request sequence and pending work
are authoritative under this mutex; do not pack full-width counters into spare
phase bits. The process control bit is the release-published notification to
running vCPUs, not a second admission authority. No mutex or reference-count
operation is added to generated RAM operations or resolved native edges.

Dispatch/unit/family handles are `(slot, generation)` in reusable slabs. Slot
generations are checked and never wrap; they are not CodeVersion identities.
Reader registrations contain stable heap-resident AtomicU64 announcements,
with zero meaning inactive. Registration, deregistration and admission use the
state mutex; a record is not reused while an invocation/fault transport can
still address it. One vCPU cannot enter two invocations concurrently.

Gateway ordering for step 2:

1. Save caller FP with the shared owner, then acquire the state mutex.
2. Require Open. Release-store the current execution epoch to the registered
   reader BEFORE resolving a BlockKey, slot, payload or executable address.
3. Resolve the generational slot and acquire-load its AtomicPtr<DispatchPayload>.
   Copy the selected immutable entry/version and revalidate Open/admission and
   ReachabilityVersion before releasing the mutex. No borrowed slot/payload
   pointer survives that critical section; no 128-bit atomic is required.
4. Release the mutex before native entry and activate guest FP only after cold
   Rust work. Keep the announced epoch through execution and fault dispatch.
   On a miss, finish FP and leave the epoch before compiling/waiting; restart
   admission for the eventual entry. Never wait for Closing while still active.
5. After canonical state, pending FPSR, caller FP and captured metadata use are
   complete, acquire the state mutex, release-store zero and notify waiters.
   This notification shares the waiter's predicate mutex, avoiding lost wakes.
   Apply the same cleanup to every failed admission/return path.

The mutex is the store-to-lookup proof, not an assumed Release/Acquire pairing
on unrelated atomics: if admission wins, its unlock happens-before a closer or
collector acquires the mutex and sees its reader announcement (or a later
completed exit). If closure/removal wins, subsequent admission sees Closed/
Closing or the replacement entry. A closer arriving after the final admission
check must wait for that already-announced invocation, even if its machine jump
has not happened yet. Carry this proof into the step 2 implementation comments.
See the [Rust acquire/release model](https://doc.rust-lang.org/nomicon/atomics.html#acquire-release).

### Publication, closure and references

Prepare all fallible storage, payloads and directory tables outside the state
mutex. Under it, validate the captured admission epoch and exact handles,
install complete metadata, then release-store the payload pointer: that store
is the key's publication linearization point. Replaced payload boxes may be
dropped outside the mutex once their serialized readers have copied the entry;
the copied executable address remains protected by the reader epoch. Removed
DispatchSlots still follow the spec's epoch retirement rule.

Register transition work and its checked request sequence under the state
mutex, signal the existing control bit, and change Open to Closing with a fresh
checked admission epoch there.
That state change is the closure linearization point and excludes publication.
One transition owner waits via a condition variable which releases the mutex;
fault dispatch remains part of its vCPU's active epoch. Closed work uses cache
or memory locks separately. Reacquire state to acknowledge exactly the drained
sequence and check pending work before reopening at a fresh admission epoch.
Concurrent requests either join that stop or start the next one; Shutdown is
terminal and cannot reopen. Counter exhaustion never wraps or fabricates a
successful lifecycle transition.

The unit registry retains one Arc<CodeUnit> through retirement. Indexes and
pending transition records use checked handles; compiler/link snapshots acquire
strong references under state while the unit is eligible. Existing snapshots
may clone their own reference. Never upgrade an unprotected raw/weak owner, and
do not let a retired unit acquire new snapshot references from an index.
Reclamation scans are cold operations, not a new background collector.

### Directory and reclamation ordering

The fixed directory stores AtomicPtr<SegmentFaultTable>, with release swaps
and acquire signal-time loads. A table contains a generation and sorted exact
intervals naming unit/version and immutable fault data. No signal-time locks,
allocation, Arc increment or last-owner destruction is permitted. Its borrowed
metadata is protected by the invocation's ALREADY active epoch through the
normal-stack resolver and retry/escape; a handler cannot announce protection
only after obtaining a raw pointer. Diagnostic readers must register before
lookup too. The process/directory outlive all reader registrations.

After removing future entry roots, stamp retirement with the current execution
epoch and advance that epoch before admitting later readers. Under state,
require all announcements to be zero/newer and no external strong references
before detaching a unit's directory/dependency records. Directory swaps and
their retirement stamps serialize under state with admission. Replacing a
table needs its own retirement epoch: even a reader of another unit in the
segment may hold that snapshot. Keep detached unit metadata alive until every
old table that can name it has completed this grace period. Directory tables
do not hold CodeUnit Arcs that would create a reclamation dependency cycle.

Final eligibility is selected under state, making the exact retired handle
unavailable to new references or another collector. Release state before
returning its span under the cache lock: insertion into the free-span map is
the span-reuse linearization point. Only afterward return registry slots under
state; no slot becomes reusable while its old allocation is still owned.
Charge retired table snapshots and metadata until actually released. For a
wholly free segment, publish null, complete the directory grace period, release backing storage,
and use a fresh segment generation on its next publication. Generations alone
do not make stale raw pointers safe.

### Geometry and implementation order

The executable window has 128 directory slots: 127 times 16 MiB plus 15 MiB.
Reserve the final 64 KiB of each actual segment for islands; bounds checks
exclude the absent last MiB before index calculation. The RW alias is outside
this executable window, maps the same backing and does not double its physical
charge. Soft and hard limits both include metadata. These are clarifications
of existing limits, not increased capacity or a new policy.

Step 2 implements the above state/reader protocol with nonexecuted records;
step 3 adds actual storage; step 4 adds complete-unit publication and directory
snapshots;
step 5 completes physical reclamation; step 6 replaces temporary protection in
unlinked native proofs. No new fork hook has been identified as necessary;
retain the current pin `e2a984d96678207094c0fc50057c8b6bcfd68715`.

## Validation

`lifetime/tests.rs` and the slab's unit tests cover competing publishers,
address/version coherence, admission versus closure, requests arriving during
Closed, deferred links, shutdown, stale handles, epoch-delayed reuse, newer
readers, registration churn, FP completion, poison and counter exhaustion.
Interleavings use barriers/channels, not sleeps. These metadata-only protocol
tests do not execute their fake addresses. No new test framework or fork hook
was introduced.

The storage tests additionally cover actual RX execution, closed RW permissions,
span fragmentation/reuse, cross-thread execution, relocation encodings/ranges,
backend context reset, metadata budget failures and real backing release
(`fstat.st_blocks`, not just permission changes).

`lifetime/unit/tests.rs` adds complete-unit publication and fault-directory
proofs, including execution of a synthetic System-ABI leaf while protected by
the real admission guard. Its unreachable faulting instruction has an exact
record; actual signal capture/retry with this directory is still integration
work, not claimed by these attribution tests.

Native x86-64: 234 unit tests and four integration tests pass, including 24
admission, 15 storage, 19 unit-publication and 17 reclamation tests. Clippy and
formatting checks pass.

AArch64 QEMU 11.1.1 (`-cpu max`): all 124 nonlegacy tests pass, including the two storage
reuse tests that executed stale translations under QEMU 6.2.0. No runtime
workaround or test exclusion was added to resolve those failures. Native Arm
hardware validation remains pending.

The [QEMU build and test recipe](../../aarch64-tests.md) pins the release archive,
configuration and explicit CPU model. Native validation commands:

```sh
cargo test --offline --locked -p nixe-cpu-jit --quiet
cargo clippy --offline --locked -p nixe-cpu-jit --lib --tests --no-deps -- -D warnings
cargo fmt --all --check
git diff --check
```

## Specification reading by step

Read the applicable section when implementing each step, not the entire spec
for every edit:

- Steps 1–2: [runtime data model](spec.md#runtime-data-model),
  [gateway](spec.md#gateway-and-fast-mode),
  [coordinator](spec.md#maintenance-coordinator) and
  [concurrency rules](spec.md#failure-policy-and-concurrency-rules).
- Steps 3–4: [cache ownership](spec.md#executable-cache-and-backend-ownership),
  [publication](spec.md#compilation-and-publication-pipeline),
  [bounded indexes](spec.md#bounded-registries-and-indexes) and the lookup/lifetime
  obligations in [fault retry](spec.md#native-fault-retry).
- Steps 5–6: [reclamation](spec.md#epoch-reclamation), cache pressure/shutdown
  rules, [migration map](spec.md#migration-map) and the Task 2 exit criterion.
