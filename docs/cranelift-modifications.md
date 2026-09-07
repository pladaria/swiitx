# Cranelift modifications for Nixe

This document inventories the changes maintained in Nixe's Wasmtime fork and
explains their purpose in the [tiered JIT](specs/tiered-jit/spec.md). It describes
the implemented backend capabilities; production integration progresses through
the specification's tasks. Performance benefits below are design motivations,
not measured speedups.

## Baseline and revision

- Upstream: Wasmtime `v48.0.1`, Cranelift `0.135.1`, commit
  `7bac2c2775808aaec5d4aa5627a5e447b51102cf`.
- Fork: `pladaria/wasmtime`, branch `nixe`; Nixe currently pins
  `e2a984d96678207094c0fc50057c8b6bcfd68715` in
  [the JIT manifest](../crates/cpu-jit/Cargo.toml) and `Cargo.lock`.
- The maintained changes are in two commits:
  [leaf ABI and canonical multi-entry support][abi-commit], and
  [boundary maps and patchable native exits][boundary-commit].

The [complete diff][fork-diff] is the source inventory. Paths below are relative
to that Wasmtime checkout. Update this document when advancing the dependency
pin, including any changes removed because upstream now supplies them.

## Implemented modifications and benefits

### 1. Opt-in leaf-fragment ABI

Added `enable_nixe_abi` and validation for a fragment ABI on x86-64 and AArch64.
With this mode enabled, the backend emits no ordinary function prologue and
rejects system-ABI arguments, calls, returns and dynamic frames, including
backend-introduced calls. The existing register allocator and machine backends
remain in use.

**Benefit to Nixe:** the gateway can save the host context once, then execute a
chain of guest units using jumps. Each unit avoids repeated function setup and
teardown, and guest calls do not grow a host call chain. Helper transitions and
the final return remain Nixe's responsibility.

Implementation: `cranelift/codegen/meta/src/shared/settings.rs`,
`cranelift/codegen/src/nixe.rs`, and `cranelift/codegen/src/machinst/abi.rs`.

### 2. Reserved registers for the execution context

Changed the allocation environments to exclude Nixe's persistent registers and
link scratch registers:

| Role | x86-64 | AArch64 |
| --- | --- | --- |
| NativeFrame pointer | `r15` | `x21` |
| Poll budget | `r14` | `x20` |
| Direct guest-arena base | `r13` | `x19` |
| Link scratch | `r11` | `x16` / `x17` |

Some registers were already unavailable in the ordinary backend, notably
AArch64's link scratch registers. The change reserves the additional Nixe
registers while preserving the ordinary allocation policy outside this mode.

**Benefit to Nixe:** memory accesses, polling and transitions can use stable
registers throughout the chain without reloading their context at each boundary.
Link code can use scratch registers without overwriting allocated guest values.
This deliberately reduces the register pool available to guest computations.

Implementation: `cranelift/codegen/src/isa/{x64,aarch64}/abi.rs`.

### 3. Fixed NativeFrame-relative spills

Redirected spills, reloads and explicit stack-slot addresses to the pinned
NativeFrame base instead of the host stack. The fixed area is 16 KiB, aligned
to 64 bytes; its first 2 KiB are reserved for Nixe's transfers, leaving 14 KiB
for backend storage. The backend reports `nixe_frame_size` and rejects overflow.

**Benefit to Nixe:** independently compiled units share a bounded storage
contract without adjusting SP. Bridges can address spilled values directly,
while cycle-breaking and helper/fault transfers cannot collide with backend
spills. Nixe can reject or split oversized units before publication.

Implementation: `cranelift/codegen/src/machinst/{abi,buffer}.rs` and the
target-specific ABI files.

### 4. Independent entries into one optimized body

Added `nixe::set_entries` and function metadata for selected external entries.
An analysis-only root makes all entries reachable for dominance and allocation;
the root and its outgoing critical edges are omitted from executable output.
Optimization is constrained so loop-invariant computations cannot move into
that unexecuted root. Validation rejects dependencies on root-defined values.
Final entry offsets include the entry's landing instruction and allocator edits.

**Benefit to Nixe:** an optimized region can expose multiple valid entry points
without duplicating its body or executing a selector/prologue on every entry.
Each entry works independently, including when the region contains loops.

Implementation: `cranelift/codegen/src/nixe.rs`,
`cranelift/codegen/src/egraph/elaborate.rs`, and
`cranelift/codegen/src/machinst/{blockorder,lower,vcode}.rs`.

### 5. Allocation-visible inputs and final state maps

Added the CLIF operations `nixe_entry`, `nixe_state` and `nixe_exit`, with
lowering for both targets. Entry results define simultaneous physical inputs;
state and exit operands remain live through allocation. Final `StateMap`
records retain operand order, types, physical registers or NativeFrame spill
offsets, and exact native positions, independently of debug information.
Eliminated entry inputs are explicitly reported as `Location::Unused`.

Entry constraints allow allocation to choose a location (`Any`) or require a
specific register. Allocation-chosen spills are supported; forced spills and
caller-selected spill offsets are not.

**Benefit to Nixe:** canonical adapters and bridges can transfer precisely the
values required by each boundary, even between units compiled by different
allocators or tiers. Compatible boundaries can avoid transfers. Nixe can also
reconstruct dirty guest state and lazy flag operands from actual allocations,
without writing every guest register back at every edge. The fork tracks SSA
operands; Nixe supplies their guest meaning and NZCV recipes.

Implementation: `cranelift/codegen/src/nixe/boundary.rs`, CLIF instruction
definitions in `cranelift/codegen/meta/src/shared/`, and target lowering,
operand collection and final emission in `cranelift/codegen/src/`.
Nixe consumes these maps in
[native/backend.rs](../crates/cpu-jit/src/native/backend.rs).

### 6. Patchable exits and indirect-entry landings

Added aligned exit patch units: 8 bytes on x86-64 and 4 bytes on AArch64.
`StateMap::patch_exit` encodes an in-range direct branch into caller-owned
bytes and validates the patch bounds, alignment and branch reach. Boundary
metadata is protected from subsequent branch truncation that could invalidate
its recorded position.

Added `enable_nixe_ibt` for x86-64 `ENDBR64` landings and integrated selected
entries with the existing AArch64 BTI machinery. Entry offsets point to the
landing when one is emitted.

**Benefit to Nixe:** the code cache can link a unit directly to another unit or
a bridge, avoiding dispatcher lookup on linked edges. Indirect entry targets
can include the landing instructions required by host control-flow policy.
Nixe still owns far-branch islands, W^X mappings, synchronization, instruction
cache maintenance and safe live-code publication; patch encoding supplies none
of those protocols.

Implementation: `cranelift/codegen/src/nixe/boundary.rs`,
`cranelift/codegen/src/machinst/{buffer,vcode}.rs`, and target instruction emitters.

### 7. Prefault state at actual memory-instruction PCs

Added `nixe_fault_start` / `nixe_fault_end` around ordinary trapping CLIF memory
operations. Their operands stay live at the machine instructions that can
fault, and `nixe_faults` exports the final locations at each actual trap PC,
including multiple trapping instructions in compound atomic lowering.
The x86 assembler generator now exposes memory-trap annotations for this
tracking. Invalid spans, such as nested spans or `notrap` memory operations,
are rejected.

**Benefit to Nixe:** a direct-memory fault can recover the guest state preceding
the operation from the saved native registers and frame. A marker placed before
the access would be insufficient because allocation and compound lowering can
change locations before the actual fault. These maps provide the backend data
for Nixe's fault protocol; retry and partial-commit semantics remain in Nixe.

Implementation: `cranelift/codegen/src/machinst/{lower,vcode,buffer}.rs`,
`cranelift/codegen/src/nixe.rs`, `cranelift/codegen/src/nixe/boundary.rs`,
target instruction handling, and
`cranelift/assembler-x64/meta/src/generate.rs`.

### 8. IR persistence, diagnostics and regression coverage

Extended CLIF function metadata, parsing, printing and verification for the new
entries, constraints and instructions. Added regressions for both encoders,
both allocation algorithms, independent optimized entries, frame limits,
boundary maps, patching, chaining and native x86-64 fault capture.

**Benefit to Nixe:** the custom contracts can be inspected and reproduced in
backend tests, and fork upgrades can detect failures in the mechanisms on which
the JIT depends.

Implementation: `cranelift/codegen/src/{ir,verifier,write.rs,nixe.rs,nixe/}`,
`cranelift/reader/src/parser.rs`, and `cranelift/jit/tests/nixe_faults.rs`.

## Existing APIs reused and integration status

`Context::take_compiled_code`, ordinary code bytes and relocations, the two
register allocation algorithms, and Wasmtime's instruction-cache coherence
implementation already exist upstream. Reusing them is not a fork modification.
The custom output consists of entry offsets, boundary/fault maps and frame
extent metadata carried alongside the existing compilation output.

The [Task 1 handoff](specs/tiered-jit/task-01-plan.md#integration-handoff)
records backend integration and its tests. Its validation includes native
x86-64 execution, both encoders and AArch64 execution under QEMU; native AArch64
hardware validation remains outstanding. Production publication and lifetime
management are tracked in [Task 2](specs/tiered-jit/task-02-plan.md), with frontend
cutover and the complete fault protocol in later specification tasks.

[abi-commit]: https://github.com/pladaria/wasmtime/commit/2f8ccabacf
[boundary-commit]: https://github.com/pladaria/wasmtime/commit/e2a984d96678207094c0fc50057c8b6bcfd68715
[fork-diff]: https://github.com/pladaria/wasmtime/compare/7bac2c2775808aaec5d4aa5627a5e447b51102cf...e2a984d96678207094c0fc50057c8b6bcfd68715
