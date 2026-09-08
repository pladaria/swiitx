//! System-ABI half of the gateway. Admission, protected dispatch lookup and
//! epoch publication/quiescence belong to the lifetime owner, not this module.

use crate::abi::{BudgetError, NativeExitReason, NativeFrame, PollBudget, PollOutcome};
use std::mem::offset_of;

/// Check minimum native-JIT CPU requirements once during process setup, not
/// per invocation or link. Cross-emission still targets the documented minimum;
/// the execution owner on the destination host performs this check.
pub fn check_host() -> Result<(), &'static str> {
    #[cfg(target_arch = "x86_64")]
    {
        // Intel SDM, CPUID.80000001H:ECX[0], LAHF_SAHF_64:
        // https://cdrdv2-public.intel.com/868137/325462-089-sdm-vol-1-2abcd-3abcd-4.pdf
        // CPUID is available on every x86-64 processor. Don't query an extended
        // leaf the host doesn't expose (including virtualized host features).
        let max = core::arch::x86_64::__cpuid(0x80000000).eax;
        if max < 0x80000001 || core::arch::x86_64::__cpuid(0x80000001).ecx & 1 == 0 {
            return Err(
                "Nixe requires LAHF/SAHF in x86-64 mode (CPUID.80000001H:ECX[0]); this host or VM does not expose it",
            );
        }
    }
    Ok(())
}

#[cfg(all(test, target_arch = "x86_64"))]
#[test]
fn host_requirement_matches_exposed_cpuid() {
    // Also run under qemu-x86_64 -cpu max,lahf-lm=off to exercise rejection
    // against real CPUID results, without a production feature-override switch.
    let max = core::arch::x86_64::__cpuid(0x80000000).eax;
    let supported = max >= 0x80000001 && core::arch::x86_64::__cpuid(0x80000001).ecx & 1 != 0;
    let result = check_host();
    assert_eq!(result.is_ok(), supported);
    if let Err(message) = result {
        assert!(message.contains("LAHF/SAHF"));
        assert!(message.contains("CPUID.80000001H:ECX[0]"));
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeReturn {
    pub reason: NativeExitReason,
    pub poll: PollOutcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeReturnError {
    InvalidExitReason(u32),
    Budget(BudgetError),
}

/// Enter an already protected canonical entry, then complete its canonical exit.
/// Caller nonvolatiles and the host stack are saved once, regardless of how many
/// native links execute. Guest units jump to the frame's exit continuation;
/// they never return or adjust SP. FP completion precedes all result handling.
/// A Control exit discards a crossed sample but charges both budget balances;
/// other reasons report the crossed sample to the cold-path caller.
/// This leaves the execution epoch active, including on error: the lifetime
/// owner may announce quiescence only after consuming the canonical result.
///
/// # Safety
/// The execution owner must have validated this host with [check_host] during
/// setup. Generated code assumes the minimum ISA, including x86-64 SAHF.
/// `begin_fp` must have run on this OS thread before dispatch lookup. The caller
/// must have published its execution epoch BEFORE reading `entry` or any other
/// executable address, and revalidated admission/reachability afterward. All
/// reachable code and `arena` must remain valid until this function returns.
/// `entry` uses this host's NativeFrame ABI and ends through an emitted canonical
/// exit (not RET), preserving the pinned registers and host SP. Every exit map
/// must make observed state canonical, including deferred flags. No unwinding
/// may cross native code. The frame must not be moved or entered recursively.
/// If native FP is already active, no general Rust work may intervene between
/// activation and this call. This is not a substitute for protected dispatch.
#[inline(always)]
pub unsafe fn enter_protected(
    frame: &mut NativeFrame<'_>,
    arena: *mut u8,
    entry: *const u8,
) -> Result<NativeReturn, NativeReturnError> {
    frame.exit_reason = NativeExitReason::None as u32;
    let remaining = unsafe { enter(frame, arena, entry) };
    // Only the bounded, shared FP-owner leaf may run before caller restoration.
    // Software FPSR writeback has already completed in the generated exit.
    unsafe { frame.finish_fp() };
    frame.gateway_exit = 0;
    let reason = NativeExitReason::try_from(frame.exit_reason)
        .ok()
        .filter(|reason| *reason != NativeExitReason::None)
        .ok_or(NativeReturnError::InvalidExitReason(frame.exit_reason))?;
    let poll = frame
        .budget
        .reconcile(remaining, reason == NativeExitReason::Control)
        .map_err(NativeReturnError::Budget)?;
    Ok(NativeReturn { reason, poll })
}

// Both entries keep SP aligned for eventual cold system-ABI helper calls.
// Save all allocator-visible nonvolatiles as well as the pinned registers.
// ABI references: https://gitlab.com/x86-psABIs/x86-64-ABI and
// https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn enter(
    frame: *mut NativeFrame<'_>,
    arena: *mut u8,
    entry: *const u8,
) -> i64 {
    core::arch::naked_asm!(
        "endbr64",
        "push rbp",
        "push rbx",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        "sub rsp, 8",
        "mov r15, rdi",
        "mov r13, rsi",
        "mov r14, [r15 + {deadline}]",
        "lea r11, [rip + 2f]",
        "mov [r15 + {continuation}], r11",
        "jmp rdx",
        "2:",
        "endbr64",
        "mov rax, r14",
        "add rsp, 8",
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbx",
        "pop rbp",
        "ret",
        deadline = const offset_of!(NativeFrame<'static>, budget) + offset_of!(PollBudget, armed_span),
        continuation = const offset_of!(NativeFrame<'static>, gateway_exit),
    );
}

#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn enter(frame: *mut NativeFrame<'_>, arena: *mut u8, entry: *const u8) -> i64 {
    core::arch::naked_asm!(
        "bti c",
        "stp x29, x30, [sp, #-160]!",
        "stp x19, x20, [sp, #16]",
        "stp x21, x22, [sp, #32]",
        "stp x23, x24, [sp, #48]",
        "stp x25, x26, [sp, #64]",
        "stp x27, x28, [sp, #80]",
        "stp d8, d9, [sp, #96]",
        "stp d10, d11, [sp, #112]",
        "stp d12, d13, [sp, #128]",
        "stp d14, d15, [sp, #144]",
        "mov x21, x0",
        "mov x19, x1",
        "ldr x20, [x21, #{deadline}]",
        "adr x16, 2f",
        "str x16, [x21, #{continuation}]",
        "br x2",
        "2:",
        "bti jc",
        "mov x0, x20",
        "ldp d8, d9, [sp, #96]",
        "ldp d10, d11, [sp, #112]",
        "ldp d12, d13, [sp, #128]",
        "ldp d14, d15, [sp, #144]",
        "ldp x19, x20, [sp, #16]",
        "ldp x21, x22, [sp, #32]",
        "ldp x23, x24, [sp, #48]",
        "ldp x25, x26, [sp, #64]",
        "ldp x27, x28, [sp, #80]",
        "ldp x29, x30, [sp], #160",
        "ret",
        deadline = const offset_of!(NativeFrame<'static>, budget) + offset_of!(PollBudget, armed_span),
        continuation = const offset_of!(NativeFrame<'static>, gateway_exit),
    );
}
