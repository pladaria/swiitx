//! One host-FP owner for the current compiler and the tiered native boundary.
//! Host encodings and status translation are shared, not duplicated per tier.

use crate::abi::{HostFpState, NativeFrame};
use crate::fp_policy::native_fpcr_supported;

#[cfg(test)]
pub(crate) mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsupportedFpControl(pub u32);

impl std::fmt::Display for UnsupportedFpControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FPCR {:#010x} requires exact FP semantics, not native FP activation",
            self.0
        )
    }
}
impl std::error::Error for UnsupportedFpControl {}

impl HostFpState {
    /// Save the caller environment once, before any guest segment.
    ///
    /// # Safety
    /// This owner must remain on this OS thread until `finish`. No other owner
    /// may manage this thread's FP environment during the invocation.
    pub unsafe fn begin(&mut self) {
        assert_eq!(self.active, 0, "guest FP segment is already active");
        assert_eq!(self.saved, 0, "caller FP environment was already saved");
        let (control, status) = host::read();
        self.saved_control = control;
        self.saved_status = status;
        self.saved = 1;
        self.suspended = 0;
    }

    /// Lazily activate supported native FP. Repeated activation on a compatible
    /// link does not clear accumulated guest status or replace the caller save.
    ///
    /// # Safety
    /// `begin` must have run on this thread. An active segment's FPCR must not
    /// change without `end`. No general Rust/helper work may execute while the
    /// guest environment is active; suspend first.
    pub unsafe fn ensure(&mut self, fpcr: u32) -> Result<(), UnsupportedFpControl> {
        assert_ne!(self.saved, 0, "guest FP activation before gateway save");
        if !native_fpcr_supported(fpcr) {
            return Err(UnsupportedFpControl(fpcr));
        }
        if self.active != 0 {
            return Ok(());
        }
        host::install_guest(fpcr);
        self.active = 1;
        self.suspended = 0;
        Ok(())
    }

    /// Restore the caller before a helper and return this segment's sticky
    /// guest FPSR contribution. The caller ORs it into authoritative software
    /// FPSR after any physical software-value writeback.
    ///
    /// # Safety
    /// Same invocation/thread as `begin`; all live native values are protected
    /// before this system-ABI operation.
    pub unsafe fn suspend(&mut self) -> u32 {
        if self.active == 0 {
            return 0;
        }
        let status = host::guest_status();
        host::restore(self.saved_control, self.saved_status);
        self.active = 0;
        self.suspended = 1;
        status
    }

    /// Resume only a successfully suspended segment, with cleared host status.
    ///
    /// # Safety
    /// Same invocation/thread, and the helper succeeded. FP mode replacement
    /// must use `end`, not resume a segment from the old mode.
    pub unsafe fn resume(&mut self, fpcr: u32) -> Result<(), UnsupportedFpControl> {
        if self.suspended == 0 {
            return Ok(());
        }
        unsafe { self.ensure(fpcr) }
    }

    /// Finish a segment without making it resumable. Commit the returned status
    /// before replacing architectural FPCR/FPSR.
    ///
    /// # Safety
    /// Same requirements as `suspend`.
    pub unsafe fn end(&mut self) -> u32 {
        let status = unsafe { self.suspend() };
        self.suspended = 0;
        status
    }

    /// End the invocation and restore the original caller environment, even
    /// when no native FP ran or a helper exited without resuming.
    ///
    /// # Safety
    /// Same invocation/thread as `begin`; canonical software state has been
    /// written. Commit the returned FPSR before announcing epoch quiescence.
    pub unsafe fn finish(&mut self) -> u32 {
        let was_active = self.active != 0;
        let status = unsafe { self.suspend() };
        if !was_active && self.saved != 0 {
            host::restore(self.saved_control, self.saved_status);
        }
        self.saved = 0;
        self.suspended = 0;
        status
    }
}

impl NativeFrame<'_> {
    /// Save the caller FP environment at gateway entry.
    ///
    /// # Safety
    /// Remain on this OS thread until `finish_fp`; no nested FP owner.
    pub unsafe fn begin_fp(&mut self) {
        unsafe { self.host_fp.begin() };
    }

    /// Activate the guest environment lazily, after the compiler's native-FP
    /// eligibility guard. Unsupported controls leave the owner unchanged.
    ///
    /// # Safety
    /// Same invocation/thread as `begin_fp`. Canonical FPCR is current and
    /// no general Rust work runs while active.
    pub unsafe fn ensure_fp(&mut self) -> Result<(), UnsupportedFpControl> {
        unsafe { self.host_fp.ensure(*self.canonical.fpcr) }
    }

    /// Suspend before general Rust/helper work.
    ///
    /// # Safety
    /// Same invocation/thread as `begin_fp`; all observed software state,
    /// especially mapped FPSR, is canonical first.
    pub unsafe fn suspend_fp(&mut self) {
        let status = unsafe { self.host_fp.suspend() };
        unsafe { *self.canonical.fpsr |= status };
    }

    /// Resume the guest environment only on successful helper continuation.
    ///
    /// # Safety
    /// Same invocation/thread; canonical FPCR is current and unchanged in an
    /// existing segment. A mode replacement must use `end_fp`.
    pub unsafe fn resume_fp(&mut self) -> Result<(), UnsupportedFpControl> {
        unsafe { self.host_fp.resume(*self.canonical.fpcr) }
    }

    /// End the current segment before an FPCR/FPSR replacement.
    ///
    /// # Safety
    /// Same invocation/thread as `begin_fp`; mapped software FPSR is canonical.
    /// Perform the replacement only afterward.
    pub unsafe fn end_fp(&mut self) {
        let status = unsafe { self.host_fp.end() };
        unsafe { *self.canonical.fpsr |= status };
    }

    /// Complete canonical FP writeback and restore the original caller.
    /// Does not clear the execution epoch or reconcile the poll budget.
    ///
    /// # Safety
    /// Same invocation/thread as `begin_fp`; generated canonical data writeback
    /// has completed. No later write may overwrite this merged FPSR before
    /// announcing epoch quiescence.
    pub unsafe fn finish_fp(&mut self) {
        let status = unsafe { self.host_fp.finish() };
        unsafe { *self.canonical.fpsr |= status };
    }
}

#[cfg(target_arch = "x86_64")]
mod host {
    use core::arch::asm;

    const STATUS_MASK: u32 = 0x3f;
    const DENORMALS_ARE_ZERO: u32 = 1 << 6;
    const EXCEPTION_MASKS: u32 = 0x1f80;
    const ROUNDING_MASK: u32 = 3 << 13;
    const FLUSH_TO_ZERO: u32 = 1 << 15;

    pub(super) fn read() -> (u64, u64) {
        let mxcsr = read_mxcsr();
        (
            u64::from(mxcsr & !STATUS_MASK),
            u64::from(mxcsr & STATUS_MASK),
        )
    }

    pub(super) fn install_guest(fpcr: u32) {
        let current = read_mxcsr();
        let arm_rounding = (fpcr >> 22) & 3;
        let host_rounding = match arm_rounding {
            0 => 0,
            1 => 2,
            2 => 1,
            3 => 3,
            _ => unreachable!(),
        };
        let flush = if fpcr & (1 << 24) != 0 {
            DENORMALS_ARE_ZERO | FLUSH_TO_ZERO
        } else {
            0
        };
        let guest = (current & !(STATUS_MASK | DENORMALS_ARE_ZERO | ROUNDING_MASK | FLUSH_TO_ZERO))
            | EXCEPTION_MASKS
            | (host_rounding << 13)
            | flush;
        write_mxcsr(guest);
    }

    pub(super) fn guest_status() -> u32 {
        let status = read_mxcsr() & STATUS_MASK;
        // x86: IE, DE, ZE, OE, UE, PE. Arm: IOC, DZC, OFC, UFC, IXC, IDC.
        (status & 1)
            | ((status & (1 << 2)) >> 1)
            | ((status & (1 << 3)) >> 1)
            | ((status & (1 << 4)) >> 1)
            | ((status & (1 << 5)) >> 1)
            | ((status & (1 << 1)) << 6)
    }

    pub(super) fn restore(control: u64, status: u64) {
        write_mxcsr((control | status) as u32);
    }

    fn read_mxcsr() -> u32 {
        let mut value = 0_u32;
        unsafe { asm!("stmxcsr [{value}]", value = in(reg) &mut value, options(nostack)) };
        value
    }

    fn write_mxcsr(value: u32) {
        unsafe { asm!("ldmxcsr [{value}]", value = in(reg) &value, options(nostack)) };
    }
}

#[cfg(target_arch = "aarch64")]
mod host {
    use core::arch::asm;

    const GUEST_STATUS_MASK: u64 = 0x0800_009f;

    pub(super) fn read() -> (u64, u64) {
        let control: u64;
        let status: u64;
        unsafe {
            asm!("mrs {control}, fpcr", control = out(reg) control, options(nomem, nostack));
            asm!("mrs {status}, fpsr", status = out(reg) status, options(nomem, nostack));
        }
        (control, status)
    }

    pub(super) fn install_guest(fpcr: u32) {
        let guest = u64::from(fpcr & crate::fp_policy::NATIVE_FPCR_MASK);
        unsafe {
            asm!("msr fpcr, {guest}", guest = in(reg) guest, options(nomem, nostack));
            asm!("msr fpsr, xzr", options(nomem, nostack));
        }
    }

    pub(super) fn guest_status() -> u32 {
        let status: u64;
        unsafe { asm!("mrs {status}, fpsr", status = out(reg) status, options(nomem, nostack)) };
        (status & GUEST_STATUS_MASK) as u32
    }

    pub(super) fn restore(control: u64, status: u64) {
        unsafe {
            asm!("msr fpcr, {control}", control = in(reg) control, options(nomem, nostack));
            asm!("msr fpsr, {status}", status = in(reg) status, options(nomem, nostack));
        }
    }
}
