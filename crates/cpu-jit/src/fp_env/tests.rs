use super::*;
use crate::abi::PollBudget;
use nixe_cpu::state::a64::A64State;

// Restore even on a failed assertion; host FP state belongs to the test thread.
pub(crate) struct RestoreHost((u64, u64));
impl RestoreHost {
    pub(crate) fn new() -> Self {
        Self(host::read())
    }
}
impl Drop for RestoreHost {
    fn drop(&mut self) {
        host::restore(self.0.0, self.0.1);
    }
}

fn distinct_caller() -> (u64, u64) {
    let original = host::read();
    #[cfg(target_arch = "x86_64")]
    let caller = ((original.0 & !(3 << 13)) | (2 << 13) | 0x1f80, 0x21);
    #[cfg(target_arch = "aarch64")]
    let caller = ((original.0 & !(3 << 22)) | (1 << 22), 0x81);
    host::restore(caller.0, caller.1);
    host::read()
}

/// Execute real status-producing arithmetic in the active guest environment.
#[inline(always)]
pub(crate) unsafe fn divide_by_zero() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::asm!("divsd {a}, {b}", a = inout(xmm_reg) 1.0f64 => _, b = in(xmm_reg) 0.0f64, options(nostack, preserves_flags));
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!("fdiv {a:d}, {a:d}, {b:d}", a = inout(vreg) 1.0f64 => _, b = in(vreg) 0.0f64, options(nostack, preserves_flags));
    }
}

#[test]
fn caller_is_saved_at_entry_even_without_native_fp() {
    let _restore = RestoreHost::new();
    let caller = distinct_caller();
    let mut owner = HostFpState::default();
    unsafe {
        owner.begin();
    }
    assert_eq!((owner.saved_control, owner.saved_status), caller);
    assert_eq!(owner.active, 0);
    // A helper does not get to replace the invocation's original caller save.
    host::restore(caller.0, 0);
    let contribution = unsafe { owner.finish() };
    assert_eq!(host::read(), caller);
    assert_eq!(contribution, 0);
    assert_eq!((owner.active, owner.saved, owner.suspended), (0, 0, 0));
}

#[test]
fn compatible_segments_keep_status_and_helpers_restore_the_caller() {
    let _restore = RestoreHost::new();
    let caller = distinct_caller();
    let mut owner = HostFpState::default();
    // SAFETY: thread-local owner, integer-only management around explicit FP,
    // with the caller restored before assertions/general Rust work.
    let (first, second, repeated_suspend) = unsafe {
        owner.begin();
        owner.ensure(0).unwrap();
        divide_by_zero();
        owner.ensure(0).unwrap(); // Must not clear the pending divide-by-zero.
        let first = owner.suspend();
        assert_eq!(host::read(), caller);
        let repeated_suspend = owner.suspend();
        host::restore(caller.0, 0); // Helper-local status cannot enter the guest.
        owner.resume(0).unwrap();
        let clean_status = host::guest_status();
        divide_by_zero();
        let second = owner.finish();
        assert_eq!(clean_status, 0);
        (first, second, repeated_suspend)
    };
    assert_eq!(host::read(), caller);
    assert_eq!((first, second, repeated_suspend), (2, 2, 0));
}

#[test]
fn native_frame_ends_old_fp_segment_before_architectural_replacement() {
    let _restore = RestoreHost::new();
    let caller = distinct_caller();
    let mut state = A64State::default();
    state.set_fpsr(1 << 27);
    {
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(77, 1000).unwrap());
        frame.execution_epoch = 7;
        let (old_fpsr, resumed, new_status) = unsafe {
            frame.begin_fp();
            frame.ensure_fp().unwrap();
            divide_by_zero();
            frame.end_fp();
            let old_fpsr = *frame.canonical.fpsr;
            *frame.canonical.fpsr = 0;
            *frame.canonical.fpcr = 1 << 22;
            frame.resume_fp().unwrap(); // Ended segments are not resumable.
            let resumed = frame.host_fp.active;
            frame.ensure_fp().unwrap();
            let new_status = host::guest_status();
            frame.finish_fp();
            (old_fpsr, resumed, new_status)
        };
        assert_eq!(host::read(), caller);
        assert_eq!(old_fpsr, (1 << 27) | 2);
        assert_eq!((resumed, new_status), (0, 0));
        assert_eq!(
            frame.execution_epoch, 7,
            "FP completion cannot announce quiescence"
        );
    }
    assert_eq!(state.fpsr(), 0);
    assert_eq!(state.fpcr(), 1 << 22);
}

#[test]
fn unsupported_fpcr_does_not_install_a_masked_native_mode() {
    let _restore = RestoreHost::new();
    let caller = distinct_caller();
    let mut owner = HostFpState::default();
    unsafe {
        owner.begin();
    }
    assert_eq!(
        unsafe { owner.ensure(1 << 8) },
        Err(UnsupportedFpControl(1 << 8))
    );
    assert_eq!(host::read(), caller);
    assert_eq!(owner.active, 0);
    assert_eq!(unsafe { owner.finish() }, 0);
    assert_eq!(host::read(), caller);
}
