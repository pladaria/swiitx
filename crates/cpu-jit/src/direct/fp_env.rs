//! Current compiler's typed FP veneers. Ownership and host instructions live
//! in the shared HostFpState implementation used by NativeFrame as well.

use super::NativeContext;
pub(super) use crate::abi::HostFpState;

/// # Safety
/// Live native invocation on this OS thread, with no nested FP owner.
pub(super) unsafe fn begin(context: &mut NativeContext) {
    unsafe { context.host_fp.begin() };
}

/// # Safety
/// Canonical software state is committed; same invocation/thread as begin.
pub(super) unsafe fn finish(context: &mut NativeContext) {
    let status = unsafe { context.host_fp.finish() };
    unsafe { *context.fpsr |= status };
}

pub(super) unsafe extern "C" fn ensure(context: *mut NativeContext) {
    let Some(context) = (unsafe { context.as_mut() }) else {
        return;
    };
    // Generated eligibility guards must reject unsupported controls first.
    unsafe { context.host_fp.ensure(context.guest_fpcr) }
        .expect("invalid guarded native FP activation");
}

pub(super) unsafe extern "C" fn suspend(context: *mut NativeContext) {
    let Some(context) = (unsafe { context.as_mut() }) else {
        return;
    };
    let status = unsafe { context.host_fp.suspend() };
    unsafe { *context.fpsr |= status };
}

pub(super) unsafe extern "C" fn end(context: *mut NativeContext) {
    let Some(context) = (unsafe { context.as_mut() }) else {
        return;
    };
    let status = unsafe { context.host_fp.end() };
    unsafe { *context.fpsr |= status };
}

pub(super) unsafe extern "C" fn resume(context: *mut NativeContext) {
    let Some(context) = (unsafe { context.as_mut() }) else {
        return;
    };
    unsafe { context.host_fp.resume(context.guest_fpcr) }
        .expect("invalid guarded native FP resumption");
}
