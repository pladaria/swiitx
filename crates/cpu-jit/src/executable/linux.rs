//! Linux shared backing and W^X views. These are host operations, independent
//! of guest-memory mappings. All permission changes are cache-lock serialized.

use super::{Error, WINDOW_BYTES};
use std::{
    ffi::c_void,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    ptr::NonNull,
};

pub(super) struct Mapping {
    pub base: NonNull<u8>,
}
// The owner moves between compiler threads; mapping mutation is serialized by
// the executable cache, and leases keep the RX reservation alive while used.
unsafe impl Send for Mapping {}
impl Mapping {
    fn new(fd: &OwnedFd) -> Result<Self, Error> {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                WINDOW_BYTES,
                libc::PROT_NONE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(Error::host("reserve executable alias"));
        }
        Ok(Self {
            base: NonNull::new(ptr.cast()).expect("Linux mmap returned address zero"),
        })
    }
    pub fn protect(&self, offset: usize, bytes: usize, protection: i32) -> Result<(), Error> {
        if unsafe { libc::mprotect(self.base.as_ptr().add(offset).cast(), bytes, protection) } != 0
        {
            return Err(Error::host("change executable alias permissions"));
        }
        Ok(())
    }
}
impl Drop for Mapping {
    fn drop(&mut self) {
        if unsafe { libc::munmap(self.base.as_ptr().cast(), WINDOW_BYTES) } != 0 {
            // An owner cannot report successful teardown while retaining a
            // callable mapping. Valid owned ranges make this an OS failure.
            eprintln!(
                "fatal JIT alias unmap failure: {}",
                std::io::Error::last_os_error()
            );
            std::process::abort();
        }
    }
}

pub(super) struct Backing {
    pub rx: Mapping,
    pub rw: Option<Mapping>,
    pub fd: OwnedFd,
}
impl Backing {
    pub fn new() -> Result<Self, Error> {
        // Sparse memfd: reservation is not committed capacity. fallocate below
        // commits whole segments; PUNCH_HOLE releases their real backing.
        // https://man7.org/linux/man-pages/man2/memfd_create.2.html
        // https://man7.org/linux/man-pages/man2/fallocate.2.html
        let fd = unsafe { libc::memfd_create(c"nixe-jit".as_ptr(), libc::MFD_CLOEXEC) };
        if fd < 0 {
            return Err(Error::host("create executable memfd"));
        }
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        if unsafe { libc::ftruncate(fd.as_raw_fd(), WINDOW_BYTES as libc::off_t) } != 0 {
            return Err(Error::host("size executable memfd"));
        }
        Ok(Self {
            rx: Mapping::new(&fd)?,
            rw: Some(Mapping::new(&fd)?),
            fd,
        })
    }
    pub fn commit(&self, offset: usize, bytes: usize) -> Result<(), Error> {
        if unsafe {
            libc::fallocate(
                self.fd.as_raw_fd(),
                0,
                offset as libc::off_t,
                bytes as libc::off_t,
            )
        } != 0
        {
            return Err(Error::host("commit executable segment backing"));
        }
        Ok(())
    }
    pub fn decommit(&self, offset: usize, bytes: usize) -> Result<(), Error> {
        self.rx.protect(offset, bytes, libc::PROT_NONE)?;
        if unsafe {
            libc::fallocate(
                self.fd.as_raw_fd(),
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                offset as libc::off_t,
                bytes as libc::off_t,
            )
        } != 0
        {
            return Err(Error::host("release executable segment backing"));
        }
        Ok(())
    }
}

/// Clean the writable alias to PoU, then invalidate the executable alias.
/// This must precede cross-thread pipeline synchronization and publication.
#[cfg(target_arch = "aarch64")]
pub(super) unsafe fn synchronize(rw: *mut u8, rx: *const u8, bytes: usize) {
    use core::arch::asm;
    // Same CTR_EL0-derived line sizes / DC CVAU, DSB, IC IVAU, DSB, ISB
    // sequence as GCC's aarch64 sync-cache.c, verified against local libgcc.
    // The two addresses differ here because Nixe uses dual aliases.
    // Arm's implementation and CTR_EL0.IDC/DIC rules:
    // https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/caches-self-modifying-code-implementing-clear-cache
    // https://gcc.gnu.org/git/?p=gcc.git;a=blob;f=libgcc/config/aarch64/sync-cache.c
    let ctr: u64;
    unsafe {
        asm!("mrs {ctr}, ctr_el0", ctr = out(reg) ctr, options(nostack, preserves_flags));
    }
    let dline = 4usize << ((ctr >> 16) & 15);
    let iline = 4usize << (ctr & 15);
    let mut address = (rw as usize) & !(dline - 1);
    while ctr & (1 << 28) == 0 && address < rw as usize + bytes {
        unsafe {
            asm!("dc cvau, {address}", address = in(reg) address, options(nostack, preserves_flags));
        }
        address += dline;
    }
    unsafe {
        asm!("dsb ish", options(nostack, preserves_flags));
    }
    address = (rx as usize) & !(iline - 1);
    while ctr & (1 << 29) == 0 && address < rx as usize + bytes {
        unsafe {
            asm!("ic ivau, {address}", address = in(reg) address, options(nostack, preserves_flags));
        }
        address += iline;
    }
    unsafe {
        asm!("dsb ish", "isb", options(nostack, preserves_flags));
    }
}
#[cfg(target_arch = "x86_64")]
pub(super) unsafe fn synchronize(_rw: *mut u8, _rx: *const u8, _bytes: usize) {
    // Coherent x86 instruction/data caches; no live instruction is overwritten
    // here. Release publication follows the finalized stores.
    // https://github.com/pladaria/wasmtime/blob/e2a984d96678207094c0fc50057c8b6bcfd68715/crates/jit-icache-coherence/src/libc.rs
}

pub(super) unsafe fn copy(rw: *mut u8, rx: *const u8, bytes: &[u8]) {
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), rw, bytes.len());
        synchronize(rw, rx, bytes.len());
    }
}

const _: () = assert!(std::mem::size_of::<*const c_void>() == 8);
