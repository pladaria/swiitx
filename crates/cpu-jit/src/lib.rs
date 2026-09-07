//! Concrete Cranelift JIT backend.
//!
//! Normalized A64 instructions lower directly to CLIF. The execution frame,
//! compiler, native lookup table, and slow paths remain private to the backend.
//! Both tiers share the architectural analysis and native ABI contracts.

#[cfg(not(target_os = "linux"))]
compile_error!("nixe-cpu-jit requires Linux direct-memory support");

pub mod abi;
pub mod analysis;
mod direct;
mod fp_env;
mod fp_policy;
// Task 2 builds this owner before Task 3 replaces the legacy production path.
#[allow(dead_code)]
mod lifetime;
// Storage is assembled into published CodeUnits in Task 2, then used by Task 3.
#[allow(dead_code)]
mod executable;
pub mod native;

pub use direct::{JitProcess, JitThread};
