//! Bounded Linux executable storage. Allocation/population use only the cache
//! mutex, never JIT state. A lease owns an exact span; published CodeUnit owners
//! must retain it through unlink, reader quiescence and all strong references.

mod linux;
pub(crate) mod output;
#[cfg(test)]
mod tests;

use crate::abi::{CheckedCounter, SegmentGeneration};
use cranelift_codegen::binemit::Reloc;
use linux::Backing;
use output::{Metadata, Output, Target};
use std::sync::{Arc, Mutex, MutexGuard};

const MIB: usize = 1024 * 1024;
pub(crate) const WINDOW_BYTES: usize = 2047 * MIB;
pub(crate) const SEGMENT_BYTES: usize = 16 * MIB;
pub(crate) const SEGMENTS: usize = 128;
const ISLAND_BYTES: usize = 64 * 1024;
pub(crate) const SOFT_BYTES: usize = 512 * MIB;
pub(crate) const HARD_BYTES: usize = 640 * MIB;
const LCQ_RESERVE: usize = 32 * MIB;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tier {
    Lcq,
    Hcq,
}

#[derive(Debug)]
pub(crate) enum Error {
    Host {
        operation: &'static str,
        error: std::io::Error,
    },
    Capacity(&'static str),
    Output(String),
    Relocation {
        offset: u32,
        kind: Reloc,
        detail: &'static str,
    },
    Poisoned,
    Closed,
}
impl Error {
    fn host(operation: &'static str) -> Self {
        Self::Host {
            operation,
            error: std::io::Error::last_os_error(),
        }
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => f.write_str("JIT executable cache is closed"),
            Self::Host { operation, error } => write!(f, "JIT {operation}: {error}"),
            Self::Capacity(detail) => write!(f, "JIT executable capacity: {detail}"),
            Self::Output(detail) => write!(f, "JIT staged output: {detail}"),
            Self::Relocation {
                offset,
                kind,
                detail,
            } => write!(f, "JIT {kind:?} relocation at {offset:#x}: {detail}"),
            Self::Poisoned => f.write_str(
                "JIT executable cache failed; further allocation/publication is disabled",
            ),
        }
    }
}
impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Usage {
    pub committed: usize,
    pub metadata: usize,
}
impl Usage {
    pub fn total(self) -> usize {
        self.committed + self.metadata
    }
    pub fn needs_reclamation(self) -> bool {
        self.total() >= SOFT_BYTES
    }
    pub(crate) fn check(self, bytes: usize, tier: Tier) -> Result<(), Error> {
        let total = self
            .total()
            .checked_add(bytes)
            .ok_or(Error::Capacity("byte count overflow"))?;
        if total > HARD_BYTES {
            return Err(Error::Capacity("640 MiB code+metadata hard limit"));
        }
        if tier == Tier::Hcq {
            if total > HARD_BYTES - LCQ_RESERVE {
                return Err(Error::Capacity("32 MiB LCQ reserve is unavailable to HCQ"));
            }
            if total > SOFT_BYTES {
                return Err(Error::Capacity("HCQ requires soft-limit reclamation"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Span {
    start: usize,
    len: usize,
}
impl Span {
    fn end(self) -> usize {
        self.start + self.len
    }
}

#[derive(Default)]
struct Segment {
    generation: Option<SegmentGeneration>,
    tier: Option<Tier>,
    bump: usize,
    live: usize,
    // Sorted, coalesced ranges. Capacity is charged, including replacement
    // overlap; fixed boxed storage makes that extent exact. No release allocates.
    free: Box<[Span]>,
    free_len: usize,
}
impl Segment {
    fn remove_free(&mut self, index: usize) -> Span {
        let span = self.free[index];
        self.free.copy_within(index + 1..self.free_len, index);
        self.free_len -= 1;
        span
    }
    fn insert_free(&mut self, span: Span) {
        if span.len == 0 {
            return;
        }
        let mut index = self.free[..self.free_len].partition_point(|free| free.start < span.start);
        assert!(
            self.free_len < self.free.len(),
            "allocation reserved release metadata"
        );
        self.free.copy_within(index..self.free_len, index + 1);
        self.free[index] = span;
        self.free_len += 1;
        if index > 0 && self.free[index - 1].end() == self.free[index].start {
            let right = self.remove_free(index);
            index -= 1;
            self.free[index].len += right.len;
        }
        if index + 1 < self.free_len && self.free[index].end() == self.free[index + 1].start {
            let right = self.remove_free(index + 1);
            self.free[index].len += right.len;
        }
    }
    fn release(&mut self, span: Span) {
        self.insert_free(span);
        self.live -= 1;
        if self.free_len != 0 && self.free[self.free_len - 1].end() == self.bump {
            self.bump = self.free[self.free_len - 1].start;
            self.free_len -= 1;
        }
    }
}

struct State {
    backing: Option<Backing>,
    segments: [Segment; SEGMENTS],
    generations: CheckedCounter<SegmentGeneration>,
    usage: Usage,
    failed: bool,
}
impl State {
    fn ensure_free_storage(&mut self, index: usize, tier: Tier) -> Result<(), Error> {
        let segment = &mut self.segments[index];
        let needed = (segment.live + 2).max(segment.free_len + 2);
        if segment.free.len() >= needed {
            return Ok(());
        }
        let count = needed.max(segment.free.len() * 2).max(8);
        let bytes = count
            .checked_mul(std::mem::size_of::<Span>())
            .ok_or(Error::Capacity("free span metadata overflow"))?;
        self.usage.check(bytes, tier)?; // Includes old/new allocation overlap.
        let mut replacement = vec![Span::default(); count].into_boxed_slice();
        replacement[..segment.free_len].copy_from_slice(&segment.free[..segment.free_len]);
        self.usage.metadata += bytes;
        let old = std::mem::replace(&mut segment.free, replacement);
        let old_bytes = std::mem::size_of_val(&*old);
        drop(old);
        self.usage.metadata -= old_bytes;
        Ok(())
    }
}

pub(crate) struct Cache {
    state: Mutex<State>,
    base: usize,
}
impl Cache {
    pub fn new() -> Result<Arc<Self>, Error> {
        let backing = Backing::new()?;
        let base = backing.rx.base.as_ptr() as usize;
        Ok(Arc::new(Self {
            base,
            state: Mutex::new(State {
                backing: Some(backing),
                segments: std::array::from_fn(|_| Segment::default()),
                generations: CheckedCounter::default(),
                usage: Usage {
                    committed: 0,
                    metadata: std::mem::size_of::<Self>() + 2 * std::mem::size_of::<usize>(),
                },
                failed: false,
            }),
        }))
    }
    fn lock(&self) -> Result<MutexGuard<'_, State>, Error> {
        let state = self.state.lock().map_err(|_| Error::Poisoned)?;
        if state.failed {
            return Err(Error::Poisoned);
        }
        Ok(state)
    }
    pub fn usage(&self) -> Result<Usage, Error> {
        Ok(self.lock()?.usage)
    }

    pub(crate) fn account<T>(
        self: &Arc<Self>,
        value: T,
        bytes: usize,
        tier: Tier,
    ) -> Result<Accounted<T>, Error> {
        Ok(Accounted {
            value,
            charge: self.charge_metadata(bytes, tier)?,
        })
    }
    pub fn executable_base(&self) -> usize {
        self.base
    }

    /// Bounds first: the final slot represents 15 MiB, not a full 16 MiB.
    pub fn segment_for_pc(&self, pc: usize) -> Option<usize> {
        let offset = pc.checked_sub(self.base)?;
        (offset < WINDOW_BYTES).then_some(offset / SEGMENT_BYTES)
    }

    /// Charge actual metadata storage before transferring it into cache-owned
    /// live/retired records. Replacement keeps both charges until old storage
    /// is freed. Registry/directory owners use this same budget in later steps.
    pub fn charge_metadata(
        self: &Arc<Self>,
        bytes: usize,
        tier: Tier,
    ) -> Result<MetadataLease, Error> {
        let mut state = self.lock()?;
        state.usage.check(bytes, tier)?;
        if state.backing.is_none() {
            return Err(Error::Closed);
        }
        state.usage.metadata += bytes;
        Ok(MetadataLease {
            cache: Arc::clone(self),
            bytes,
        })
    }

    fn allocate(
        self: &Arc<Self>,
        size: usize,
        alignment: usize,
        tier: Tier,
    ) -> Result<Allocation, Error> {
        if size == 0 || size > SEGMENT_BYTES - ISLAND_BYTES {
            return Err(Error::Capacity(
                "unit is empty or exceeds one segment's code area",
            ));
        }
        if !alignment.is_power_of_two() || alignment > SEGMENT_BYTES {
            return Err(Error::Capacity("invalid code alignment"));
        }
        let mut state = self.lock()?;
        if state.backing.is_none() {
            return Err(Error::Closed);
        }
        // Free-span policy is best fit, then lowest RX address. A tier can
        // borrow an unused segment, never another tier's live active segment.
        let mut best: Option<(usize, usize, usize, usize)> = None;
        for (index, segment) in state.segments.iter().enumerate() {
            if segment.live != 0 && segment.tier != Some(tier) {
                continue;
            }
            for (free_index, free) in segment.free[..segment.free_len].iter().enumerate() {
                let start = align(self.base + index * SEGMENT_BYTES + free.start, alignment)?
                    - self.base
                    - index * SEGMENT_BYTES;
                if start + size <= free.end() {
                    let candidate = (free.len, index, start, free_index);
                    if best.is_none_or(|old| candidate < old) {
                        best = Some(candidate);
                    }
                }
            }
        }
        let (index, start, free_index) = if let Some((_, index, start, free_index)) = best {
            (index, start, Some(free_index))
        } else {
            let mut candidate = None;
            // Prefer already committed tier/unused segments to new backing.
            for committed in [true, false] {
                for (index, segment) in state.segments.iter().enumerate() {
                    if segment.generation.is_some() != committed
                        || (segment.live != 0 && segment.tier != Some(tier))
                    {
                        continue;
                    }
                    let start = align(self.base + index * SEGMENT_BYTES + segment.bump, alignment)?
                        - self.base
                        - index * SEGMENT_BYTES;
                    if start + size <= segment_size(index) - ISLAND_BYTES {
                        candidate = Some((index, start, None));
                        break;
                    }
                }
                if candidate.is_some() {
                    break;
                }
            }
            candidate.ok_or(Error::Capacity(
                "2047 MiB executable window has no fitting span",
            ))?
        };
        state.ensure_free_storage(index, tier)?;
        if state.segments[index].generation.is_none() {
            let bytes = segment_size(index);
            state.usage.check(bytes, tier)?;
            let generation = state
                .generations
                .next_id()
                .map_err(|_| Error::Capacity("segment generation exhausted"))?;
            state.usage.committed += bytes;
            if let Err(error) = state
                .backing
                .as_ref()
                .unwrap()
                .commit(index * SEGMENT_BYTES, bytes)
            {
                // fallocate can fail after partial allocation. Refund only
                // after releasing that unpublished backing, not on errno alone.
                if state
                    .backing
                    .as_ref()
                    .unwrap()
                    .decommit(index * SEGMENT_BYTES, bytes)
                    .is_ok()
                {
                    state.usage.committed -= bytes;
                } else {
                    state.failed = true;
                }
                return Err(error);
            }
            state.segments[index].generation = Some(generation);
            // Failure after commitment keeps the charge and disables use; it
            // never reports backing released when only permissions changed.
            if let Err(error) = state.backing.as_ref().unwrap().rx.protect(
                index * SEGMENT_BYTES,
                bytes,
                libc::PROT_READ | libc::PROT_EXEC,
            ) {
                state.failed = true;
                return Err(error);
            }
        }
        let segment = &mut state.segments[index];
        if let Some(free_index) = free_index {
            let free = segment.remove_free(free_index);
            segment.insert_free(Span {
                start: free.start,
                len: start - free.start,
            });
            segment.insert_free(Span {
                start: start + size,
                len: free.end() - start - size,
            });
        } else {
            segment.insert_free(Span {
                start: segment.bump,
                len: start - segment.bump,
            });
            segment.bump = start + size;
        }
        segment.live += 1;
        segment.tier = Some(tier);
        Ok(Allocation {
            cache: Arc::clone(self),
            tier,
            segment: index,
            generation: segment.generation.unwrap(),
            span: Span { start, len: size },
        })
    }

    pub fn install(
        self: &Arc<Self>,
        output: Output,
        tier: Tier,
        resolve: impl FnMut(&Target) -> Option<usize>,
    ) -> Result<Installed, Error> {
        if !output.alignment.is_power_of_two() {
            return Err(Error::Capacity("invalid code alignment"));
        }
        let metadata_bytes = output.metadata.bytes() - std::mem::size_of::<Metadata>()
            + std::mem::size_of::<Installed>();
        // Staging bytes overlap the actual executable backing during transfer.
        let charge = self.charge_metadata(metadata_bytes + output.bytes.len(), tier)?;
        // On every error, destroy storage before releasing its budget. Locals
        // alone would drop the later charge before the earlier output argument.
        struct Staging {
            output: Output,
            charge: MetadataLease,
        }
        let mut staging = Staging { output, charge };
        let allocation = self.allocate(
            staging.output.bytes.len(),
            staging.output.alignment.max(16),
            tier,
        )?;
        staging.output.relocate(allocation.address(), resolve)?; // No cache/JIT lock.
        {
            let mut state = self.lock()?;
            let offset = allocation.segment * SEGMENT_BYTES;
            let rw = state
                .backing
                .as_ref()
                .ok_or(Error::Closed)?
                .rw
                .as_ref()
                .ok_or(Error::Poisoned)?;
            rw.protect(
                offset,
                segment_size(allocation.segment),
                libc::PROT_READ | libc::PROT_WRITE,
            )?;
            unsafe {
                linux::copy(
                    rw.base.as_ptr().add(offset + allocation.span.start),
                    allocation.address() as *const u8,
                    &staging.output.bytes,
                );
            }
            if let Err(error) =
                rw.protect(offset, segment_size(allocation.segment), libc::PROT_NONE)
            {
                // Disable the entire nonexecutable view, rather than leave it
                // accessible outside a write window. RX leases stay intact.
                state.failed = true;
                drop(state.backing.as_mut().unwrap().rw.take());
                return Err(error);
            }
        }
        // Reused addresses may have been fetched by a different core. Broadcast
        // pipeline synchronization is the fork's existing Linux implementation.
        wasmtime_internal_jit_icache_coherence::pipeline_flush_mt().map_err(|error| {
            Error::Host {
                operation: "synchronize executable pipelines",
                error,
            }
        })?;
        let Staging { output, mut charge } = staging;
        let staging_bytes = output.bytes.len();
        drop(output.bytes);
        charge.reduce(staging_bytes);
        Ok(Installed {
            allocation,
            metadata: output.metadata,
            charge,
        })
    }

    /// # Safety
    /// For a previously published segment the owner must first publish a null
    /// native-PC directory slot and complete its snapshot grace period. No
    /// callable root, reader or compiler/link/fault reference may remain.
    pub unsafe fn decommit_empty(&self, index: usize) -> Result<bool, Error> {
        if index >= SEGMENTS {
            return Err(Error::Capacity("segment index outside reservation"));
        }
        let mut state = self.lock()?;
        let segment = &state.segments[index];
        if segment.live != 0 {
            return Ok(false);
        }
        if segment.generation.is_some() {
            if let Err(error) = state
                .backing
                .as_ref()
                .unwrap()
                .decommit(index * SEGMENT_BYTES, segment_size(index))
            {
                state.failed = true;
                return Err(error);
            }
            state.usage.committed -= segment_size(index);
        }
        let old = std::mem::take(&mut state.segments[index]);
        let metadata = std::mem::size_of_val(&*old.free);
        drop(old);
        state.usage.metadata -= metadata;
        Ok(true)
    }

    /// # Safety
    /// Called only after the publication owner has drained all directory
    /// readers. Staged or externally retained allocations postpone unmapping.
    pub unsafe fn try_close(&self) -> Result<bool, Error> {
        let mut state = self.lock()?;
        if state.segments.iter().any(|segment| segment.live != 0) {
            return Ok(false);
        }
        drop(state.backing.take());
        state.usage.committed = 0;
        for index in 0..SEGMENTS {
            let old = std::mem::take(&mut state.segments[index]);
            let bytes = std::mem::size_of_val(&*old.free);
            drop(old);
            state.usage.metadata -= bytes;
        }
        Ok(true)
    }
}

fn align(address: usize, alignment: usize) -> Result<usize, Error> {
    address
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(Error::Capacity("aligned address overflow"))
}
fn segment_size(index: usize) -> usize {
    (WINDOW_BYTES - index * SEGMENT_BYTES).min(SEGMENT_BYTES)
}

pub(crate) struct Allocation {
    cache: Arc<Cache>,
    pub tier: Tier,
    pub segment: usize,
    pub generation: SegmentGeneration,
    span: Span,
}
impl Allocation {
    pub fn belongs_to(&self, cache: &Arc<Cache>) -> bool {
        Arc::ptr_eq(&self.cache, cache)
    }
    pub fn address(&self) -> usize {
        self.cache.base + self.segment * SEGMENT_BYTES + self.span.start
    }
    pub fn len(&self) -> usize {
        self.span.len
    }
}
impl Drop for Allocation {
    fn drop(&mut self) {
        if let Ok(mut state) = self.cache.lock() {
            let segment = &mut state.segments[self.segment];
            assert_eq!(segment.generation, Some(self.generation));
            segment.release(self.span);
        }
    }
}

pub(crate) struct MetadataLease {
    cache: Arc<Cache>,
    bytes: usize,
}

/// Coupled storage/charge ownership: all error and retirement paths destroy
/// the actual value before returning its budget, including replaced indexes.
pub(crate) struct Accounted<T> {
    pub value: T,
    pub charge: MetadataLease,
}
impl<T> std::ops::Deref for Accounted<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}
impl MetadataLease {
    fn reduce(&mut self, bytes: usize) {
        assert!(bytes <= self.bytes);
        if let Ok(mut state) = self.cache.lock() {
            state.usage.metadata -= bytes;
        }
        self.bytes -= bytes;
    }
}
impl Drop for MetadataLease {
    fn drop(&mut self) {
        self.reduce(self.bytes);
    }
}

pub(crate) struct Installed {
    pub allocation: Allocation,
    pub metadata: Metadata,
    // Field order releases the actual metadata before returning its charge.
    charge: MetadataLease,
}
