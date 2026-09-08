//! Signal-time attribution. Ownership and all mutations belong to JIT state;
//! a reader MUST already protect an invocation epoch before touching a slot.

use super::unit::{CodeUnit, FaultRecord};
use crate::abi::SegmentGeneration;
use crate::executable::{Accounted, SEGMENT_BYTES, SEGMENTS, WINDOW_BYTES};
use std::sync::atomic::{AtomicPtr, Ordering};

#[derive(Clone, Copy)]
pub(super) struct Interval {
    pub start: usize,
    pub end: usize,
    pub unit: *const CodeUnit,
    pub fault: usize,
}
// Raw pointers name immutable units retained by the unit registry. Copying an
// interval during table preparation never dereferences it. Dereferencing is
// restricted to lookup with an already announced epoch; removal must complete
// the table's grace period before releasing any unit it names.
unsafe impl Send for Interval {}
unsafe impl Sync for Interval {}

pub(super) struct Table {
    pub generation: SegmentGeneration,
    pub intervals: Vec<Interval>,
}

pub(super) struct Directory {
    base: usize,
    slots: [AtomicPtr<Accounted<Table>>; SEGMENTS],
}
impl Directory {
    pub fn new(base: usize) -> Self {
        Self {
            base,
            slots: std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut())),
        }
    }

    /// State owns both the new table and every replaced snapshot. Call only
    /// after all metadata owners are installed and before exposing dispatch.
    pub unsafe fn publish(&self, segment: usize, table: *const Accounted<Table>) {
        self.slots[segment].store(table.cast_mut(), Ordering::Release);
    }

    /// # Safety
    /// Caller already announced an execution epoch under this process's state
    /// mutex. Keep it active through the last use of the returned record,
    /// including normal-stack fault resolution/retry. Never announce only after
    /// reading a pointer. The directory and process must outlive the caller.
    ///
    /// Bounds plus one atomic load and a binary search: no mutable registry,
    /// reference counting, allocation, lock, or unbounded signal-time scan.
    pub unsafe fn lookup(&self, pc: usize) -> Option<Fault<'_>> {
        let offset = pc.checked_sub(self.base)?;
        if offset >= WINDOW_BYTES {
            return None;
        }
        let pointer = self.slots[offset / SEGMENT_BYTES].load(Ordering::Acquire);
        let table = unsafe { pointer.as_ref()? };
        let index = table
            .intervals
            .partition_point(|interval| interval.start <= pc);
        let interval = table.intervals.get(index.checked_sub(1)?)?;
        if pc >= interval.end {
            return None;
        }
        let unit = unsafe { &*interval.unit };
        // The generation is published in the same immutable table as its
        // intervals, never in an independently observed atomic word.
        if unit.code.allocation.generation != table.generation {
            return None;
        }
        Some(Fault {
            unit,
            record: &unit.faults[interval.fault],
        })
    }
}

pub(crate) struct Fault<'a> {
    pub unit: &'a CodeUnit,
    pub record: &'a FaultRecord,
}
