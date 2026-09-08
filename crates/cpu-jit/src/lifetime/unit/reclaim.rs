//! Unlinked-unit retirement and two-stage fault-directory grace periods.
//! Compiler snapshots own code; raw directory pointers never acquire ownership.

use super::*;
use crate::lifetime::{Phase, Ticket, Transition};

#[cfg(test)]
mod tests;

#[derive(Clone)]
pub(crate) struct Snapshot {
    unit: Arc<Accounted<CodeUnit>>,
}
impl std::ops::Deref for Snapshot {
    type Target = CodeUnit;
    fn deref(&self) -> &CodeUnit {
        &self.unit
    }
}

fn names(entry: Option<PublishedEntry>, unit: &CodeUnit) -> bool {
    entry.is_some_and(|entry| entry.unit == unit.id && entry.version == unit.version)
}
fn rooted(state: &crate::lifetime::State, record: &UnitRecord) -> bool {
    record.slots.iter().any(|slot| {
        let payload = state.dispatch.get(*slot).unwrap().snapshot();
        names(payload.lcq(), &record.code)
            || names(payload.hcq().map(|entry| entry.entry), &record.code)
    })
}

struct Collector<'a>(&'a Lifetime);
impl Drop for Collector<'_> {
    fn drop(&mut self) {
        self.0.lock().units.collecting = false;
        self.0.changed.notify_all();
    }
}

impl Lifetime {
    /// Acquire compiler/link ownership while the exact version is still
    /// eligible. A retired/invalidating unit cannot gain a new snapshot from
    /// an index; an existing snapshot may clone its own strong reference.
    pub(crate) fn snapshot(&self, handle: UnitHandle) -> Result<super::Snapshot, Error> {
        let state = self.lock();
        state.open()?;
        if handle.1 != self.identity {
            return Err(Error::StaleUnit);
        }
        let record = state.units.records.get(handle.0).ok_or(Error::StaleUnit)?;
        if !matches!(
            record.lifecycle,
            Lifecycle::Published | Lifecycle::Superseded
        ) {
            return Err(Error::StaleUnit);
        }
        Ok(Snapshot {
            unit: Arc::clone(&record.code),
        })
    }

    /// Register the exact target and closure under the same mutex. Snapshot
    /// references delay reclamation, not unlink; baseline promises additionally
    /// prevent LCQ eviction until their active/in-flight family releases them.
    pub(crate) fn retire_unit(&self, handle: UnitHandle) -> Result<Ticket<'_>, Error> {
        let mut state = self.lock();
        state.healthy()?;
        if handle.1 != self.identity {
            return Err(Error::StaleUnit);
        }
        let record = state.units.records.get(handle.0).ok_or(Error::StaleUnit)?;
        if !matches!(
            record.lifecycle,
            Lifecycle::Published | Lifecycle::Superseded
        ) {
            return Err(Error::StaleUnit);
        }
        if record.code.baseline_pins.load(Ordering::Relaxed) != 0 {
            return Err(Error::PinnedBaseline);
        }
        let ticket = self.request_locked(&mut state, Reason::Eviction)?;
        let record = state.units.records.get_mut(handle.0).unwrap();
        record.lifecycle = Lifecycle::Invalidating;
        record.retirement = Some((Reason::Eviction, ticket.sequence));
        Ok(ticket)
    }

    /// No waiting, allocation, or destruction under JIT state. A collector
    /// owns each removed slot until dropping code has returned its actual span.
    /// Concurrent callers leave the single cold collector to finish its scan.
    pub(crate) fn reclaim_units(&self) -> Result<usize, Error> {
        {
            let mut state = self.lock();
            state.healthy()?;
            if state.units.collecting {
                return Ok(0);
            }
            state.units.collecting = true;
        }
        let _collector = Collector(self);
        self.collect_tables()?;
        self.retire_rootless()?;
        // Detach candidates one at a time. A short-lived strong reference
        // protects their code while replacement tables are built outside state.
        loop {
            let candidate = {
                let state = self.lock();
                let handle = state.units.records.find(|record| {
                    matches!(record.lifecycle, Lifecycle::Retired(epoch) if state.quiescent(epoch))
                        && record.detached_epoch.is_none()
                        && Arc::strong_count(&record.code) == 1
                });
                handle.map(|handle| {
                    let record = state.units.records.get(handle).unwrap();
                    (
                        handle,
                        Arc::clone(&record.code),
                        state.units.tables[record.code.code.allocation.segment].clone(),
                    )
                })
            };
            let Some((handle, code, table)) = candidate else {
                break;
            };
            if !self.detach_faults(handle, &code, table)? {
                break;
            }
        }
        let mut reclaimed = 0;
        loop {
            let removed = {
                let mut state = self.lock();
                let handle = state.units.records.find(|record| {
                    record
                        .detached_epoch
                        .is_some_and(|epoch| state.quiescent(epoch))
                        && Arc::strong_count(&record.code) == 1
                });
                let Some(handle) = handle else {
                    break;
                };
                let record = state.units.records.take_held(handle).unwrap();
                for page in &*record.code.dependencies {
                    let hash = state.units.dependencies.hash.hash_one(page.page);
                    if let Ok(entry) = state.units.dependencies.entries.find_entry(hash, |entry| {
                        entry.unit == UnitHandle(handle, self.identity) && entry.page == *page
                    }) {
                        entry.remove();
                    }
                }
                (handle, record)
            };
            let (handle, record) = removed;
            let UnitRecord {
                code,
                slots,
                family,
                detached_table,
                ..
            } = record;
            drop(detached_table);
            drop(code); // Returns the span under cache only, BEFORE releasing slots.
            {
                let mut state = self.lock();
                for slot in slots.iter() {
                    state.dispatch.get_mut(*slot).unwrap().units -= 1;
                }
                assert!(state.units.records.release_held(handle));
                if let Some(family) = family {
                    assert!(state.units.families.release_held(family));
                }
            }
            drop(slots);
            reclaimed += 1;
        }
        self.collect_tables()?;
        self.collect_dispatch()?;
        self.decommit_unused()?;
        Ok(reclaimed)
    }

    fn retire_rootless(&self) -> Result<(), Error> {
        let mut state = self.lock();
        loop {
            let handle = state.units.records.find(|record| {
                record.lifecycle == Lifecycle::Superseded
                    && record.retirement.is_none()
                    && record.code.baseline_pins.load(Ordering::Relaxed) == 0
                    && !rooted(&state, record)
            });
            let Some(handle) = handle else {
                return Ok(());
            };
            let retired = state.execution;
            let result = state.executions.next_id();
            state.execution = self.checked(&mut state, result)?;
            let record = state.units.records.get_mut(handle).unwrap();
            record.lifecycle = Lifecycle::Unlinked;
            record.lifecycle = Lifecycle::Retired(retired);
        }
    }

    fn detach_faults(
        &self,
        handle: Handle<UnitRecord>,
        code: &Arc<Accounted<CodeUnit>>,
        mut previous: Option<Arc<Accounted<Table>>>,
    ) -> Result<bool, Error> {
        let segment = code.code.allocation.segment;
        let pointer = &code.value as *const CodeUnit;
        // At a Closed rendezvous, a table with no compiler-held snapshot can
        // be withdrawn, compacted in place and republished. This is required
        // for progress at the hard budget: eviction must not require new RAM.
        {
            let mut state = self.lock();
            if state.phase == Phase::Closed
                && state.idle()
                && same_snapshot(&state.units.tables[segment], &previous)
                && previous
                    .as_ref()
                    .is_some_and(|table| Arc::strong_count(table) == 2)
            {
                let retired = state.execution;
                let result = state.executions.next_id();
                let next = self.checked(&mut state, result)?;
                // All previous table readers are quiescent. The only Arcs are
                // current owner and this local snapshot; no compiler can hold it.
                drop(previous.take()); // Current owner remains: no destructor.
                unsafe {
                    self.directory.publish(segment, std::ptr::null());
                }
                let table = state.units.tables[segment].as_mut().unwrap();
                Arc::get_mut(table)
                    .unwrap()
                    .value
                    .intervals
                    .retain(|interval| interval.unit != pointer);
                let empty = table.intervals.is_empty();
                if !empty {
                    unsafe {
                        self.directory.publish(segment, Arc::as_ptr(table));
                    }
                }
                let removed = if empty {
                    state.units.tables[segment].take()
                } else {
                    None
                };
                let record = state.units.records.get_mut(handle).unwrap();
                record.detached_epoch = Some(retired);
                record.detached_table = removed;
                state.execution = next;
                return Ok(true);
            }
        }
        let mut replacement = if let Some(table) = &previous {
            let intervals: Vec<_> = table
                .intervals
                .iter()
                .copied()
                .filter(|interval| interval.unit != pointer)
                .collect();
            if intervals.is_empty() {
                None
            } else {
                let bytes = size_of::<Accounted<Table>>()
                    + 2 * size_of::<usize>()
                    + intervals.capacity() * size_of::<Interval>();
                match self.cache.account(
                    Table {
                        generation: table.generation,
                        intervals,
                    },
                    bytes,
                    Tier::Lcq,
                ) {
                    Ok(table) => Some(Arc::new(table)),
                    Err(crate::executable::Error::Capacity(_)) => return Ok(false), // Retry at Closed without allocation.
                    Err(error) => return Err(error.into()),
                }
            }
        } else {
            None
        };
        let mut state = self.lock();
        if !same_snapshot(&state.units.tables[segment], &previous) {
            return Ok(false);
        }
        let retired = state.execution;
        let result = state.executions.next_id();
        let next = self.checked(&mut state, result)?;
        std::mem::swap(&mut state.units.tables[segment], &mut replacement);
        unsafe {
            self.directory.publish(
                segment,
                state.units.tables[segment]
                    .as_ref()
                    .map_or(std::ptr::null(), Arc::as_ptr),
            );
        }
        let record = state.units.records.get_mut(handle).unwrap();
        record.detached_epoch = Some(retired);
        record.detached_table = replacement.take();
        state.execution = next;
        Ok(true)
    }

    fn decommit_unused(&self) -> Result<(), Error> {
        for segment in 0..SEGMENTS {
            {
                let mut state = self.lock();
                if state.units.tables[segment].is_some()
                    || state
                        .units
                        .records
                        .values()
                        .any(|record| record.code.code.allocation.segment == segment)
                {
                    continue;
                }
                state.units.decommitting[segment] = true;
            }
            // A staging allocation can win the cache lock; its live lease then
            // prevents decommit. Publication cannot cross this marked interval.
            let result = unsafe { self.cache.decommit_empty(segment) };
            let mut state = self.lock();
            state.units.decommitting[segment] = false;
            if let Err(error) = result {
                let error = Error::from(error);
                self.fail(&mut state, error);
                return Err(error);
            }
        }
        Ok(())
    }
}

impl Transition<'_> {
    /// Cold pressure pass. `additional` is the pending allocation's full
    /// incremental code/metadata charge (including any new segment). This does
    /// not reserve it: the allocator must still enforce admission on retry.
    /// No wait on compiler references, and no guest-loop budget checks.
    pub(crate) fn relieve_pressure(&mut self, additional: usize, tier: Tier) -> Result<(), Error> {
        {
            let state = self.process.lock();
            self.require_closed(&state)?;
            if state.shutdown {
                return Err(Error::Shutdown);
            }
        }
        self.drain_retirements()?;
        self.retire_empty_dispatch()?;
        loop {
            self.process.reclaim_units()?;
            let usage = self.process.cache.usage()?;
            if usage
                .total()
                .checked_add(additional)
                .is_some_and(|total| total <= crate::executable::SOFT_BYTES)
            {
                return Ok(());
            }
            let candidate = {
                let state = self.process.lock();
                // Stop scheduling evictions once retiring whole segments can
                // cover the shortage. Compiler/staging leases can postpone
                // their actual decommit; this is NOT a budget refund or a
                // promise that the allocator's retry will succeed.
                let mut pending = [false; SEGMENTS];
                let mut resident = [false; SEGMENTS];
                for record in state.units.records.values() {
                    let segment = record.code.code.allocation.segment;
                    if matches!(record.lifecycle, Lifecycle::Retired(_)) {
                        pending[segment] = true;
                    } else {
                        resident[segment] = true;
                    }
                }
                let pending_bytes: usize = (0..SEGMENTS)
                    .filter(|&index| pending[index] && !resident[index])
                    .map(|index| {
                        (crate::executable::WINDOW_BYTES - index * crate::executable::SEGMENT_BYTES)
                            .min(crate::executable::SEGMENT_BYTES)
                    })
                    .sum();
                if usage
                    .total()
                    .saturating_sub(pending_bytes)
                    .checked_add(additional)
                    .is_some_and(|total| total <= crate::executable::SOFT_BYTES)
                {
                    return usage.check(additional, tier).map_err(Error::from);
                }
                [Tier::Hcq, Tier::Lcq].into_iter().find_map(|kind| {
                    let id = state
                        .units
                        .records
                        .values()
                        .filter(|record| {
                            record.code.tier == kind
                                && matches!(
                                    record.lifecycle,
                                    Lifecycle::Published | Lifecycle::Superseded
                                )
                                && record.code.baseline_pins.load(Ordering::Relaxed) == 0
                        })
                        .map(|record| record.code.id)
                        .min()?;
                    state.units.records.find(|record| record.code.id == id)
                })
            };
            let Some(handle) = candidate else {
                // LCQ may consume the hard-limit headroom; HCQ must abandon
                // this attempt if snapshots prevent returning below soft.
                return usage.check(additional, tier).map_err(Error::from);
            };
            self.process
                .retire_unit(UnitHandle(handle, self.process.identity))?;
            self.drain_retirements()?;
        }
    }

    fn retire_empty_dispatch(&self) -> Result<(), Error> {
        let mut state = self.process.lock();
        self.require_closed(&state)?;
        let mut retired = None;
        loop {
            let empty = state.keys.entries.iter().copied().find(|(_, slot)| {
                state
                    .dispatch
                    .get(*slot)
                    .unwrap()
                    .snapshot()
                    .preferred()
                    .is_none()
            });
            let Some((key, slot)) = empty else {
                return Ok(());
            };
            let epoch = match retired {
                Some(epoch) => epoch,
                None => {
                    let epoch = state.execution;
                    let result = state.executions.next_id();
                    state.execution = self.process.checked(&mut state, result)?;
                    retired = Some(epoch);
                    epoch
                }
            };
            // Closing already invalidated compiler publications which reserved
            // these keys. Retained CodeUnit users still postpone slot reuse.
            state.keys.remove(&key);
            state.dispatch.get_mut(slot).unwrap().retired = Some(epoch);
        }
    }

    /// Nonblocking shutdown progress. Call again after outstanding compiler
    /// outputs/snapshots have been dropped. Reason acknowledgement is forbidden
    /// until this has released mappings and all foundation-owned indexes.
    pub(crate) fn try_finish_shutdown(&mut self) -> Result<bool, Error> {
        {
            let state = self.process.lock();
            self.require_closed(&state)?;
            if !state.shutdown {
                return Err(Error::InvalidUnit("shutdown was not requested"));
            }
            if state.units.shutdown_finished {
                return Ok(true);
            }
        }
        self.drain_retirements()?;
        self.process.reclaim_units()?;
        {
            let state = self.process.lock();
            if state.units.collecting
                || !state.units.records.is_empty()
                || !state.units.families.is_empty()
            {
                return Ok(false);
            }
        }
        // No published code remains. Cache leases also cover unpublished
        // outputs, which may still be finishing on another compiler thread.
        if !unsafe { self.process.cache.try_close()? } {
            return Ok(false);
        }
        let empty_units = Units::default();
        let empty_keys = crate::lifetime::KeyIndex::with_capacity(0);
        let removed = {
            let mut state = self.process.lock();
            // Admission is terminal; resetting empty slab counters cannot
            // make an old handle valid in a new publication.
            (
                std::mem::replace(&mut state.units, empty_units),
                std::mem::take(&mut state.dispatch),
                std::mem::replace(&mut state.keys, empty_keys),
                std::mem::take(&mut state.readers),
                state.dispatch_storage.take(),
                state.key_storage.take(),
                state.reader_storage.take(),
            )
        };
        drop(removed);
        self.process.lock().units.shutdown_finished = true;
        Ok(true)
    }

    /// Drain exact queued unit records before acknowledging their reason batch.
    /// Newly arriving requests remain in records and are drained in this stop.
    pub(crate) fn drain_retirements(&mut self) -> Result<usize, Error> {
        let mut count = 0;
        loop {
            let handle = {
                let state = self.process.lock();
                self.require_closed(&state)?;
                // Release active HCQ baseline promises before selecting LCQ.
                state
                    .units
                    .records
                    .find(|record| record.retirement.is_some() && record.code.tier == Tier::Hcq)
                    .or_else(|| {
                        state
                            .units
                            .records
                            .find(|record| record.retirement.is_some())
                    })
            };
            let Some(handle) = handle else {
                return Ok(count);
            };
            count += usize::from(self.unlink(handle)?);
        }
    }

    fn require_closed(&self, state: &crate::lifetime::State) -> Result<(), Error> {
        state.healthy()?;
        if !self.active || state.phase != Phase::Closed {
            return Err(Error::Closed);
        }
        Ok(())
    }

    fn unlink(&mut self, handle: Handle<UnitRecord>) -> Result<bool, Error> {
        let removed_family = {
            let mut state = self.process.lock();
            self.require_closed(&state)?;
            let record = state.units.records.get(handle).ok_or(Error::StaleUnit)?;
            let (reason, _) = record.retirement.ok_or(Error::StaleUnit)?;
            if reason == Reason::TierCutover
                && (rooted(&state, record)
                    || record.code.baseline_pins.load(Ordering::Relaxed) != 0)
            {
                state.units.records.get_mut(handle).unwrap().retirement = None;
                return Ok(false);
            }
            if !state.shutdown && record.code.baseline_pins.load(Ordering::Relaxed) != 0 {
                return Err(Error::PinnedBaseline);
            }
            let count = record
                .slots
                .iter()
                .filter(|slot| {
                    let payload = state.dispatch.get(**slot).unwrap().snapshot();
                    names(payload.lcq(), &record.code)
                        || names(payload.hcq().map(|entry| entry.entry), &record.code)
                })
                .count();
            let result = state.reachabilities.take_ids(count);
            let mut identities = self.process.checked(&mut state, result)?;
            let retired = state.execution;
            let result = state.executions.next_id();
            let next = self.process.checked(&mut state, result)?;
            let length = state.units.records.get(handle).unwrap().slots.len();
            for index in 0..length {
                let record = state.units.records.get(handle).unwrap();
                let slot = record.slots[index];
                let key = record.code.entries[index].key;
                let payload = state.dispatch.get(slot).unwrap().snapshot();
                let lcq = payload
                    .lcq()
                    .filter(|entry| !names(Some(*entry), &record.code));
                let hcq = payload
                    .hcq()
                    .filter(|entry| !names(Some(entry.entry), &record.code));
                if lcq == payload.lcq() && hcq == payload.hcq() {
                    continue;
                }
                let empty = lcq.is_none() && hcq.is_none();
                state
                    .dispatch
                    .get_mut(slot)
                    .unwrap()
                    .rewrite_closed(DispatchPayload::new(identities.next().unwrap(), lcq, hcq));
                if empty {
                    if state.keys.get(&key) == Some(&slot) {
                        state.keys.remove(&key);
                    }
                    state.dispatch.get_mut(slot).unwrap().retired = Some(retired);
                }
            }
            let record = state.units.records.get_mut(handle).unwrap();
            record.lifecycle = Lifecycle::Unlinked;
            record.lifecycle = Lifecycle::Retired(retired);
            record.retirement = None;
            let family = record.family;
            state.execution = next;
            family.map(|family| state.units.families.take_held(family).unwrap())
        };
        // Last-family destruction releases baseline pins outside state. Its
        // registry slot remains held until the HCQ unit's span is actually freed.
        drop(removed_family);
        Ok(true)
    }
}
