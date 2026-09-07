//! Production admission/publication owner for the tiered JIT. Executable
//! storage and coupled CodeUnits share its publication/reader protocol;
//! this protocol does not delegate lifetime to the legacy JITModule path.

mod directory;
mod registry;
#[cfg(test)]
mod tests;
pub(crate) mod unit;

use crate::abi::{
    AdmissionEpoch, BlockKey, CheckedCounter, DispatchPayload, ExecutionEpoch, IdentityExhausted,
    MaintenanceSequence, NativeFrame, ReachabilityVersion,
};
#[cfg(test)]
use crate::abi::{HcqEntry, PublishedEntry};
use crate::executable::{Accounted, Cache, MetadataLease, Tier};
use registry::{Handle, Registry};
use std::hash::{BuildHasher, RandomState};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Error {
    Exhausted(IdentityExhausted),
    Poisoned,
    Closed,
    Shutdown,
    StalePublication,
    OccupiedDispatch,
    ActiveReader,
    Capacity(&'static str),
    CacheFailed,
    InvalidUnit(&'static str),
    StaleUnit,
    PinnedBaseline,
    MaintenancePending,
    UnsupportedHost(&'static str),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedHost(detail) => f.write_str(detail),
            Self::Exhausted(error) => error.fmt(f),
            Self::Poisoned => f.write_str("JIT lifetime state poisoned; admission is disabled"),
            Self::Closed => f.write_str("JIT admission is closed for maintenance"),
            Self::Shutdown => f.write_str("JIT process is shutting down"),
            Self::StalePublication => f.write_str(
                "JIT publication has a stale process, admission, slot or reachability identity",
            ),
            Self::OccupiedDispatch => {
                f.write_str("cannot retire a dispatch slot with resident entries")
            }
            Self::ActiveReader => f.write_str("JIT reader already protects an invocation"),
            Self::Capacity(detail) => write!(f, "JIT capacity: {detail}"),
            Self::CacheFailed => f.write_str("JIT executable cache has failed"),
            Self::InvalidUnit(detail) => write!(f, "JIT unit publication: {detail}"),
            Self::StaleUnit => {
                f.write_str("JIT unit handle is stale or belongs to another process")
            }
            Self::PinnedBaseline => {
                f.write_str("LCQ unit is pinned by an active or in-flight HCQ family")
            }
            Self::MaintenancePending => {
                f.write_str("JIT maintenance still has unapplied unit work")
            }
        }
    }
}
impl std::error::Error for Error {}
impl From<IdentityExhausted> for Error {
    fn from(error: IdentityExhausted) -> Self {
        Self::Exhausted(error)
    }
}
impl From<crate::executable::Error> for Error {
    fn from(error: crate::executable::Error) -> Self {
        match error {
            crate::executable::Error::Closed => Self::Shutdown,
            crate::executable::Error::Capacity(detail) => Self::Capacity(detail),
            _ => Self::CacheFailed,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Open,
    Closing,
    Closed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum Reason {
    LinkPatch,
    TierCutover,
    Eviction,
    MappingChange,
    Shutdown,
}
const REASONS: [Reason; 5] = [
    Reason::LinkPatch,
    Reason::TierCutover,
    Reason::Eviction,
    Reason::MappingChange,
    Reason::Shutdown,
];

struct DispatchSlot {
    payload: AtomicPtr<Accounted<DispatchPayload>>,
    retired: Option<ExecutionEpoch>,
    units: usize,
}
impl DispatchSlot {
    /// Withdraw and republish the uniquely owned payload during Closed. Readers
    /// copy under state and none retains its pointer. This avoids allocating
    /// replacement boxes to evict code when the hard budget is already full.
    fn rewrite_closed(&mut self, payload: DispatchPayload) {
        let pointer = self.payload.swap(std::ptr::null_mut(), Ordering::Relaxed);
        unsafe {
            (*pointer).value = payload;
        }
        self.payload.store(pointer, Ordering::Release);
    }
    fn new(payload: Box<Accounted<DispatchPayload>>) -> Self {
        Self {
            payload: AtomicPtr::new(Box::into_raw(payload)),
            retired: None,
            units: 0,
        }
    }

    // Only called with the owning state mutex. Copying cannot allocate: the
    // payload contains only value fields. No pointer/reference escapes to a
    // reader, so replacement can drop the old box after releasing that mutex.
    fn snapshot(&self) -> DispatchPayload {
        unsafe { (&*self.payload.load(Ordering::Acquire)).value.clone() }
    }

    fn reachability(&self) -> ReachabilityVersion {
        unsafe { (&*self.payload.load(Ordering::Acquire)).reachability() }
    }

    fn replace(
        &mut self,
        payload: Box<Accounted<DispatchPayload>>,
    ) -> Box<Accounted<DispatchPayload>> {
        // Publication linearizes at this release swap, not at separate stores
        // of address/version. The state lock has already validated admission.
        let old = self.payload.swap(Box::into_raw(payload), Ordering::Release);
        unsafe { Box::from_raw(old) }
    }
}
impl Drop for DispatchSlot {
    fn drop(&mut self) {
        // Exclusive ownership, after all serialized payload readers finished.
        unsafe { drop(Box::from_raw(*self.payload.get_mut())) };
    }
}

// HashTable exposes its actual allocation size, unlike std::HashMap. Keep
// randomized hashing and account the bucket/control allocation without guessing
// its load factor, SIMD group width or allocator layout.
// https://docs.rs/hashbrown/0.16.1/hashbrown/struct.HashTable.html#method.allocation_size
struct KeyIndex {
    entries: hashbrown::HashTable<(BlockKey, Handle<DispatchSlot>)>,
    hash: RandomState,
}
impl KeyIndex {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: hashbrown::HashTable::with_capacity(capacity),
            hash: RandomState::new(),
        }
    }
    fn capacity(&self) -> usize {
        self.entries.capacity()
    }
    fn allocation_size(&self) -> usize {
        self.entries.allocation_size()
    }
    fn len(&self) -> usize {
        self.entries.len()
    }
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    fn get(&self, key: &BlockKey) -> Option<&Handle<DispatchSlot>> {
        self.entries
            .find(self.hash.hash_one(key), |entry| entry.0 == *key)
            .map(|entry| &entry.1)
    }
    fn contains_key(&self, key: &BlockKey) -> bool {
        self.get(key).is_some()
    }
    fn insert(&mut self, key: BlockKey, handle: Handle<DispatchSlot>) {
        debug_assert!(!self.contains_key(&key));
        self.entries
            .insert_unique(self.hash.hash_one(key), (key, handle), |entry| {
                self.hash.hash_one(entry.0)
            });
    }
    fn remove(&mut self, key: &BlockKey) {
        if let Ok(entry) = self
            .entries
            .find_entry(self.hash.hash_one(key), |entry| entry.0 == *key)
        {
            entry.remove();
        }
    }
    fn drain(&mut self) -> impl Iterator<Item = (BlockKey, Handle<DispatchSlot>)> + '_ {
        self.entries.drain()
    }
    fn extend(&mut self, entries: impl Iterator<Item = (BlockKey, Handle<DispatchSlot>)>) {
        for (key, handle) in entries {
            self.insert(key, handle);
        }
    }
}

// Replaced storage is returned here so it drops after releasing JIT state,
// before returning its charge (which acquires the separate cache mutex).
struct PreparedStorage<T> {
    value: T,
    charge: Option<MetadataLease>,
}
impl<T> PreparedStorage<T> {
    fn new(value: T, bytes: usize, cache: &Arc<Cache>) -> Result<Self, Error> {
        Self::for_tier(value, bytes, cache, Tier::Lcq)
    }
    fn for_tier(value: T, bytes: usize, cache: &Arc<Cache>, tier: Tier) -> Result<Self, Error> {
        Ok(Self {
            value,
            charge: Some(cache.charge_metadata(bytes, tier)?),
        })
    }
}

struct State {
    units: unit::Units,
    phase: Phase,
    admission: AdmissionEpoch,
    execution: ExecutionEpoch,
    admissions: CheckedCounter<AdmissionEpoch>,
    executions: CheckedCounter<ExecutionEpoch>,
    reachabilities: CheckedCounter<ReachabilityVersion>,
    sequences: CheckedCounter<MaintenanceSequence>,
    // Latest unacknowledged sequence per reason. Target records belong to the
    // corresponding operation (added with code/mapping work), not to a second
    // generic work queue. Coalescing a reason never acknowledges its records.
    pending: [Option<MaintenanceSequence>; 5],
    completed: [Option<MaintenanceSequence>; 5],
    transition_owned: bool,
    shutdown: bool,
    failure: Option<Error>,
    dispatch: Registry<DispatchSlot>,
    keys: KeyIndex,
    readers: Registry<Arc<Accounted<AtomicU64>>>,
    dispatch_storage: Option<MetadataLease>,
    key_storage: Option<MetadataLease>,
    reader_storage: Option<MetadataLease>,
}
impl State {
    fn healthy(&self) -> Result<(), Error> {
        self.failure.map_or(Ok(()), Err)
    }

    fn open(&self) -> Result<AdmissionEpoch, Error> {
        self.healthy()?;
        if self.shutdown {
            return Err(Error::Shutdown);
        }
        if self.phase != Phase::Open {
            return Err(Error::Closed);
        }
        Ok(self.admission)
    }

    fn quiescent(&self, retired: ExecutionEpoch) -> bool {
        self.readers.values().all(|reader| {
            let epoch = reader.load(Ordering::Acquire);
            epoch == 0 || epoch > retired.get()
        })
    }

    fn idle(&self) -> bool {
        self.readers
            .values()
            .all(|reader| reader.load(Ordering::Acquire) == 0)
    }

    fn validate(&self, publication: &Publication<'_>) -> Result<(), Error> {
        if self.open()? != publication.admission {
            return Err(Error::StalePublication);
        }
        let slot = self
            .dispatch
            .get(publication.slot)
            .ok_or(Error::StalePublication)?;
        if slot.retired.is_some()
            || self.keys.get(&publication.key) != Some(&publication.slot)
            || slot.reachability() != publication.reachability
        {
            return Err(Error::StalePublication);
        }
        Ok(())
    }
}

pub(crate) struct Lifetime {
    identity: u64,
    state: Mutex<State>,
    directory: directory::Directory,
    changed: Condvar,
    // Same single process-pending word consumed by the native control poll.
    // It notifies; it is never used as an independent admission authority.
    pending: AtomicU32,
    cache: Arc<Cache>,
    // Dropped after the state's actual metadata owners and allocations.
    storage: MetadataLease,
}
impl Lifetime {
    pub(crate) fn new(cache: Arc<Cache>) -> Result<Self, Error> {
        crate::native::check_host().map_err(Error::UnsupportedHost)?;
        static IDENTITIES: AtomicU64 = AtomicU64::new(0);
        let identity = IDENTITIES
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |last| {
                last.checked_add(1)
            })
            .map_err(|_| Error::Exhausted(IdentityExhausted("JIT process")))?
            + 1;
        let storage = cache.charge_metadata(
            std::mem::size_of::<Self>() + 2 * std::mem::size_of::<usize>(),
            Tier::Lcq,
        )?;
        let mut admissions = CheckedCounter::default();
        let mut executions = CheckedCounter::default();
        Ok(Self {
            identity,
            state: Mutex::new(State {
                units: unit::Units::default(),
                phase: Phase::Open,
                admission: admissions.next_id().unwrap(),
                execution: executions.next_id().unwrap(),
                admissions,
                executions,
                reachabilities: CheckedCounter::default(),
                sequences: CheckedCounter::default(),
                pending: [None; 5],
                completed: [None; 5],
                transition_owned: false,
                shutdown: false,
                failure: None,
                dispatch: Registry::default(),
                keys: KeyIndex::with_capacity(0),
                readers: Registry::default(),
                dispatch_storage: None,
                key_storage: None,
                reader_storage: None,
            }),
            changed: Condvar::new(),
            directory: directory::Directory::new(cache.executable_base()),
            pending: AtomicU32::new(0),
            cache,
            storage,
        })
    }

    fn fail(&self, state: &mut State, error: Error) {
        state.failure.get_or_insert(error);
        if state.phase == Phase::Open {
            state.phase = Phase::Closing;
        }
        self.pending.store(1, Ordering::Release);
        self.changed.notify_all();
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.recover(self.state.lock())
    }

    fn recover<'a>(
        &self,
        result: std::sync::LockResult<MutexGuard<'a, State>>,
    ) -> MutexGuard<'a, State> {
        match result {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                // Recovery only permits cold cleanup. Never resume publication
                // after an unwind may have interrupted a state mutation.
                self.fail(&mut state, Error::Poisoned);
                state
            }
        }
    }

    fn checked<T>(
        &self,
        state: &mut State,
        result: Result<T, IdentityExhausted>,
    ) -> Result<T, Error> {
        result.map_err(|error| {
            let error = Error::Exhausted(error);
            self.fail(state, error);
            error
        })
    }

    pub(crate) fn control_word(&self) -> &AtomicU32 {
        &self.pending
    }

    pub(crate) fn register(self: &Arc<Self>) -> Result<Reader, Error> {
        let announcement = Arc::new(self.cache.account(
            AtomicU64::new(0),
            std::mem::size_of::<Accounted<AtomicU64>>() + 2 * std::mem::size_of::<usize>(),
            Tier::Lcq,
        )?);
        let mut registration = Some(Arc::clone(&announcement));
        loop {
            let capacity = {
                let mut state = self.lock();
                state.healthy()?;
                if state.shutdown {
                    return Err(Error::Shutdown);
                }
                if state.readers.has_space() {
                    let result = state.readers.insert(&mut registration);
                    let handle = result.inspect_err(|error| self.fail(&mut state, *error))?;
                    return Ok(Reader {
                        process: Arc::clone(self),
                        handle,
                        announcement,
                    });
                }
                state.readers.capacity()
            };
            let spare = Vec::with_capacity(capacity.saturating_mul(2).max(16));
            let bytes =
                spare.capacity() * std::mem::size_of::<registry::Slot<Arc<Accounted<AtomicU64>>>>();
            let mut spare = PreparedStorage::new(spare, bytes, &self.cache)?;
            let mut state = self.lock();
            state.healthy()?;
            if state.shutdown {
                return Err(Error::Shutdown);
            }
            if spare.value.capacity() > state.readers.capacity() {
                state.readers.grow(&mut spare.value);
                std::mem::swap(&mut state.reader_storage, &mut spare.charge);
            }
            // The old empty allocation is dropped without holding state.
        }
    }

    /// Capture a publication identity, creating an unavailable slot on a miss.
    /// No entry is read by a gateway through this compiler-side operation.
    pub(crate) fn reserve(&self, key: BlockKey) -> Result<Publication<'_>, Error> {
        loop {
            let (admission, version, slot_capacity, key_capacity) = {
                let mut state = self.lock();
                let admission = state.open()?;
                if let Some(&slot) = state.keys.get(&key) {
                    let reachability = state.dispatch.get(slot).unwrap().reachability();
                    return Ok(Publication {
                        process: self,
                        key,
                        slot,
                        admission,
                        reachability,
                    });
                }
                let result = state.reachabilities.next_id();
                let version = self.checked(&mut state, result)?;
                let slots = if state.dispatch.has_space() {
                    0
                } else {
                    state.dispatch.capacity().saturating_mul(2).max(16)
                };
                let keys = if state.keys.len() < state.keys.capacity() {
                    0
                } else {
                    state.keys.capacity().saturating_mul(2).max(16)
                };
                (admission, version, slots, keys)
            };
            let mut slot = Some(DispatchSlot::new(Box::new(self.cache.account(
                DispatchPayload::new(version, None, None),
                std::mem::size_of::<Accounted<DispatchPayload>>(),
                Tier::Lcq,
            )?)));
            let slots = Vec::with_capacity(slot_capacity);
            let bytes = slots.capacity() * std::mem::size_of::<registry::Slot<DispatchSlot>>();
            let mut slots = PreparedStorage::new(slots, bytes, &self.cache)?;
            let keys = KeyIndex::with_capacity(key_capacity);
            let bytes = keys.allocation_size();
            let mut keys = PreparedStorage::new(keys, bytes, &self.cache)?;
            let mut state = self.lock();
            if state.open()? != admission {
                return Err(Error::StalePublication);
            }
            if state.keys.contains_key(&key) {
                continue;
            }
            if slots.value.capacity() > state.dispatch.capacity() {
                state.dispatch.grow(&mut slots.value);
                std::mem::swap(&mut state.dispatch_storage, &mut slots.charge);
            }
            if keys.value.capacity() > state.keys.capacity() {
                keys.value.extend(state.keys.drain());
                std::mem::swap(&mut state.keys, &mut keys.value);
                std::mem::swap(&mut state.key_storage, &mut keys.charge);
            }
            if !state.dispatch.has_space() || state.keys.len() == state.keys.capacity() {
                continue;
            }
            let result = state.dispatch.insert(&mut slot);
            let handle = result.inspect_err(|error| self.fail(&mut state, *error))?;
            state.keys.insert(key, handle);
            return Ok(Publication {
                process: self,
                key,
                slot: handle,
                admission,
                reachability: version,
            });
        }
    }

    /// Isolated protocol tests use nonexecuted addresses. Production publication
    /// goes exclusively through PreparedUnit, which owns code and metadata.
    #[cfg(test)]
    pub(crate) fn publish(
        &self,
        publication: Publication<'_>,
        lcq: Option<PublishedEntry>,
        hcq: Option<HcqEntry>,
    ) -> Result<ReachabilityVersion, Error> {
        if !std::ptr::eq(self, publication.process) {
            return Err(Error::StalePublication);
        }
        let version = {
            let mut state = self.lock();
            state.validate(&publication)?;
            let result = state.reachabilities.next_id();
            self.checked(&mut state, result)?
        };
        let tier = if hcq.is_some() { Tier::Hcq } else { Tier::Lcq };
        let payload = Box::new(self.cache.account(
            DispatchPayload::new(version, lcq, hcq),
            std::mem::size_of::<Accounted<DispatchPayload>>(),
            tier,
        )?);
        let old = {
            let mut state = self.lock();
            state.validate(&publication)?;
            state
                .dispatch
                .get_mut(publication.slot)
                .unwrap()
                .replace(payload)
        };
        drop(old);
        Ok(version)
    }

    pub(crate) fn retire_dispatch(&self, publication: Publication<'_>) -> Result<(), Error> {
        if !std::ptr::eq(self, publication.process) {
            return Err(Error::StalePublication);
        }
        let mut state = self.lock();
        state.validate(&publication)?;
        if state
            .dispatch
            .get(publication.slot)
            .unwrap()
            .snapshot()
            .preferred()
            .is_some()
        {
            return Err(Error::OccupiedDispatch);
        }
        let result = state.executions.next_id();
        let next = self.checked(&mut state, result)?;
        state.keys.remove(&publication.key);
        let retired = state.execution;
        state.dispatch.get_mut(publication.slot).unwrap().retired = Some(retired);
        state.execution = next;
        Ok(())
    }

    pub(crate) fn collect_dispatch(&self) -> Result<usize, Error> {
        let mut count = 0;
        loop {
            let removed = {
                let mut state = self.lock();
                state.healthy()?;
                let Some(handle) = state.dispatch.find(|slot| {
                    slot.units == 0 && slot.retired.is_some_and(|epoch| state.quiescent(epoch))
                }) else {
                    return Ok(count);
                };
                state.dispatch.remove(handle).unwrap()
            };
            drop(removed);
            count += 1;
        }
    }

    /// The operation registers its exact target records before announcing its
    /// reason. Until their consumers are implemented, this is only coordination
    /// and does not pretend to patch code, alter mappings or release storage.
    pub(crate) fn request(&self, reason: Reason) -> Result<Ticket<'_>, Error> {
        let mut state = self.lock();
        self.request_locked(&mut state, reason)
    }

    fn request_locked(&self, state: &mut State, reason: Reason) -> Result<Ticket<'_>, Error> {
        state.healthy()?;
        if state.shutdown {
            return Err(Error::Shutdown);
        }
        let result = state.sequences.next_id();
        let sequence = self.checked(state, result)?;
        if state.phase == Phase::Open {
            let result = state.admissions.next_id();
            state.admission = self.checked(state, result)?;
            // Closure linearizes under the same lock as reader announcement
            // and publication. No packed/shortened epoch counter is involved.
            state.phase = Phase::Closing;
        }
        state.pending[reason as usize] = Some(sequence);
        state.shutdown |= reason == Reason::Shutdown;
        if reason == Reason::Shutdown {
            state.units.mark_shutdown(sequence);
        }
        self.pending
            .fetch_or(1 << reason as usize, Ordering::Release);
        self.changed.notify_all();
        Ok(Ticket {
            process: self,
            reason,
            sequence,
        })
    }

    pub(crate) fn try_transition(&self) -> Result<Option<Transition<'_>>, Error> {
        let mut state = self.lock();
        state.healthy()?;
        if state.transition_owned || (state.shutdown && state.pending.iter().all(Option::is_none)) {
            return Ok(None);
        }
        if state.phase == Phase::Open {
            // Deferred performance-only links retain their control request and
            // start a later coalesced stop; no fabricated request/sequence.
            if state.pending.iter().all(Option::is_none) {
                return Ok(None);
            }
            let result = state.admissions.next_id();
            state.admission = self.checked(&mut state, result)?;
            state.phase = Phase::Closing;
        }
        state.transition_owned = true;
        Ok(Some(Transition {
            process: self,
            active: true,
            defer_links: false,
        }))
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Publication<'a> {
    process: &'a Lifetime,
    key: BlockKey,
    slot: Handle<DispatchSlot>,
    admission: AdmissionEpoch,
    reachability: ReachabilityVersion,
}

pub(crate) struct Reader {
    process: Arc<Lifetime>,
    handle: Handle<Arc<Accounted<AtomicU64>>>,
    announcement: Arc<Accounted<AtomicU64>>,
}
impl Reader {
    /// Save FP before lookup and protect the whole invocation, including fault
    /// capture/dispatch/retry. No guest FP activation happens in cold Rust.
    ///
    /// # Safety
    /// No other FP owner is active on this OS thread. Any native execution must
    /// complete canonical writeback before the returned guard is dropped, even
    /// on errors. All borrowed fault metadata must be consumed before drop.
    /// Use the protected entry only with the native gateway's safety contract.
    pub(crate) unsafe fn admit<'r, 'f, 's>(
        &'r mut self,
        frame: &'f mut NativeFrame<'s>,
        key: BlockKey,
    ) -> Result<Option<Invocation<'r, 'f, 's>>, Error> {
        if self.announcement.load(Ordering::Acquire) != 0 {
            return Err(Error::ActiveReader);
        }
        unsafe { frame.begin_fp() };
        let mut invocation = Invocation {
            reader: self,
            frame,
            payload: None,
            thread: PhantomData,
        };
        {
            let state = invocation.reader.process.lock();
            let admission = state.open()?;
            let execution = state.execution;
            invocation.frame.execution_epoch = execution.get();
            invocation.frame.admission_epoch = admission.get();
            invocation
                .reader
                .announcement
                .store(execution.get(), Ordering::Release);
            // This lock closes the store-to-lookup race: admission's unlock
            // happens-before a later collector/closer lock, so it sees either
            // this announcement or a completed exit. If closure/removal wins
            // first, admission sees it before lookup. Release/Acquire on two
            // unrelated atomics alone would NOT establish that ordering.
            // https://doc.rust-lang.org/nomicon/atomics.html#acquire-release
            // A closer after unlock but before the machine jump must still
            // wait for this already-announced invocation.
            if let Some(handle) = state.keys.get(&key) {
                let payload = state.dispatch.get(*handle).unwrap().snapshot();
                if payload.preferred().is_some() {
                    // Both checks are stable while holding state; no slot or
                    // payload pointer is retained past this critical section.
                    debug_assert_eq!(state.open(), Ok(admission));
                    debug_assert_eq!(
                        state.dispatch.get(*handle).unwrap().reachability(),
                        payload.reachability()
                    );
                    invocation.payload = Some(payload);
                }
            }
        }
        if invocation.payload.is_none() {
            return Ok(None);
        }
        Ok(Some(invocation))
    }
}
impl Drop for Reader {
    fn drop(&mut self) {
        let removed = {
            let mut state = self.process.lock();
            // Safe borrowing prevents dropping a reader with a live guard.
            // A forgotten guard leaks its announcement rather than permitting
            // reclamation of code that might still be used.
            if self.announcement.load(Ordering::Acquire) != 0 {
                return;
            }
            state.readers.remove(self.handle)
        };
        drop(removed);
    }
}

pub(crate) struct Invocation<'r, 'f, 's> {
    reader: &'r mut Reader,
    frame: &'f mut NativeFrame<'s>,
    payload: Option<DispatchPayload>,
    // FP save/restore must remain on one OS thread, even though an inactive
    // reader registration can migrate between threads.
    thread: PhantomData<Rc<()>>,
}
impl<'s> Invocation<'_, '_, 's> {
    /// The borrow prevents normal-stack fault dispatch from outliving this
    /// invocation's epoch. No lock, allocation or Arc operation occurs here.
    pub(crate) fn fault(&self, pc: usize) -> Option<directory::Fault<'_>> {
        unsafe { self.reader.process.directory.lookup(pc) }
    }
    pub(crate) fn payload(&self) -> &DispatchPayload {
        self.payload.as_ref().unwrap()
    }
    pub(crate) fn frame(&mut self) -> &mut NativeFrame<'s> {
        self.frame
    }
}
impl Drop for Invocation<'_, '_, '_> {
    fn drop(&mut self) {
        // FP/status completion must precede even acquiring state. The native
        // gateway may already have finished FP; finish_fp is idempotent then.
        unsafe { self.frame.finish_fp() };
        let _state = self.reader.process.lock();
        self.frame.execution_epoch = 0;
        self.frame.admission_epoch = 0;
        self.reader.announcement.store(0, Ordering::Release);
        // Same predicate mutex as wait_closed: no exit notification can fall
        // between its predicate test and releasing the mutex to wait.
        self.reader.process.changed.notify_all();
    }
}

pub(crate) struct Ticket<'a> {
    process: &'a Lifetime,
    reason: Reason,
    sequence: MaintenanceSequence,
}
impl Ticket<'_> {
    pub(crate) fn is_complete(&self) -> Result<bool, Error> {
        let state = self.process.lock();
        state.healthy()?;
        Ok(state.completed[self.reason as usize].is_some_and(|seq| seq >= self.sequence))
    }
}

pub(crate) struct Transition<'a> {
    process: &'a Lifetime,
    active: bool,
    defer_links: bool,
}
impl<'p> Transition<'p> {
    /// Call only from canonical mode, never while protecting one's own native
    /// invocation or holding a code-cache/memory lock. Condvar::wait releases
    /// state while other vCPUs finish, including normal-stack fault dispatch.
    pub(crate) fn wait_closed(&mut self) -> Result<(), Error> {
        if !self.active {
            return Err(Error::Closed);
        }
        let mut state = self.process.lock();
        loop {
            state.healthy()?;
            if state.idle() {
                state.phase = Phase::Closed;
                return Ok(());
            }
            state = self.process.recover(self.process.changed.wait(state));
        }
    }

    pub(crate) fn batch(&mut self) -> Result<Batch<'_, 'p>, Error> {
        if !self.active {
            return Err(Error::Closed);
        }
        let state = self.process.lock();
        state.healthy()?;
        if state.phase != Phase::Closed {
            return Err(Error::Closed);
        }
        let sequences = state.pending;
        drop(state);
        Ok(Batch {
            transition: self,
            sequences,
        })
    }

    /// False means new/unacknowledged work must still be drained while Closed.
    /// Shutdown completes coordination but leaves admission permanently Closed.
    pub(crate) fn try_reopen(&mut self) -> Result<bool, Error> {
        if !self.active {
            return Err(Error::Closed);
        }
        let mut state = self.process.lock();
        state.healthy()?;
        if state.phase != Phase::Closed {
            return Err(Error::Closed);
        }
        if state.pending.iter().enumerate().any(|(index, sequence)| {
            sequence.is_some()
                && !(index == Reason::LinkPatch as usize && self.defer_links && !state.shutdown)
        }) {
            return Ok(false);
        }
        if !state.shutdown {
            let result = state.admissions.next_id();
            state.admission = self.process.checked(&mut state, result)?;
            state.phase = Phase::Open;
            let reasons = state
                .pending
                .iter()
                .enumerate()
                .fold(0, |bits, (index, sequence)| {
                    bits | (u32::from(sequence.is_some()) << index)
                });
            self.process.pending.store(reasons, Ordering::Release);
        }
        state.transition_owned = false;
        self.active = false;
        self.process.changed.notify_all();
        Ok(true)
    }
}
impl Drop for Transition<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut state = self.process.lock();
        state.transition_owned = false;
        // Abandonment is not completion. Leave Closed/Closing and all pending
        // work intact for another owner; never implicitly reopen on an error.
        self.process.changed.notify_all();
    }
}

pub(crate) struct Batch<'t, 'p> {
    transition: &'t mut Transition<'p>,
    sequences: [Option<MaintenanceSequence>; 5],
}
impl Batch<'_, '_> {
    pub(crate) fn reasons(&self) -> impl Iterator<Item = Reason> + '_ {
        REASONS
            .into_iter()
            .filter(|reason| self.sequences[*reason as usize].is_some())
    }

    /// Acknowledge only after all records covered by this batch were applied
    /// without JIT-state/cache/memory lock nesting. An error/drop acknowledges
    /// nothing. A concurrent request of the same reason remains pending.
    pub(crate) fn complete(self) -> Result<(), Error> {
        self.acknowledge(false)
    }

    /// Task 4's installer uses this after its 4096-record limit. Only optional
    /// installation may be deferred; safety-critical unlinks are completed as
    /// safety work. Uninstalled links must retain their valid native fallback.
    /// Their tickets remain incomplete and the control request survives reopen.
    pub(crate) fn complete_with_links_deferred(self) -> Result<(), Error> {
        self.acknowledge(true)
    }

    fn acknowledge(self, defer_links: bool) -> Result<(), Error> {
        let mut state = self.transition.process.lock();
        state.healthy()?;
        if self.sequences.iter().enumerate().any(|(index, sequence)| {
            sequence
                .is_some_and(|sequence| state.units.pending_retirement(REASONS[index], sequence))
        }) {
            return Err(Error::MaintenancePending);
        }
        self.transition.defer_links |= defer_links;
        for (index, sequence) in self.sequences.into_iter().enumerate() {
            if index == Reason::LinkPatch as usize && defer_links {
                continue;
            }
            if let Some(sequence) = sequence {
                state.completed[index] = Some(sequence);
                if state.pending[index] == Some(sequence) {
                    state.pending[index] = None;
                }
            }
        }
        self.transition.process.changed.notify_all();
        Ok(())
    }
}
