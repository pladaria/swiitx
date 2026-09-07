use super::*;
use crate::abi::{
    BlockKey, ExitSiteKey, FpSpecialization, NativeFrame, PollBudget, TRANSFER_BYTES,
};
use crate::analysis::StateSet;
use crate::executable::{
    Cache, HARD_BYTES, SEGMENT_BYTES, WINDOW_BYTES,
    output::{Metadata, Output},
};
use crate::lifetime::Reason;
use cranelift_codegen::{ir, nixe::StateMap};
use nixe_cpu::{platform::TargetPlatform, profile::ProcessCpuContext, state::a64::A64State};
use nixe_memory::{AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress, MappingGeneration};
use std::sync::{Barrier, mpsc};

pub(super) fn key(pc: u64) -> BlockKey {
    BlockKey::new(
        ProcessCpuContext::new(TargetPlatform::Switch1, AddressSpaceId::new(1)),
        GuestVirtualAddress::new(pc),
        FpSpecialization::Dynamic,
    )
    .unwrap()
}
pub(super) fn process() -> Arc<Lifetime> {
    Arc::new(Lifetime::new(Cache::new().unwrap()).unwrap())
}
pub(super) fn input(process: &Lifetime, pcs: &[u64], tier: Tier) -> Input {
    let identity = process.begin_unit(tier).unwrap();
    let version = identity.version();
    let abi = if cfg!(target_arch = "x86_64") {
        HostAbi::X86_64
    } else {
        HostAbi::Aarch64
    };
    // Callable System-ABI leaf returning 42, followed by an unreachable load.
    // These tests prove ownership/attribution, not the Task 1 gateway (step 6).
    let (bytes, end) = match abi {
        HostAbi::X86_64 => (
            vec![
                0xf3, 0x0f, 0x1e, 0xfa, 0xb8, 42, 0, 0, 0, 0xc3, 0x90, 0x90, 0x48, 0x8b, 0x07, 0x90,
            ],
            15,
        ),
        HostAbi::Aarch64 => (
            [0xd503245f_u32, 0x52800540, 0xd65f03c0, 0xf9400000]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
            16,
        ),
    };
    let code = process
        .cache
        .install(
            Output {
                bytes: bytes.into_boxed_slice(),
                alignment: 16,
                metadata: Metadata {
                    abi,
                    frame_extent: TRANSFER_BYTES,
                    entries: Box::new([(ir::Block::from_u32(0), 0)]),
                    states: Box::new([]),
                    faults: Box::new([StateMap {
                        id: 1,
                        offset: 12,
                        entry: false,
                        patch_bytes: 0,
                        values: vec![],
                    }]),
                    traps: Box::new([]),
                    relocations: Box::new([]),
                },
            },
            tier,
            |_| None,
        )
        .unwrap();
    Input {
        identity,
        code,
        tier,
        instructions: pcs
            .iter()
            .map(|pc| Instruction {
                key: InstructionKey::new(key(*pc)).unwrap(),
                bits: 0xd503201f,
            })
            .collect(),
        entries: pcs
            .iter()
            .map(|pc| Entry {
                key: key(*pc),
                canonical_offset: 0,
                fast_offset: 0,
                contract: EntryContract {
                    live_in: StateSet::default(),
                    abi,
                    bindings: Box::new([]),
                    nzcv: NzcvLocation::Canonical,
                },
            })
            .collect(),
        dependencies: Box::new([CodePageDependency {
            page: GuestPhysicalPageId::new(7),
            mapping_generation: MappingGeneration::new(2),
        }]),
        cursor: MemoryInvalidationCursor::INITIAL,
        states: Box::new([StateRecord {
            native_offset: 12,
            state: ExitStateMap {
                site: ExitSiteKey {
                    source: version,
                    state_map: 0,
                },
                abi,
                live: StateSet::default(),
                dirty_live: StateSet::default(),
                bindings: Box::new([]),
                nzcv: NzcvLocation::Canonical,
                host_fpsr_pending: false,
            },
        }]),
        faults: Box::new([FaultRecord {
            native_start: 12,
            native_end: end,
            instruction: InstructionKey::new(key(pcs[0])).unwrap(),
            access: Access::Read,
            bytes: 8,
            subaccess: 0,
            commit_stage: 0,
            state_map: 0,
        }]),
    }
}
pub(super) fn publish(
    process: &Lifetime,
    cursor: &AtomicU64,
    pcs: &[u64],
    tier: Tier,
) -> UnitHandle {
    let publications: Vec<_> = pcs
        .iter()
        .map(|pc| process.reserve(key(*pc)).unwrap())
        .collect();
    process
        .prepare_unit(&publications, input(process, pcs, tier), cursor)
        .unwrap()
        .publish()
        .unwrap()
}
pub(super) fn frame(state: &mut A64State) -> NativeFrame<'_> {
    NativeFrame::new(state, PollBudget::new(77, 1000).unwrap())
}

#[test]
fn metadata_and_exact_faults_are_live_before_multi_entry_dispatch() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let handle = publish(&process, &cursor, &[0, 4], Tier::Lcq);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    for pc in [0, 4] {
        let invocation = unsafe { reader.admit(&mut frame, key(pc)) }
            .unwrap()
            .unwrap();
        let entry = invocation.payload().preferred().unwrap();
        let base = entry.canonical.get();
        // Holding state here also ensures the lookup itself cannot take it.
        let state = process.lock();
        let owner = &state.units.records.get(handle.0).unwrap().code;
        assert_eq!((entry.unit, entry.version), (owner.id, owner.version));
        assert_eq!(owner.abi_version, NATIVE_ABI_VERSION);
        let fault = invocation.fault(base + 12).unwrap();
        assert_eq!(fault.unit.id, owner.id);
        assert_eq!(fault.record.access, Access::Read);
        assert_eq!(
            fault.unit.states[fault.record.state_map as usize]
                .state
                .site
                .source,
            entry.version
        );
        for offset in [0, 11, 16] {
            assert!(invocation.fault(base + offset).is_none());
        }
        if cfg!(target_arch = "x86_64") {
            assert!(invocation.fault(base + 15).is_none());
        }
        let deps = &state.units.dependencies;
        assert_eq!(deps.entries.len(), 1); // Per unit, not per selected entry.
        assert_eq!(
            deps.for_page(GuestPhysicalPageId::new(7))
                .next()
                .unwrap()
                .unit,
            handle
        );
        drop(state);
        let leaf: unsafe extern "C" fn() -> u32 = unsafe { std::mem::transmute(base) };
        assert_eq!(unsafe { leaf() }, 42);
    }
}

#[test]
fn uncommitted_units_never_appear_in_dispatch_or_fault_directory() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    let candidate = input(&process, &[4], Tier::Lcq);
    let address = candidate.code.allocation.address();
    let prepared = process
        .prepare_unit(&[process.reserve(key(4)).unwrap()], candidate, &cursor)
        .unwrap();
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    assert!(invocation.fault(address + 12).is_none());
    {
        let state = process.lock();
        let slot = *state.keys.get(&key(4)).unwrap();
        assert!(
            state
                .dispatch
                .get(slot)
                .unwrap()
                .snapshot()
                .preferred()
                .is_none()
        );
    }
    drop(prepared);
    assert!(process.lock().units.records.get(old.0).is_some());
    let replacement = input(&process, &[4], Tier::Lcq);
    assert_eq!(replacement.code.allocation.address(), address);
}

#[test]
fn table_replacement_retains_snapshots_through_fault_dispatch() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[0], Tier::Lcq);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let address = invocation.payload().preferred().unwrap().canonical.get();
    let fault = invocation.fault(address + 12).unwrap();
    let id = fault.unit.id;
    let old = Arc::downgrade(process.lock().units.tables[0].as_ref().unwrap());
    publish(&process, &cursor, &[4], Tier::Lcq);
    assert_eq!(process.collect_tables().unwrap(), 0);
    assert!(old.upgrade().is_some());
    assert_eq!(fault.unit.id, id); // Captured metadata survives normal-stack work.
    assert_eq!(invocation.fault(address + 12).unwrap().unit.id, id);
    drop(invocation);
    assert_eq!(process.collect_tables().unwrap(), 1);
    assert!(old.upgrade().is_none());
}

#[test]
fn newer_reader_does_not_pin_an_older_table_snapshot() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[0], Tier::Lcq);
    let old = Arc::downgrade(process.lock().units.tables[0].as_ref().unwrap());
    publish(&process, &cursor, &[4], Tier::Lcq);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    assert_eq!(process.collect_tables().unwrap(), 1);
    assert!(old.upgrade().is_none());
    let address = invocation.payload().preferred().unwrap().canonical.get();
    assert!(invocation.fault(address + 12).is_some());
}

#[test]
fn closing_after_preparation_rejects_every_entry_and_returns_exact_span() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let publications = [
        process.reserve(key(0)).unwrap(),
        process.reserve(key(4)).unwrap(),
    ];
    let candidate = input(&process, &[0, 4], Tier::Lcq);
    let address = candidate.code.allocation.address();
    let prepared = process
        .prepare_unit(&publications, candidate, &cursor)
        .unwrap();
    let ticket = process.request(Reason::MappingChange).unwrap();
    assert_eq!(prepared.publish(), Err(Error::Closed));
    let state = process.lock();
    assert!(state.units.records.values().next().is_none());
    assert!(state.units.dependencies.entries.is_empty());
    assert!(state.units.tables.iter().all(Option::is_none));
    for publication in publications {
        assert!(
            state
                .dispatch
                .get(publication.slot)
                .unwrap()
                .snapshot()
                .preferred()
                .is_none()
        );
    }
    drop(state);
    assert!(!ticket.is_complete().unwrap());
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(
        input(&process, &[0], Tier::Lcq).code.allocation.address(),
        address
    );
}

#[test]
fn changed_cursor_rejects_prepared_output_without_replacing_old_roots() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    let prepared = process
        .prepare_unit(
            &[process.reserve(key(0)).unwrap()],
            input(&process, &[0], Tier::Lcq),
            &cursor,
        )
        .unwrap();
    cursor.store(1, Ordering::Release);
    assert_eq!(prepared.publish(), Err(Error::StalePublication));
    let state = process.lock();
    assert_eq!(state.units.records.values().count(), 1);
    assert_eq!(
        state
            .units
            .dependencies
            .for_page(GuestPhysicalPageId::new(7))
            .next()
            .unwrap()
            .unit,
        old
    );
}

#[test]
fn competing_preparations_have_one_winner_and_no_partial_multi_entry_output() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let publications = [
        process.reserve(key(0)).unwrap(),
        process.reserve(key(4)).unwrap(),
    ];
    let first = process
        .prepare_unit(&publications, input(&process, &[0, 4], Tier::Lcq), &cursor)
        .unwrap();
    let second = process
        .prepare_unit(&publications, input(&process, &[0, 4], Tier::Lcq), &cursor)
        .unwrap();
    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        let a = scope.spawn(|| {
            barrier.wait();
            first.publish()
        });
        let b = scope.spawn(|| {
            barrier.wait();
            second.publish()
        });
        let results = [a.join().unwrap(), b.join().unwrap()];
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert!(results.contains(&Err(Error::StalePublication)));
    });
    let state = process.lock();
    assert_eq!(state.units.records.values().count(), 1);
    let entries: Vec<_> = publications
        .iter()
        .map(|publication| {
            state
                .dispatch
                .get(publication.slot)
                .unwrap()
                .snapshot()
                .preferred()
                .unwrap()
        })
        .collect();
    assert_eq!(entries[0], entries[1]);
}

#[test]
fn different_keys_cannot_publish_a_table_built_from_a_stale_snapshot() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let prepared = process
        .prepare_unit(
            &[process.reserve(key(0)).unwrap()],
            input(&process, &[0], Tier::Lcq),
            &cursor,
        )
        .unwrap();
    publish(&process, &cursor, &[4], Tier::Lcq);
    assert_eq!(prepared.publish(), Err(Error::StalePublication));
    assert_eq!(process.lock().units.records.values().count(), 1);
}

#[test]
fn independent_segments_do_not_conflict_on_a_predicted_registry_slot() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[0], Tier::Lcq);
    let lcq = process
        .prepare_unit(
            &[process.reserve(key(4)).unwrap()],
            input(&process, &[4], Tier::Lcq),
            &cursor,
        )
        .unwrap();
    let hcq = process
        .prepare_unit(
            &[process.reserve(key(0)).unwrap()],
            input(&process, &[0], Tier::Hcq),
            &cursor,
        )
        .unwrap();
    let lcq = lcq.publish().unwrap();
    let hcq = hcq.publish().unwrap();
    assert_ne!(lcq, hcq);
    assert_eq!(process.lock().units.records.values().count(), 3);
}

#[test]
fn registry_and_dependency_growth_preserve_all_owners_and_aliases() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let mut handles = Vec::new();
    for i in 0..40 {
        let pc = i * 4;
        let mut candidate = input(&process, &[pc], Tier::Lcq);
        candidate.dependencies[0].mapping_generation = MappingGeneration::new(i);
        handles.push(
            process
                .prepare_unit(&[process.reserve(key(pc)).unwrap()], candidate, &cursor)
                .unwrap()
                .publish()
                .unwrap(),
        );
    }
    let state = process.lock();
    let deps: Vec<_> = state
        .units
        .dependencies
        .for_page(GuestPhysicalPageId::new(7))
        .collect();
    assert_eq!(deps.len(), 40);
    assert_eq!(state.units.records.values().count(), 40);
    assert_eq!(state.units.retired_tables.len(), 1); // Collected/reused each preparation.
    for (i, handle) in handles.iter().enumerate() {
        let unit = &state.units.records.get(handle.0).unwrap().code;
        assert_eq!(unit.entries[0].key, key(i as u64 * 4));
        assert!(deps.iter().any(|dependency| dependency.unit == *handle
            && dependency.page.mapping_generation.get() == i as u64));
    }
}

#[test]
fn dropping_the_process_releases_coupled_metadata_and_span_leases() {
    let cache = Cache::new().unwrap();
    let initial = cache.usage().unwrap();
    let cursor = AtomicU64::new(0);
    let process = Lifetime::new(Arc::clone(&cache)).unwrap();
    publish(&process, &cursor, &[0], Tier::Lcq);
    publish(&process, &cursor, &[0], Tier::Hcq);
    let unit = Arc::downgrade(&process.lock().units.records.values().next().unwrap().code);
    drop(process); // No reader or compiler reference survives this scope.
    assert!(unit.upgrade().is_none());
    // Process/directory are gone; all leases must actually have been released.
    for index in 0..SEGMENTS {
        assert!(unsafe { cache.decommit_empty(index) }.unwrap());
    }
    assert_eq!(cache.usage().unwrap(), initial);
}

#[test]
fn hcq_family_owns_baselines_and_preserves_lcq_dispatch() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let lcq = publish(&process, &cursor, &[0, 4], Tier::Lcq);
    let hcq = publish(&process, &cursor, &[0, 4], Tier::Hcq);
    let state = process.lock();
    let family = state.units.families.values().next().unwrap();
    assert!(Arc::ptr_eq(
        &family.unit,
        &state.units.records.get(hcq.0).unwrap().code
    ));
    assert_eq!(family.baselines.len(), 1);
    let baseline = &state.units.records.get(lcq.0).unwrap().code;
    assert!(Arc::ptr_eq(&family.baselines[0], baseline));
    assert_eq!(state.units.dependencies.entries.len(), 2);
    for pc in [0, 4] {
        let handle = *state.keys.get(&key(pc)).unwrap();
        let payload = state.dispatch.get(handle).unwrap().snapshot();
        assert_eq!(payload.lcq().unwrap().unit, baseline.id);
        assert_eq!(payload.hcq().unwrap().family, family.id);
        assert_eq!(
            payload.preferred().unwrap().unit,
            state.units.records.get(hcq.0).unwrap().code.id
        );
    }
    drop(state);
    assert!(matches!(
        process.prepare_unit(
            &[process.reserve(key(0)).unwrap()],
            input(&process, &[0], Tier::Hcq),
            &cursor
        ),
        Err(Error::InvalidUnit(_))
    ));
}

#[test]
fn malformed_metadata_and_foreign_allocations_are_rejected_before_publication() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let publication = process.reserve(key(0)).unwrap();
    for case in 0..8 {
        let mut candidate = input(&process, &[0], Tier::Lcq);
        match case {
            0 => candidate.entries[0].fast_offset = 16,
            1 => candidate.faults[0].native_start = 11,
            2 => candidate.faults[0].state_map = 1,
            3 => candidate.faults = Box::new([]),
            4 => candidate.entries[0].contract.live_in.integer.x[0] = true,
            5 => candidate.tier = Tier::Hcq,
            6 => candidate.instructions[0].key = InstructionKey::new(key(8)).unwrap(),
            7 => candidate.states[0].state.site.source = CodeVersion::new(u64::MAX).unwrap(),
            _ => unreachable!(),
        }
        assert!(matches!(
            process.prepare_unit(&[publication], candidate, &cursor),
            Err(Error::InvalidUnit(_))
        ));
    }
    let other = Lifetime::new(Cache::new().unwrap()).unwrap();
    assert!(matches!(
        process.prepare_unit(&[publication], input(&other, &[0], Tier::Lcq), &cursor),
        Err(Error::InvalidUnit(_))
    ));
    assert!(process.lock().units.records.values().next().is_none());
}

#[test]
fn metadata_pressure_and_epoch_exhaustion_never_leave_reachable_fragments() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let publication = process.reserve(key(0)).unwrap();
    let candidate = input(&process, &[0], Tier::Lcq);
    let usage = process.cache.usage().unwrap();
    let pressure = process
        .cache
        .charge_metadata(HARD_BYTES - usage.total(), Tier::Lcq)
        .unwrap();
    assert!(matches!(
        process.prepare_unit(&[publication], candidate, &cursor),
        Err(Error::Capacity(_))
    ));
    drop(pressure);
    let prepared = process
        .prepare_unit(&[publication], input(&process, &[0], Tier::Lcq), &cursor)
        .unwrap();
    process.lock().executions = CheckedCounter::exhausted();
    assert!(matches!(prepared.publish(), Err(Error::Exhausted(_))));
    let state = process.lock();
    assert!(state.units.records.values().next().is_none());
    assert!(state.units.tables.iter().all(Option::is_none));
    assert!(
        state
            .dispatch
            .get(publication.slot)
            .unwrap()
            .snapshot()
            .preferred()
            .is_none()
    );
}

#[test]
fn optimizer_identity_exhaustion_keeps_baseline_admission_and_publication_open() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let baseline = publish(&process, &cursor, &[0], Tier::Lcq);
    process.lock().units.family_ids = CheckedCounter::exhausted();
    assert!(matches!(
        process.prepare_unit(
            &[process.reserve(key(0)).unwrap()],
            input(&process, &[0], Tier::Hcq),
            &cursor
        ),
        Err(Error::Exhausted(_))
    ));
    let state = process.lock();
    assert!(state.open().is_ok());
    assert!(state.units.hcq_failure.is_some());
    assert!(state.units.families.values().next().is_none());
    assert!(state.units.records.get(baseline.0).is_some());
    drop(state);
    publish(&process, &cursor, &[4], Tier::Lcq);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    assert!(
        unsafe { reader.admit(&mut frame, key(0)) }
            .unwrap()
            .is_some()
    );
}

#[test]
fn emission_identity_is_final_before_codegen_and_old_admission_cannot_publish() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let candidate = input(&process, &[0], Tier::Lcq);
    let version = candidate.identity.version();
    assert_eq!(candidate.states[0].state.site.source, version);
    process.request(Reason::MappingChange).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    // Even a fresh dispatch reservation cannot launder bytes emitted against
    // the previous admission. No metadata relabels their embedded exit version.
    assert!(matches!(
        process.prepare_unit(&[process.reserve(key(0)).unwrap()], candidate, &cursor),
        Err(Error::StalePublication)
    ));
    let handle = publish(&process, &cursor, &[0], Tier::Lcq);
    assert!(process.snapshot(handle).unwrap().version > version);
}

#[test]
fn emission_identity_exhaustion_disables_hcq_but_lcq_fails_closed() {
    for tier in [Tier::Lcq, Tier::Hcq] {
        let process = process();
        process.lock().units.versions = CheckedCounter::exhausted();
        assert!(matches!(process.begin_unit(tier), Err(Error::Exhausted(_))));
        let state = process.lock();
        assert_eq!(state.open().is_ok(), tier == Tier::Hcq);
        assert!(state.units.records.is_empty());
    }
}

#[test]
fn directory_bounds_exclude_final_missing_mebibyte_and_unpublished_segments() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[0], Tier::Lcq);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let base = process.cache.executable_base();
    for pc in [
        0,
        base - 1,
        base + SEGMENT_BYTES,
        base + WINDOW_BYTES - 1,
        base + WINDOW_BYTES,
        base + SEGMENTS * SEGMENT_BYTES - 1,
        usize::MAX,
    ] {
        assert!(invocation.fault(pc).is_none());
    }
}

#[test]
fn publication_before_closing_remains_attributable_until_reader_exit() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[0], Tier::Lcq);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let address = invocation.payload().preferred().unwrap().canonical.get();
    process.request(Reason::Eviction).unwrap();
    let (tx, rx) = mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut transition = process.try_transition().unwrap().unwrap();
            transition.wait_closed().unwrap();
            tx.send(()).unwrap();
        });
        assert!(invocation.fault(address + 12).is_some());
        assert_eq!(rx.try_recv(), Err(mpsc::TryRecvError::Empty));
        drop(invocation);
        rx.recv().unwrap();
    });
}
