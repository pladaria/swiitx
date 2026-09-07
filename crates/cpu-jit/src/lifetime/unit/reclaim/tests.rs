use super::super::tests::{frame, input, key, process, publish};
use super::*;
use crate::executable::{HARD_BYTES, SEGMENT_BYTES, SOFT_BYTES};
use nixe_cpu::state::a64::A64State;

fn drain(process: &Lifetime) {
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn snapshots_pin_actual_code_dependencies_and_slots_until_last_release() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    let survivor = publish(&process, &cursor, &[4], Tier::Lcq);
    let snapshot = process.snapshot(old).unwrap();
    let address = snapshot.code.allocation.address();
    let ticket = process.retire_unit(old).unwrap();
    drain(&process);
    assert!(ticket.is_complete().unwrap());
    assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
    let clone = snapshot.clone(); // Existing compiler ownership remains valid.
    drop(snapshot);
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert_eq!(process.lock().units.dependencies.entries.len(), 2);
    let unrelated = input(&process, &[8], Tier::Lcq);
    assert_ne!(unrelated.code.allocation.address(), address);
    drop(unrelated);
    drop(clone);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert_eq!(process.lock().units.dependencies.entries.len(), 1);
    assert!(process.lock().units.records.get(old.0).is_none());
    assert!(process.lock().units.records.get(survivor.0).is_some());
    let new = publish(&process, &cursor, &[0], Tier::Lcq);
    assert_ne!(new, old);
    assert_eq!(
        process.snapshot(new).unwrap().code.allocation.address(),
        address
    );
    assert!(matches!(process.retire_unit(old), Err(Error::StaleUnit)));
    assert!(process.snapshot(new).is_ok());
}

#[test]
fn newer_fault_reader_needs_a_second_grace_period_after_unlink() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    publish(&process, &cursor, &[4], Tier::Lcq);
    let address = process.snapshot(old).unwrap().code.allocation.address();
    process.retire_unit(old).unwrap();
    drain(&process);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(4)) }
        .unwrap()
        .unwrap();
    let fault = invocation.fault(address + 12).unwrap();
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert!(invocation.fault(address + 12).is_none());
    assert_eq!(fault.unit.code.allocation.address(), address); // Old table borrow survives.
    assert!(process.lock().units.records.get(old.0).is_some());
    drop(invocation);
    assert_eq!(process.reclaim_units().unwrap(), 1);
}

#[test]
fn closure_waits_for_native_invocation_before_unlink() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    process.retire_unit(old).unwrap();
    assert_eq!(process.reclaim_units().unwrap(), 0);
    std::thread::scope(|scope| {
        let (send, receive) = std::sync::mpsc::channel();
        let process = &process;
        let closer = scope.spawn(move || {
            let mut transition = process.try_transition().unwrap().unwrap();
            send.send(()).unwrap();
            transition.wait_closed().unwrap();
            transition.drain_retirements().unwrap();
            assert_eq!(process.reclaim_units().unwrap(), 1);
            transition.batch().unwrap().complete().unwrap();
            assert!(transition.try_reopen().unwrap());
        });
        receive.recv().unwrap();
        assert_eq!(process.lock().phase, Phase::Closing);
        assert!(
            invocation
                .fault(invocation.payload().preferred().unwrap().canonical.get() + 12)
                .is_some()
        );
        drop(invocation);
        closer.join().unwrap();
    });
}

#[test]
fn empty_segment_republication_uses_same_address_but_fresh_generation() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    let snapshot = process.snapshot(old).unwrap();
    let address = snapshot.code.allocation.address();
    let generation = snapshot.code.allocation.generation;
    drop(snapshot);
    process.retire_unit(old).unwrap();
    drain(&process);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert_eq!(process.cache.usage().unwrap().committed, 0);
    let new = publish(&process, &cursor, &[0], Tier::Lcq);
    let snapshot = process.snapshot(new).unwrap();
    assert_eq!(snapshot.code.allocation.address(), address);
    assert_ne!(snapshot.code.allocation.generation, generation);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    assert_eq!(invocation.fault(address + 12).unwrap().unit.id, snapshot.id);
}

#[test]
fn hcq_retirement_restores_all_baselines_and_releases_family_pins() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let lcq = publish(&process, &cursor, &[0, 4], Tier::Lcq);
    let hcq = publish(&process, &cursor, &[0, 4], Tier::Hcq);
    assert!(matches!(
        process.retire_unit(lcq),
        Err(Error::PinnedBaseline)
    ));
    let family = process
        .lock()
        .units
        .records
        .get(hcq.0)
        .unwrap()
        .family
        .unwrap();
    let compiler = process.snapshot(hcq).unwrap();
    process.retire_unit(hcq).unwrap();
    drain(&process);
    for pc in [0, 4] {
        let state = process.lock();
        let payload = state
            .dispatch
            .get(*state.keys.get(&key(pc)).unwrap())
            .unwrap()
            .snapshot();
        assert!(payload.hcq().is_none());
        assert_eq!(payload.preferred(), payload.lcq());
    }
    assert!(!process.lock().units.families.is_empty()); // Held until span returned.
    process.retire_unit(lcq).unwrap();
    drain(&process);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    drop(compiler);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert!(process.lock().units.families.is_empty());
    assert!(process.lock().units.families.get(family).is_none());
    assert!(process.lock().dispatch.is_empty());
}

#[test]
fn in_flight_family_prevents_baseline_eviction_until_cancelled() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let lcq = publish(&process, &cursor, &[0], Tier::Lcq);
    let prepared = process
        .prepare_unit(
            &[process.reserve(key(0)).unwrap()],
            input(&process, &[0], Tier::Hcq),
            &cursor,
        )
        .unwrap();
    assert!(matches!(
        process.retire_unit(lcq),
        Err(Error::PinnedBaseline)
    ));
    drop(prepared);
    process.retire_unit(lcq).unwrap();
    drain(&process);
    assert_eq!(process.reclaim_units().unwrap(), 1);
}

#[test]
fn lcq_cutover_cannot_be_acknowledged_without_draining_exact_old_owner() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    let new = publish(&process, &cursor, &[0], Tier::Lcq);
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert_eq!(
        transition.batch().unwrap().complete(),
        Err(Error::MaintenancePending)
    );
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
    assert!(process.snapshot(new).is_ok());
}

#[test]
fn handles_are_process_scoped_and_exhaustion_never_partially_unlinks() {
    let process = process();
    let other = super::super::tests::process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0, 4], Tier::Lcq);
    publish(&other, &cursor, &[0, 4], Tier::Lcq);
    assert!(matches!(other.snapshot(old), Err(Error::StaleUnit)));
    assert!(matches!(other.retire_unit(old), Err(Error::StaleUnit)));
    process.retire_unit(old).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    process.lock().reachabilities = CheckedCounter::exhausted();
    assert!(matches!(
        transition.drain_retirements(),
        Err(Error::Exhausted(_))
    ));
    let state = process.lock();
    for pc in [0, 4] {
        assert!(
            state
                .dispatch
                .get(*state.keys.get(&key(pc)).unwrap())
                .unwrap()
                .snapshot()
                .lcq()
                .is_some()
        );
    }
}

#[test]
fn closed_reclamation_makes_progress_with_no_free_metadata_budget() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    publish(&process, &cursor, &[4], Tier::Lcq);
    // Model unrelated live charged storage filling the real configured budget.
    let charge = process
        .cache
        .charge_metadata(
            HARD_BYTES - process.cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    process.retire_unit(old).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    assert_eq!(process.reclaim_units().unwrap(), 1);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(charge);
}

#[test]
fn pressure_evicts_oldest_hcq_before_any_lcq_without_waiting_on_snapshots() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let lcq = publish(&process, &cursor, &[0, 4], Tier::Lcq);
    let first = publish(&process, &cursor, &[0], Tier::Hcq);
    let second = publish(&process, &cursor, &[4], Tier::Hcq);
    let retained = process.snapshot(first).unwrap();
    // Both HCQ units share a segment; neither eviction refunds that backing
    // while the first compiler snapshot remains. The pass must not wait.
    let charge = process
        .cache
        .charge_metadata(
            SOFT_BYTES + SEGMENT_BYTES / 2 - process.cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    process.request(Reason::Eviction).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert!(matches!(
        transition.relieve_pressure(0, Tier::Hcq),
        Err(Error::Capacity(_))
    ));
    assert!(process.lock().units.records.get(second.0).is_none());
    assert!(process.lock().units.records.get(lcq.0).is_some());
    assert!(process.lock().units.records.get(first.0).is_some());
    drop(retained);
    transition.relieve_pressure(0, Tier::Hcq).unwrap();
    assert!(process.lock().units.records.get(first.0).is_none());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(charge);
}

#[test]
fn shutdown_waits_for_snapshots_and_unpublished_outputs_then_unmaps() {
    let process = process();
    let cache = Arc::clone(&process.cache);
    let cursor = AtomicU64::new(0);
    let lcq = publish(&process, &cursor, &[0], Tier::Lcq);
    let snapshot = process.snapshot(lcq).unwrap();
    let staged = input(&process, &[4], Tier::Lcq);
    let reader = process.register().unwrap();
    let address = snapshot.code.allocation.address();
    let ticket = process.request(Reason::Shutdown).unwrap();
    // Identify this backing, not merely its address: another parallel test
    // may reserve the same virtual range as soon as shutdown unmaps it.
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
    let backing = maps
        .lines()
        .find_map(|line| {
            let fields: Vec<_> = line.split_whitespace().collect();
            let (start, end) = fields[0].split_once('-').unwrap();
            let start = usize::from_str_radix(start, 16).unwrap();
            let end = usize::from_str_radix(end, 16).unwrap();
            (start <= address && address < end)
                .then(|| (fields[3].to_owned(), fields[4].to_owned()))
        })
        .unwrap();
    assert_ne!(backing.1, "0");
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert!(!transition.try_finish_shutdown().unwrap());
    assert_eq!(
        transition.batch().unwrap().complete(),
        Err(Error::MaintenancePending)
    );
    drop(snapshot);
    assert!(!transition.try_finish_shutdown().unwrap());
    drop(staged);
    assert!(transition.try_finish_shutdown().unwrap());
    transition.batch().unwrap().complete().unwrap();
    assert!(ticket.is_complete().unwrap());
    assert!(transition.try_reopen().unwrap());
    assert_eq!(process.cache.usage().unwrap().committed, 0);
    assert_eq!(process.lock().units.records.capacity(), 0);
    assert_eq!(process.lock().dispatch.capacity(), 0);
    assert_eq!(process.lock().readers.capacity(), 0);
    assert!(matches!(
        process.cache.charge_metadata(1, Tier::Lcq),
        Err(crate::executable::Error::Closed)
    ));
    for line in std::fs::read_to_string("/proc/self/maps").unwrap().lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert!(fields[3] != backing.0 || fields[4] != backing.1);
    }
    drop(reader); // Inactive registrations can outlive terminal cleanup.
    drop(transition);
    drop(process);
    assert_eq!(
        cache.usage().unwrap().metadata,
        size_of::<crate::executable::Cache>() + 2 * size_of::<usize>()
    );
}

#[test]
fn partial_lcq_replacement_retains_other_roots_until_final_cutover() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0, 4], Tier::Lcq);
    publish(&process, &cursor, &[0], Tier::Lcq);
    drain(&process);
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert!(process.snapshot(old).is_ok());
    publish(&process, &cursor, &[4], Tier::Lcq);
    drain(&process);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
}

#[test]
fn requests_arriving_during_closed_keep_their_exact_targets_pending() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let first = publish(&process, &cursor, &[0], Tier::Lcq);
    let second = publish(&process, &cursor, &[4], Tier::Lcq);
    let first_ticket = process.retire_unit(first).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    let batch = transition.batch().unwrap();
    let second_ticket = process.retire_unit(second).unwrap();
    batch.complete().unwrap();
    assert!(first_ticket.is_complete().unwrap());
    assert!(!second_ticket.is_complete().unwrap());
    assert!(!transition.try_reopen().unwrap());
    assert_eq!(
        transition.batch().unwrap().complete(),
        Err(Error::MaintenancePending)
    );
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    assert_eq!(process.reclaim_units().unwrap(), 2);
    transition.batch().unwrap().complete().unwrap();
    assert!(second_ticket.is_complete().unwrap());
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn pressure_returns_empty_compiler_reservations_without_reviving_old_handles() {
    let process = process();
    let old = process.reserve(key(0)).unwrap();
    process.request(Reason::Eviction).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.relieve_pressure(0, Tier::Lcq).unwrap();
    assert!(process.lock().keys.is_empty());
    assert!(process.lock().dispatch.is_empty());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    let new = process.reserve(key(0)).unwrap();
    assert_ne!(old.slot, new.slot);
    assert_eq!(process.retire_dispatch(old), Err(Error::StalePublication));
}

#[test]
fn repeated_lcq_hcq_churn_reuses_bounded_metadata_and_executable_storage() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let mut expected = None;
    for _ in 0..128 {
        let lcq = publish(&process, &cursor, &[0, 4], Tier::Lcq);
        let hcq = publish(&process, &cursor, &[0, 4], Tier::Hcq);
        process.retire_unit(hcq).unwrap();
        drain(&process);
        process.retire_unit(lcq).unwrap();
        drain(&process);
        assert_eq!(process.reclaim_units().unwrap(), 2);
        let state = process.lock();
        assert!(state.units.records.is_empty());
        assert!(state.units.families.is_empty());
        assert!(state.dispatch.is_empty());
        assert!(state.keys.is_empty());
        assert!(state.units.dependencies.entries.is_empty());
        let capacities = (
            state.units.records.capacity(),
            state.units.families.capacity(),
            state.dispatch.capacity(),
        );
        drop(state);
        let usage = process.cache.usage().unwrap();
        assert_eq!(usage.committed, 0);
        assert_eq!(
            *expected.get_or_insert((capacities, usage)),
            (capacities, usage)
        );
    }
}

#[test]
fn compiler_table_snapshot_blocks_in_place_mutation_not_unlink() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let old = publish(&process, &cursor, &[0], Tier::Lcq);
    publish(&process, &cursor, &[4], Tier::Lcq);
    let prepared = process
        .prepare_unit(
            &[process.reserve(key(8)).unwrap()],
            input(&process, &[8], Tier::Lcq),
            &cursor,
        )
        .unwrap();
    let charge = process
        .cache
        .charge_metadata(
            HARD_BYTES - process.cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    process.retire_unit(old).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert!(process.lock().units.records.get(old.0).is_some());
    drop(prepared);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(charge);
}

#[test]
fn pressure_uses_creation_order_and_lcq_reports_hard_capacity_precisely() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let lcq = publish(&process, &cursor, &[0, 4], Tier::Lcq);
    let first = publish(&process, &cursor, &[0], Tier::Hcq);
    let second = publish(&process, &cursor, &[4], Tier::Hcq);
    let snapshots = [
        process.snapshot(first).unwrap(),
        process.snapshot(second).unwrap(),
    ];
    let charge = process
        .cache
        .charge_metadata(
            SOFT_BYTES + SEGMENT_BYTES / 2 - process.cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    process.request(Reason::Eviction).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert!(matches!(
        transition.relieve_pressure(0, Tier::Hcq),
        Err(Error::Capacity(_))
    ));
    {
        let state = process.lock();
        let Lifecycle::Retired(a) = state.units.records.get(first.0).unwrap().lifecycle else {
            panic!()
        };
        let Lifecycle::Retired(b) = state.units.records.get(second.0).unwrap().lifecycle else {
            panic!()
        };
        assert!(a < b);
        assert_eq!(
            state.units.records.get(lcq.0).unwrap().lifecycle,
            Lifecycle::Published
        );
    }
    assert!(matches!(
        transition.relieve_pressure(HARD_BYTES, Tier::Lcq),
        Err(Error::Capacity("640 MiB code+metadata hard limit"))
    ));
    drop(snapshots);
    assert_eq!(process.reclaim_units().unwrap(), 2);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(charge);
}
