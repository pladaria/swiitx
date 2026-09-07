use super::*;
use crate::abi::{
    CodeUnitId, CodeVersion, FamilyVersion, FpSpecialization, HcqFamilyId, PollBudget,
};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::A64State;
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};
use std::num::NonZeroUsize;
use std::sync::{Barrier, mpsc};

fn new_process() -> Lifetime {
    Lifetime::new(Cache::new().unwrap()).unwrap()
}

fn key(pc: u64) -> BlockKey {
    BlockKey::new(
        ProcessCpuContext::new(TargetPlatform::Switch1, AddressSpaceId::new(1)),
        GuestVirtualAddress::new(pc),
        FpSpecialization::Dynamic,
    )
    .unwrap()
}

// These deliberately are not executable pointers. Tests inspect publication
// and lifetime only; storage and real gateway execution belong to later steps.
fn entry(version: u64) -> PublishedEntry {
    PublishedEntry {
        unit: CodeUnitId::new(version + 10).unwrap(),
        version: CodeVersion::new(version).unwrap(),
        canonical: NonZeroUsize::new(version as usize * 16).unwrap(),
        fast: NonZeroUsize::new(version as usize * 16 + 8).unwrap(),
    }
}

fn install(process: &Lifetime, key: BlockKey, version: u64) -> ReachabilityVersion {
    process
        .publish(process.reserve(key).unwrap(), Some(entry(version)), None)
        .unwrap()
}

fn frame(state: &mut A64State) -> NativeFrame<'_> {
    NativeFrame::new(state, PollBudget::new(77, 1000).unwrap())
}

fn drain(process: &Lifetime) {
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    if process.lock().shutdown {
        assert!(transition.try_finish_shutdown().unwrap());
    }
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn reader_announces_before_lookup_and_protects_copied_payload_until_exit() {
    let process = Arc::new(new_process());
    let reachability = install(&process, key(0), 1);
    let mut reader = process.register().unwrap();
    let announcement = Arc::clone(&reader.announcement);
    let mut state = A64State::default();
    let mut frame = frame(&mut state);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    assert_eq!(announcement.load(Ordering::Acquire), 1);
    assert_eq!(invocation.payload().reachability(), reachability);
    assert_eq!(invocation.payload().preferred(), Some(entry(1)));
    install(&process, key(0), 2);
    assert_eq!(invocation.payload().preferred(), Some(entry(1)));
    drop(invocation);
    assert_eq!(announcement.load(Ordering::Acquire), 0);
    assert_eq!(
        (
            frame.execution_epoch,
            frame.admission_epoch,
            frame.host_fp.saved
        ),
        (0, 0, 0)
    );
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    assert_eq!(invocation.payload().preferred(), Some(entry(2)));
}

#[test]
fn miss_unavailable_slot_and_closed_admission_finish_fp_and_clear_epochs() {
    let process = Arc::new(new_process());
    let mut reader = process.register().unwrap();
    let mut state = A64State::default();
    let mut frame = frame(&mut state);
    for available_slot in [false, true] {
        if available_slot {
            process.reserve(key(0)).unwrap();
        }
        assert!(
            unsafe { reader.admit(&mut frame, key(0)) }
                .unwrap()
                .is_none()
        );
        assert_eq!(reader.announcement.load(Ordering::Acquire), 0);
        assert_eq!(
            (
                frame.execution_epoch,
                frame.admission_epoch,
                frame.host_fp.saved
            ),
            (0, 0, 0)
        );
    }
    process.request(Reason::MappingChange).unwrap();
    assert!(matches!(
        unsafe { reader.admit(&mut frame, key(0)) },
        Err(Error::Closed)
    ));
    assert_eq!(reader.announcement.load(Ordering::Acquire), 0);
    assert_eq!(
        (
            frame.execution_epoch,
            frame.admission_epoch,
            frame.host_fp.saved
        ),
        (0, 0, 0)
    );
}

#[test]
fn stale_and_foreign_publications_never_replace_current_payload() {
    let process = new_process();
    let old = process.reserve(key(0)).unwrap();
    process.publish(old, Some(entry(1)), None).unwrap();
    assert_eq!(
        process.publish(old, Some(entry(2)), None),
        Err(Error::StalePublication)
    );
    let captured = process.reserve(key(0)).unwrap();
    let foreign = new_process();
    foreign.reserve(key(0)).unwrap();
    assert_eq!(
        foreign.publish(captured, Some(entry(3)), None),
        Err(Error::StalePublication)
    );
    process.request(Reason::Eviction).unwrap();
    assert_eq!(
        process.publish(captured, Some(entry(4)), None),
        Err(Error::Closed)
    );
    drain(&process);
    assert_eq!(
        process.publish(captured, Some(entry(5)), None),
        Err(Error::StalePublication)
    );
    let state = process.lock();
    assert_eq!(
        state
            .dispatch
            .get(captured.slot)
            .unwrap()
            .snapshot()
            .preferred(),
        Some(entry(1))
    );
}

#[test]
fn competing_publishers_have_one_winner_for_the_same_exact_snapshot() {
    let process = new_process();
    let captured = process.reserve(key(0)).unwrap();
    let start = Barrier::new(3);
    std::thread::scope(|scope| {
        let a = scope.spawn(|| {
            start.wait();
            process.publish(captured, Some(entry(1)), None)
        });
        let b = scope.spawn(|| {
            start.wait();
            process.publish(captured, Some(entry(2)), None)
        });
        start.wait();
        let results = [a.join().unwrap(), b.join().unwrap()];
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| **result == Err(Error::StalePublication))
                .count(),
            1
        );
    });
}

#[test]
fn coherent_tier_payloads_survive_concurrent_replacement() {
    let process = Arc::new(new_process());
    install(&process, key(0), 1);
    let start = Barrier::new(2);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            start.wait();
            for version in 2..502 {
                let hcq = HcqEntry {
                    entry: entry(version),
                    family: HcqFamilyId::new(version + 1000).unwrap(),
                    family_version: FamilyVersion::new(version + 2000).unwrap(),
                };
                process
                    .publish(process.reserve(key(0)).unwrap(), Some(entry(1)), Some(hcq))
                    .unwrap();
            }
        });
        let mut reader = process.register().unwrap();
        let mut state = A64State::default();
        let mut frame = frame(&mut state);
        start.wait();
        for _ in 0..500 {
            let invocation = unsafe { reader.admit(&mut frame, key(0)) }
                .unwrap()
                .unwrap();
            let payload = invocation.payload();
            assert_eq!(payload.lcq(), Some(entry(1)));
            let preferred = payload.preferred().unwrap();
            assert_eq!(preferred, entry(preferred.version.get()));
            if let Some(hcq) = payload.hcq() {
                assert_eq!(hcq.entry, preferred);
                assert_eq!(hcq.family.get(), preferred.version.get() + 1000);
                assert_eq!(hcq.family_version.get(), preferred.version.get() + 2000);
            }
        }
    });
}

#[test]
fn closing_waits_for_admitted_reader_even_before_its_machine_jump() {
    let process = Arc::new(new_process());
    install(&process, key(0), 1);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let ticket = process.request(Reason::Eviction).unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (closed_tx, closed_rx) = mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut transition = process.try_transition().unwrap().unwrap();
            started_tx.send(()).unwrap();
            transition.wait_closed().unwrap();
            closed_tx.send(()).unwrap();
            transition.batch().unwrap().complete().unwrap();
            assert!(transition.try_reopen().unwrap());
        });
        started_rx.recv().unwrap();
        assert_eq!(process.lock().phase, Phase::Closing);
        assert!(process.try_transition().unwrap().is_none());
        assert_eq!(closed_rx.try_recv(), Err(mpsc::TryRecvError::Empty));
        assert!(!ticket.is_complete().unwrap());
        drop(invocation);
        closed_rx.recv().unwrap();
    });
    assert!(ticket.is_complete().unwrap());
    assert_eq!(process.control_word().load(Ordering::Acquire), 0);
}

#[test]
fn requests_arriving_in_closed_survive_old_batch_acknowledgement() {
    let process = new_process();
    let first = process.request(Reason::MappingChange).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    let batch = transition.batch().unwrap();
    assert_eq!(batch.reasons().collect::<Vec<_>>(), [Reason::MappingChange]);
    let second = process.request(Reason::MappingChange).unwrap();
    let third = process.request(Reason::TierCutover).unwrap();
    batch.complete().unwrap();
    assert!(first.is_complete().unwrap());
    assert!(!second.is_complete().unwrap());
    assert!(!third.is_complete().unwrap());
    assert!(!transition.try_reopen().unwrap());
    assert_ne!(process.control_word().load(Ordering::Acquire), 0);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(second.is_complete().unwrap());
    assert!(third.is_complete().unwrap());
}

#[test]
fn abandoned_transition_and_batch_do_not_acknowledge_or_reopen() {
    let process = new_process();
    let ticket = process.request(Reason::MappingChange).unwrap();
    {
        let mut transition = process.try_transition().unwrap().unwrap();
        transition.wait_closed().unwrap();
        let _batch = transition.batch().unwrap();
    }
    assert!(!ticket.is_complete().unwrap());
    assert_eq!(process.lock().phase, Phase::Closed);
    drain(&process);
    assert!(ticket.is_complete().unwrap());
}

#[test]
fn completed_transition_cannot_clear_a_new_owners_claim() {
    let process = new_process();
    process.request(Reason::Eviction).unwrap();
    let mut first = process.try_transition().unwrap().unwrap();
    first.wait_closed().unwrap();
    first.batch().unwrap().complete().unwrap();
    assert!(first.try_reopen().unwrap());
    process.request(Reason::MappingChange).unwrap();
    let second = process.try_transition().unwrap().unwrap();
    assert_eq!(first.wait_closed(), Err(Error::Closed));
    drop(first);
    assert!(process.try_transition().unwrap().is_none());
    drop(second);
    drain(&process);
}

#[test]
fn shutdown_is_terminal_and_drains_earlier_work() {
    let process = Arc::new(new_process());
    let old = process.request(Reason::Eviction).unwrap();
    let shutdown = process.request(Reason::Shutdown).unwrap();
    drain(&process);
    assert!(old.is_complete().unwrap());
    assert!(shutdown.is_complete().unwrap());
    assert_eq!(process.lock().phase, Phase::Closed);
    assert!(matches!(process.reserve(key(0)), Err(Error::Shutdown)));
    assert!(matches!(process.register(), Err(Error::Shutdown)));
    assert!(matches!(
        process.request(Reason::MappingChange),
        Err(Error::Shutdown)
    ));
    assert!(process.try_transition().unwrap().is_none());
    assert_ne!(process.control_word().load(Ordering::Acquire), 0);
}

#[test]
fn removed_dispatch_slots_wait_for_epochs_then_reuse_actual_storage() {
    let process = Arc::new(new_process());
    install(&process, key(0), 1);
    let old = process.reserve(key(0)).unwrap();
    assert_eq!(process.retire_dispatch(old), Err(Error::OccupiedDispatch));
    let mut reader = process.register().unwrap();
    let mut state = A64State::default();
    let mut frame = frame(&mut state);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    process.publish(old, None, None).unwrap();
    let empty = process.reserve(key(0)).unwrap();
    process.retire_dispatch(empty).unwrap();
    assert_eq!(process.collect_dispatch().unwrap(), 0);
    let new = process.reserve(key(0)).unwrap();
    assert_ne!(old.slot, new.slot);
    drop(invocation);
    assert_eq!(process.collect_dispatch().unwrap(), 1);
    let reused = process.reserve(key(4)).unwrap();
    assert_ne!(old.slot, reused.slot);
    assert!(process.lock().dispatch.get(old.slot).is_none());
    assert_eq!(process.retire_dispatch(empty), Err(Error::StalePublication));
    assert_eq!(process.collect_dispatch().unwrap(), 0);
}

#[test]
fn readers_and_empty_dispatch_churn_reuse_bounded_slots() {
    let process = Arc::new(new_process());
    for i in 0..1000 {
        let reader = process.register().unwrap();
        let publication = process.reserve(key(i * 4)).unwrap();
        process.retire_dispatch(publication).unwrap();
        assert_eq!(process.collect_dispatch().unwrap(), 1);
        drop(reader);
    }
    let state = process.lock();
    assert_eq!(state.readers.capacity(), 16);
    assert_eq!(state.dispatch.capacity(), 16);
    assert!(state.keys.is_empty());
    assert!(state.keys.capacity() < 64);
    assert!(state.readers.values().next().is_none());
}

#[test]
fn poison_disables_admission_but_reader_cleanup_still_completes() {
    let process = Arc::new(new_process());
    install(&process, key(0), 1);
    let mut reader = process.register().unwrap();
    let mut state = A64State::default();
    let mut frame = frame(&mut state);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let _ = std::panic::catch_unwind(|| {
        let _state = process.state.lock().unwrap();
        panic!("test an interrupted state mutation");
    });
    drop(invocation);
    assert_eq!(reader.announcement.load(Ordering::Acquire), 0);
    assert_eq!(frame.host_fp.saved, 0);
    assert!(matches!(process.reserve(key(4)), Err(Error::Poisoned)));
    assert_ne!(process.control_word().load(Ordering::Acquire), 0);
}

#[test]
fn deferred_links_keep_their_ticket_and_control_request_for_the_next_stop() {
    let process = new_process();
    let link = process.request(Reason::LinkPatch).unwrap();
    let mut first = process.try_transition().unwrap().unwrap();
    first.wait_closed().unwrap();
    let batch = first.batch().unwrap();
    let safety = process.request(Reason::MappingChange).unwrap();
    batch.complete_with_links_deferred().unwrap();
    assert!(!first.try_reopen().unwrap());
    first
        .batch()
        .unwrap()
        .complete_with_links_deferred()
        .unwrap();
    assert!(first.try_reopen().unwrap());
    assert!(!link.is_complete().unwrap());
    assert!(safety.is_complete().unwrap());
    assert_eq!(
        process.control_word().load(Ordering::Acquire),
        1 << Reason::LinkPatch as usize
    );
    // No new request is needed to service the retained LinkPatch reason.
    drain(&process);
    assert!(link.is_complete().unwrap());
    assert_eq!(process.control_word().load(Ordering::Acquire), 0);
}

#[test]
fn shutdown_cannot_leave_deferred_link_records_behind() {
    let process = new_process();
    process.request(Reason::LinkPatch).unwrap();
    process.request(Reason::Shutdown).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert!(transition.try_finish_shutdown().unwrap());
    transition
        .batch()
        .unwrap()
        .complete_with_links_deferred()
        .unwrap();
    assert!(!transition.try_reopen().unwrap());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(process.lock().phase, Phase::Closed);
}

#[test]
fn fp_status_and_caller_environment_complete_before_reader_quiescence() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    let process = Arc::new(new_process());
    install(&process, key(0), 1);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    cpu.set_fpsr(1 << 27);
    let mut frame = frame(&mut cpu);
    let mut invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let caller = (
        invocation.frame().host_fp.saved_control,
        invocation.frame().host_fp.saved_status,
    );
    unsafe {
        invocation.frame().ensure_fp().unwrap();
        crate::fp_env::tests::divide_by_zero();
        // No general Rust/helper work may intervene before Drop restores FP.
        drop(invocation);
    }
    let mut probe = crate::abi::HostFpState::default();
    unsafe {
        probe.begin();
        probe.finish();
    }
    assert_eq!((probe.saved_control, probe.saved_status), caller);
    assert_eq!(reader.announcement.load(Ordering::Acquire), 0);
    assert_eq!(unsafe { *frame.canonical.fpsr }, (1 << 27) | 2);
}

#[test]
fn counter_exhaustion_fails_closed_without_success_or_wrapping() {
    let process = new_process();
    let publication = process.reserve(key(0)).unwrap();
    process.lock().reachabilities = CheckedCounter::exhausted();
    assert!(matches!(
        process.publish(publication, Some(entry(1)), None),
        Err(Error::Exhausted(_))
    ));
    assert!(
        process
            .lock()
            .dispatch
            .get(publication.slot)
            .unwrap()
            .snapshot()
            .preferred()
            .is_none()
    );
    assert_ne!(process.control_word().load(Ordering::Acquire), 0);

    let process = new_process();
    process.lock().sequences = CheckedCounter::exhausted();
    assert!(matches!(
        process.request(Reason::MappingChange),
        Err(Error::Exhausted(_))
    ));
    assert_eq!(process.lock().phase, Phase::Closing);

    let process = new_process();
    process.lock().admissions = CheckedCounter::exhausted();
    assert!(matches!(
        process.request(Reason::Eviction),
        Err(Error::Exhausted(_))
    ));
    assert_eq!(process.lock().admission.get(), 1);
    assert!(matches!(process.reserve(key(0)), Err(Error::Exhausted(_))));

    let process = new_process();
    let publication = process.reserve(key(0)).unwrap();
    process.lock().executions = CheckedCounter::exhausted();
    assert!(matches!(
        process.retire_dispatch(publication),
        Err(Error::Exhausted(_))
    ));
    let state = process.lock();
    assert_eq!(state.keys.get(&key(0)), Some(&publication.slot));
    assert!(
        state
            .dispatch
            .get(publication.slot)
            .unwrap()
            .retired
            .is_none()
    );
    assert_eq!(state.execution.get(), 1);
}

#[test]
fn reopen_epoch_exhaustion_keeps_admission_closed() {
    let process = new_process();
    process.request(Reason::Eviction).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.batch().unwrap().complete().unwrap();
    process.lock().admissions = CheckedCounter::exhausted();
    assert!(matches!(transition.try_reopen(), Err(Error::Exhausted(_))));
    assert_eq!(process.lock().phase, Phase::Closed);
    assert_ne!(process.control_word().load(Ordering::Acquire), 0);
}

#[test]
fn a_newer_invocation_does_not_pin_an_already_retired_slot() {
    let process = Arc::new(new_process());
    install(&process, key(0), 1);
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    process
        .publish(process.reserve(key(0)).unwrap(), None, None)
        .unwrap();
    process
        .retire_dispatch(process.reserve(key(0)).unwrap())
        .unwrap();
    assert_eq!(process.collect_dispatch().unwrap(), 0);
    drop(invocation);
    install(&process, key(4), 2);
    let mut invocation = unsafe { reader.admit(&mut frame, key(4)) }
        .unwrap()
        .unwrap();
    assert_eq!(invocation.frame().execution_epoch, 2);
    assert_eq!(process.collect_dispatch().unwrap(), 1);
    assert_eq!(invocation.payload().preferred(), Some(entry(2)));
}

#[test]
fn concurrent_same_key_reservations_install_one_slot() {
    let process = new_process();
    let start = Barrier::new(8);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    start.wait();
                    process.reserve(key(0)).unwrap().slot
                })
            })
            .collect();
        let slots: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect();
        assert!(slots.iter().all(|slot| *slot == slots[0]));
    });
    let state = process.lock();
    assert_eq!(state.keys.len(), 1);
    assert_eq!(state.dispatch.values().count(), 1);
    assert_eq!(state.dispatch.capacity(), 16);
}

#[test]
fn dispatch_readers_and_index_storage_share_the_executable_budget() {
    let cache = Cache::new().unwrap();
    let initial = cache.usage().unwrap();
    {
        let process = Arc::new(Lifetime::new(Arc::clone(&cache)).unwrap());
        let root = cache.usage().unwrap();
        assert!(root.metadata > initial.metadata);
        let reader = process.register().unwrap();
        let registered = cache.usage().unwrap();
        assert!(registered.metadata > root.metadata);
        drop(reader);
        assert!(cache.usage().unwrap().metadata < registered.metadata);
        install(&process, key(0), 1);
        let resident = cache.usage().unwrap();
        for version in 2..32 {
            install(&process, key(0), version);
        }
        assert_eq!(cache.usage().unwrap(), resident);
        process
            .publish(process.reserve(key(0)).unwrap(), None, None)
            .unwrap();
        process
            .retire_dispatch(process.reserve(key(0)).unwrap())
            .unwrap();
        assert_eq!(cache.usage().unwrap(), resident);
        assert_eq!(process.collect_dispatch().unwrap(), 1);
        assert!(cache.usage().unwrap().metadata < resident.metadata);
    }
    assert_eq!(cache.usage().unwrap(), initial);
}

#[test]
fn publication_capacity_failure_keeps_the_old_entry_and_admission_works_without_allocation() {
    let cache = Cache::new().unwrap();
    let process = Arc::new(Lifetime::new(Arc::clone(&cache)).unwrap());
    let mut reader = process.register().unwrap();
    let reachability = install(&process, key(0), 1);
    let captured = process.reserve(key(0)).unwrap();
    let available = crate::executable::HARD_BYTES - cache.usage().unwrap().total();
    let _charge = cache.charge_metadata(available, Tier::Lcq).unwrap();
    assert!(matches!(
        process.publish(captured, Some(entry(2)), None),
        Err(Error::Capacity(_))
    ));
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    assert_eq!(invocation.payload().preferred(), Some(entry(1)));
    assert_eq!(invocation.payload().reachability(), reachability);
    assert_eq!(
        cache.usage().unwrap().total(),
        crate::executable::HARD_BYTES
    );
}
