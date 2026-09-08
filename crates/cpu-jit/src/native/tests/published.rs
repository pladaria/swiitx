//! Synthetic inputs to the production cache, publication owner and gateway.
use super::*;
use crate::executable::{
    Cache, Tier,
    output::{Metadata, Output},
};
use crate::lifetime::{
    Lifetime, Reason,
    unit::{EmissionIdentity, Entry, Input, Instruction, StateRecord, UnitHandle},
};
use nixe_cpu::{platform::TargetPlatform, profile::ProcessCpuContext, state::a64::A64State};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress, MemoryInvalidationCursor};
use std::sync::{Arc, atomic::AtomicU64};

pub(super) fn key() -> BlockKey {
    BlockKey::new(
        ProcessCpuContext::new(TargetPlatform::Switch1, AddressSpaceId::new(1)),
        GuestVirtualAddress::new(0),
        FpSpecialization::Dynamic,
    )
    .unwrap()
}

pub(super) struct Published {
    pub process: Arc<Lifetime>,
    pub cache: Arc<Cache>,
    pub handle: UnitHandle,
}
impl Published {
    pub fn new(build: impl FnOnce(&Lifetime, &Arc<Cache>) -> Input) -> Self {
        let cache = Cache::new().unwrap();
        let process = Arc::new(Lifetime::new(Arc::clone(&cache)).unwrap());
        let handle = Self::publish(&process, build(&process, &cache));
        Self {
            process,
            cache,
            handle,
        }
    }
    pub fn publish(process: &Lifetime, input: Input) -> UnitHandle {
        let cursor = AtomicU64::new(0);
        process
            .prepare_unit(&[process.reserve(key()).unwrap()], input, &cursor)
            .unwrap()
            .publish()
            .unwrap()
    }
    pub fn shutdown(&self) {
        self.process.request(Reason::Shutdown).unwrap();
        let mut transition = self.process.try_transition().unwrap().unwrap();
        transition.wait_closed().unwrap();
        assert!(transition.try_finish_shutdown().unwrap());
        transition.batch().unwrap().complete().unwrap();
        assert!(transition.try_reopen().unwrap());
        assert_eq!(self.cache.usage().unwrap().committed, 0);
    }
}

fn faulting_input(process: &Lifetime, cache: &Arc<Cache>, pc: u64) -> Input {
    let abi = if cfg!(target_arch = "x86_64") {
        HostAbi::X86_64
    } else {
        HostAbi::Aarch64
    };
    let (exit, entry) = contracts(abi, &[]);
    let mut emitter = moves::Emitter::new(abi);
    // A real, valid load from the frame, with exact signal-attribution metadata.
    emitter.memory(
        true,
        RegisterClass::Integer,
        abi.reserved().link_scratch[0],
        0,
        8,
    );
    let body = emitter.finish();
    let length = u8::try_from(body.len()).unwrap();
    synthetic(
        process.begin_unit(Tier::Lcq).unwrap(),
        cache,
        entry,
        exit,
        (body, Some(length)),
        ValueLocation::Constant(u128::from(pc)),
        NativeExitReason::Dispatch,
    )
}

#[test]
fn native_replacement_fault_borrows_pressure_reuse_and_shutdown_share_one_owner() {
    let published = Published::new(|process, cache| faulting_input(process, cache, 4));
    let process = &published.process;
    let snapshot = process.snapshot(published.handle).unwrap();
    let old_base = snapshot.code.allocation.address();
    let generation = snapshot.code.allocation.generation;
    let fault_pc = old_base + snapshot.faults[0].native_start as usize;
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = NativeFrame::new(&mut cpu, PollBudget::new(17, 23).unwrap());
    frame.spill[..8].fill(MaybeUninit::new(0));
    let mut invocation = unsafe { reader.admit(&mut frame, key()) }.unwrap().unwrap();
    let old = invocation.payload().preferred().unwrap();
    std::thread::scope(|scope| {
        let (send, receive) = std::sync::mpsc::channel();
        let cache = &published.cache;
        let publisher = scope.spawn(move || {
            let replacement = Published::publish(process, faulting_input(process, cache, 8));
            send.send(replacement).unwrap(); // Cutover is now Closing.
            let mut transition = process.try_transition().unwrap().unwrap();
            transition.wait_closed().unwrap();
            transition.drain_retirements().unwrap();
            assert_eq!(process.reclaim_units().unwrap(), 0); // Compiler snapshot remains.
            transition.batch().unwrap().complete().unwrap();
            assert!(transition.try_reopen().unwrap());
        });
        receive.recv().unwrap();
        assert_eq!(
            invocation.fault(fault_pc).unwrap().unit.version,
            old.version
        );
        assert!(matches!(
            process.begin_unit(Tier::Lcq),
            Err(crate::lifetime::Error::Closed)
        ));
        // The reader won admission before Closing. Its old code remains valid
        // even though another thread has already published a replacement.
        let result = unsafe {
            enter_protected(
                invocation.frame(),
                std::ptr::null_mut(),
                old.canonical.get() as *const u8,
            )
        }
        .unwrap();
        assert_eq!(result.reason, NativeExitReason::Dispatch);
        assert_eq!(invocation.frame().exit_source_version, old.version.get());
        assert_eq!(invocation.frame().exit_pc, 4);
        drop(invocation);
        publisher.join().unwrap();
    });
    assert_eq!(frame.execution_epoch, 0);
    let mut invocation = unsafe { reader.admit(&mut frame, key()) }.unwrap().unwrap();
    let new = invocation.payload().preferred().unwrap();
    assert_ne!(old.version, new.version);
    let fault = invocation.fault(fault_pc).unwrap();
    drop(snapshot);
    assert_eq!(process.reclaim_units().unwrap(), 0); // Newer reader can hold the old table.
    assert!(invocation.fault(fault_pc).is_none());
    assert_eq!(fault.unit.version, old.version);
    unsafe {
        enter_protected(
            invocation.frame(),
            std::ptr::null_mut(),
            new.canonical.get() as *const u8,
        )
    }
    .unwrap();
    assert_eq!(invocation.frame().exit_pc, 8);
    assert_eq!(invocation.frame().exit_source_version, new.version.get());
    drop(invocation);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    // Force the configured hard budget, without changing production limits.
    let charge = published
        .cache
        .charge_metadata(
            crate::executable::HARD_BYTES - published.cache.usage().unwrap().total(),
            Tier::Lcq,
        )
        .unwrap();
    process.request(Reason::Eviction).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.relieve_pressure(0, Tier::Lcq).unwrap();
    assert_eq!(published.cache.usage().unwrap().committed, 0);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(transition);
    drop(charge);
    let replacement = faulting_input(process, &published.cache, 12);
    assert_eq!(replacement.code.allocation.address(), old_base);
    assert_ne!(replacement.code.allocation.generation, generation);
    let handle = Published::publish(process, replacement);
    assert!(matches!(
        process.snapshot(published.handle),
        Err(crate::lifetime::Error::StaleUnit)
    ));
    let mut invocation = unsafe { reader.admit(&mut frame, key()) }.unwrap().unwrap();
    let entry = invocation.payload().preferred().unwrap();
    assert_eq!(
        invocation.fault(fault_pc).unwrap().unit.version,
        entry.version
    );
    unsafe {
        enter_protected(
            invocation.frame(),
            std::ptr::null_mut(),
            entry.canonical.get() as *const u8,
        )
    }
    .unwrap();
    assert_eq!(invocation.frame().exit_pc, 12);
    assert_eq!(invocation.frame().exit_source_version, entry.version.get());
    drop(invocation);
    drop(reader);
    let held = process.snapshot(handle).unwrap();
    process.request(Reason::Shutdown).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert!(!transition.try_finish_shutdown().unwrap());
    drop(held);
    assert!(transition.try_finish_shutdown().unwrap());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(published.cache.usage().unwrap().committed, 0);
}

/// Hand-emitted unlinked unit: the ingress and exit use the Task 1 encoders.
/// The closure supplies only its body and exact exit state, not another runtime.
pub(super) fn synthetic(
    identity: EmissionIdentity,
    cache: &Arc<Cache>,
    entry: EntryContract,
    mut exit: ExitStateMap,
    body: (Vec<u8>, Option<u8>),
    pc: ValueLocation,
    reason: NativeExitReason,
) -> Input {
    exit.site = ExitSiteKey {
        source: identity.version(),
        state_map: 0,
    };
    let abi = entry.abi;
    let mut bytes = gateway::landing(abi);
    bytes.extend(emit_canonical_entry(&entry).unwrap());
    let fault_offset = bytes.len() as u32;
    let (body, fault_bytes) = body;
    bytes.extend(body);
    let offset = bytes.len() as u32;
    bytes.extend(emit_canonical_exit(&exit, pc, reason).unwrap());
    let code = cache
        .install(
            Output {
                bytes: bytes.into_boxed_slice(),
                alignment: 16,
                metadata: Metadata {
                    abi,
                    frame_extent: SPILL_BYTES,
                    entries: Box::new([(cranelift_codegen::ir::Block::from_u32(0), 0)]),
                    states: Box::new([cranelift_codegen::nixe::StateMap {
                        id: 0,
                        offset,
                        entry: false,
                        patch_bytes: 0,
                        values: vec![],
                    }]),
                    faults: fault_bytes
                        .map(|_| cranelift_codegen::nixe::StateMap {
                            id: 1,
                            offset: fault_offset,
                            entry: false,
                            patch_bytes: 0,
                            values: vec![],
                        })
                        .into_iter()
                        .collect(),
                    traps: Box::new([]),
                    relocations: Box::new([]),
                },
            },
            Tier::Lcq,
            |_| None,
        )
        .unwrap();
    let mut states = vec![StateRecord {
        native_offset: offset,
        state: exit.clone(),
    }];
    let faults = fault_bytes
        .map(|length| {
            exit.site.state_map = 1;
            states.push(StateRecord {
                native_offset: fault_offset,
                state: exit,
            });
            crate::lifetime::unit::FaultRecord {
                native_start: fault_offset,
                native_end: fault_offset + u32::from(length),
                instruction: InstructionKey::new(key()).unwrap(),
                access: crate::lifetime::unit::Access::Read,
                bytes: 8,
                subaccess: 0,
                commit_stage: 0,
                state_map: 1,
            }
        })
        .into_iter()
        .collect();
    Input {
        identity,
        code,
        tier: Tier::Lcq,
        instructions: Box::new([Instruction {
            key: InstructionKey::new(key()).unwrap(),
            bits: 0xd503201f,
        }]),
        entries: Box::new([Entry {
            key: key(),
            canonical_offset: 0,
            fast_offset: 0,
            contract: entry,
        }]),
        dependencies: Box::new([]),
        cursor: MemoryInvalidationCursor::INITIAL,
        states: states.into_boxed_slice(),
        faults,
    }
}

#[test]
fn published_gateway_preserves_full_state_flags_fp_and_pins_before_quiescence() {
    let abi = if cfg!(target_arch = "x86_64") {
        HostAbi::X86_64
    } else {
        HostAbi::Aarch64
    };
    for deferred in [false, true] {
        let (mut exit, entry) = canonical::complete(abi);
        if deferred {
            exit.nzcv = NzcvLocation::Deferred(LazyFlags::Logical {
                result: entry.bindings[0].location,
                width: 64,
            });
        }
        exit.host_fpsr_pending = true;
        let (mut cpu, _) = canonical::pattern(&entry);
        cpu.set_fpcr(0);
        cpu.set_fpsr(1 << 27);
        let mut expected = cpu.clone();
        expected.set_pc(0x12345678);
        expected.set_fpsr((1 << 27) | 2);
        if deferred {
            let x0 = expected.general_register_storage_mut()[0];
            expected.set_nzcv(nixe_cpu::state::a64::Nzcv::from_bits(
                ((x0 >> 63) as u32) << 31 | (u32::from(x0 == 0) << 30),
            ));
        }
        let published = Published::new(|process, cache| {
            let mut emitter = moves::Emitter::new(abi);
            // Capture the gateway's actual reserved registers without clobbering live guest state.
            for (reg, offset) in [
                (abi.reserved().arena, 3504),
                (abi.reserved().poll, 3512),
                (abi.reserved().frame, 3520),
            ] {
                emitter.memory(false, RegisterClass::Integer, reg, offset, 8);
            }
            synthetic(
                process.begin_unit(Tier::Lcq).unwrap(),
                cache,
                entry,
                exit,
                (emitter.finish(), None),
                ValueLocation::Constant(0x12345678),
                NativeExitReason::Dispatch,
            )
        });
        let mut reader = published.process.register().unwrap();
        let mut frame = NativeFrame::new(&mut cpu, PollBudget::new(7, 11).unwrap());
        let frame_address = &frame as *const NativeFrame as u64;
        let mut invocation = unsafe { reader.admit(&mut frame, key()) }.unwrap().unwrap();
        let payload = invocation.payload().preferred().unwrap();
        let epoch = invocation.frame().execution_epoch;
        assert_ne!(epoch, 0);
        let mut arena = [0u8; 8];
        let result = unsafe {
            invocation.frame().ensure_fp().unwrap();
            crate::fp_env::tests::divide_by_zero();
            enter_protected(
                invocation.frame(),
                arena.as_mut_ptr(),
                payload.canonical.get() as *const u8,
            )
        }
        .unwrap();
        assert_eq!(result.reason, NativeExitReason::Dispatch);
        {
            let frame = invocation.frame();
            assert_eq!(frame.execution_epoch, epoch);
            assert_eq!(
                (
                    frame.host_fp.saved,
                    frame.host_fp.active,
                    frame.gateway_exit
                ),
                (0, 0, 0)
            );
            assert_eq!(
                (frame.exit_source_version, frame.exit_state_map),
                (payload.version.get(), 0)
            );
            for (offset, expected) in [
                (3504, arena.as_ptr() as u64),
                (3512, 7),
                (3520, frame_address),
            ] {
                assert_eq!(
                    u64::from_le_bytes(std::array::from_fn(|i| unsafe {
                        frame.spill[offset + i].assume_init()
                    })),
                    expected
                );
            }
        }
        drop(invocation);
        assert_eq!(frame.execution_epoch, 0);
        assert_eq!(cpu, expected);
        drop(reader);
        published.shutdown();
    }
}
