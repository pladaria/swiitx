use super::*;
use canonical::{complete, native, pattern};
use nixe_cpu::state::a64::A64State;

pub(super) fn landing(abi: HostAbi) -> Vec<u8> {
    match abi {
        HostAbi::X86_64 => vec![0xf3, 0x0f, 0x1e, 0xfa],
        HostAbi::Aarch64 => 0xd50324dfu32.to_le_bytes().to_vec(), // BTI jc
    }
}

fn compile(bytes: &[u8]) -> (JITModule, cranelift_module::FuncId) {
    check_host().unwrap();
    let mut module = JITModule::new(JITBuilder::new(default_libcall_names()).unwrap());
    let id = module
        .declare_function("gateway_unit", Linkage::Local, &module.make_signature())
        .unwrap();
    module.define_function_bytes(id, 16, bytes, &[]).unwrap();
    module.finalize_definitions().unwrap();
    (module, id)
}

#[test]
fn real_gateway_links_independent_units_and_completes_canonical_exit() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        // Both encoders are exercised; only host-compatible bytes execute.
        for cycle in [false, true] {
            for fp in [false, true] {
                let (mut source, entry) = complete(abi);
                let mut target = entry.clone();
                if cycle {
                    let a = target.bindings[0].location;
                    target.bindings[0].location = target.bindings[1].location;
                    target.bindings[1].location = a;
                    target.nzcv = NzcvLocation::Host {
                        carry_inverted: true,
                    };
                }
                let mut exit = source.clone();
                exit.bindings = target.bindings.clone();
                exit.nzcv = target.nzcv.clone();
                exit.site.source = CodeVersion::new(93).unwrap();
                exit.site.state_map = 0x12345678;
                exit.host_fpsr_pending = fp;
                // Compose lazy NZCV materialization with the real cyclic link,
                // active guest FP and the complete gateway return protocol.
                if cycle {
                    source.nzcv = NzcvLocation::Deferred(LazyFlags::Logical {
                        result: source.bindings[0].location,
                        width: 64,
                    });
                }
                let reason = if cycle {
                    NativeExitReason::Control
                } else {
                    NativeExitReason::Dispatch
                };
                let mut second = landing(abi);
                second.extend(
                    emit_canonical_exit(&exit, ValueLocation::Constant(0xfedcba9876543210), reason)
                        .unwrap(),
                );
                let transfer = emit_fast_transfer(&source, &target).unwrap();
                assert_eq!(transfer.is_empty(), !cycle);
                if !native(abi) {
                    continue;
                }
                // This is compiler-owned code, not a dispatch read. Both owners
                // stay alive through execution; production ownership is Task 2.
                let (second_owner, second_id) = compile(&second);
                let target_address = second_owner.get_finalized_function(second_id);
                let mut first = landing(abi);
                if fp {
                    let mut emitter = moves::Emitter::new(abi);
                    emitter.copy(moves::Copy {
                        source: ValueLocation::Constant(1.0f64.to_bits() as u128),
                        destination: vector(0),
                        bytes: 8,
                    });
                    emitter.copy(moves::Copy {
                        source: ValueLocation::Constant(0),
                        destination: vector(1),
                        bytes: 8,
                    });
                    first.extend(emitter.finish());
                    match abi {
                        HostAbi::X86_64 => first.extend([0xf2, 0x0f, 0x5e, 0xc1]), // DIVSD xmm0,xmm1
                        HostAbi::Aarch64 => first.extend(0x1e611800u32.to_le_bytes()), // FDIV d0,d0,d1
                    }
                }
                first.extend(emit_canonical_entry(&entry).unwrap());
                let mut capture = moves::Emitter::new(abi);
                for (reg, offset) in [
                    (abi.reserved().arena, 3504),
                    (abi.reserved().poll, 3512),
                    (abi.reserved().frame, 3520),
                ] {
                    capture.memory(false, RegisterClass::Integer, reg, offset, 8);
                }
                first.extend(capture.finish());
                // Capture the helper-call stack alignment without changing SP.
                match abi {
                    HostAbi::X86_64 => first.extend([0x49, 0x89, 0xe3]), // MOV r11,rsp
                    HostAbi::Aarch64 => first.extend(0x910003f0u32.to_le_bytes()), // MOV x16,sp
                }
                let mut capture = moves::Emitter::new(abi);
                capture.memory(
                    false,
                    RegisterClass::Integer,
                    abi.reserved().link_scratch[0],
                    3528,
                    8,
                );
                first.extend(capture.finish());
                match abi {
                    HostAbi::X86_64 => first.extend([0x49, 0x83, 0xee, 20]), // SUB r14,20
                    HostAbi::Aarch64 => first.extend((0xd1000294u32 | (20 << 10)).to_le_bytes()), // SUB x20,x20,20
                }
                first.extend(transfer);
                let mut branch = moves::Emitter::new(abi);
                let scratch = abi.reserved().link_scratch[0];
                branch.constant(scratch, target_address as u64, 8);
                branch.jump_register(scratch);
                first.extend(branch.finish());
                let (first_owner, first_id) = compile(&first);
                let (mut state, _) = pattern(&entry);
                state.set_fpcr(0);
                state.set_fpsr(1 << 27);
                let mut before = state.clone();
                {
                    let mut frame = NativeFrame::new(&mut state, PollBudget::new(7, 11).unwrap());
                    unsafe { frame.begin_fp() };
                    // Isolated invocation proof only: actual publication and
                    // reachability revalidation arrive with Task 2's coordinator.
                    frame.execution_epoch = 7;
                    let address = first_owner.get_finalized_function(first_id);
                    let outcome = unsafe {
                        if fp {
                            frame.ensure_fp().unwrap();
                        }
                        enter_protected(&mut frame, std::ptr::dangling_mut(), address)
                    }
                    .unwrap();
                    assert_eq!(outcome.reason, reason);
                    assert_eq!(
                        outcome.poll,
                        PollOutcome {
                            sample: !cycle,
                            exhausted: true
                        }
                    );
                    assert_eq!(
                        (frame.budget.sample_remaining, frame.budget.slice_remaining),
                        (4083, -9)
                    );
                    assert_eq!(frame.exit_pc, 0xfedcba9876543210);
                    assert_eq!(
                        (frame.exit_source_version, frame.exit_state_map),
                        (93, 0x12345678)
                    );
                    assert_eq!(frame.execution_epoch, 7);
                    assert_eq!(frame.gateway_exit, 0);
                    assert_eq!((frame.host_fp.saved, frame.host_fp.active), (0, 0));
                    // FP and software state must be complete BEFORE quiescence.
                    assert_eq!(
                        unsafe { *frame.canonical.fpsr },
                        (1 << 27) | if fp { 2 } else { 0 }
                    );
                    let mut caller = HostFpState::default();
                    unsafe {
                        caller.begin();
                        caller.finish();
                    }
                    assert_eq!(
                        (caller.saved_control, caller.saved_status),
                        (frame.host_fp.saved_control, frame.host_fp.saved_status)
                    );
                    for (offset, expected) in [
                        (3504, 1),
                        (3512, 7),
                        (3520, &frame as *const NativeFrame as u64),
                    ] {
                        let mut bytes = [0; 8];
                        for (i, byte) in bytes.iter_mut().enumerate() {
                            *byte = unsafe { frame.spill[offset + i].assume_init() };
                        }
                        assert_eq!(u64::from_le_bytes(bytes), expected);
                    }
                    assert_eq!(unsafe { frame.spill[3528].assume_init() } & 15, 0);
                    frame.execution_epoch = 0;
                }
                assert_eq!(
                    state.general_register_storage_mut(),
                    before.general_register_storage_mut()
                );
                for i in 0..32 {
                    assert_eq!(state.vector(i), before.vector(i));
                }
                assert_eq!(
                    state.stack_pointer_storage_mut(),
                    before.stack_pointer_storage_mut()
                );
                let expected_flags = if cycle {
                    let value = before.general_register_storage_mut()[0];
                    ((value >> 63) as u32) << 31 | (u32::from(value == 0) << 30)
                } else {
                    before.nzcv().bits()
                };
                assert_eq!(state.nzcv().bits(), expected_flags);
                assert_eq!(state.fpsr(), (1 << 27) | if fp { 2 } else { 0 });
                assert_eq!(state.pc(), 0xfedcba9876543210);
                unsafe {
                    first_owner.free_memory();
                    second_owner.free_memory();
                }
            }
        }
    }
}

#[test]
fn canonical_exit_validates_pc_and_reason() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (source, _) = contracts(abi, &[]);
        for pc in [
            integer(abi.reserved().frame),
            spill(0, 8),
            ValueLocation::Constant(u128::MAX),
        ] {
            assert!(emit_canonical_exit(&source, pc, NativeExitReason::Dispatch).is_err());
        }
        assert!(
            emit_canonical_exit(&source, ValueLocation::Constant(0), NativeExitReason::None)
                .is_err()
        );
    }
}

#[test]
fn canonical_exit_preserves_dynamic_pc_until_writeback_finishes() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for pc in [
            integer(0),
            vector(0),
            spill(2048, 8),
            ValueLocation::Constant(0x123456789abcdef0),
        ] {
            let (mut source, entry) = contracts(abi, &[(GuestValue::General(0), pc, pc)]);
            // Constant exit sources are valid, constant ingress destinations aren't.
            let mut code = landing(abi);
            if !matches!(pc, ValueLocation::Constant(_)) {
                code.extend(emit_canonical_entry(&entry).unwrap());
            }
            source.dirty_live = source.live;
            code.extend(emit_canonical_exit(&source, pc, NativeExitReason::Control).unwrap());
            if !native(abi) {
                continue;
            }
            let owner = published::Published::new(|process, cache| {
                let ingress = if matches!(pc, ValueLocation::Constant(_)) {
                    contracts(abi, &[]).1
                } else {
                    entry
                };
                published::synthetic(
                    process.begin_unit(crate::executable::Tier::Lcq).unwrap(),
                    cache,
                    ingress,
                    source,
                    (vec![], None),
                    pc,
                    NativeExitReason::Control,
                )
            });
            let mut state = A64State::default();
            state.general_register_storage_mut()[0] = 0x123456789abcdef0;
            {
                let mut frame = NativeFrame::new(&mut state, PollBudget::new(7, 11).unwrap());
                let mut reader = owner.process.register().unwrap();
                let mut invocation = unsafe { reader.admit(&mut frame, published::key()) }
                    .unwrap()
                    .unwrap();
                let address =
                    invocation.payload().preferred().unwrap().canonical.get() as *const u8;
                let epoch = invocation.frame().execution_epoch;
                let result =
                    unsafe { enter_protected(invocation.frame(), std::ptr::null_mut(), address) }
                        .unwrap();
                assert_eq!(
                    result.poll,
                    PollOutcome {
                        sample: false,
                        exhausted: false
                    }
                );
                assert_eq!(invocation.frame().exit_pc, 0x123456789abcdef0);
                assert_eq!(invocation.frame().budget.slice_remaining, 11);
                assert_eq!(invocation.frame().execution_epoch, epoch);
                drop(invocation);
                assert_eq!(frame.execution_epoch, 0);
            }
            assert_eq!(state.pc(), 0x123456789abcdef0);
            owner.shutdown();
        }
    }
}

#[test]
fn invalid_native_budget_restores_fp_without_announcing_quiescence() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (source, entry) = contracts(abi, &[]);
        let mut code = landing(abi);
        let mut invalid = moves::Emitter::new(abi);
        invalid.constant(abi.reserved().poll, 8, 8); // Armed span is only seven.
        let body = invalid.finish();
        code.extend(&body);
        code.extend(
            emit_canonical_exit(
                &source,
                ValueLocation::Constant(4),
                NativeExitReason::Internal,
            )
            .unwrap(),
        );
        if !native(abi) {
            continue;
        }
        let owner = published::Published::new(|process, cache| {
            published::synthetic(
                process.begin_unit(crate::executable::Tier::Lcq).unwrap(),
                cache,
                entry,
                source,
                (body, None),
                ValueLocation::Constant(4),
                NativeExitReason::Internal,
            )
        });
        let mut state = A64State::default();
        {
            let mut frame = NativeFrame::new(&mut state, PollBudget::new(7, 11).unwrap());
            let mut reader = owner.process.register().unwrap();
            let mut invocation = unsafe { reader.admit(&mut frame, published::key()) }
                .unwrap()
                .unwrap();
            let address = invocation.payload().preferred().unwrap().canonical.get() as *const u8;
            let epoch = invocation.frame().execution_epoch;
            let result = unsafe {
                invocation.frame().ensure_fp().unwrap();
                crate::fp_env::tests::divide_by_zero();
                enter_protected(invocation.frame(), std::ptr::null_mut(), address)
            };
            assert_eq!(
                result,
                Err(NativeReturnError::Budget(BudgetError::InvalidDeadline))
            );
            assert_eq!(
                (
                    invocation.frame().host_fp.saved,
                    invocation.frame().host_fp.active
                ),
                (0, 0)
            );
            assert_eq!(unsafe { *invocation.frame().canonical.fpsr }, 2);
            assert_eq!(
                (
                    invocation.frame().budget.sample_remaining,
                    invocation.frame().budget.slice_remaining
                ),
                (7, 11)
            );
            assert_eq!(invocation.frame().execution_epoch, epoch);
            drop(invocation);
            assert_eq!(frame.execution_epoch, 0);
        }
        owner.shutdown();
    }
}
