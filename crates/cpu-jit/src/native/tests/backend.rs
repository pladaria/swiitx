//! Real fork output -> shared contracts -> Nixe adapters -> system gateway.
//! Unlinked proofs use production publication; linked fixtures await Task 4.
use super::*;
use cranelift_codegen::{
    CompiledCode, Context,
    control::ControlPlane,
    cursor::{Cursor, FuncCursor},
    ir::{self, InstBuilder, MemFlagsData, types},
    isa::{self, CallConv},
    nixe::{EntryConstraint, Location, StateMap},
    settings::{self, Configurable},
};
use nixe_cpu::state::a64::{A64State, Nzcv};

mod arithmetic;

fn publish_unlinked(
    abi: HostAbi,
    code: CompiledCode,
    contract: EntryContract,
    mut state: ExitStateMap,
) -> published::Published {
    use crate::executable::{Tier, output::Output};
    use crate::lifetime::unit::{Entry, Input, Instruction, StateRecord};
    published::Published::new(|process, cache| {
        let identity = process.begin_unit(Tier::Lcq).unwrap();
        state.site = ExitSiteKey {
            source: identity.version(),
            state_map: 0,
        };
        let input = boundary(abi, &code, 10).map.offset;
        let output = boundary(abi, &code, 20).map.clone();
        // compile() rejects all relocations in these fixtures, so there are no
        // function-local symbol indices to resolve from the emptied function.
        let mut staged = Output::from_backend(abi, code, &ir::Function::new()).unwrap();
        let mut bytes = staged.bytes.into_vec();
        let end = append_exit(&mut bytes, &state);
        output.patch_exit(&mut bytes, 0, end as u64).unwrap();
        let start = canonical_ingress(abi, &mut bytes, &contract, input as usize);
        staged.bytes = bytes.into_boxed_slice();
        let code = cache.install(staged, Tier::Lcq, |_| None).unwrap();
        Input {
            identity,
            code,
            tier: Tier::Lcq,
            instructions: Box::new([Instruction {
                key: InstructionKey::new(published::key()).unwrap(),
                bits: 0xd503201f,
            }]),
            entries: Box::new([Entry {
                key: published::key(),
                canonical_offset: start as u32,
                fast_offset: input,
                contract,
            }]),
            dependencies: Box::new([]),
            cursor: nixe_memory::MemoryInvalidationCursor::INITIAL,
            states: Box::new([StateRecord {
                native_offset: output.offset,
                state,
            }]),
            faults: Box::new([]),
        }
    })
}

fn run_unlinked(unit: &published::Published, cpu: &mut A64State, frame_extent: u32) {
    let mut reader = unit.process.register().unwrap();
    let mut frame = NativeFrame::new(cpu, PollBudget::new(17, 23).unwrap());
    let frame_address = &frame as *const NativeFrame as u64;
    frame.spill.fill(MaybeUninit::new(0xa5));
    let mut invocation = unsafe { reader.admit(&mut frame, published::key()) }
        .unwrap()
        .unwrap();
    let entry = invocation.payload().preferred().unwrap();
    let epoch = invocation.frame().execution_epoch;
    let result = unsafe {
        invocation.frame().ensure_fp().unwrap();
        enter_protected(
            invocation.frame(),
            std::ptr::dangling_mut(),
            entry.canonical.get() as *const u8,
        )
    }
    .unwrap();
    let native_frame = invocation.frame();
    assert_eq!(result.reason, NativeExitReason::Dispatch);
    assert_eq!(
        (
            native_frame.exit_source_version,
            native_frame.exit_state_map
        ),
        (entry.version.get(), 0)
    );
    assert_eq!(native_frame.execution_epoch, epoch);
    assert_ne!(epoch, 0);
    assert_eq!(
        (
            native_frame.host_fp.saved,
            native_frame.host_fp.active,
            native_frame.gateway_exit
        ),
        (0, 0, 0)
    );
    assert_eq!(
        (
            native_frame.budget.sample_remaining,
            native_frame.budget.slice_remaining
        ),
        (17, 23)
    );
    let word = |offset: usize| {
        u64::from_le_bytes(std::array::from_fn(|i| unsafe {
            native_frame.spill[offset + i].assume_init()
        }))
    };
    assert_eq!(word(1024), 1);
    assert_eq!(word(1032), 17);
    assert_eq!(word(1040), frame_address);
    assert_eq!(word(1056), word(1064));
    assert_eq!(word(1056) % 16, 0);
    assert!(
        native_frame.spill[64..1024]
            .iter()
            .all(|byte| unsafe { byte.assume_init() } == 0xa5)
    );
    assert!(
        native_frame.spill[frame_extent as usize..]
            .iter()
            .all(|byte| unsafe { byte.assume_init() } == 0xa5)
    );
    drop(invocation);
    assert_eq!(frame.execution_epoch, 0);
}

fn compile(abi: HostAbi, allocator: &str, mut function: ir::Function) -> CompiledCode {
    let mut flags = settings::builder();
    for (name, value) in [
        ("enable_pinned_reg", "true"),
        ("enable_nixe_abi", "true"),
        ("regalloc_checker", "true"),
        ("machine_code_cfg_info", "true"),
        ("regalloc_algorithm", allocator),
        (
            "opt_level",
            if allocator == "single_pass" {
                "none"
            } else {
                "speed"
            },
        ),
    ] {
        flags.set(name, value).unwrap();
    }
    let triple = if abi == HostAbi::X86_64 {
        "x86_64-unknown-linux-gnu"
    } else {
        "aarch64-unknown-linux-gnu"
    };
    let mut builder = isa::lookup(triple.parse().unwrap()).unwrap();
    if abi == HostAbi::X86_64 {
        flags.set("enable_nixe_ibt", "true").unwrap();
    } else {
        builder.set("use_bti", "true").unwrap();
    }
    let target = builder.finish(settings::Flags::new(flags)).unwrap();
    if function.nixe_entries.is_empty() {
        let entry = function.layout.entry_block().unwrap();
        cranelift_codegen::nixe::set_entries(&mut function, &[entry]).unwrap();
    }
    let mut context = Context::for_function(function);
    context.set_disasm(true);
    context
        .compile(&*target, &mut ControlPlane::default())
        .unwrap();
    let code = context.take_compiled_code().unwrap();
    // These fixtures intentionally have no external symbols. Never drop real
    // relocations when transferring compiled bytes to their eventual owner.
    assert!(code.buffer.relocs().is_empty());
    let text = code.vcode.as_ref().unwrap();
    assert!(!text.contains("%rsp"), "{text}");
    assert!(!text.contains(", sp"), "{text}");
    assert!(
        !text.lines().any(|line| {
            let line = line.trim_start();
            line.starts_with("call") || line.starts_with("ret")
        }),
        "{text}"
    );
    code
}

fn operands(count: u8) -> Vec<(GuestValue, usize)> {
    let mut operands = Vec::new();
    for i in 0..count {
        operands.push((GuestValue::General(i), operands.len()));
        operands.push((GuestValue::Vector(i), operands.len()));
    }
    operands.push((GuestValue::Fpsr, operands.len()));
    operands
}

fn signature(operands: &[(GuestValue, usize)]) -> ir::Signature {
    let mut signature = ir::Signature::new(CallConv::SystemV);
    signature.returns = operands
        .iter()
        .map(|(value, _)| {
            ir::AbiParam::new(match value {
                GuestValue::Vector(_) => types::I8X16,
                GuestValue::Fpsr => types::I32,
                _ => types::I64,
            })
        })
        .chain([ir::AbiParam::new(types::I32)])
        .collect();
    signature
}

fn constraints(locations: &[Location]) -> Vec<EntryConstraint> {
    locations
        .iter()
        .map(|location| match *location {
            Location::Register { index, vector } => EntryConstraint::Register { index, vector },
            _ => panic!("small link fixture must fit in registers"),
        })
        .collect()
}

fn live(operands: &[(GuestValue, usize)]) -> StateSet {
    operands.iter().fold(
        StateSet {
            nzcv: NZCV,
            ..StateSet::default()
        },
        |live, &(value, _)| live.union(value.state().unwrap()),
    )
}

fn fragment(
    operands: &[(GuestValue, usize)],
    ingress: Option<&[Location]>,
    fp: bool,
) -> ir::Function {
    let mut function = ir::Function::new();
    let block = function.dfg.make_block();
    function.layout.append_block(block);
    let signature = function.import_signature(signature(operands));
    if let Some(ingress) = ingress {
        function
            .nixe_entry_constraints
            .insert(10, constraints(ingress));
    }
    let mut cursor = FuncCursor::new(&mut function).at_bottom(block);
    let entry = cursor.ins().nixe_entry(signature, 10);
    let mut values = cursor.func.dfg.inst_results(entry).to_vec();
    for &(value, index) in operands {
        if matches!(value, GuestValue::General(_)) {
            values[index] = cursor.ins().iadd_imm_s(values[index], 1);
        }
    }
    if fp {
        let vector = cursor.ins().bitcast(
            types::F64X2,
            MemFlagsData::new().with_endianness(ir::Endianness::Little),
            values[1],
        );
        let numerator = cursor.ins().extractlane(vector, 0);
        let zero = cursor.ins().f64const(0.0);
        let quotient = cursor.ins().fdiv(numerator, zero);
        let vector = cursor.ins().insertlane(vector, quotient, 0);
        values[1] = cursor.ins().bitcast(
            types::I8X16,
            MemFlagsData::new().with_endianness(ir::Endianness::Little),
            vector,
        );
    }
    cursor.ins().nixe_exit(20, &values);
    function
}

fn boundary(abi: HostAbi, code: &CompiledCode, id: u64) -> AllocatedBoundary<'_> {
    let map = code
        .buffer
        .nixe_states
        .iter()
        .find(|map| map.id == id)
        .unwrap();
    if map.entry {
        assert!(
            code.buffer
                .nixe_entries
                .iter()
                .any(|&(_, offset)| offset == map.offset)
        );
        let landing = if abi == HostAbi::X86_64 {
            [0xf3, 0x0f, 0x1e, 0xfa]
        } else {
            0xd503249fu32.to_le_bytes()
        };
        assert_eq!(
            &code.code_buffer()[map.offset as usize..map.offset as usize + 4],
            &landing
        );
    }
    AllocatedBoundary::new(abi, code, map).unwrap()
}

fn entry(
    abi: HostAbi,
    map: &AllocatedBoundary<'_>,
    operands: &[(GuestValue, usize)],
) -> EntryContract {
    let entry = EntryContract {
        abi,
        live_in: live(operands),
        bindings: map.bindings(operands).unwrap(),
        nzcv: NzcvLocation::Packed(map.location(operands.len(), types::I32).unwrap()),
    };
    entry.validate().unwrap();
    entry
}

fn exit(
    abi: HostAbi,
    map: &AllocatedBoundary<'_>,
    operands: &[(GuestValue, usize)],
    version: u64,
    deferred: bool,
) -> ExitStateMap {
    let mut dirty = live(operands);
    // Clean vectors must remain canonical, although they are physically live.
    dirty.vector.fill(false);
    dirty.vector[0] = true;
    let exit = ExitStateMap {
        abi,
        site: ExitSiteKey {
            source: CodeVersion::new(version).unwrap(),
            state_map: map.map.id.try_into().unwrap(),
        },
        live: live(operands),
        dirty_live: dirty,
        bindings: map.bindings(operands).unwrap(),
        nzcv: if deferred {
            NzcvLocation::Deferred(LazyFlags::Logical {
                result: map.location(0, types::I64).unwrap(),
                width: 64,
            })
        } else {
            NzcvLocation::Packed(map.location(operands.len(), types::I32).unwrap())
        },
        host_fpsr_pending: true,
    };
    exit.validate().unwrap();
    exit
}

fn append(bytes: &mut Vec<u8>, part: &[u8]) -> usize {
    bytes.resize(bytes.len().next_multiple_of(16), 0);
    let offset = bytes.len();
    bytes.extend_from_slice(part);
    offset
}

fn branch(abi: HostAbi, bytes: &mut [u8], from: usize, to: usize) {
    StateMap {
        id: 0,
        offset: from.try_into().unwrap(),
        entry: false,
        patch_bytes: if abi == HostAbi::X86_64 { 8 } else { 4 },
        values: vec![],
    }
    .patch_exit(bytes, 0, to as u64)
    .unwrap();
}

fn reserve_branch(abi: HostAbi, bytes: &mut Vec<u8>) -> usize {
    while !bytes.len().is_multiple_of(8) {
        match abi {
            HostAbi::X86_64 => bytes.push(0x90),
            HostAbi::Aarch64 => bytes.extend_from_slice(&0xd503201fu32.to_le_bytes()),
        }
    }
    let offset = bytes.len();
    bytes.resize(offset + 8, 0);
    offset
}

fn canonical_ingress(
    abi: HostAbi,
    bytes: &mut Vec<u8>,
    contract: &EntryContract,
    fast: usize,
) -> usize {
    let mut ingress = gateway::landing(abi);
    ingress.extend(capture_sp(abi, 1064));
    ingress.extend(emit_canonical_entry(contract).unwrap());
    let jump = reserve_branch(abi, &mut ingress);
    let offset = append(bytes, &ingress);
    branch(abi, bytes, offset + jump, fast);
    offset
}

fn capture_sp(abi: HostAbi, offset: u32) -> Vec<u8> {
    let mut bytes = match abi {
        HostAbi::X86_64 => vec![0x49, 0x89, 0xe3], // MOV r11,rsp
        HostAbi::Aarch64 => 0x910003f0u32.to_le_bytes().to_vec(), // MOV x16,sp
    };
    let mut emitter = moves::Emitter::new(abi);
    emitter.memory(
        false,
        RegisterClass::Integer,
        abi.reserved().link_scratch[0],
        offset,
        8,
    );
    bytes.extend(emitter.finish());
    bytes
}

fn append_exit(bytes: &mut Vec<u8>, state: &ExitStateMap) -> usize {
    let mut emitter = moves::Emitter::new(state.abi);
    for (reg, offset) in [
        (state.abi.reserved().arena, 1024),
        (state.abi.reserved().poll, 1032),
        (state.abi.reserved().frame, 1040),
    ] {
        emitter.memory(false, RegisterClass::Integer, reg, offset, 8);
    }
    let mut exit = emitter.finish();
    exit.extend(capture_sp(state.abi, 1056));
    exit.extend(
        emit_canonical_exit(
            state,
            ValueLocation::Constant(0x12345678),
            NativeExitReason::Dispatch,
        )
        .unwrap(),
    );
    append(bytes, &exit)
}

fn run(
    abi: HostAbi,
    bytes: &[u8],
    address_offset: usize,
    state: &mut A64State,
    identity: (u64, u32),
    frame_extent: u32,
) {
    if !canonical::native(abi) {
        return;
    }
    let _restore = crate::fp_env::tests::RestoreHost::new();
    check_host().unwrap();
    let mut owner = JITModule::new(JITBuilder::new(default_libcall_names()).unwrap());
    let id = owner
        .declare_function(
            "compiled_boundary_proof",
            Linkage::Local,
            &owner.make_signature(),
        )
        .unwrap();
    // Patches above are relative within one test-owned allocation, so they
    // can be applied BEFORE finalization makes this memory executable.
    owner.define_function_bytes(id, 16, bytes, &[]).unwrap();
    owner.finalize_definitions().unwrap();
    {
        let mut frame = NativeFrame::new(state, PollBudget::new(17, 23).unwrap());
        frame.spill.fill(MaybeUninit::new(0xa5));
        unsafe { frame.begin_fp() };
        // Test-only invocation protection, not Task 2's publication protocol.
        frame.execution_epoch = 7;
        let address = unsafe { owner.get_finalized_function(id).add(address_offset) };
        let result = unsafe {
            frame.ensure_fp().unwrap();
            enter_protected(&mut frame, std::ptr::dangling_mut(), address)
        }
        .unwrap();
        assert_eq!(result.reason, NativeExitReason::Dispatch);
        assert_eq!((frame.exit_source_version, frame.exit_state_map), identity);
        assert_eq!(
            (frame.budget.sample_remaining, frame.budget.slice_remaining),
            (17, 23)
        );
        assert_eq!(frame.execution_epoch, 7);
        assert_eq!(
            (
                frame.host_fp.saved,
                frame.host_fp.active,
                frame.gateway_exit
            ),
            (0, 0, 0)
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
        let word = |offset: usize| {
            u64::from_le_bytes(std::array::from_fn(|i| unsafe {
                frame.spill[offset + i].assume_init()
            }))
        };
        assert_eq!(word(1024), 1, "arena pin changed");
        assert_eq!(word(1032), 17, "poll pin changed");
        assert_eq!(word(1040), &raw const frame as u64, "frame pin changed");
        assert_eq!(
            word(1056),
            word(1064),
            "SP changed between ingress and exit"
        );
        assert_eq!(word(1056) % 16, 0);
        // These fixtures use at most 64 bytes for cycle scratch. The rest of
        // this range is neither adapter scratch nor backend-owned storage.
        assert!(
            frame.spill[64..1024]
                .iter()
                .all(|byte| unsafe { byte.assume_init() } == 0xa5),
            "backend or bridge overwrote the reserved transfer partition"
        );
        assert!(
            frame.spill[frame_extent as usize..]
                .iter()
                .all(|byte| unsafe { byte.assume_init() } == 0xa5),
            "backend wrote beyond its reported extent"
        );
        frame.execution_epoch = 0;
        assert_eq!(frame.execution_epoch, 0);
    }
    unsafe { owner.free_memory() };
}

fn execute(
    abi: HostAbi,
    bytes: &[u8],
    address_offset: usize,
    count: u8,
    increments: u64,
    frame_extent: u32,
) {
    if !canonical::native(abi) {
        return;
    }
    let mut state = A64State::default();
    for (index, value) in state.general_register_storage_mut().iter_mut().enumerate() {
        *value = 100 + index as u64;
    }
    state.general_register_storage_mut()[0] = 0u64.wrapping_sub(increments);
    for (index, value) in state.vector_register_storage_mut().iter_mut().enumerate() {
        *value = (0x123456789abcdef0u128 << 64) | (index as u128 + 123);
    }
    state.vector_register_storage_mut()[0] =
        (0x123456789abcdef0u128 << 64) | u128::from(1.0f64.to_bits());
    state.set_nzcv(Nzcv::from_bits(0xb0000000));
    state.set_fpsr(1 << 27);
    state.set_tpidr_el0(0xaabbccdd);
    let mut expected = state.clone();
    for value in &mut expected.general_register_storage_mut()[..usize::from(count)] {
        *value = value.wrapping_add(increments);
    }
    expected.vector_register_storage_mut()[0] =
        (0x123456789abcdef0u128 << 64) | u128::from(f64::INFINITY.to_bits());
    expected.set_nzcv(Nzcv::from_bits(Nzcv::Z));
    expected.set_fpsr((1 << 27) | 2);
    expected.set_pc(0x12345678);
    run(
        abi,
        bytes,
        address_offset,
        &mut state,
        (2, 20),
        frame_extent,
    );
    assert_eq!(state, expected);
}

fn extent(code: &CompiledCode) -> u32 {
    code.buffer.frame_layout().unwrap().nixe_frame_size.unwrap()
}

#[test]
fn one_gateway_crosses_empty_then_cyclic_edges_between_two_compiled_units() {
    let operands = operands(3);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for a_allocator in ["single_pass", "backtracking"] {
            for b_allocator in ["single_pass", "backtracking"] {
                // A has two external labels. A:first -> B -> A:second crosses
                // both edge shapes without another gateway or canonical load.
                let mut function = ir::Function::new();
                let sig = function.import_signature(signature(&operands));
                let mut entries = Vec::new();
                for (entry_id, exit_id, registers) in [(10, 40, [0, 1, 2]), (30, 20, [1, 2, 0])] {
                    let block = function.dfg.make_block();
                    function.layout.append_block(block);
                    entries.push(block);
                    let mut locations = Vec::new();
                    for index in registers {
                        locations.push(Location::Register {
                            index,
                            vector: false,
                        });
                        locations.push(Location::Register {
                            index,
                            vector: true,
                        });
                    }
                    locations.extend([
                        Location::Register {
                            index: 3,
                            vector: false,
                        },
                        Location::Register {
                            index: 6,
                            vector: false,
                        },
                    ]);
                    function
                        .nixe_entry_constraints
                        .insert(entry_id, constraints(&locations));
                    let mut cursor = FuncCursor::new(&mut function).at_bottom(block);
                    let entry = cursor.ins().nixe_entry(sig, entry_id as i64);
                    let values = cursor.func.dfg.inst_results(entry).to_vec();
                    cursor.ins().nixe_exit(exit_id, &values);
                }
                cranelift_codegen::nixe::set_entries(&mut function, &entries).unwrap();
                let a = compile(abi, a_allocator, function.clone());
                let a_link_map = boundary(abi, &a, 40);
                let locations: Vec<_> = a_link_map
                    .map
                    .values
                    .iter()
                    .map(|value| value.location)
                    .collect();
                let b = compile(
                    abi,
                    b_allocator,
                    fragment(&operands, Some(&locations), true),
                );
                let b_entry = entry(abi, &boundary(abi, &b, 10), &operands);
                let b_map = boundary(abi, &b, 20);
                let b_source = exit(abi, &b_map, &operands, 1, false);
                // Select the return contract from B's FINAL output, not a
                // guess that arithmetic reuses its input registers. Recompile
                // A and verify its independent first entry still matches B.
                let mut locations: Vec<_> = b_map
                    .map
                    .values
                    .iter()
                    .map(|value| value.location)
                    .collect();
                locations.swap(0, 2);
                locations.swap(2, 4);
                locations.swap(1, 3);
                locations.swap(3, 5);
                function
                    .nixe_entry_constraints
                    .insert(30, constraints(&locations));
                let a = compile(abi, a_allocator, function);
                let a_first = entry(abi, &boundary(abi, &a, 10), &operands);
                let a_second = entry(abi, &boundary(abi, &a, 30), &operands);
                let a_link_map = boundary(abi, &a, 40);
                let a_source = exit(abi, &a_link_map, &operands, 2, false);
                assert!(emit_fast_transfer(&a_source, &b_entry).unwrap().is_empty());
                for i in 0..3 {
                    assert_eq!(
                        a_second.bindings[i * 2].location,
                        b_source.bindings[((i + 1) % 3) * 2].location,
                        "expected an actual three-GPR cycle"
                    );
                }
                let mut bridge = emit_fast_transfer(&b_source, &a_second).unwrap();
                assert!(!bridge.is_empty());
                let jump = reserve_branch(abi, &mut bridge);
                let mut bytes = a.code_buffer().to_vec();
                let b_offset = append(&mut bytes, b.code_buffer());
                a_link_map
                    .map
                    .patch_exit(
                        &mut bytes,
                        0,
                        (b_offset + boundary(abi, &b, 10).map.offset as usize) as u64,
                    )
                    .unwrap();
                let bridge_offset = append(&mut bytes, &bridge);
                branch(
                    abi,
                    &mut bytes,
                    bridge_offset + jump,
                    boundary(abi, &a, 30).map.offset as usize,
                );
                b_map
                    .map
                    .patch_exit(
                        &mut bytes[b_offset..],
                        b_offset as u64,
                        bridge_offset as u64,
                    )
                    .unwrap();
                let final_map = boundary(abi, &a, 20);
                let final_state = exit(abi, &final_map, &operands, 2, true);
                let exit_offset = append_exit(&mut bytes, &final_state);
                final_map
                    .map
                    .patch_exit(&mut bytes, 0, exit_offset as u64)
                    .unwrap();
                let start = canonical_ingress(
                    abi,
                    &mut bytes,
                    &a_first,
                    boundary(abi, &a, 10).map.offset as usize,
                );
                execute(abi, &bytes, start, 3, 1, extent(&a).max(extent(&b)));
            }
        }
    }
}

#[test]
fn final_allocations_drive_empty_and_cyclic_links_through_nixe_gateway() {
    let operands = operands(3);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for source_allocator in ["single_pass", "backtracking"] {
            for target_allocator in ["single_pass", "backtracking"] {
                for cycle in [false, true] {
                    let a = compile(abi, source_allocator, fragment(&operands, None, true));
                    let a_entry = entry(abi, &boundary(abi, &a, 10), &operands);
                    let a_map = boundary(abi, &a, 20);
                    let a_exit = exit(abi, &a_map, &operands, 1, false);
                    let mut locations: Vec<_> = a_map
                        .map
                        .values
                        .iter()
                        .map(|value| value.location)
                        .collect();
                    if cycle {
                        locations.swap(0, 2);
                        locations.swap(2, 4);
                        locations.swap(1, 3);
                        locations.swap(3, 5);
                    }
                    let b = compile(
                        abi,
                        target_allocator,
                        fragment(&operands, Some(&locations), false),
                    );
                    let b_entry = entry(abi, &boundary(abi, &b, 10), &operands);
                    let b_map = boundary(abi, &b, 20);
                    let b_exit = exit(abi, &b_map, &operands, 2, true);
                    let transfer = emit_fast_transfer(&a_exit, &b_entry).unwrap();
                    assert_eq!(transfer.is_empty(), !cycle);
                    let mut bytes = a.code_buffer().to_vec();
                    let b_offset = append(&mut bytes, b.code_buffer());
                    let exit_offset = append_exit(&mut bytes, &b_exit);
                    b_map
                        .map
                        .patch_exit(&mut bytes[b_offset..], b_offset as u64, exit_offset as u64)
                        .unwrap();
                    let destination = b_offset + boundary(abi, &b, 10).map.offset as usize;
                    let target = if cycle {
                        let mut bridge = transfer;
                        let jump = reserve_branch(abi, &mut bridge);
                        let offset = append(&mut bytes, &bridge);
                        branch(abi, &mut bytes, offset + jump, destination);
                        offset
                    } else {
                        destination
                    };
                    a_map.map.patch_exit(&mut bytes, 0, target as u64).unwrap();
                    let address_offset = canonical_ingress(
                        abi,
                        &mut bytes,
                        &a_entry,
                        boundary(abi, &a, 10).map.offset as usize,
                    );
                    execute(
                        abi,
                        &bytes,
                        address_offset,
                        3,
                        2,
                        extent(&a).max(extent(&b)),
                    );
                }
            }
        }
    }
}

#[test]
fn randomized_final_maps_reconstruct_live_state_with_bounded_spills() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for allocator in ["single_pass", "backtracking"] {
            for (case, count) in [3, 9, 17, 23, 31, 31, 31, 31].into_iter().enumerate() {
                let mut seed = 0x12345678u64 + case as u64;
                let mut operands = operands(count);
                // Keep X0/V0 (the lazy producer and FP operation) first; vary the
                // simultaneous definitions and allocation order of all other data.
                for index in (3..operands.len() - 1).rev() {
                    operands.swap(index, 2 + next(&mut seed) as usize % (index - 1));
                }
                for (index, operand) in operands.iter_mut().enumerate() {
                    operand.1 = index;
                }
                let mut function = fragment(&operands, None, true);
                let slot_bytes = if case >= 4 { 12 * 1024 } else { 0 };
                if slot_bytes != 0 {
                    // An explicit slot reserves real frame space ahead of regalloc
                    // spills; this pushes their final offsets close to the limit.
                    function.create_sized_stack_slot(ir::StackSlotData::new(
                        ir::StackSlotKind::ExplicitSlot,
                        slot_bytes,
                        4,
                    ));
                }
                let code = compile(abi, allocator, function);
                assert!((TRANSFER_BYTES + slot_bytes..=SPILL_BYTES).contains(&extent(&code)));
                let input = boundary(abi, &code, 10);
                let output = boundary(abi, &code, 20);
                for map in [&input, &output] {
                    if count == 31 {
                        assert!(
                            map.map
                                .values
                                .iter()
                                .any(|value| matches!(value.location, Location::Spill { .. }))
                        );
                    }
                }
                let contract = entry(abi, &input, &operands);
                let mut state = exit(abi, &output, &operands, 2, true);
                // Require physical reconstruction of EVERY live vector here;
                // leaving the unchanged vectors canonical would hide bad spills.
                state.dirty_live.vector = state.live.vector;
                if !canonical::native(abi) {
                    continue;
                }
                let frame_extent = extent(&code);
                let unit = publish_unlinked(abi, code, contract, state);
                let mut actual = A64State::default();
                for value in actual.general_register_storage_mut() {
                    *value = next(&mut seed);
                }
                actual.general_register_storage_mut()[0] =
                    [u64::MAX, 0, i64::MAX as u64, i64::MIN as u64][case % 4];
                for value in actual.vector_register_storage_mut() {
                    *value = u128::from(next(&mut seed)) << 64 | u128::from(next(&mut seed));
                }
                let high = actual.vector_register_storage_mut()[0] & (u128::from(u64::MAX) << 64);
                actual.vector_register_storage_mut()[0] = high | u128::from(1.0f64.to_bits());
                actual.set_nzcv(Nzcv::from_bits(next(&mut seed) as u32));
                actual.set_fpsr(1 << 27);
                actual.set_tpidr_el0(next(&mut seed));
                let mut expected = actual.clone();
                for value in &mut expected.general_register_storage_mut()[..usize::from(count)] {
                    *value = value.wrapping_add(1);
                }
                let result = expected.general_register_storage_mut()[0];
                expected.set_nzcv(Nzcv::from_bits(if result == 0 {
                    Nzcv::Z
                } else if result >> 63 != 0 {
                    Nzcv::N
                } else {
                    0
                }));
                expected.vector_register_storage_mut()[0] =
                    high | u128::from(f64::INFINITY.to_bits());
                expected.set_fpsr((1 << 27) | 2);
                expected.set_pc(0x12345678);
                run_unlinked(&unit, &mut actual, frame_extent);
                assert_eq!(actual, expected, "{abi:?}/{allocator}/case={case}");
                unit.shutdown();
            }
        }
    }
}

#[test]
fn instruction_attached_prefault_maps_translate_real_registers_and_spills() {
    let operands = operands(31);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for allocator in ["single_pass", "backtracking"] {
            let mut function = fragment(&operands, None, false);
            let block = function.layout.entry_block().unwrap();
            let entry = function.layout.first_inst(block).unwrap();
            let values = function.dfg.inst_results(entry).to_vec();
            // The fault must describe PRE-instruction SSA values, even though
            // the rest of this unit later increments all architectural GPRs.
            let mut cursor = FuncCursor::new(&mut function).at_inst(entry);
            cursor.next_inst();
            cursor.ins().nixe_fault_start(30, &values);
            let address = cursor.ins().get_pinned_reg(types::I64);
            let loaded = cursor
                .ins()
                .load(types::I64, MemFlagsData::new(), address, 0);
            cursor.ins().store(MemFlagsData::new(), loaded, address, 8);
            cursor.ins().nixe_fault_end(30, &[]);
            let code = compile(abi, allocator, function);
            assert_eq!(code.buffer.nixe_faults.len(), 2);
            for map in &code.buffer.nixe_faults {
                assert_eq!(map.id, 30);
                assert!(!map.entry);
                assert_eq!(map.patch_bytes, 0);
                assert!(
                    code.buffer
                        .traps()
                        .iter()
                        .any(|trap| trap.offset == map.offset)
                );
                let allocated = AllocatedBoundary::new(abi, &code, map).unwrap();
                let mut state = exit(abi, &allocated, &operands, 2, false);
                state.host_fpsr_pending = false;
                state.dirty_live = StateSet::default();
                state.validate().unwrap();
                assert!(
                    state
                        .bindings
                        .iter()
                        .any(|binding| matches!(binding.location, ValueLocation::Spill { .. }))
                );
            }
        }
    }
}

#[test]
fn backend_mapping_rejects_missing_mistyped_reserved_and_out_of_extent_operands() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let code = compile(abi, "single_pass", fragment(&operands(3), None, false));
        let allocated = boundary(abi, &code, 20);
        assert!(allocated.location(100, types::I64).is_err());
        assert!(allocated.location(0, types::I32).is_err());
        assert!(allocated.bindings(&[(GuestValue::Vector(0), 0)]).is_err());
        let mut bad = allocated.map.clone();
        for location in [
            Location::Unused,
            Location::Register {
                index: abi.reserved().poll,
                vector: false,
            },
            Location::Register {
                index: 0,
                vector: true,
            },
            Location::Spill { offset: 0 },
            Location::Spill {
                offset: SPILL_BYTES - 8,
            },
        ] {
            bad.values[0].location = location;
            assert!(
                AllocatedBoundary::new(abi, &code, &bad).is_err(),
                "{location:?}"
            );
        }
        bad = allocated.map.clone();
        bad.offset = code.code_buffer().len() as u32;
        assert!(AllocatedBoundary::new(abi, &code, &bad).is_err());
        // An optimized-away definition is allowed only on entry, and cannot
        // be consumed as a physical input by a retained contract.
        bad = boundary(abi, &code, 10).map.clone();
        bad.values[0].location = Location::Unused;
        let unused = AllocatedBoundary::new(abi, &code, &bad).unwrap();
        assert!(unused.location(0, types::I64).is_err());
    }
}
