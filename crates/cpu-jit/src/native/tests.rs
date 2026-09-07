use super::*;
use crate::abi::*;
use crate::analysis::{NZCV, StateSet};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};
use std::mem::MaybeUninit;

mod backend;
mod canonical;
mod flags;
mod gateway;
mod published;

fn register(class: RegisterClass, index: u8) -> ValueLocation {
    ValueLocation::Register { class, index }
}
fn integer(index: u8) -> ValueLocation {
    register(RegisterClass::Integer, index)
}
fn vector(index: u8) -> ValueLocation {
    register(RegisterClass::Vector, index)
}
fn spill(offset: u32, bytes: u8) -> ValueLocation {
    ValueLocation::Spill { offset, bytes }
}

fn contracts(
    abi: HostAbi,
    values: &[(GuestValue, ValueLocation, ValueLocation)],
) -> (ExitStateMap, EntryContract) {
    let live = values
        .iter()
        .fold(StateSet::default(), |state, (value, _, _)| {
            state.union(value.state().unwrap())
        });
    let source = ExitStateMap {
        site: ExitSiteKey {
            source: CodeVersion::new(1).unwrap(),
            state_map: 0,
        },
        abi,
        live,
        dirty_live: live,
        bindings: values
            .iter()
            .map(|&(value, location, _)| ValueBinding { value, location })
            .collect(),
        nzcv: NzcvLocation::Canonical,
        host_fpsr_pending: false,
    };
    let target = EntryContract {
        abi,
        live_in: live,
        bindings: values
            .iter()
            .map(|&(value, _, location)| ValueBinding { value, location })
            .collect(),
        nzcv: NzcvLocation::Canonical,
    };
    (source, target)
}

#[test]
fn identical_contracts_emit_no_transfer() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (mut source, mut target) = contracts(
            abi,
            &[
                (GuestValue::General(0), integer(0), integer(0)),
                (GuestValue::Vector(0), vector(7), vector(7)),
                (GuestValue::Sp, spill(2048, 8), spill(2048, 8)),
            ],
        );
        source.live.nzcv = NZCV;
        target.live_in.nzcv = NZCV;
        source.nzcv = NzcvLocation::Host {
            carry_inverted: true,
        };
        target.nzcv = source.nzcv.clone();
        assert!(emit_fast_transfer(&source, &target).unwrap().is_empty());
        execute(&source, &target, 11);
    }
}

#[test]
fn integer_cycles_use_reserved_scratch_without_memory_traffic() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (source, target) = contracts(
            abi,
            &[
                (GuestValue::General(0), integer(0), integer(1)),
                (GuestValue::General(1), integer(1), integer(2)),
                (GuestValue::General(2), integer(2), integer(0)),
            ],
        );
        // One save and three register moves: no stores/reloads or final branch.
        let expected = if abi == HostAbi::X86_64 { 12 } else { 16 };
        assert_eq!(
            emit_fast_transfer(&source, &target).unwrap().len(),
            expected
        );
        execute(&source, &target, 13);
    }
}

#[test]
fn cycles_aliases_constants_and_cross_bank_transfers() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (source, target) = contracts(
            abi,
            &[
                (GuestValue::General(0), integer(0), integer(1)),
                (GuestValue::General(1), integer(1), integer(2)),
                (GuestValue::General(2), integer(2), integer(0)),
                (GuestValue::General(3), integer(0), integer(12)),
                (GuestValue::Vector(0), vector(0), vector(7)),
                (GuestValue::Vector(1), vector(7), vector(15)),
                (GuestValue::Vector(2), vector(15), vector(0)),
                (GuestValue::Vector(3), vector(15), spill(2304, 16)),
                (GuestValue::General(4), integer(8), vector(8)),
                (GuestValue::General(5), vector(8), integer(8)),
                (GuestValue::Fpcr, vector(9), integer(9)),
                (GuestValue::Fpsr, integer(9), vector(9)),
                (
                    GuestValue::General(6),
                    ValueLocation::Constant(0xfeed123456789abc),
                    vector(10),
                ),
                (
                    GuestValue::Vector(4),
                    ValueLocation::Constant(0x8123456789abcdef_fedcba9876543210),
                    vector(11),
                ),
                (
                    GuestValue::Vector(5),
                    ValueLocation::Constant(u128::MAX),
                    spill(2320, 16),
                ),
                (
                    GuestValue::General(7),
                    ValueLocation::Constant(0x8000000000000001),
                    integer(10),
                ),
            ],
        );
        execute(&source, &target, 17);
        execute(&source, &target, u64::MAX - 103);
    }
}

#[test]
fn partially_overlapping_spills_are_snapshotted_before_writes() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (source, target) = contracts(
            abi,
            &[
                (GuestValue::Vector(0), spill(2048, 16), spill(2064, 16)),
                (GuestValue::General(0), spill(2064, 8), spill(2056, 8)),
                (GuestValue::General(1), spill(2056, 8), integer(8)),
                (GuestValue::Vector(1), spill(2048, 16), vector(15)),
                (GuestValue::Fpcr, spill(2080, 4), spill(2084, 4)),
                (GuestValue::Fpsr, spill(2084, 4), spill(2080, 4)),
            ],
        );
        execute(&source, &target, 31);
    }
}

#[test]
fn packed_flags_participate_in_the_copy_graph() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (mut source, mut target) =
            contracts(abi, &[(GuestValue::Fpcr, integer(1), integer(0))]);
        source.live.nzcv = NZCV;
        source.dirty_live.nzcv = NZCV;
        target.live_in.nzcv = NZCV;
        source.nzcv = NzcvLocation::Packed(integer(0));
        target.nzcv = NzcvLocation::Packed(integer(1));
        execute(&source, &target, 43);
        source.nzcv = NzcvLocation::Packed(ValueLocation::Constant(0xa0000000));
        target.nzcv = NzcvLocation::Packed(spill(2048, 4));
        execute(&source, &target, 47);
    }
}

#[test]
fn unavailable_inputs_fail_and_valid_flag_conversions_emit() {
    let (mut source, mut target) = contracts(HostAbi::X86_64, &[]);
    target.abi = HostAbi::Aarch64;
    assert_eq!(
        emit_fast_transfer(&source, &target),
        Err(TransferError::DifferentHostAbis)
    );
    target.abi = source.abi;
    target.live_in.integer.x[0] = true;
    target.bindings = Box::new([ValueBinding {
        value: GuestValue::General(0),
        location: integer(0),
    }]);
    assert_eq!(
        emit_fast_transfer(&source, &target),
        Err(TransferError::MissingValue(GuestValue::General(0)))
    );
    target.live_in = StateSet {
        nzcv: NZCV,
        ..StateSet::default()
    };
    target.bindings = Box::new([]);
    target.nzcv = NzcvLocation::Packed(integer(0));
    assert_eq!(
        emit_fast_transfer(&source, &target),
        Err(TransferError::MissingFlags)
    );
    source.live.nzcv = NZCV;
    for flags in [
        NzcvLocation::Canonical,
        NzcvLocation::Host {
            carry_inverted: false,
        },
        NzcvLocation::Deferred(LazyFlags::Logical {
            result: integer(0),
            width: 64,
        }),
    ] {
        source.nzcv = flags;
        assert!(emit_fast_transfer(&source, &target).is_ok());
    }
    target.nzcv = NzcvLocation::Host {
        carry_inverted: true,
    };
    source.nzcv = NzcvLocation::Host {
        carry_inverted: false,
    };
    assert!(emit_fast_transfer(&source, &target).is_ok());
    source.nzcv = NzcvLocation::Packed(integer(0));
    assert!(emit_fast_transfer(&source, &target).is_ok());
}

fn next(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed
}
fn shuffle(values: &mut [ValueLocation], seed: &mut u64) {
    for index in (1..values.len()).rev() {
        values.swap(index, (next(seed) as usize) % (index + 1));
    }
}

#[test]
fn randomized_full_register_files_with_aliases_and_spills() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let mut integers: Vec<_> = (0..32)
            .map(integer)
            .filter(|location| location.valid(abi, 8))
            .collect();
        for i in integers.len()..31 {
            integers.push(spill(2048 + i as u32 * 8, 8));
        }
        let mut vectors: Vec<_> = (0..32)
            .map(vector)
            .filter(|location| location.valid(abi, 16))
            .collect();
        for i in vectors.len()..32 {
            vectors.push(spill(2560 + i as u32 * 16, 16));
        }
        let mut seed = 59;
        for iteration in 0..128 {
            let mut source_ints = integers.clone();
            let mut source_vectors = vectors.clone();
            shuffle(&mut integers, &mut seed);
            shuffle(&mut vectors, &mut seed);
            if iteration % 2 == 0 {
                source_ints[3] = source_ints[0];
                source_vectors[7] = source_vectors[4];
            }
            if iteration % 3 == 0 {
                source_ints[5] = ValueLocation::Constant(u128::from(next(&mut seed)));
                source_vectors[9] = ValueLocation::Constant(
                    u128::from(next(&mut seed)) | (u128::from(next(&mut seed)) << 64),
                );
            }
            let values: Vec<_> = (0..31)
                .map(|i| (GuestValue::General(i as u8), source_ints[i], integers[i]))
                .chain(
                    (0..32).map(|i| (GuestValue::Vector(i as u8), source_vectors[i], vectors[i])),
                )
                .collect();
            let (source, target) = contracts(abi, &values);
            execute(&source, &target, seed);
        }
    }
}

// Conventional native test fixture, independent of the move encoder. Assembly
// loads/captures every allocatable register at its architectural index. This
// avoids using the emitter under test to initialize or interpret its own maps.
// These isolated register fixtures use a direct allocation lease, not dispatch;
// complete canonical units exercise protected publication in published.rs.
fn execute(source: &ExitStateMap, target: &EntryContract, seed: u64) {
    let bytes = emit_fast_transfer(source, target).unwrap();
    if cfg!(target_arch = "x86_64") && source.abi != HostAbi::X86_64
        || cfg!(target_arch = "aarch64") && source.abi != HostAbi::Aarch64
    {
        if source.abi == HostAbi::Aarch64 {
            assert_eq!(bytes.len() % 4, 0);
        }
        return; // Encoding only; native AArch64 is not claimed on an x86-64 host.
    }
    let mut state = nixe_cpu::state::a64::A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(77, 1000).unwrap());
    let mut seed = seed;
    for byte in &mut frame.spill {
        *byte = MaybeUninit::new((next(&mut seed) >> 32) as u8);
    }
    // SAFETY: every byte in the array has just been initialized.
    let before = unsafe { *frame.spill.as_ptr().cast::<[u8; SPILL_BYTES as usize]>() };
    let mut expected: Vec<_> = target
        .bindings
        .iter()
        .map(|output| {
            let input = source
                .bindings
                .iter()
                .find(|input| input.value == output.value)
                .unwrap();
            (
                output.location,
                read(&before, input.location, input.value.bytes(), false),
            )
        })
        .collect();
    if let (NzcvLocation::Packed(input), NzcvLocation::Packed(output)) =
        (&source.nzcv, &target.nzcv)
    {
        expected.push((*output, read(&before, *input, 4, false)));
    }
    invoke(source.abi, bytes, &mut frame);
    // SAFETY: the fixture and transfer only wrote to the fully initialized array.
    let after = unsafe { &*frame.spill.as_ptr().cast::<[u8; SPILL_BYTES as usize]>() };
    for (location, value) in expected {
        assert_eq!(
            read(after, location, value.len() as u8, true),
            value,
            "{location:?}"
        );
    }
    let mut destinations: Vec<_> = target
        .bindings
        .iter()
        .map(|binding| binding.location)
        .collect();
    if let NzcvLocation::Packed(location) = target.nzcv {
        destinations.push(location);
    }
    for (class, width) in [(RegisterClass::Integer, 8), (RegisterClass::Vector, 16)] {
        for index in 0..32 {
            let location = register(class, index);
            if !location.valid(source.abi, width)
                || destinations
                    .iter()
                    .any(|&destination| locations_overlap(location, destination))
            {
                continue;
            }
            assert_eq!(
                read(after, location, width, true),
                read(&before, location, width, false),
                "unmentioned register {location:?}"
            );
        }
    }
    for offset in 2048..4096 {
        if !destinations
            .iter()
            .any(|&destination| locations_overlap(spill(offset, 1), destination))
        {
            assert_eq!(
                after[offset as usize], before[offset as usize],
                "unmentioned spill byte {offset}"
            );
        }
    }
    for (offset, expected) in [
        (5160, 0x1234u64),
        (5168, 77),
        (5176, (&frame as *const NativeFrame) as u64),
    ] {
        assert_eq!(
            &after[offset..offset + 8],
            &expected.to_le_bytes(),
            "reserved register"
        );
    }
    assert_eq!(&after[5120..5128], &after[5128..5136], "host flags changed");
    assert_eq!(
        &after[5136..5144],
        &after[5144..5152],
        "host FP control changed"
    );
    if source.abi == HostAbi::Aarch64 {
        assert_eq!(
            &after[5184..5192],
            &after[5192..5200],
            "host FP status changed"
        );
    }
}

fn invoke(abi: HostAbi, bytes: Vec<u8>, frame: &mut NativeFrame<'_>) {
    invoke_inner(abi, bytes, frame, false);
}

fn invoke_with_fp(abi: HostAbi, bytes: Vec<u8>, frame: &mut NativeFrame<'_>) {
    invoke_inner(abi, bytes, frame, true);
}

fn invoke_inner(abi: HostAbi, mut bytes: Vec<u8>, frame: &mut NativeFrame<'_>, fp: bool) {
    check_host().unwrap();
    match abi {
        HostAbi::X86_64 => bytes.push(0xc3),
        HostAbi::Aarch64 => bytes.extend_from_slice(&0xd65f03c0u32.to_le_bytes()),
    }
    let cache = crate::executable::Cache::new().unwrap();
    let code = cache
        .install(
            crate::executable::output::Output {
                bytes: bytes.into_boxed_slice(),
                alignment: 16,
                metadata: crate::executable::output::Metadata {
                    abi,
                    frame_extent: SPILL_BYTES,
                    entries: Box::new([]),
                    states: Box::new([]),
                    faults: Box::new([]),
                    traps: Box::new([]),
                    relocations: Box::new([]),
                },
            },
            crate::executable::Tier::Lcq,
            |_| None,
        )
        .unwrap();
    let entry = code.allocation.address() as *const u8;
    // SAFETY: live frame with initialized physical inputs, RX fragment ending
    // in RET, and fixture preserves the platform's callee-saved registers.
    unsafe {
        if fp {
            frame.begin_fp();
            frame.ensure_fp().unwrap();
            crate::fp_env::tests::divide_by_zero();
        }
        nixe_transfer_enter((frame as *mut NativeFrame).cast(), entry);
        if fp {
            frame.finish_fp();
        }
    }
    // The exact storage lease survives the assembly fixture; no raw address
    // escapes it and no callable dispatch root was ever published.
    drop(code);
}

fn read(frame: &[u8], location: ValueLocation, bytes: u8, output: bool) -> Vec<u8> {
    let offset = match location {
        ValueLocation::Constant(value) => {
            return value.to_le_bytes()[..usize::from(bytes)].to_vec();
        }
        ValueLocation::Spill { offset, .. } => offset as usize,
        ValueLocation::Register { class, index } => {
            let base = if output { 8192 } else { 4096 };
            match class {
                RegisterClass::Integer => base + usize::from(index) * 8,
                RegisterClass::Vector => base + 256 + usize::from(index) * 16,
            }
        }
    };
    frame[offset..offset + usize::from(bytes)].to_vec()
}

unsafe extern "C" {
    fn nixe_transfer_enter(frame: *mut u8, code: *const u8);
}

#[cfg(target_arch = "x86_64")]
std::arch::global_asm!(
    r#"
.text
.global nixe_transfer_enter
.type nixe_transfer_enter,@function
nixe_transfer_enter:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    mov [rsp], rsi
    mov r15, rdi
    mov r13, 0x1234
    mov r14, 77
    .macro transfer_gpr reg, index, base, store
        .if \store
            mov [r15 + \base + 8 * \index], \reg
        .else
            mov \reg, [r15 + \base + 8 * \index]
        .endif
    .endm
    .macro transfer_gprs base, store
        transfer_gpr rax, 0, \base, \store
        transfer_gpr rcx, 1, \base, \store
        transfer_gpr rdx, 2, \base, \store
        transfer_gpr rbx, 3, \base, \store
        transfer_gpr rsi, 6, \base, \store
        transfer_gpr rdi, 7, \base, \store
        transfer_gpr r8, 8, \base, \store
        transfer_gpr r9, 9, \base, \store
        transfer_gpr r10, 10, \base, \store
        transfer_gpr r12, 12, \base, \store
    .endm
    transfer_gprs 4096, 0
    .irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        movdqu xmm\n, [r15 + 4352 + 16 * \n]
    .endr
    cmp r15, r15
    pushfq
    pop qword ptr [r15 + 5120]
    mov qword ptr [r15 + 5136], 0
    mov qword ptr [r15 + 5144], 0
    stmxcsr [r15 + 5136]
    call [rsp]
    pushfq
    pop qword ptr [r15 + 5128]
    stmxcsr [r15 + 5144]
    transfer_gprs 8192, 1
    .irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        movdqu [r15 + 8448 + 16 * \n], xmm\n
    .endr
    mov [r15 + 5160], r13
    mov [r15 + 5168], r14
    mov [r15 + 5176], r15
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret
    .purgem transfer_gprs
    .purgem transfer_gpr
.size nixe_transfer_enter, .-nixe_transfer_enter
"#
);

#[cfg(target_arch = "aarch64")]
std::arch::global_asm!(
    r#"
.text
.global nixe_transfer_enter
.type nixe_transfer_enter,%function
nixe_transfer_enter:
    stp x29, x30, [sp, #-176]!
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    stp d8, d9, [sp, #96]
    stp d10, d11, [sp, #112]
    stp d12, d13, [sp, #128]
    stp d14, d15, [sp, #144]
    str x1, [sp, #160]
    mov x21, x0
    mov x19, #0x1234
    mov x20, #77
    .irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,27,28
        ldr x\n, [x21, #(4096 + 8 * \n)]
    .endr
    .irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
        ldr q\n, [x21, #(4352 + 16 * \n)]
    .endr
    cmp x21, x21
    mrs x16, nzcv
    str x16, [x21, #5120]
    mrs x16, fpcr
    str x16, [x21, #5136]
    mrs x16, fpsr
    str x16, [x21, #5184]
    ldr x17, [sp, #160]
    blr x17
    mrs x16, nzcv
    str x16, [x21, #5128]
    mrs x16, fpcr
    str x16, [x21, #5144]
    mrs x16, fpsr
    str x16, [x21, #5192]
    .irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,27,28
        str x\n, [x21, #(8192 + 8 * \n)]
    .endr
    .irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
        str q\n, [x21, #(8448 + 16 * \n)]
    .endr
    str x19, [x21, #5160]
    str x20, [x21, #5168]
    str x21, [x21, #5176]
    ldp d14, d15, [sp, #144]
    ldp d12, d13, [sp, #128]
    ldp d10, d11, [sp, #112]
    ldp d8, d9, [sp, #96]
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #176
    ret
.size nixe_transfer_enter, .-nixe_transfer_enter
"#
);
