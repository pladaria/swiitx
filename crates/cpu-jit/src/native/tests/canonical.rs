use super::*;
use nixe_cpu::state::a64::{A64State, Nzcv};

pub(super) fn native(abi: HostAbi) -> bool {
    matches!(
        (std::env::consts::ARCH, abi),
        ("x86_64", HostAbi::X86_64) | ("aarch64", HostAbi::Aarch64)
    )
}

pub(super) fn complete(abi: HostAbi) -> (ExitStateMap, EntryContract) {
    let mut integers = (0..32)
        .map(integer)
        .filter(|location| location.valid(abi, 8));
    let mut vectors = (0..32)
        .map(vector)
        .filter(|location| location.valid(abi, 16));
    let mut values = Vec::new();
    for index in 0..31 {
        let location = integers
            .next()
            .unwrap_or(spill(2048 + u32::from(index) * 8, 8));
        values.push((GuestValue::General(index), location, location));
    }
    for index in 0..32 {
        let location = vectors
            .next()
            .unwrap_or(spill(2560 + u32::from(index) * 16, 16));
        values.push((GuestValue::Vector(index), location, location));
    }
    for (value, offset) in [
        (GuestValue::Sp, 3200),
        (GuestValue::TpidrEl0, 3208),
        (GuestValue::TpidrroEl0, 3216),
        (GuestValue::Fpcr, 3232),
        (GuestValue::Fpsr, 3236),
    ] {
        let location = spill(offset, value.bytes());
        values.push((value, location, location));
    }
    let (mut source, mut target) = contracts(abi, &values);
    source.live.nzcv = NZCV;
    source.dirty_live.nzcv = NZCV;
    target.live_in.nzcv = NZCV;
    source.nzcv = NzcvLocation::Packed(spill(3240, 4));
    target.nzcv = source.nzcv.clone();
    (source, target)
}

fn set(state: &mut A64State, value: GuestValue, bits: u128) {
    match value {
        GuestValue::General(index) => {
            state.general_register_storage_mut()[usize::from(index)] = bits as u64
        }
        GuestValue::Vector(index) => {
            assert!(state.set_vector(index, bits));
        }
        GuestValue::Sp => *state.stack_pointer_storage_mut() = bits as u64,
        GuestValue::Fpcr => state.set_fpcr(bits as u32),
        GuestValue::Fpsr => state.set_fpsr(bits as u32),
        GuestValue::TpidrEl0 => *state.tpidr_el0_storage_mut() = bits as u64,
        GuestValue::TpidrroEl0 => *state.tpidrro_el0_storage_mut() = bits as u64,
    }
}

pub(super) fn pattern(target: &EntryContract) -> (A64State, Vec<(GuestValue, u128)>) {
    let mut state = A64State::default();
    let mut seed = 73;
    let values = target
        .bindings
        .iter()
        .map(|binding| {
            let bits = u128::from(next(&mut seed)) | (u128::from(next(&mut seed)) << 64);
            set(&mut state, binding.value, bits);
            (binding.value, bits)
        })
        .collect();
    state.set_nzcv(Nzcv::from_bits(0xb0000000));
    state.set_pc(0xdeadbeef);
    (state, values)
}

fn initialize(frame: &mut NativeFrame<'_>) {
    let mut seed = 97;
    for byte in &mut frame.spill {
        *byte = MaybeUninit::new((next(&mut seed) >> 32) as u8);
    }
}
fn arena<'a>(frame: &'a NativeFrame<'_>) -> &'a [u8; SPILL_BYTES as usize] {
    // SAFETY: every caller initialized the complete array before invoking code.
    unsafe { &*frame.spill.as_ptr().cast() }
}

fn pins_and_fp(frame: &NativeFrame<'_>, flags: bool) {
    let data = arena(frame);
    for (offset, value) in [
        (5160, 0x1234u64),
        (5168, 77),
        (5176, frame as *const NativeFrame as u64),
    ] {
        assert_eq!(&data[offset..offset + 8], &value.to_le_bytes());
    }
    if flags {
        assert_eq!(&data[5120..5128], &data[5128..5136]);
    }
    assert_eq!(&data[5136..5144], &data[5144..5152]);
    if cfg!(target_arch = "aarch64") {
        assert_eq!(&data[5184..5192], &data[5192..5200]);
    }
}

#[test]
fn canonical_entry_loads_real_state_into_registers_and_spills() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for flags_in_rax in [false, true] {
            let (_, mut target) = complete(abi);
            if flags_in_rax {
                target.bindings[0].location = spill(3248, 8);
                target.nzcv = NzcvLocation::Packed(integer(0));
            }
            let code = emit_canonical_entry(&target).unwrap();
            if !native(abi) {
                continue;
            }
            let (mut state, values) = pattern(&target);
            let original = state.clone();
            {
                let mut frame = NativeFrame::new(&mut state, PollBudget::new(77, 1000).unwrap());
                initialize(&mut frame);
                invoke(abi, code, &mut frame);
                for (binding, (_, bits)) in target.bindings.iter().zip(&values) {
                    assert_eq!(
                        read(arena(&frame), binding.location, binding.value.bytes(), true),
                        bits.to_le_bytes()[..usize::from(binding.value.bytes())]
                    );
                }
                let NzcvLocation::Packed(location) = target.nzcv else {
                    unreachable!()
                };
                assert_eq!(
                    read(arena(&frame), location, 4, true),
                    original.nzcv().bits().to_le_bytes()
                );
                pins_and_fp(&frame, true);
            }
            assert_eq!(
                state, original,
                "canonical ingress must not modify A64State"
            );
        }
    }
}

#[test]
fn canonical_host_flag_ingress_preserves_all_loaded_inputs() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for nibble in 0..16 {
            for inverted in [false, true] {
                for bits in 1..16 {
                    let (_, mut target) = complete(abi);
                    target.live_in.nzcv = bits;
                    target.nzcv = NzcvLocation::Host {
                        carry_inverted: inverted,
                    };
                    let code = emit_canonical_entry(&target).unwrap();
                    if !native(abi) {
                        continue;
                    }
                    let (mut state, values) = pattern(&target);
                    state.set_nzcv(Nzcv::from_bits(nibble << 28));
                    let original = state.clone();
                    {
                        let mut frame =
                            NativeFrame::new(&mut state, PollBudget::new(77, 1000).unwrap());
                        initialize(&mut frame);
                        invoke(abi, code, &mut frame);
                        for (binding, (_, value)) in target.bindings.iter().zip(&values) {
                            assert_eq!(
                                read(arena(&frame), binding.location, binding.value.bytes(), true),
                                value.to_le_bytes()[..usize::from(binding.value.bytes())]
                            );
                        }
                        let flags = super::flags::captured_host_nzcv(abi, arena(&frame))
                            ^ if inverted { 1 << 29 } else { 0 };
                        assert_eq!(
                            flags & (u32::from(bits) << 28),
                            (nibble << 28) & (u32::from(bits) << 28)
                        );
                        pins_and_fp(&frame, false);
                    }
                    assert_eq!(state, original);
                }
            }
        }
    }
}

#[test]
fn canonical_entry_does_not_access_unneeded_state_pointers() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (_, target) = contracts(abi, &[(GuestValue::General(30), integer(0), integer(0))]);
        let code = emit_canonical_entry(&target).unwrap();
        if !native(abi) {
            continue;
        }
        let mut state = A64State::default();
        state.general_register_storage_mut()[30] = 0x8123456789abcdef;
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(77, 1000).unwrap());
        initialize(&mut frame);
        // Only the X-array pointer is usable: neither a full-state load nor a
        // native assumption about A64State's Rust layout can pass this case.
        frame.canonical.state = std::ptr::null_mut();
        frame.canonical.sp = std::ptr::null_mut();
        frame.canonical.vector = std::ptr::null_mut();
        frame.canonical.pc = std::ptr::null_mut();
        frame.canonical.nzcv = std::ptr::null_mut();
        frame.canonical.fpcr = std::ptr::null_mut();
        frame.canonical.fpsr = std::ptr::null_mut();
        frame.canonical.tpidr_el0 = std::ptr::null_mut();
        frame.canonical.tpidrro_el0 = std::ptr::null_mut();
        invoke(abi, code, &mut frame);
        assert_eq!(
            read(arena(&frame), integer(0), 8, true),
            0x8123456789abcdefu64.to_le_bytes()
        );
        pins_and_fp(&frame, true);
    }
}

#[test]
fn canonical_writeback_commits_only_dirty_values_and_selected_flag_bits() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for mask in 0..=NZCV {
            let (mut source, target) = complete(abi);
            source.dirty_live = StateSet {
                nzcv: mask,
                ..StateSet::default()
            };
            for (index, binding) in source.bindings.iter_mut().enumerate() {
                if index % 3 == usize::from(mask % 3) {
                    source.dirty_live = source.dirty_live.union(binding.value.state().unwrap());
                }
                if index == 7 {
                    binding.location = ValueLocation::Constant(0xabcdef0123456789);
                }
                if index == 40 {
                    binding.location = ValueLocation::Constant(u128::MAX);
                }
            }
            let flags = match mask % 3 {
                0 => integer(0),
                1 => vector(15),
                _ => spill(3240, 4),
            };
            source.nzcv = NzcvLocation::Packed(flags);
            let code = emit_canonical_writeback(&source).unwrap();
            if !native(abi) {
                continue;
            }
            let (mut state, _) = pattern(&target);
            let mut expected = state.clone();
            {
                let mut frame = NativeFrame::new(&mut state, PollBudget::new(77, 1000).unwrap());
                initialize(&mut frame);
                let before = *arena(&frame);
                for binding in &source.bindings {
                    if source
                        .dirty_live
                        .intersection(binding.value.state().unwrap())
                        .is_empty()
                    {
                        continue;
                    }
                    let bytes = read(&before, binding.location, binding.value.bytes(), false);
                    let mut bits = [0u8; 16];
                    bits[..bytes.len()].copy_from_slice(&bytes);
                    set(&mut expected, binding.value, u128::from_le_bytes(bits));
                }
                let flags = u32::from_le_bytes(read(&before, flags, 4, false).try_into().unwrap());
                let mask = u32::from(mask) << 28;
                expected.set_nzcv(Nzcv::from_bits(
                    (expected.nzcv().bits() & !mask) | (flags & mask),
                ));
                invoke(abi, code, &mut frame);
                for binding in &source.bindings {
                    assert_eq!(
                        read(arena(&frame), binding.location, binding.value.bytes(), true),
                        read(&before, binding.location, binding.value.bytes(), false),
                        "writeback destroyed a physical source"
                    );
                }
                pins_and_fp(&frame, mask == 0 || abi == HostAbi::Aarch64);
            }
            assert_eq!(state, expected, "dirty mask {mask:x}");
        }
    }
}

#[test]
fn canonical_adapters_compose_with_a_real_fast_transfer() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (source, entry) = complete(abi);
        let mut destination = entry.clone();
        // Reverse physical assignments within each width, including spills.
        for bytes in [4, 8, 16] {
            let locations: Vec<_> = entry
                .bindings
                .iter()
                .filter(|b| b.value.bytes() == bytes)
                .map(|b| b.location)
                .rev()
                .collect();
            for (binding, location) in destination
                .bindings
                .iter_mut()
                .filter(|b| b.value.bytes() == bytes)
                .zip(locations)
            {
                binding.location = location;
            }
        }
        let exit = ExitStateMap {
            bindings: destination.bindings.clone(),
            ..source.clone()
        };
        let mut code = emit_canonical_entry(&entry).unwrap();
        code.extend(emit_fast_transfer(&source, &destination).unwrap());
        code.extend(emit_canonical_writeback(&exit).unwrap());
        if !native(abi) {
            continue;
        }
        let (mut state, _) = pattern(&entry);
        let original = state.clone();
        {
            let mut frame = NativeFrame::new(&mut state, PollBudget::new(77, 1000).unwrap());
            initialize(&mut frame);
            invoke(abi, code, &mut frame);
            pins_and_fp(&frame, abi == HostAbi::Aarch64);
        }
        assert_eq!(state, original);
    }
}

#[test]
fn canonical_adapters_accept_host_and_deferred_flag_contracts() {
    let (mut source, mut target) = contracts(HostAbi::X86_64, &[]);
    assert!(emit_canonical_entry(&target).unwrap().is_empty());
    assert!(emit_canonical_writeback(&source).unwrap().is_empty());
    target.live_in.nzcv = NZCV;
    target.nzcv = NzcvLocation::Host {
        carry_inverted: false,
    };
    assert!(emit_canonical_entry(&target).is_ok());
    source.live.nzcv = NZCV;
    source.dirty_live.nzcv = NZCV;
    source.nzcv = NzcvLocation::Deferred(LazyFlags::Logical {
        result: integer(0),
        width: 64,
    });
    assert!(emit_canonical_writeback(&source).is_ok());
}

#[test]
fn pending_host_fpsr_is_merged_after_software_writeback() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for location in [
            Some(integer(0)),
            Some(spill(3240, 4)),
            Some(ValueLocation::Constant((1 << 27) | (1 << 4))),
            None,
        ] {
            let values: Vec<_> = location
                .into_iter()
                .map(|location| (GuestValue::Fpsr, location, spill(3240, 4)))
                .collect();
            let (mut source, _) = contracts(abi, &values);
            source.live.fpsr = true;
            source.dirty_live.fpsr = true;
            source.host_fpsr_pending = true;
            let code = emit_canonical_writeback(&source).unwrap();
            if !native(abi) {
                continue;
            }
            let mut state = A64State::default();
            state.set_fpsr((1 << 27) | 8);
            let software = if location.is_some() {
                (1 << 27) | (1 << 4)
            } else {
                state.fpsr()
            };
            {
                let mut frame = NativeFrame::new(&mut state, PollBudget::new(77, 1000).unwrap());
                initialize(&mut frame);
                for offset in [4096, 3240] {
                    for (index, byte) in software.to_le_bytes().into_iter().enumerate() {
                        frame.spill[offset + index] = MaybeUninit::new(byte);
                    }
                }
                // The fixture raises a real divide-by-zero, then executes data
                // writeback before returning directly to shared FP completion.
                invoke_with_fp(abi, code, &mut frame);
                assert_eq!(frame.execution_epoch, 0); // Direct lease, no dispatch lookup.
                assert_eq!(
                    (
                        frame.host_fp.active,
                        frame.host_fp.saved,
                        frame.host_fp.suspended
                    ),
                    (0, 0, 0)
                );
            }
            assert_eq!(
                state.fpsr(),
                software | 2,
                "software store lost pending DZC"
            );
        }
    }
}
