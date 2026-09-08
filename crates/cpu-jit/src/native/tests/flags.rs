use super::*;
use nixe_cpu::semantics::arithmetic::{add_with_carry, subtract_with_carry};
use nixe_cpu::semantics::bits::BitWidth;
use nixe_cpu::state::a64::{A64State, Nzcv};

// Exercise production emitters with the existing independent register fixture.
// Arithmetic expectations use the CPU's shared semantics, not a copy of the
// native materializer. Values occupy borrowed GPRs, SIMD and byte-wide spills.
fn check(
    abi: HostAbi,
    nzcv: NzcvLocation,
    inputs: &[(ValueLocation, u128, u8)],
    bits: u8,
    expected: u32,
    fast: bool,
    prefix: &[u8],
) {
    check_boundary(
        abi,
        nzcv,
        inputs,
        bits,
        expected,
        if fast {
            Boundary::Packed
        } else {
            Boundary::Canonical
        },
        prefix,
    );
}

enum Boundary {
    Canonical,
    Packed,
    Host { inverted: bool },
}

pub(super) fn captured_host_nzcv(abi: HostAbi, snapshot: &[u8]) -> u32 {
    let flags = u64::from_le_bytes(snapshot[5128..5136].try_into().unwrap()) as u32;
    match abi {
        HostAbi::Aarch64 => flags,
        HostAbi::X86_64 => {
            ((flags & 0x80) << 24)
                | ((flags & 0x40) << 24)
                | ((flags & 1) << 29)
                | ((flags & 0x800) << 17)
        }
    }
}

fn check_boundary(
    abi: HostAbi,
    nzcv: NzcvLocation,
    inputs: &[(ValueLocation, u128, u8)],
    bits: u8,
    expected: u32,
    boundary: Boundary,
    prefix: &[u8],
) {
    let fast = !matches!(boundary, Boundary::Canonical);
    let (mut source, mut target) = contracts(
        abi,
        &[
            (GuestValue::General(0), integer(0), integer(1)),
            (GuestValue::General(1), integer(1), integer(0)),
        ],
    );
    source.live.nzcv = NZCV;
    source.dirty_live.nzcv = if matches!(nzcv, NzcvLocation::Canonical) {
        0
    } else {
        bits
    };
    source.nzcv = nzcv;
    // Don't commit data when checking selective NZCV writeback.
    source.dirty_live.integer = Default::default();
    target.live_in.nzcv = bits;
    target.nzcv = if bits == 0 {
        NzcvLocation::Canonical
    } else if let Boundary::Host { inverted } = boundary {
        NzcvLocation::Host {
            carry_inverted: inverted,
        }
    } else {
        NzcvLocation::Packed(spill(4000, 4))
    };
    let mut code = prefix.to_vec();
    code.extend(
        if fast {
            emit_fast_transfer(&source, &target)
        } else {
            emit_canonical_writeback(&source)
        }
        .unwrap(),
    );
    if !canonical::native(abi) {
        return;
    }
    let mut state = A64State::default();
    state.set_nzcv(Nzcv::from_bits(0xa0000000));
    {
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(77, 1000).unwrap());
        let mut seed = 42;
        for byte in &mut frame.spill {
            *byte = MaybeUninit::new((next(&mut seed) >> 32) as u8);
        }
        for &(location, value, bytes) in inputs {
            let offset = match location {
                ValueLocation::Register {
                    class: RegisterClass::Integer,
                    index,
                } => 4096 + usize::from(index) * 8,
                ValueLocation::Register {
                    class: RegisterClass::Vector,
                    index,
                } => 4352 + usize::from(index) * 16,
                ValueLocation::Spill { offset, .. } => offset as usize,
                ValueLocation::Constant(_) => continue,
            };
            for (i, byte) in value.to_le_bytes()[..usize::from(bytes)].iter().enumerate() {
                frame.spill[offset + i] = MaybeUninit::new(*byte);
            }
        }
        let before: Vec<u8> = frame
            .spill
            .iter()
            .map(|byte| unsafe { byte.assume_init() })
            .collect();
        invoke(abi, code, &mut frame);
        let after: Vec<u8> = frame
            .spill
            .iter()
            .map(|byte| unsafe { byte.assume_init() })
            .collect();
        for index in 0..32 {
            if integer(index).valid(abi, 8) {
                let original = if fast && index < 2 { 1 - index } else { index };
                assert_eq!(
                    read(&after, integer(index), 8, true),
                    read(&before, integer(original), 8, false),
                    "clobbered GPR {index}"
                );
            }
            if vector(index).valid(abi, 16) {
                assert_eq!(
                    read(&after, vector(index), 16, true),
                    read(&before, vector(index), 16, false)
                );
            }
        }
        assert_eq!(
            &before[16376..16384],
            &after[16376..16384],
            "byte-wide inputs were overwritten"
        );
        assert_eq!(&after[5136..5144], &after[5144..5152], "FP control changed");
        let mask = u32::from(bits) << 28;
        if fast && bits != 0 {
            let actual = if let Boundary::Host { inverted } = boundary {
                captured_host_nzcv(abi, &after) ^ if inverted { 1 << 29 } else { 0 }
            } else {
                u32::from_le_bytes(after[4000..4004].try_into().unwrap())
            };
            assert_eq!(actual & mask, expected & mask, "fast flags mask={bits:x}");
        } else if !fast {
            assert_eq!(
                unsafe { (*frame.canonical.nzcv).bits() },
                (0xa0000000 & !mask) | (expected & mask),
                "canonical mask={bits:x}"
            );
        }
    }
}

#[test]
fn values_to_host_flags_survive_overwriting_cycles() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for nibble in 0u32..16 {
            for inverted in [false, true] {
                for bits in 0..16 {
                    for value in [
                        integer(0),
                        vector(3),
                        spill(2048, 4),
                        ValueLocation::Constant(u128::from(nibble << 28)),
                    ] {
                        for deferred in [false, true] {
                            let location = if deferred {
                                NzcvLocation::Deferred(LazyFlags::Packed(value))
                            } else {
                                NzcvLocation::Packed(value)
                            };
                            check_boundary(
                                abi,
                                location,
                                &[(value, u128::from(nibble << 28), 4)],
                                bits,
                                nibble << 28,
                                Boundary::Host { inverted },
                                &[],
                            );
                        }
                    }
                    check_boundary(
                        abi,
                        NzcvLocation::Canonical,
                        &[],
                        bits,
                        0xa0000000,
                        Boundary::Host { inverted },
                        &[],
                    );
                }
            }
        }
    }
}

#[test]
fn lazy_arithmetic_matches_shared_semantics_at_both_boundaries() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for width in [32, 64] {
            let w = BitWidth::new(width).unwrap();
            let edge = [
                0,
                1,
                (1u128 << (width - 1)) - 1,
                1u128 << (width - 1),
                w.mask(),
            ];
            let mut seed = 71;
            for sample in 0..64 {
                let lhs = if sample < 25 {
                    edge[sample / 5]
                } else {
                    u128::from(next(&mut seed)) & w.mask()
                };
                let rhs = if sample < 25 {
                    edge[sample % 5]
                } else {
                    u128::from(next(&mut seed)) & w.mask()
                };
                for kind in 0..4 {
                    let carry = match kind {
                        0 => false,
                        1 => true,
                        _ => sample % 2 != 0,
                    };
                    let subtract = kind % 2 == 1;
                    let result = if subtract {
                        subtract_with_carry(lhs, rhs, carry, w)
                    } else {
                        add_with_carry(lhs, rhs, carry, w)
                    };
                    let packed = (((result.result >> (width - 1)) as u32) << 31)
                        | (u32::from(result.result == 0) << 30)
                        | (u32::from(result.carry_out) << 29)
                        | (u32::from(result.overflow) << 28);
                    let a = integer(0);
                    let b = if sample % 2 == 0 {
                        integer(1)
                    } else {
                        vector(3)
                    };
                    let r = integer(2);
                    let c = spill(16383, 1);
                    let recipe = match kind {
                        0 => LazyFlags::Add {
                            lhs: a,
                            rhs: b,
                            result: r,
                            width,
                        },
                        1 => LazyFlags::Subtract {
                            lhs: a,
                            rhs: b,
                            result: r,
                            width,
                        },
                        2 => LazyFlags::AddCarry {
                            lhs: a,
                            rhs: b,
                            result: r,
                            carry: c,
                            width,
                        },
                        _ => LazyFlags::SubtractCarry {
                            lhs: a,
                            rhs: b,
                            result: r,
                            carry: c,
                            width,
                        },
                    };
                    check(
                        abi,
                        NzcvLocation::Deferred(recipe),
                        &[
                            (a, lhs, width / 8),
                            (b, rhs, width / 8),
                            (r, result.result, width / 8),
                            (c, u128::from(carry), 1),
                        ],
                        if sample < 25 {
                            NZCV
                        } else {
                            (sample % 16) as u8
                        },
                        packed,
                        sample % 2 == 0,
                        &[],
                    );
                }
            }
        }
    }
}

#[test]
fn conditional_logical_and_packed_recipes_keep_partial_flags() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for bits in 0..16 {
            for predicate in [0u128, 1, 0x80] {
                for fast in [false, true] {
                    let recipe = LazyFlags::Conditional {
                        predicate: spill(16382, 1),
                        when_true: Box::new(LazyFlags::Conditional {
                            predicate: vector(4),
                            when_true: Box::new(LazyFlags::Logical {
                                result: integer(0),
                                width: 32,
                            }),
                            when_false: 3,
                        }),
                        when_false: 9,
                    };
                    check(
                        abi,
                        NzcvLocation::Deferred(recipe),
                        &[
                            (spill(16382, 1), predicate, 1),
                            (vector(4), if fast { 0 } else { predicate }, 1),
                            (integer(0), 0xffff_ffff_0000_0000, 8),
                        ],
                        bits,
                        if predicate == 0 {
                            0x90000000
                        } else if fast {
                            0x30000000
                        } else {
                            0x40000000
                        },
                        fast,
                        &[],
                    );
                }
            }
            for recipe in [
                LazyFlags::Canonical(ValueLocation::Constant(0xd1234567)),
                LazyFlags::Packed(integer(0)),
            ] {
                check(
                    abi,
                    NzcvLocation::Deferred(recipe),
                    &[(integer(0), 0xd1234567, 4)],
                    bits,
                    0xd0000000,
                    bits % 2 == 0,
                    &[],
                );
            }
            check(
                abi,
                NzcvLocation::Canonical,
                &[],
                bits,
                0xa0000000,
                bits % 2 == 0,
                &[],
            );
        }
    }
}

#[test]
fn host_flags_are_captured_before_boundary_arithmetic() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for nibble in 0u32..16 {
            for inverted in [false, true] {
                for bits in 0..16 {
                    // Only the TEST prefix uses the host stack to seed arbitrary
                    // x86 flags. The production emitter has no stack operation.
                    let mut prefix = moves::Emitter::new(abi);
                    let scratch = abi.reserved().link_scratch[0];
                    let native = if abi == HostAbi::X86_64 {
                        2 | ((nibble & 8) << 4)
                            | ((nibble & 4) << 4)
                            | ((nibble & 2) >> 1)
                            | ((nibble & 1) << 11)
                    } else {
                        nibble << 28
                    };
                    prefix.constant(scratch, u64::from(native), 8);
                    let mut prefix = prefix.finish();
                    match abi {
                        HostAbi::X86_64 => prefix.extend([0x41, 0x53, 0x9d]), // PUSH r11; POPFQ
                        HostAbi::Aarch64 => prefix.extend(0xd51b4210u32.to_le_bytes()), // MSR NZCV,x16
                    }
                    check(
                        abi,
                        NzcvLocation::Host {
                            carry_inverted: inverted,
                        },
                        &[],
                        bits,
                        (nibble ^ if inverted { 2 } else { 0 }) << 28,
                        bits % 2 == 0,
                        &prefix,
                    );
                }
            }
        }
    }
}

#[test]
fn host_to_host_carry_conversion_changes_only_carry() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (mut source, mut target) = contracts(abi, &[]);
        source.live.nzcv = NZCV;
        target.live_in.nzcv = NZCV;
        source.nzcv = NzcvLocation::Host {
            carry_inverted: false,
        };
        target.nzcv = NzcvLocation::Host {
            carry_inverted: true,
        };
        let code = emit_fast_transfer(&source, &target).unwrap();
        if abi == HostAbi::X86_64 {
            assert_eq!(code, [0xf5]);
        }
        if !canonical::native(abi) {
            continue;
        }
        let mut state = A64State::default();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(77, 1000).unwrap());
        frame.spill.fill(MaybeUninit::new(0));
        invoke(abi, code, &mut frame);
        let snapshot: Vec<u8> = frame
            .spill
            .iter()
            .map(|byte| unsafe { byte.assume_init() })
            .collect();
        let before = u64::from_le_bytes(snapshot[5120..5128].try_into().unwrap());
        let after = u64::from_le_bytes(snapshot[5128..5136].try_into().unwrap());
        assert_eq!(
            before ^ after,
            if abi == HostAbi::X86_64 { 1 } else { 1 << 29 }
        );
    }
}
