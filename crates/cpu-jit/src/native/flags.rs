//! Compile the shared lazy recipes to boundary instructions, not runtime helpers.
//! The formulas match the existing SSA lowering and Arm AddWithCarry:
//! https://developer.arm.com/documentation/ddi0602/2025-12/Shared-Pseudocode
//! https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85

use super::moves::{Copy, Emitter};
use crate::abi::{
    CanonicalState, HostAbi, LazyFlags, NativeFrame, NzcvLocation, RegisterClass::Integer,
    ValueLocation,
};
use std::mem::offset_of;

// The packed result survives subsequent parallel copies and canonical writeback.
// Other slots below are live only during materialization; no guest map can name
// the ABI-owned transfer partition. BORROW_SAVE and SCRATCH_SLOT remain separate.
pub(super) const RESULT: u32 = crate::abi::TRANSFER_BYTES - 64;
const SAVES: u32 = RESULT - 32;
const HOST_BITS: u32 = RESULT - 8;

/// Install the protected packed result after all physical input copies. Only
/// link scratch is clobbered on AArch64. x86-64 borrows and restores RAX;
/// unrequested NZCV bits are unspecified. SAHF is a minimum host requirement.
pub(super) fn install_host(e: &mut Emitter, carry_inverted: bool) {
    if e.abi == HostAbi::X86_64 {
        // SAHF installs SF/ZF/CF from AH and leaves OF unchanged. Construct AH
        // first, then set OF with 0x7f + V in AL, without carrying into AH.
        // MOV restores the borrowed RAX without changing any flags. No PUSHF,
        // POPF or host stack use: https://cdrdv2-public.intel.com/782151/253667-sdm-vol-2b.pdf
        e.memory(false, Integer, 0, SAVES, 8);
        e.memory(true, Integer, 0, RESULT, 4);
        e.shift(0, 16, false, 4); // N,Z -> AH[7:6]
        e.x64(&[], false, &[0x81], 4, 0, None); // AND eax,0xc000
        e.word(0xc000);
        e.memory(true, Integer, 11, RESULT, 4);
        e.shift(11, 21, false, 4); // C -> AH[0]
        e.x64(&[], false, &[0x81], 4, 11, None);
        e.word(0x100);
        e.logic(Op::Or, 0, 11, 4);
        if carry_inverted {
            e.x64(&[], false, &[0x81], 6, 0, None); // XOR eax,0x100
            e.word(0x100);
        }
        e.x64(&[], false, &[0x81], 1, 0, None); // OR eax,0x7f
        e.word(0x7f);
        e.memory(true, Integer, 11, RESULT, 4);
        e.shift(11, 28, false, 4);
        e.x64(&[], false, &[0x81], 4, 11, None);
        e.word(1);
        e.x64(&[], false, &[0x00], 11, 0, None); // ADD al,r11b: OF = V
        e.code_byte(0x9e); // SAHF: SF=N, ZF=Z, CF=C (or !C), OF remains V
        e.memory(true, Integer, 0, SAVES, 8);
        return;
    }
    e.memory(true, Integer, 16, RESULT, 4);
    if carry_inverted {
        e.constant(17, 1 << 29, 4);
        e.word(0xca110210); // EOR x16,x16,x17
    }
    e.word(0xd51b4210); // MSR NZCV,x16
}

/// Convert only the carry convention while retaining host-resident NZV.
/// Uses only globally reserved scratch and never touches SP or guest values.
pub(super) fn invert_host_carry(e: &mut Emitter) {
    if e.abi == HostAbi::X86_64 {
        e.code_byte(0xf5); // CMC
    } else {
        e.word(0xd53b4210); // MRS x16,NZCV
        e.constant(17, 1 << 29, 4);
        e.word(0xca110210); // EOR x16,x16,x17 (flag transparent)
        e.word(0xd51b4210); // MSR NZCV,x16
    }
}

#[derive(Clone, Copy)]
enum Op {
    And,
    Or,
    Xor,
}
#[derive(Clone, Copy)]
enum Cond {
    Eq,
    Lt,
    Ge,
    Gt,
    Negative,
    Overflow,
    Carry,
}

/// Preserve every guest register/spill. Host flags may be destroyed, but never
/// FP state or SP. The caller has validated the recipe and consumes RESULT
/// before reusing the transfer partition. Only requested architectural bits
/// (NZCV nibble order) are computed; other packed bits are zero.
pub(super) fn materialize(emitter: &mut Emitter, location: &NzcvLocation, bits: u8) {
    for reg in 0..3 {
        emitter.memory(false, Integer, reg, SAVES + u32::from(reg) * 8, 8);
    }
    if let NzcvLocation::Host { carry_inverted } = location {
        if emitter.abi == HostAbi::Aarch64 {
            emitter.word(0xd53b4200); // MRS x0,NZCV
            if *carry_inverted {
                emitter.constant(1, 1 << 29, 4);
                emitter.logic(Op::Xor, 0, 1, 4);
            }
            emitter.constant(1, u64::from(bits) << 28, 4);
            emitter.logic(Op::And, 0, 1, 4);
            emitter.memory(false, Integer, 0, RESULT, 4);
        } else {
            // SETcc and MOVZX are flag transparent. Capture every requested
            // condition BEFORE arithmetic can destroy the native producer.
            for (bit, cond) in [
                (3, Cond::Negative),
                (2, Cond::Eq),
                (1, Cond::Carry),
                (0, Cond::Overflow),
            ] {
                if bits & (1 << bit) != 0 {
                    emitter.condition(0, cond);
                    emitter.x64(&[], false, &[0x88], 0, 15, Some(HOST_BITS + bit));
                }
            }
            emitter.constant(0, 0, 4);
            emitter.memory(false, Integer, 0, RESULT, 4);
            for bit in 0..4 {
                if bits & (1 << bit) == 0 {
                    continue;
                }
                emitter.byte_load(0, HOST_BITS + bit);
                if bit == 1 && *carry_inverted {
                    emitter.constant(1, 1, 4);
                    emitter.logic(Op::Xor, 0, 1, 4);
                }
                accumulate(emitter, bit);
            }
        }
    } else if matches!(
        location,
        NzcvLocation::Canonical
            | NzcvLocation::Packed(_)
            | NzcvLocation::Deferred(LazyFlags::Canonical(_) | LazyFlags::Packed(_))
    ) {
        // An already packed source is one load and mask, not four independent
        // recipe evaluations. Read its authoritative storage only once.
        match location {
            NzcvLocation::Packed(value)
            | NzcvLocation::Deferred(LazyFlags::Canonical(value) | LazyFlags::Packed(value)) => {
                load(emitter, *value, 0, 4)
            }
            NzcvLocation::Canonical => {
                emitter.memory(
                    true,
                    Integer,
                    0,
                    (offset_of!(NativeFrame<'static>, canonical) + offset_of!(CanonicalState, nzcv))
                        as u32,
                    8,
                );
                emitter.memory_at(true, Integer, 0, 0, 0, 4);
            }
            _ => unreachable!(),
        }
        emitter.constant(1, u64::from(bits) << 28, 4);
        emitter.logic(Op::And, 0, 1, 4);
        emitter.memory(false, Integer, 0, RESULT, 4);
    } else {
        emitter.constant(0, 0, 4);
        emitter.memory(false, Integer, 0, RESULT, 4);
        let NzcvLocation::Deferred(recipe) = location else {
            unreachable!()
        };
        for bit in 0..4 {
            if bits & (1 << bit) == 0 {
                continue;
            }
            recipe_bit(emitter, recipe, bit);
            emitter.constant(1, 1, 4);
            emitter.logic(Op::And, 0, 1, 4);
            accumulate(emitter, bit);
        }
    }
    for reg in 0..3 {
        emitter.memory(true, Integer, reg, SAVES + u32::from(reg) * 8, 8);
    }
}

fn accumulate(emitter: &mut Emitter, bit: u32) {
    emitter.shift(0, bit as u8 + 28, true, 4);
    emitter.memory(true, Integer, 1, RESULT, 4);
    emitter.logic(Op::Or, 0, 1, 4);
    emitter.memory(false, Integer, 0, RESULT, 4);
}

fn load(emitter: &mut Emitter, mut source: ValueLocation, register: u8, bytes: u8) {
    if let ValueLocation::Register {
        class: Integer,
        index: 0..3,
    } = source
    {
        let ValueLocation::Register { index, .. } = source else {
            unreachable!()
        };
        source = ValueLocation::Spill {
            offset: SAVES + u32::from(index) * 8,
            bytes,
        };
    }
    if bytes == 1 {
        if let ValueLocation::Spill { offset, .. } = source {
            emitter.byte_load(register, offset);
        } else {
            // Scalar/vector registers contain at least 32 physical bits; a byte
            // operand consumes only its low eight, even if upper bits are dirty.
            emitter.copy(Copy {
                source,
                destination: ValueLocation::Register {
                    class: Integer,
                    index: register,
                },
                bytes: 4,
            });
            let scratch = emitter.abi.reserved().link_scratch[0];
            emitter.constant(scratch, 255, 4);
            emitter.logic(Op::And, register, scratch, 4);
        }
    } else {
        emitter.copy(Copy {
            source,
            destination: ValueLocation::Register {
                class: Integer,
                index: register,
            },
            bytes,
        });
    }
}

// All recursive arms return one bit in register 0. Recipes use original source
// locations, including saved values of borrowed registers, never intermediates.
fn recipe_bit(e: &mut Emitter, recipe: &LazyFlags<ValueLocation>, bit: u32) {
    use LazyFlags::*;
    match recipe {
        Canonical(value) | Packed(value) => {
            load(e, *value, 0, 4);
            e.shift(0, bit as u8 + 28, false, 4);
        }
        Conditional {
            predicate,
            when_true,
            when_false,
        } => {
            recipe_bit(e, when_true, bit);
            load(e, *predicate, 1, 1);
            // Predicate is an i8 boolean (zero/nonzero), not a packed flag bit.
            e.constant(2, 0, 4);
            e.compare(1, 2, 4);
            e.condition(1, Cond::Eq);
            // XOR-select with a 0/1 mask: normalize true value before selecting.
            e.constant(2, 1, 4);
            e.logic(Op::And, 0, 2, 4);
            let literal = u64::from((when_false >> bit) & 1);
            if literal == 0 {
                e.logic(Op::Xor, 1, 2, 4);
                e.logic(Op::And, 0, 1, 4);
            } else {
                e.logic(Op::Or, 0, 1, 4);
            }
        }
        Add { result, width, .. }
        | Subtract { result, width, .. }
        | AddCarry { result, width, .. }
        | SubtractCarry { result, width, .. }
        | Logical { result, width }
            if bit >= 2 =>
        {
            load(e, *result, 0, width / 8);
            if bit == 3 {
                e.shift(0, width - 1, false, width / 8);
            } else {
                e.constant(1, 0, width / 8);
                e.compare(0, 1, width / 8);
                e.condition(0, Cond::Eq);
            }
        }
        Logical { .. } => e.constant(0, 0, 4),
        Add {
            lhs,
            rhs,
            result,
            width,
        }
        | Subtract {
            lhs,
            rhs,
            result,
            width,
        }
        | AddCarry {
            lhs,
            rhs,
            result,
            width,
            ..
        }
        | SubtractCarry {
            lhs,
            rhs,
            result,
            width,
            ..
        } => {
            let add = matches!(recipe, Add { .. } | AddCarry { .. });
            if bit == 0 {
                load(e, *lhs, 0, width / 8);
                load(e, *rhs, 1, width / 8);
                load(e, *result, 2, width / 8);
                e.logic(Op::Xor, 1, 0, width / 8);
                if add {
                    e.invert(1, width / 8);
                }
                e.logic(Op::Xor, 2, 0, width / 8);
                e.logic(Op::And, 1, 2, width / 8);
                e.shift(1, width - 1, false, width / 8);
                e.copy(Copy {
                    source: ValueLocation::Register {
                        class: Integer,
                        index: 1,
                    },
                    destination: ValueLocation::Register {
                        class: Integer,
                        index: 0,
                    },
                    bytes: 4,
                });
            } else {
                load(e, if add { *result } else { *lhs }, 0, width / 8);
                load(e, if add { *lhs } else { *rhs }, 1, width / 8);
                e.compare(0, 1, width / 8);
                if let AddCarry { carry, .. } | SubtractCarry { carry, .. } = recipe {
                    e.condition(2, if add { Cond::Lt } else { Cond::Gt });
                    e.condition(0, Cond::Eq);
                    load(e, *carry, 1, 1);
                    e.logic(Op::And, 0, 1, 4);
                    e.logic(Op::Or, 0, 2, 4);
                } else {
                    e.condition(0, if add { Cond::Lt } else { Cond::Ge });
                }
            }
        }
    }
}

impl Emitter {
    fn byte_load(&mut self, register: u8, offset: u32) {
        if self.abi == HostAbi::X86_64 {
            self.x64(
                &[],
                false,
                &[0x0f, 0xb6],
                register,
                self.abi.reserved().frame,
                Some(offset),
            );
        } else {
            if offset < 4096 {
                self.word(0x39400000 | (offset << 10) | (21 << 5) | u32::from(register));
            } else {
                self.constant(16, u64::from(offset), 8);
                self.word(0x8b1002b0); // ADD x16,x21,x16
                self.word(0x39400200 | u32::from(register)); // LDRB Wt,[x16]
            }
        }
    }
    fn logic(&mut self, op: Op, dst: u8, src: u8, bytes: u8) {
        if self.abi == HostAbi::X86_64 {
            self.x64(
                &[],
                bytes == 8,
                &[match op {
                    Op::And => 0x21,
                    Op::Or => 0x09,
                    Op::Xor => 0x31,
                }],
                src,
                dst,
                None,
            );
        } else {
            let base = match op {
                Op::And => 0x0a000000,
                Op::Or => 0x2a000000,
                Op::Xor => 0x4a000000,
            };
            self.word(
                base | (u32::from(bytes == 8) << 31)
                    | (u32::from(src) << 16)
                    | (u32::from(dst) << 5)
                    | u32::from(dst),
            );
        }
    }
    fn invert(&mut self, reg: u8, bytes: u8) {
        if self.abi == HostAbi::X86_64 {
            self.x64(&[], bytes == 8, &[0xf7], 2, reg, None);
        } else {
            self.word(
                0x2a2003e0
                    | (u32::from(bytes == 8) << 31)
                    | (u32::from(reg) << 16)
                    | u32::from(reg),
            );
        }
    }
    fn shift(&mut self, reg: u8, amount: u8, left: bool, bytes: u8) {
        if self.abi == HostAbi::X86_64 {
            self.x64(
                &[],
                bytes == 8,
                &[0xc1],
                if left { 4 } else { 5 },
                reg,
                None,
            );
            self.code_byte(amount);
        } else {
            let width = u32::from(bytes) * 8;
            let (immr, imms) = if left {
                (
                    (width - u32::from(amount)) % width,
                    width - 1 - u32::from(amount),
                )
            } else {
                (u32::from(amount), width - 1)
            };
            self.word(
                0x53000000
                    | (if bytes == 8 { 0x80400000 } else { 0 })
                    | (immr << 16)
                    | (imms << 10)
                    | (u32::from(reg) << 5)
                    | u32::from(reg),
            );
        }
    }
    fn compare(&mut self, lhs: u8, rhs: u8, bytes: u8) {
        if self.abi == HostAbi::X86_64 {
            self.x64(&[], bytes == 8, &[0x39], rhs, lhs, None);
        } else {
            self.word(
                0x6b00001f
                    | (u32::from(bytes == 8) << 31)
                    | (u32::from(rhs) << 16)
                    | (u32::from(lhs) << 5),
            );
        }
    }
    fn condition(&mut self, reg: u8, cond: Cond) {
        if self.abi == HostAbi::X86_64 {
            let cc = match cond {
                Cond::Eq => 4,
                Cond::Lt | Cond::Carry => 2,
                Cond::Ge => 3,
                Cond::Gt => 7,
                Cond::Negative => 8,
                Cond::Overflow => 0,
            };
            self.x64(&[], false, &[0x0f, 0x90 | cc], 0, reg, None);
            self.x64(&[], false, &[0x0f, 0xb6], reg, reg, None);
        } else {
            let cc = match cond {
                Cond::Eq => 0,
                Cond::Lt => 3,
                Cond::Ge | Cond::Carry => 2,
                Cond::Gt => 8,
                Cond::Negative => 4,
                Cond::Overflow => 6,
            };
            self.word(0x1a9f07e0 | ((cc ^ 1) << 12) | u32::from(reg));
        }
    }
}
