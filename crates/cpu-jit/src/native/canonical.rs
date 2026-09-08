//! Canonical data adapters over NativeFrame's borrowed architectural pointers.
//! No A64State layout offsets, duplicate state image or system-ABI calls.

use super::TransferError;
use super::moves::{Copy, Emitter};
use crate::abi::{
    CanonicalState, EntryContract, ExitStateMap, GuestValue, HostAbi, NativeExitReason,
    NativeFrame, NzcvLocation, RegisterClass, TRANSFER_BYTES, ValueLocation,
};
use std::mem::offset_of;

// Does not overlap constant materialization's final 16-byte slot. Canonical
// adapters do not run concurrently with an in-progress fast transfer.
const BORROW_SAVE: u32 = TRANSFER_BYTES - 32;
const CANONICAL: u32 = offset_of!(NativeFrame<'static>, canonical) as u32;
const NZCV_POINTER: u32 = CANONICAL + offset_of!(CanonicalState, nzcv) as u32;

#[derive(Clone, Copy)]
struct Operand {
    pointer: u32,
    offset: u32,
    bytes: u8,
    location: ValueLocation,
}

fn operand(value: GuestValue, location: ValueLocation) -> Operand {
    let (pointer, offset) = match value {
        GuestValue::General(index) => (offset_of!(CanonicalState, x), u32::from(index) * 8),
        GuestValue::Vector(index) => (offset_of!(CanonicalState, vector), u32::from(index) * 16),
        GuestValue::Sp => (offset_of!(CanonicalState, sp), 0),
        GuestValue::Fpcr => (offset_of!(CanonicalState, fpcr), 0),
        GuestValue::Fpsr => (offset_of!(CanonicalState, fpsr), 0),
        GuestValue::TpidrEl0 => (offset_of!(CanonicalState, tpidr_el0), 0),
        GuestValue::TpidrroEl0 => (offset_of!(CanonicalState, tpidrro_el0), 0),
    };
    Operand {
        pointer: CANONICAL + pointer as u32,
        offset,
        bytes: value.bytes(),
        location,
    }
}

/// Load precisely the target's live physical inputs from canonical A64State.
/// Append the fast-entry branch separately. This adapter assumes canonical state
/// is authoritative; it neither publishes an epoch nor installs the guest FP
/// environment. Host flags change only when the target requests host NZCV;
/// packed ingress leaves host flags intact on both hosts. x86-64 host-flag ingress
/// assumes the documented SAHF minimum, checked by the execution owner at setup.
/// On x86-64 spill ingress can use RAX temporarily; if RAX is a target input it
/// is loaded last. Unbound registers need not survive canonical ingress.
pub fn emit_canonical_entry(target: &EntryContract) -> Result<Vec<u8>, TransferError> {
    target.validate().map_err(TransferError::InvalidContract)?;
    let mut operands: Vec<_> = target
        .bindings
        .iter()
        .map(|binding| operand(binding.value, binding.location))
        .collect();
    if target.live_in.nzcv != 0 {
        let location = match target.nzcv {
            NzcvLocation::Packed(location) => location,
            NzcvLocation::Host { .. } => crate::abi::ValueLocation::Spill {
                offset: super::flags::RESULT,
                bytes: 4,
            },
            _ => unreachable!("validated live NZCV ingress is packed or host flags"),
        };
        operands.push(Operand {
            pointer: NZCV_POINTER,
            offset: 0,
            bytes: 4,
            location,
        });
    }
    let mut emitter = Emitter::new(target.abi);
    emit_operands(&mut emitter, target.abi, true, operands);
    if target.live_in.nzcv != 0
        && let NzcvLocation::Host { carry_inverted } = target.nzcv
    {
        super::flags::install_host(&mut emitter, carry_inverted);
    }
    Ok(emitter.finish())
}

/// Store only dirty live architectural values; merge only dirty NZCV bits.
/// This is the writeback portion of a canonical exit, not the complete exit:
/// PC/reason publication, budget reconciliation, FP ownership and the gateway
/// return belong to the caller. Physical source registers remain available for
/// subsequent boundary work. Recipes/host flags are materialized before data
/// writeback can clobber them, and only dirty NZCV bits are computed.
/// When host FPSR is pending, this stores the mapped
/// software contribution (or leaves canonical software FPSR alone if unmapped).
/// Then the caller MUST run [NativeFrame::finish_fp] on exit, or
/// [NativeFrame::suspend_fp] before a helper, to collect sticky host status and
/// restore the caller environment BEFORE general Rust work or epoch quiescence.
/// Collecting before this writeback would let the software store erase flags.
/// The final NZCV merge may clobber x86-64 condition flags, never host FP state.
pub fn emit_canonical_writeback(source: &ExitStateMap) -> Result<Vec<u8>, TransferError> {
    source.validate().map_err(TransferError::InvalidContract)?;
    let operands = source
        .bindings
        .iter()
        .filter(|binding| {
            !source
                .dirty_live
                .intersection(binding.value.state().unwrap())
                .is_empty()
        })
        .map(|binding| operand(binding.value, binding.location))
        .collect();
    let mut emitter = Emitter::new(source.abi);
    let nzcv = if source.dirty_live.nzcv != 0 && !matches!(source.nzcv, NzcvLocation::Packed(_)) {
        super::flags::materialize(&mut emitter, &source.nzcv, source.dirty_live.nzcv);
        NzcvLocation::Packed(ValueLocation::Spill {
            offset: super::flags::RESULT,
            bytes: 4,
        })
    } else {
        source.nzcv.clone()
    };
    emit_operands(&mut emitter, source.abi, false, operands);
    if source.dirty_live.nzcv != 0 {
        let NzcvLocation::Packed(location) = nzcv else {
            unreachable!()
        };
        let pointer = source.abi.reserved().link_scratch[0];
        let value = temporary_register(source.abi);
        if source.abi == HostAbi::X86_64 {
            emitter.memory(false, RegisterClass::Integer, value, BORROW_SAVE, 8);
        }
        emitter.copy(Copy {
            source: location,
            destination: ValueLocation::Register {
                class: RegisterClass::Integer,
                index: value,
            },
            bytes: 4,
        });
        emitter.memory(true, RegisterClass::Integer, pointer, NZCV_POINTER, 8);
        emitter.merge_nzcv(value, pointer, source.dirty_live.nzcv);
        if source.dirty_live.nzcv != crate::analysis::NZCV {
            emitter.memory(true, RegisterClass::Integer, pointer, NZCV_POINTER, 8);
        }
        emitter.memory_at(false, RegisterClass::Integer, value, pointer, 0, 4);
        if source.abi == HostAbi::X86_64 {
            emitter.memory(true, RegisterClass::Integer, value, BORROW_SAVE, 8);
        }
    }
    Ok(emitter.finish())
}

/// Emit dirty state writeback, dynamic/constant destination PC and exit identity,
/// followed by a jump to this invocation's gateway continuation. FP completion
/// and poll reconciliation run there before returning to the lifetime owner.
/// No host call, RET or SP adjustment is emitted. After data writeback no guest
/// register remains live, so the epilogue may use RAX/X0 without saving it.
/// Lazy or host NZCV is materialized by the writeback adapter before publication.
pub fn emit_canonical_exit(
    source: &ExitStateMap,
    pc: ValueLocation,
    reason: NativeExitReason,
) -> Result<Vec<u8>, TransferError> {
    if !pc.valid(source.abi, 8) {
        return Err(TransferError::InvalidContract("invalid exit PC location"));
    }
    if reason == NativeExitReason::None {
        return Err(TransferError::InvalidContract(
            "canonical exit needs a reason",
        ));
    }
    let mut code = emit_canonical_writeback(source)?;
    let mut emitter = Emitter::new(source.abi);
    let scratch = source.abi.reserved().link_scratch[0];
    emitter.copy(Copy {
        source: pc,
        destination: ValueLocation::Register {
            class: RegisterClass::Integer,
            index: 0,
        },
        bytes: 8,
    });
    emitter.memory(
        false,
        RegisterClass::Integer,
        0,
        offset_of!(NativeFrame<'static>, exit_pc) as u32,
        8,
    );
    emitter.memory(
        true,
        RegisterClass::Integer,
        scratch,
        CANONICAL + offset_of!(CanonicalState, pc) as u32,
        8,
    );
    emitter.memory_at(false, RegisterClass::Integer, 0, scratch, 0, 8);
    emitter.constant(scratch, source.site.source.get(), 8);
    emitter.memory(
        false,
        RegisterClass::Integer,
        scratch,
        offset_of!(NativeFrame<'static>, exit_source_version) as u32,
        8,
    );
    // The two adjacent u32 fields are one little-endian store. Besides being
    // smaller, this keeps the AArch64 metadata access within scaled LDR/STR's
    // 64-bit immediate range above the 16 KiB spill arena.
    const _: () = assert!(
        offset_of!(NativeFrame<'static>, exit_reason)
            == offset_of!(NativeFrame<'static>, exit_state_map) + 4
    );
    const _: () = assert!(offset_of!(NativeFrame<'static>, exit_state_map) % 8 == 0);
    emitter.constant(
        scratch,
        u64::from(source.site.state_map) | ((reason as u64) << 32),
        8,
    );
    emitter.memory(
        false,
        RegisterClass::Integer,
        scratch,
        offset_of!(NativeFrame<'static>, exit_state_map) as u32,
        8,
    );
    emitter.memory(
        true,
        RegisterClass::Integer,
        scratch,
        offset_of!(NativeFrame<'static>, gateway_exit) as u32,
        8,
    );
    emitter.jump_register(scratch);
    code.extend(emitter.finish());
    Ok(code)
}

fn temporary_register(abi: HostAbi) -> u8 {
    match abi {
        HostAbi::X86_64 => 0,
        HostAbi::Aarch64 => 17,
    }
}

fn emit_operands(emitter: &mut Emitter, abi: HostAbi, load: bool, mut operands: Vec<Operand>) {
    let pointer = abi.reserved().link_scratch[0];
    let temporary = temporary_register(abi);
    let borrow = abi == HostAbi::X86_64
        && operands
            .iter()
            .any(|operand| !matches!(operand.location, ValueLocation::Register { .. }));
    // Reuse each field pointer for contiguous X/V elements. When loading, a
    // bound RAX must be initialized after all memory-to-memory transfers.
    operands.sort_by_key(|operand| {
        (
            load && borrow
                && matches!(
                    operand.location,
                    ValueLocation::Register {
                        class: RegisterClass::Integer,
                        index: 0
                    }
                ),
            operand.pointer,
        )
    });
    if borrow && !load {
        emitter.memory(false, RegisterClass::Integer, temporary, BORROW_SAVE, 8);
        // Canonical stores may refer to RAX after it is borrowed for a spill.
        for operand in &mut operands {
            if matches!(
                operand.location,
                ValueLocation::Register {
                    class: RegisterClass::Integer,
                    index: 0
                }
            ) {
                operand.location = ValueLocation::Spill {
                    offset: BORROW_SAVE,
                    bytes: operand.bytes,
                };
            }
        }
    }
    let mut last_pointer = None;
    for operand in operands {
        if last_pointer != Some(operand.pointer) {
            emitter.memory(true, RegisterClass::Integer, pointer, operand.pointer, 8);
            last_pointer = Some(operand.pointer);
        }
        match operand.location {
            ValueLocation::Register { class, index } => {
                emitter.memory_at(load, class, index, pointer, operand.offset, operand.bytes)
            }
            ValueLocation::Spill { .. } | ValueLocation::Constant(_) => {
                let part = operand.bytes.min(8);
                for delta in (0..operand.bytes).step_by(usize::from(part)) {
                    if load {
                        emitter.memory_at(
                            true,
                            RegisterClass::Integer,
                            temporary,
                            pointer,
                            operand.offset + u32::from(delta),
                            part,
                        );
                    } else {
                        match operand.location {
                            ValueLocation::Spill { offset, .. } => emitter.memory(
                                true,
                                RegisterClass::Integer,
                                temporary,
                                offset + u32::from(delta),
                                part,
                            ),
                            ValueLocation::Constant(value) => {
                                emitter.constant(temporary, (value >> (delta * 8)) as u64, part)
                            }
                            _ => unreachable!(),
                        }
                    }
                    if load {
                        let ValueLocation::Spill { offset, .. } = operand.location else {
                            unreachable!("validated ingress cannot be a constant")
                        };
                        emitter.memory(
                            false,
                            RegisterClass::Integer,
                            temporary,
                            offset + u32::from(delta),
                            part,
                        );
                    } else {
                        emitter.memory_at(
                            false,
                            RegisterClass::Integer,
                            temporary,
                            pointer,
                            operand.offset + u32::from(delta),
                            part,
                        );
                    }
                }
            }
        }
    }
    if borrow && !load {
        emitter.memory(true, RegisterClass::Integer, temporary, BORROW_SAVE, 8);
    }
}
