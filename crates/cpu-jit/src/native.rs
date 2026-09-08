//! Native boundary emission. Transfers are compiled to bytes, never interpreted
//! on a guest edge. Code ownership, publication and the final target branch
//! belong to the caller. Transfers materialize required lazy NZCV only when
//! the target representation needs it; canonical writeback is a separate boundary.

mod backend;
mod canonical;
mod flags;
mod gateway;
mod moves;

pub use backend::AllocatedBoundary;
pub use canonical::{emit_canonical_entry, emit_canonical_exit, emit_canonical_writeback};
pub use gateway::{NativeReturn, NativeReturnError, check_host, enter_protected};

use crate::abi::{EntryContract, ExitStateMap, GuestValue, NzcvLocation, ValueLocation};
use moves::{Copy, Emitter};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TransferError {
    InvalidContract(&'static str),
    DifferentHostAbis,
    MissingValue(GuestValue),
    MissingFlags,
    TransferAreaExhausted,
}

impl std::fmt::Display for TransferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidContract(reason) => {
                write!(f, "invalid native transfer contract: {reason}")
            }
            Self::DifferentHostAbis => {
                f.write_str("native transfer contracts use different host ABIs")
            }
            Self::MissingValue(value) => {
                write!(f, "native transfer source has no binding for {value:?}")
            }
            Self::MissingFlags => {
                f.write_str("native transfer source does not cover the target NZCV bits")
            }
            Self::TransferAreaExhausted => {
                f.write_str("native transfer exceeds the fixed transfer area")
            }
        }
    }
}
impl std::error::Error for TransferError {}

/// Emit the NZCV conversion and flag-transparent parallel copies of a fast edge, including
/// cycles, aliases, constants and overlapping spill ranges. An already matching
/// contract emits no bytes. The caller appends its link branch after these bytes.
///
/// Registers, spill slots and host FP state not named as destinations remain
/// unchanged, except the ABI's link scratch and transfer area. Packed NZCV is
/// copied like any other value; identical host-flag contracts need no operation.
/// A packed target materializes only its required NZCV bits, before copies can
/// overwrite lazy operands. A matching host-flag contract is flag transparent;
/// differing carry conventions flip only host carry. Packed/deferred values
/// are installed into host flags after all physical copies. x86-64 emission
/// assumes the documented SAHF minimum; execution owners validate it at setup.
/// This function never silently drops a required value,
/// executes a helper, touches SP or allocates storage during guest execution.
pub fn emit_fast_transfer(
    source: &ExitStateMap,
    target: &EntryContract,
) -> Result<Vec<u8>, TransferError> {
    source.validate().map_err(TransferError::InvalidContract)?;
    target.validate().map_err(TransferError::InvalidContract)?;
    if source.abi != target.abi {
        return Err(TransferError::DifferentHostAbis);
    }
    let mut emitter = Emitter::new(source.abi);
    let mut install_host = None;
    let mut copies = Vec::with_capacity(target.bindings.len() + 1);
    for binding in &target.bindings {
        let input = source
            .bindings
            .iter()
            .find(|input| input.value == binding.value)
            .ok_or(TransferError::MissingValue(binding.value))?;
        copies.push(Copy {
            source: input.location,
            destination: binding.location,
            bytes: binding.value.bytes(),
        });
    }
    if target.live_in.nzcv != 0 {
        if target.live_in.nzcv & !source.live.nzcv != 0 {
            return Err(TransferError::MissingFlags);
        }
        match (&source.nzcv, &target.nzcv) {
            (NzcvLocation::Packed(source), NzcvLocation::Packed(destination)) => {
                copies.push(Copy {
                    source: *source,
                    destination: *destination,
                    bytes: 4,
                });
            }
            (
                NzcvLocation::Host { carry_inverted: a },
                NzcvLocation::Host { carry_inverted: b },
            ) => {
                if a != b {
                    flags::invert_host_carry(&mut emitter);
                }
            }
            (location, NzcvLocation::Packed(destination)) => {
                flags::materialize(&mut emitter, location, target.live_in.nzcv);
                copies.push(Copy {
                    source: ValueLocation::Spill {
                        offset: flags::RESULT,
                        bytes: 4,
                    },
                    destination: *destination,
                    bytes: 4,
                });
            }
            (location, NzcvLocation::Host { carry_inverted }) => {
                if let NzcvLocation::Packed(value) = location {
                    emitter.copy(Copy {
                        source: *value,
                        destination: ValueLocation::Spill {
                            offset: flags::RESULT,
                            bytes: 4,
                        },
                        bytes: 4,
                    });
                } else {
                    flags::materialize(&mut emitter, location, target.live_in.nzcv);
                }
                install_host = Some(*carry_inverted);
            }
            _ => unreachable!("validated live NZCV ingress is packed or host flags"),
        }
    }
    copies.retain(|copy| copy.source != copy.destination);
    let mut temporary_end = 0;
    while !copies.is_empty() {
        // A write is ready only when it cannot destroy any still-needed input.
        // Overlap uses whole registers and byte ranges, not just starting offsets.
        if let Some(index) = copies.iter().position(|copy| {
            copies
                .iter()
                .all(|other| !crate::abi::locations_overlap(copy.destination, other.source))
        }) {
            emitter.copy(copies.remove(index));
            continue;
        }
        if emit_integer_cycle(&mut copies, &mut emitter, source.abi) {
            continue;
        }
        // Break the cycle by saving every input touched by one blocked write.
        // Preserve complete operands, including partially overlapping spill reads.
        // Each saved operand leaves the dependency graph permanently. The bound
        // is the fixed ABI transfer partition, not host stack or dynamic storage.
        let destination = copies[0].destination;
        for index in 0..copies.len() {
            let input = copies[index];
            if !crate::abi::locations_overlap(destination, input.source) {
                continue;
            }
            if temporary_end + 16 > flags::RESULT {
                return Err(TransferError::TransferAreaExhausted);
            }
            let temporary = ValueLocation::Spill {
                offset: temporary_end,
                bytes: input.bytes,
            };
            temporary_end += 16;
            emitter.copy(Copy {
                destination: temporary,
                ..input
            });
            for copy in &mut copies {
                if copy.source == input.source && copy.bytes == input.bytes {
                    copy.source = temporary;
                }
            }
        }
    }
    if let Some(carry_inverted) = install_host {
        flags::install_host(&mut emitter, carry_inverted);
    }
    Ok(emitter.finish())
}

// A closed GPR cycle needs just one reserved register, not a spill round trip.
// Mixed-bank/memory cycles use the general byte-range-safe path above.
fn emit_integer_cycle(
    copies: &mut Vec<Copy>,
    emitter: &mut Emitter,
    abi: crate::abi::HostAbi,
) -> bool {
    use crate::abi::RegisterClass::Integer;
    for start in 0..copies.len() {
        let saved = copies[start].destination;
        if !matches!(saved, ValueLocation::Register { class: Integer, .. }) {
            continue;
        }
        let mut cycle = vec![start];
        let mut input = copies[start].source;
        while input != saved {
            if !matches!(input, ValueLocation::Register { class: Integer, .. }) {
                break;
            }
            let Some(index) = copies.iter().position(|copy| copy.destination == input) else {
                break;
            };
            if cycle.contains(&index) {
                break;
            }
            cycle.push(index);
            input = copies[index].source;
        }
        if input != saved
            || copies.iter().enumerate().any(|(index, copy)| {
                !cycle.contains(&index)
                    && cycle
                        .iter()
                        .any(|&index| copies[index].destination == copy.source)
            })
        {
            continue;
        }
        let scratch = ValueLocation::Register {
            class: Integer,
            index: abi.reserved().link_scratch[0],
        };
        emitter.copy(Copy {
            source: saved,
            destination: scratch,
            bytes: 8,
        });
        for &index in &cycle {
            let mut copy = copies[index];
            if copy.source == saved {
                copy.source = scratch;
            }
            emitter.copy(copy);
        }
        cycle.sort_unstable();
        for index in cycle.into_iter().rev() {
            copies.remove(index);
        }
        return true;
    }
    false
}

#[cfg(test)]
mod tests;
