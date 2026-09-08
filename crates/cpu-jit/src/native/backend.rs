//! Compile-time translation of the fork's final allocations. Operand indices
//! are the original nixe_entry results / nixe_state, nixe_exit or nixe_fault
//! arguments, NOT virtual register numbers or pre-allocation guesses.
//! See https://github.com/pladaria/wasmtime/blob/e2a984d96678207094c0fc50057c8b6bcfd68715/cranelift/codegen/src/nixe/boundary.rs

use super::TransferError;
use crate::abi::{
    GuestValue, HostAbi, RegisterClass, SPILL_BYTES, TRANSFER_BYTES, ValueBinding, ValueLocation,
};
use cranelift_codegen::{
    CompiledCode,
    ir::Type,
    nixe::{Location, StateMap},
};

const _: () = {
    assert!(TRANSFER_BYTES == cranelift_codegen::nixe::TRANSFER_BYTES);
    assert!(SPILL_BYTES == cranelift_codegen::nixe::FRAME_BYTES);
};

/// A borrowed final entry, observation, exit or instruction-attached fault map.
/// The lowering retains the semantic meaning of its ordered operands and uses
/// `bindings` / `location` to build the shared contracts and lazy recipes.
/// Backend maps describe SSA values, not persistent host condition flags;
/// never infer `NzcvLocation::Host` from a preceding machine instruction.
/// Native execution never walks this object. Code ownership and publication
/// remain the caller's responsibility; offsets are relative to `code[0]`.
pub struct AllocatedBoundary<'a> {
    pub map: &'a StateMap,
    abi: HostAbi,
    frame_extent: u32,
}

impl<'a> AllocatedBoundary<'a> {
    pub fn new(
        abi: HostAbi,
        code: &CompiledCode,
        map: &'a StateMap,
    ) -> Result<Self, TransferError> {
        let fail = TransferError::InvalidContract;
        let frame_extent = code
            .buffer
            .frame_layout()
            .and_then(|frame| frame.nixe_frame_size)
            .ok_or_else(|| fail("backend output does not use the Nixe ABI"))?;
        if !(TRANSFER_BYTES..=SPILL_BYTES).contains(&frame_extent) {
            return Err(fail("backend frame exceeds the fixed spill partition"));
        }
        let patch_bytes = if abi == HostAbi::X86_64 { 8 } else { 4 };
        if map.patch_bytes != 0
            && (map.entry
                || map.patch_bytes != patch_bytes
                || !map.offset.is_multiple_of(u32::from(patch_bytes)))
        {
            return Err(fail("invalid backend exit patch shape"));
        }
        if u64::from(map.offset) + u64::from(map.patch_bytes) > code.code_buffer().len() as u64 {
            return Err(fail("backend boundary lies outside its code"));
        }
        let result = Self {
            map,
            abi,
            frame_extent,
        };
        for (index, value) in map.values.iter().enumerate() {
            if map.entry && value.location == Location::Unused {
                continue;
            }
            result.location(index, value.ty)?;
        }
        Ok(result)
    }

    /// Resolve an operand with its lowering-known type, including recipe/PC
    /// operands that are not guest registers. Eliminated entry definitions must
    /// be explicitly omitted by the consumer, never turned into a fake value.
    pub fn location(&self, index: usize, ty: Type) -> Result<ValueLocation, TransferError> {
        let fail = TransferError::InvalidContract;
        let value = self
            .map
            .values
            .get(index)
            .ok_or_else(|| fail("missing backend boundary operand"))?;
        if value.ty != ty {
            return Err(fail("backend boundary operand type mismatch"));
        }
        let bytes =
            u8::try_from(ty.bytes()).map_err(|_| fail("unsupported boundary operand width"))?;
        let vector = ty.is_vector() || ty.is_float();
        if !((ty.is_int() && matches!(bytes, 1 | 2 | 4 | 8))
            || (ty.is_float() && matches!(bytes, 4 | 8))
            || (ty.is_vector() && bytes == 16))
        {
            return Err(fail("unsupported boundary operand type"));
        }
        let location = match value.location {
            Location::Unused => return Err(fail("required entry operand was eliminated")),
            Location::Register {
                index,
                vector: bank,
            } => {
                if bank != vector {
                    return Err(fail("backend boundary register bank mismatch"));
                }
                ValueLocation::Register {
                    class: if vector {
                        RegisterClass::Vector
                    } else {
                        RegisterClass::Integer
                    },
                    index,
                }
            }
            Location::Spill { offset } => {
                if offset
                    .checked_add(u32::from(bytes))
                    .is_none_or(|end| end > self.frame_extent)
                {
                    return Err(fail(
                        "backend boundary spill exceeds the reported frame extent",
                    ));
                }
                ValueLocation::Spill { offset, bytes }
            }
        };
        if !location.valid(self.abi, bytes) {
            return Err(fail("invalid or reserved backend boundary location"));
        }
        Ok(location)
    }

    /// Bind architectural fields to the lowering's ordered operands. Contract
    /// validation subsequently checks exact liveness coverage and entry overlap.
    pub fn bindings(
        &self,
        operands: &[(GuestValue, usize)],
    ) -> Result<Box<[ValueBinding]>, TransferError> {
        operands
            .iter()
            .map(|&(value, index)| {
                let ty = self
                    .map
                    .values
                    .get(index)
                    .ok_or(TransferError::InvalidContract(
                        "missing guest boundary operand",
                    ))?
                    .ty;
                if value.state().is_none()
                    || ty.bytes() != u32::from(value.bytes())
                    || (matches!(value, GuestValue::Vector(_)) && !ty.is_vector())
                    || (!matches!(value, GuestValue::Vector(_)) && !ty.is_int())
                {
                    return Err(TransferError::InvalidContract(
                        "guest binding has the wrong backend type",
                    ));
                }
                Ok(ValueBinding {
                    value,
                    location: self.location(index, ty)?,
                })
            })
            .collect()
    }
}
