//! Shared FP lowering/status policy for both compiler tiers.
use nixe_cpu::decode::a64::fp_simd::{FloatRoundOperation, FloatToIntegerRounding, Instruction};

// Native S/D arithmetic supports these controls. Other modes (including guest
// exception enables) require the exact typed semantics, not a masked fallback.
pub(crate) const NATIVE_FPCR_MASK: u32 = (3 << 22) | (1 << 24) | (1 << 25);

pub(crate) const fn native_fpcr_supported(fpcr: u32) -> bool {
    fpcr & !NATIVE_FPCR_MASK == 0
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FpLoweringDisposition {
    Direct,
    GuardedNative,
    GuardedExact,
    Exact,
}

impl FpLoweringDisposition {
    pub(crate) const fn accesses_status(self) -> bool {
        !matches!(self, Self::Direct)
    }

    pub(crate) const fn uses_native_status(self) -> bool {
        matches!(self, Self::GuardedNative)
    }

    pub(crate) const fn is_exact(self) -> bool {
        matches!(self, Self::Exact)
    }
}

/// Single compile-time authority for FP lowering and region status analysis.
/// `GuardedNative` admits only inputs whose result and host exception flags
/// match Arm, then uses the lazy host accumulator. `GuardedExact` has a direct
/// result-only domain but sends exceptional values to the typed edge. `Exact`
/// is used when stock CLIF cannot express the complete operation: fixed and
/// directional conversions, conditional comparison, all x86 FRINT forms
/// (whose lowering can leak MXCSR.PE), and Arm FRINTA/X/I forms not represented
/// by the selected CLIF operation. No runtime policy object repeats this table.
/// Every normalized variant is named so extending the decoder requires a
/// deliberate lowering decision instead of inheriting a permissive default.
pub(crate) fn fp_lowering_disposition(instruction: Instruction) -> FpLoweringDisposition {
    let fields = instruction.operands();
    match instruction {
        Instruction::SignedIntToFloat(_) | Instruction::UnsignedIntToFloat(_) => {
            FpLoweringDisposition::GuardedNative
        }
        Instruction::FloatToSignedInt(_) | Instruction::FloatToUnsignedInt(_)
            if fields.fixed_point_fraction_bits.is_none()
                && matches!(
                    fields.float_to_integer_rounding,
                    Some(FloatToIntegerRounding::TowardZero)
                ) =>
        {
            FpLoweringDisposition::GuardedNative
        }
        Instruction::VectorSignedIntToFloat(_)
        | Instruction::VectorUnsignedIntToFloat(_)
        | Instruction::ScalarVectorSignedIntToFloat(_)
        | Instruction::ScalarVectorUnsignedIntToFloat(_)
        | Instruction::VectorFloatDivide(_)
        | Instruction::VectorFloatMultiplyElement(_)
        | Instruction::ScalarFloatConvert(_)
        | Instruction::ScalarFloatDivide(_)
        | Instruction::ScalarFloatAdd(_)
        | Instruction::ScalarFloatMultiply(_)
        | Instruction::ScalarFloatFusedMultiplyAdd(_)
        | Instruction::ScalarFloatSquareRoot(_) => FpLoweringDisposition::GuardedNative,
        Instruction::CompareRegister(_) | Instruction::CompareZero(_) => {
            FpLoweringDisposition::GuardedExact
        }
        Instruction::ScalarFloatRound(_)
            if !cfg!(target_arch = "x86_64")
                && !matches!(
                    fields.float_round_operation,
                    Some(
                        FloatRoundOperation::NearestAway
                            | FloatRoundOperation::Exact
                            | FloatRoundOperation::CurrentMode
                    )
                ) =>
        {
            FpLoweringDisposition::GuardedExact
        }
        Instruction::FloatToSignedInt(_)
        | Instruction::FloatToUnsignedInt(_)
        | Instruction::ConditionalCompare(_)
        | Instruction::ScalarFloatRound(_) => FpLoweringDisposition::Exact,
        Instruction::DuplicateGeneral(_)
        | Instruction::DuplicateElement(_)
        | Instruction::MemoryPair(_)
        | Instruction::Bitwise(_)
        | Instruction::Integer(_)
        | Instruction::ScalarMove(_)
        | Instruction::ScalarAbsolute(_)
        | Instruction::ScalarNegate(_)
        | Instruction::VectorFloatAbsolute(_)
        | Instruction::VectorFloatNegate(_)
        | Instruction::ModifiedImmediate(_)
        | Instruction::UnsignedMoveToGeneral(_)
        | Instruction::InsertElement(_)
        | Instruction::InsertGeneral(_)
        | Instruction::MoveToGeneral(_)
        | Instruction::MoveFromGeneral(_)
        | Instruction::MemoryUnsigned(_)
        | Instruction::MemoryUnscaled(_)
        | Instruction::MemoryPostIndex(_)
        | Instruction::MemoryPreIndex(_)
        | Instruction::MemoryRegister(_)
        | Instruction::MemoryMultipleStructures(_)
        | Instruction::MemoryMultipleStructuresPostIndex(_)
        | Instruction::MemorySingleStructure(_)
        | Instruction::MemorySingleStructurePostIndex(_)
        | Instruction::PermuteTwoSource(_)
        | Instruction::Extract(_)
        | Instruction::IntegerCompare(_)
        | Instruction::IntegerPairwise(_)
        | Instruction::IntegerMinMax(_)
        | Instruction::ShiftRightNarrow(_)
        | Instruction::ScalarShiftRightImmediate(_)
        | Instruction::VectorShiftRightImmediate(_)
        | Instruction::ScalarShiftLeftImmediate(_)
        | Instruction::VectorShiftLeftImmediate(_)
        | Instruction::ShiftLeftLong(_)
        | Instruction::VectorSignedShiftRegister(_)
        | Instruction::VectorUnsignedShiftRegister(_)
        | Instruction::CountBits(_)
        | Instruction::AddAcrossVector(_)
        | Instruction::ExtractNarrow(_)
        | Instruction::VectorFloatImmediate(_)
        | Instruction::ScalarFloatImmediate(_)
        | Instruction::ScalarFloatConditionalSelect(_) => FpLoweringDisposition::Direct,
    }
}
