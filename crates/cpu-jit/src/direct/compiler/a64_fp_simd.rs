use std::mem::offset_of;

use cranelift_codegen::ir::{
    AbiParam, ConstantData, Endianness, InstBuilder, MemFlagsData, Signature, Value,
    condcodes::{FloatCC, IntCC},
    immediates::{Ieee32, Ieee64, Ieee128},
    types,
};
use nixe_cpu::decode::a64::fp_simd::{
    BitwiseOperation, FloatAddOperation, FloatConversion, FloatFusedMultiplyOperation,
    FloatMultiplyOperation, FloatRoundOperation, FloatToIntegerRounding, Instruction,
    IntegerComparison, Operands, PairwiseOperation, PermuteOperation,
};
use nixe_cpu::memory::{MemoryAccessSize, MemoryOrdering};
use nixe_cpu::semantics::a64::{
    SimdMemoryMode, simd_memory_access_size, simd_multiple_structure_shape, simd_pair_access_size,
    simd_single_structure_shape,
};
use nixe_cpu::semantics::conditions::Condition;
use nixe_memory::GuestVirtualAddress;

use super::{CraneliftTranslator, LazyFlags, a64_memory::MemoryOperation};
use crate::direct::slow;
use crate::direct::{DirectJitError, NativeContext};

use crate::fp_policy::{FpLoweringDisposition, fp_lowering_disposition};

impl CraneliftTranslator<'_, '_> {
    // A64 Advanced SIMD and floating-point behavior follows Arm DDI 0602
    // (2025-12). Pure lane and bit operations are emitted directly; only FP
    // behavior whose FPCR/FPSR contract is not represented by CLIF crosses an
    // exact typed slow boundary.
    // https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions
    pub(super) fn emit_fp_simd(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        flags: &mut LazyFlags,
    ) -> Result<(), DirectJitError> {
        let fields = instruction.operands();
        if fp_lowering_disposition(instruction).is_exact() {
            return self.emit_typed_fp_cold(source, instruction, fields, flags);
        }
        match instruction {
            Instruction::DuplicateGeneral(_) => {
                let lane_bits = 8_u32 << fields.immediate_5.trailing_zeros();
                let lane = integer_lane_type(lane_bits)?;
                let value = self.read_register(fields.rn, false)?;
                let value = cast_integer(&mut self.builder, value, lane, false);
                let vector_ty = vector_type(lane, lane_bits)?;
                let value = self.builder.ins().splat(vector_ty, value);
                let value = self.finish_vector(value, fields.vector_128);
                self.write_vector(fields.rd, value)
            }
            Instruction::DuplicateElement(_) => {
                let shift = fields.immediate_5.trailing_zeros();
                let lane_bits = 8_u32 << shift;
                let lane_index = fields.immediate_5 >> (shift + 1);
                let vector_ty = vector_type(integer_lane_type(lane_bits)?, lane_bits)?;
                let source = self.read_vector_as(fields.rn, vector_ty)?;
                let lane = self.builder.ins().extractlane(source, lane_index);
                let value = self.builder.ins().splat(vector_ty, lane);
                let value = self.finish_vector(value, fields.vector_128);
                self.write_vector(fields.rd, value)
            }
            Instruction::ModifiedImmediate(_) => {
                let immediate = expand_modified_immediate(
                    fields.cmode,
                    fields.immediate_8,
                    fields.operation_bit,
                )?;
                let bits = u128::from(immediate) | (u128::from(immediate) << 64);
                let immediate = self.vector_constant(bits);
                let value = if fields.cmode <= 11 && fields.cmode & 1 != 0 {
                    let previous = self.read_vector(fields.rd)?;
                    if fields.operation_bit {
                        self.builder.ins().band(previous, immediate)
                    } else {
                        self.builder.ins().bor(previous, immediate)
                    }
                } else {
                    immediate
                };
                let value = self.mask_vector(value, if fields.vector_128 { 128 } else { 64 });
                self.write_vector(fields.rd, value)
            }
            Instruction::UnsignedMoveToGeneral(_) => {
                let shift = fields.immediate_5.trailing_zeros();
                let lane_bits = 8_u32 << shift;
                let lane_index = fields.immediate_5 >> (shift + 1);
                let vector_ty = vector_type(integer_lane_type(lane_bits)?, lane_bits)?;
                let source = self.read_vector_as(fields.rn, vector_ty)?;
                let value = self.builder.ins().extractlane(source, lane_index);
                let value = if fields.vector_128 {
                    cast_integer(&mut self.builder, value, types::I64, false)
                } else {
                    let value = cast_integer(&mut self.builder, value, types::I32, false);
                    self.builder.ins().uextend(types::I64, value)
                };
                self.write_register(fields.rd, value)
            }
            Instruction::InsertElement(_) | Instruction::InsertGeneral(_) => {
                let shift = fields.immediate_5.trailing_zeros();
                let lane_bits = 8_u32 << shift;
                let destination_lane = fields.immediate_5 >> (shift + 1);
                let lane = integer_lane_type(lane_bits)?;
                let vector_ty = vector_type(lane, lane_bits)?;
                let previous = self.read_vector_as(fields.rd, vector_ty)?;
                let value = if matches!(instruction, Instruction::InsertElement(_)) {
                    let source_lane = fields.immediate_4 >> shift;
                    let source = self.read_vector_as(fields.rn, vector_ty)?;
                    self.builder.ins().extractlane(source, source_lane)
                } else {
                    let source = self.read_register(fields.rn, false)?;
                    cast_integer(&mut self.builder, source, lane, false)
                };
                let value = self
                    .builder
                    .ins()
                    .insertlane(previous, value, destination_lane);
                let value = self.vector_as(value, types::I8X16);
                self.write_vector(fields.rd, value)
            }
            Instruction::MoveToGeneral(_) => self.emit_move_to_general(fields),
            Instruction::MoveFromGeneral(_) => self.emit_move_from_general(fields),
            Instruction::ScalarMove(_) => {
                let width = scalar_width(fields.opc)?;
                let value = self.read_vector(fields.rn)?;
                let value = self.mask_vector(value, width);
                self.write_vector(fields.rd, value)
            }
            Instruction::ScalarAbsolute(_) | Instruction::ScalarNegate(_) => {
                let width = scalar_width(fields.opc)?;
                let source = self.read_vector(fields.rn)?;
                let source = self.mask_vector(source, width);
                let sign = self.vector_constant(1_u128 << (width - 1));
                let value = if matches!(instruction, Instruction::ScalarNegate(_)) {
                    self.builder.ins().bxor(source, sign)
                } else {
                    let sign = self.builder.ins().bnot(sign);
                    self.builder.ins().band(source, sign)
                };
                self.write_vector(fields.rd, value)
            }
            Instruction::VectorFloatAbsolute(_) | Instruction::VectorFloatNegate(_) => {
                let lane_bits = if fields.opc & 1 == 0 { 32 } else { 64 };
                let vector_bits = if fields.vector_128 { 128 } else { 64 };
                let mut sign_bits = 0_u128;
                for offset in (0..vector_bits).step_by(lane_bits as usize) {
                    sign_bits |= 1_u128 << (offset + lane_bits - 1);
                }
                let sign = self.vector_constant(sign_bits);
                let source = self.read_vector(fields.rn)?;
                let source = self.mask_vector(source, vector_bits);
                let value = if matches!(instruction, Instruction::VectorFloatNegate(_)) {
                    self.builder.ins().bxor(source, sign)
                } else {
                    let sign = self.builder.ins().bnot(sign);
                    self.builder.ins().band(source, sign)
                };
                self.write_vector(fields.rd, value)
            }
            Instruction::Integer(_) => self.emit_integer_vector(fields),
            Instruction::Bitwise(_) => self.emit_bitwise(fields),
            Instruction::IntegerCompare(_) => self.emit_integer_compare(fields),
            Instruction::IntegerPairwise(_) => self.emit_integer_pairwise(fields),
            Instruction::IntegerMinMax(_) => self.emit_integer_min_max(fields),
            Instruction::PermuteTwoSource(_) => self.emit_permute(fields),
            Instruction::Extract(_) => self.emit_vector_extract(fields),
            Instruction::ShiftRightNarrow(_) | Instruction::ExtractNarrow(_) => {
                self.emit_narrow(instruction, fields)
            }
            Instruction::ScalarShiftRightImmediate(_)
            | Instruction::VectorShiftRightImmediate(_)
            | Instruction::ScalarShiftLeftImmediate(_)
            | Instruction::VectorShiftLeftImmediate(_) => {
                self.emit_immediate_shift(instruction, fields)
            }
            Instruction::ShiftLeftLong(_) => self.emit_shift_left_long(fields),
            Instruction::VectorSignedShiftRegister(_)
            | Instruction::VectorUnsignedShiftRegister(_) => {
                self.emit_register_shift(instruction, fields)
            }
            Instruction::CountBits(_) => {
                let source = self.read_vector(fields.rn)?;
                let value = self.builder.ins().popcnt(source);
                let value = self.mask_vector(value, if fields.vector_128 { 128 } else { 64 });
                self.write_vector(fields.rd, value)
            }
            Instruction::AddAcrossVector(_) => self.emit_add_across(fields),
            Instruction::ScalarFloatImmediate(_) | Instruction::VectorFloatImmediate(_) => {
                self.emit_float_immediate(instruction, fields)
            }
            Instruction::ScalarFloatConditionalSelect(_) => {
                let predicate =
                    self.emit_condition(Condition::from_encoding(fields.condition), flags);
                let first = self.read_vector(fields.rn)?;
                let second = self.read_vector(fields.rm)?;
                let selected = self.builder.ins().select(predicate, first, second);
                let value = self.mask_vector(selected, if fields.opc == 0 { 32 } else { 64 });
                self.write_vector(fields.rd, value)
            }
            Instruction::ScalarFloatAdd(_)
            | Instruction::ScalarFloatMultiply(_)
            | Instruction::ScalarFloatFusedMultiplyAdd(_)
            | Instruction::ScalarFloatDivide(_)
            | Instruction::ScalarFloatSquareRoot(_) => {
                self.emit_common_scalar_arithmetic(source, instruction, fields, flags)
            }
            Instruction::VectorFloatDivide(_) | Instruction::VectorFloatMultiplyElement(_) => {
                self.emit_common_vector_arithmetic(source, instruction, fields, flags)
            }
            Instruction::CompareRegister(_) | Instruction::CompareZero(_) => {
                self.emit_common_scalar_compare(source, instruction, fields, flags)
            }
            Instruction::ScalarFloatRound(_) => {
                self.emit_common_scalar_round(source, instruction, fields, flags)
            }
            Instruction::ScalarFloatConvert(_) => {
                self.emit_common_scalar_convert(source, instruction, fields, flags)
            }
            Instruction::SignedIntToFloat(_) | Instruction::UnsignedIntToFloat(_) => {
                self.emit_common_integer_to_float(source, instruction, fields, flags)
            }
            Instruction::VectorSignedIntToFloat(_)
            | Instruction::VectorUnsignedIntToFloat(_)
            | Instruction::ScalarVectorSignedIntToFloat(_)
            | Instruction::ScalarVectorUnsignedIntToFloat(_) => {
                self.emit_common_vector_integer_to_float(source, instruction, fields, flags)
            }
            Instruction::FloatToSignedInt(_) | Instruction::FloatToUnsignedInt(_) => {
                self.emit_common_float_to_integer(source, instruction, fields, flags)
            }
            Instruction::MemoryUnsigned(_)
            | Instruction::MemoryUnscaled(_)
            | Instruction::MemoryPostIndex(_)
            | Instruction::MemoryPreIndex(_)
            | Instruction::MemoryRegister(_)
            | Instruction::MemoryPair(_)
            | Instruction::MemoryMultipleStructures(_)
            | Instruction::MemoryMultipleStructuresPostIndex(_)
            | Instruction::MemorySingleStructure(_)
            | Instruction::MemorySingleStructurePostIndex(_) => {
                self.emit_vector_memory(source, instruction, fields, flags)
            }
            _ => Err(DirectJitError::unsupported(
                "A64 FP/SIMD instruction has no lowering disposition",
            )),
        }
    }

    fn read_vector(&mut self, index: u8) -> Result<Value, DirectJitError> {
        let variable = self.vector_registers[usize::from(index)].ok_or_else(|| {
            DirectJitError::internal(format!("direct JIT vector V{index} was not planned"))
        })?;
        Ok(self.builder.use_var(variable))
    }

    fn emit_common_scalar_arithmetic(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &mut LazyFlags,
    ) -> Result<(), DirectJitError> {
        let width = scalar_width(fields.opc)?;
        let first_bits = self.scalar_fp_bits(fields.rn, width)?;
        let unary = matches!(instruction, Instruction::ScalarFloatSquareRoot(_));
        let second_bits = if unary {
            self.builder
                .ins()
                .iconst(if width == 32 { types::I32 } else { types::I64 }, 0)
        } else {
            self.scalar_fp_bits(fields.rm, width)?
        };
        let mut direct = self.fp_finite_or_zero(first_bits, width);
        if !unary {
            let second_direct = self.fp_finite_or_zero(second_bits, width);
            direct = self.builder.ins().band(direct, second_direct);
        }
        if matches!(instruction, Instruction::ScalarFloatDivide(_)) {
            let magnitude_mask = if width == 32 {
                0x7fff_ffff_u64
            } else {
                0x7fff_ffff_ffff_ffff
            };
            let magnitude = self
                .builder
                .ins()
                .band_imm_u(second_bits, magnitude_mask as i64);
            let nonzero = self.builder.ins().icmp_imm_s(IntCC::NotEqual, magnitude, 0);
            direct = self.builder.ins().band(direct, nonzero);
        }
        if unary {
            let sign_mask = if width == 32 {
                1_u64 << 31
            } else {
                1_u64 << 63
            };
            let sign = self.builder.ins().band_imm_u(first_bits, sign_mask as i64);
            let positive = self.builder.ins().icmp_imm_s(IntCC::Equal, sign, 0);
            let magnitude = self
                .builder
                .ins()
                .band_imm_u(first_bits, (!sign_mask) as i64);
            let zero = self.builder.ins().icmp_imm_s(IntCC::Equal, magnitude, 0);
            let valid = self.builder.ins().bor(positive, zero);
            direct = self.builder.ins().band(direct, valid);
        }
        let third_bits = if matches!(instruction, Instruction::ScalarFloatFusedMultiplyAdd(_)) {
            let bits = self.scalar_fp_bits(fields.ra, width)?;
            let third_direct = self.fp_finite_or_zero(bits, width);
            direct = self.builder.ins().band(direct, third_direct);
            Some(bits)
        } else {
            None
        };
        direct = self.builder.ins().band(direct, self.native_fp_enabled);

        let native = self.builder.create_block();
        let exact = self.cold_block();
        let done = self.builder.create_block();
        let destination_was_dirty = self.block_dirty_vector_registers[usize::from(fields.rd)];
        self.builder.ins().brif(direct, native, &[], exact, &[]);

        self.builder.switch_to_block(native);
        let ty = if width == 32 { types::F32 } else { types::F64 };
        let first = self.builder.ins().bitcast(ty, bitcast_flags(), first_bits);
        let second = self.builder.ins().bitcast(ty, bitcast_flags(), second_bits);
        let result = match instruction {
            Instruction::ScalarFloatAdd(_) => match fields
                .float_add_operation
                .expect("normalized FP add operation")
            {
                FloatAddOperation::Add => self.builder.ins().fadd(first, second),
                FloatAddOperation::Subtract => self.builder.ins().fsub(first, second),
            },
            Instruction::ScalarFloatMultiply(_) => {
                let result = self.builder.ins().fmul(first, second);
                if matches!(
                    fields
                        .float_multiply_operation
                        .expect("normalized FP multiply operation"),
                    FloatMultiplyOperation::NegatedMultiply
                ) {
                    self.builder.ins().fneg(result)
                } else {
                    result
                }
            }
            Instruction::ScalarFloatFusedMultiplyAdd(_) => {
                let third_bits = third_bits.expect("fused operation has an addend");
                let third = self.builder.ins().bitcast(ty, bitcast_flags(), third_bits);
                match fields
                    .float_fused_multiply_operation
                    .expect("normalized fused operation")
                {
                    FloatFusedMultiplyOperation::MultiplyAdd => {
                        self.builder.ins().fma(first, second, third)
                    }
                    FloatFusedMultiplyOperation::MultiplySubtract => {
                        let first = self.builder.ins().fneg(first);
                        self.builder.ins().fma(first, second, third)
                    }
                    FloatFusedMultiplyOperation::NegatedMultiplyAdd => {
                        let first = self.builder.ins().fneg(first);
                        let third = self.builder.ins().fneg(third);
                        self.builder.ins().fma(first, second, third)
                    }
                    FloatFusedMultiplyOperation::NegatedMultiplySubtract => {
                        let third = self.builder.ins().fneg(third);
                        self.builder.ins().fma(first, second, third)
                    }
                }
            }
            Instruction::ScalarFloatDivide(_) => self.builder.ins().fdiv(first, second),
            Instruction::ScalarFloatSquareRoot(_) => self.builder.ins().sqrt(first),
            _ => unreachable!(),
        };
        let integer_ty = if width == 32 { types::I32 } else { types::I64 };
        let result = self
            .builder
            .ins()
            .bitcast(integer_ty, bitcast_flags(), result);
        let result = self.builder.ins().uextend(types::I128, result);
        let result = self.vector_as(result, types::I8X16);
        self.write_vector(fields.rd, result)?;
        self.builder.ins().jump(done, &[]);

        self.builder.switch_to_block(exact);
        self.block_dirty_vector_registers[usize::from(fields.rd)] = destination_was_dirty;
        self.emit_typed_fp_cold(source, instruction, fields, flags)?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        self.block_dirty_vector_registers[usize::from(fields.rd)] = true;
        Ok(())
    }

    fn emit_common_vector_arithmetic(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &mut LazyFlags,
    ) -> Result<(), DirectJitError> {
        let lane_bits = if fields.opc == 0 { 32 } else { 64 };
        let lanes = if lane_bits == 32 {
            if fields.vector_128 { 4 } else { 2 }
        } else {
            2
        };
        let integer_ty = if lane_bits == 32 {
            types::I32X4
        } else {
            types::I64X2
        };
        let float_ty = if lane_bits == 32 {
            types::F32X4
        } else {
            types::F64X2
        };
        let vector_bits = if fields.vector_128 { 128 } else { 64 };
        let first_raw = self.read_vector(fields.rn)?;
        let second_raw = self.read_vector(fields.rm)?;
        let first_raw = self.mask_vector(first_raw, vector_bits);
        let second_active = if matches!(instruction, Instruction::VectorFloatDivide(_)) {
            let active = self.mask_vector(second_raw, vector_bits);
            if lane_bits == 32 && !fields.vector_128 {
                // CLIF has no 64-bit F32 vector shape. Keep the single packed
                // host divide, but make its two inactive lanes exact 0 / 1
                // operations so they cannot contribute host FP exceptions.
                let inactive_ones = self.vector_constant(
                    (u128::from(1.0_f32.to_bits()) << 64) | (u128::from(1.0_f32.to_bits()) << 96),
                );
                self.builder.ins().bor(active, inactive_ones)
            } else {
                active
            }
        } else {
            second_raw
        };
        let first_integer = self.vector_as(first_raw, integer_ty);
        let second_integer = self.vector_as(second_active, integer_ty);
        let mut direct = self.native_fp_enabled;
        let magnitude_mask = if lane_bits == 32 {
            0x7fff_ffff_u64
        } else {
            0x7fff_ffff_ffff_ffff
        };
        let element = if matches!(instruction, Instruction::VectorFloatMultiplyElement(_)) {
            Some(
                self.builder
                    .ins()
                    .extractlane(second_integer, fields.fp_element_lane),
            )
        } else {
            None
        };
        if let Some(element) = element {
            let valid = self.fp_finite_or_zero(element, lane_bits);
            direct = self.builder.ins().band(direct, valid);
        }
        let valid_first = self.fp_vector_finite_or_zero(first_integer, lane_bits);
        direct = self.builder.ins().band(direct, valid_first);
        if matches!(instruction, Instruction::VectorFloatDivide(_)) {
            let valid_second = self.fp_vector_finite_or_zero(second_integer, lane_bits);
            let nonzero = self.fp_vector_active_lanes_nonzero(
                second_integer,
                lane_bits,
                lanes,
                magnitude_mask,
            );
            let valid_second = self.builder.ins().band(valid_second, nonzero);
            direct = self.builder.ins().band(direct, valid_second);
        }

        let native = self.builder.create_block();
        let exact = self.cold_block();
        let done = self.builder.create_block();
        let destination_was_dirty = self.block_dirty_vector_registers[usize::from(fields.rd)];
        self.builder.ins().brif(direct, native, &[], exact, &[]);
        self.builder.switch_to_block(native);
        let first = self.vector_as(first_raw, float_ty);
        let second = if let Some(element) = element {
            let scalar_ty = if lane_bits == 32 {
                types::F32
            } else {
                types::F64
            };
            let element = self
                .builder
                .ins()
                .bitcast(scalar_ty, bitcast_flags(), element);
            self.builder.ins().splat(float_ty, element)
        } else {
            self.vector_as(second_active, float_ty)
        };
        let result = if matches!(instruction, Instruction::VectorFloatDivide(_)) {
            self.builder.ins().fdiv(first, second)
        } else {
            self.builder.ins().fmul(first, second)
        };
        let result = self.vector_as(result, types::I8X16);
        let result = self.mask_vector(result, vector_bits);
        self.write_vector(fields.rd, result)?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(exact);
        self.block_dirty_vector_registers[usize::from(fields.rd)] = destination_was_dirty;
        self.emit_typed_fp_cold(source, instruction, fields, flags)?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        self.block_dirty_vector_registers[usize::from(fields.rd)] = true;
        Ok(())
    }

    fn emit_common_scalar_compare(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &mut LazyFlags,
    ) -> Result<(), DirectJitError> {
        let width = scalar_width(fields.opc)?;
        let first_bits = self.scalar_fp_bits(fields.rn, width)?;
        let second_bits = if matches!(instruction, Instruction::CompareZero(_)) {
            self.builder
                .ins()
                .iconst(if width == 32 { types::I32 } else { types::I64 }, 0)
        } else {
            self.scalar_fp_bits(fields.rm, width)?
        };
        let first_direct = self.fp_finite_or_zero(first_bits, width);
        let second_direct = self.fp_finite_or_zero(second_bits, width);
        let direct = self.builder.ins().band(first_direct, second_direct);
        let native = self.builder.create_block();
        let exact = self.cold_block();
        let done = self.builder.create_block();
        let result_flags = self.builder.declare_var(types::I32);
        self.builder.ins().brif(direct, native, &[], exact, &[]);

        self.builder.switch_to_block(native);
        let ty = if width == 32 { types::F32 } else { types::F64 };
        let first = self.builder.ins().bitcast(ty, bitcast_flags(), first_bits);
        let second = self.builder.ins().bitcast(ty, bitcast_flags(), second_bits);
        let equal = self.builder.ins().fcmp(FloatCC::Equal, first, second);
        let less = self.builder.ins().fcmp(FloatCC::LessThan, first, second);
        let equal_flags = self.builder.ins().iconst(types::I32, 0x6000_0000);
        let less_flags = self
            .builder
            .ins()
            .iconst(types::I32, 0x8000_0000_u32 as i64);
        let greater_flags = self.builder.ins().iconst(types::I32, 0x2000_0000);
        let ordered = self.builder.ins().select(less, less_flags, greater_flags);
        let packed = self.builder.ins().select(equal, equal_flags, ordered);
        self.builder.def_var(result_flags, packed);
        self.builder.ins().jump(done, &[]);

        self.builder.switch_to_block(exact);
        self.emit_typed_fp_cold(source, instruction, fields, flags)?;
        let packed = self.packed_flags(flags);
        self.builder.def_var(result_flags, packed);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        *flags = LazyFlags::Packed(self.builder.use_var(result_flags));
        Ok(())
    }

    fn emit_common_scalar_round(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &mut LazyFlags,
    ) -> Result<(), DirectJitError> {
        let operation = fields
            .float_round_operation
            .expect("normalized FP round operation");
        debug_assert_eq!(
            fp_lowering_disposition(instruction),
            FpLoweringDisposition::GuardedExact
        );
        let width = scalar_width(fields.opc)?;
        let bits = self.scalar_fp_bits(fields.rn, width)?;
        let direct = self.fp_finite_or_zero(bits, width);
        let native = self.builder.create_block();
        let exact = self.cold_block();
        let done = self.builder.create_block();
        let destination_was_dirty = self.block_dirty_vector_registers[usize::from(fields.rd)];
        self.builder.ins().brif(direct, native, &[], exact, &[]);
        self.builder.switch_to_block(native);
        let ty = if width == 32 { types::F32 } else { types::F64 };
        let value = self.builder.ins().bitcast(ty, bitcast_flags(), bits);
        let result = match operation {
            FloatRoundOperation::NearestEven => self.builder.ins().nearest(value),
            FloatRoundOperation::TowardPositive => self.builder.ins().ceil(value),
            FloatRoundOperation::TowardNegative => self.builder.ins().floor(value),
            FloatRoundOperation::TowardZero => self.builder.ins().trunc(value),
            FloatRoundOperation::NearestAway
            | FloatRoundOperation::Exact
            | FloatRoundOperation::CurrentMode => unreachable!(),
        };
        let integer_ty = if width == 32 { types::I32 } else { types::I64 };
        let result = self
            .builder
            .ins()
            .bitcast(integer_ty, bitcast_flags(), result);
        let result = self.builder.ins().uextend(types::I128, result);
        let result = self.vector_as(result, types::I8X16);
        self.write_vector(fields.rd, result)?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(exact);
        self.block_dirty_vector_registers[usize::from(fields.rd)] = destination_was_dirty;
        self.emit_typed_fp_cold(source, instruction, fields, flags)?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        self.block_dirty_vector_registers[usize::from(fields.rd)] = true;
        Ok(())
    }

    fn emit_common_scalar_convert(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &mut LazyFlags,
    ) -> Result<(), DirectJitError> {
        let conversion = fields
            .float_conversion
            .expect("normalized scalar FP conversion");
        let source_width = if matches!(conversion, FloatConversion::SingleToDouble) {
            32
        } else {
            64
        };
        let bits = self.scalar_fp_bits(fields.rn, source_width)?;
        let mut direct = self.fp_finite_or_zero(bits, source_width);
        direct = self.builder.ins().band(direct, self.native_fp_enabled);
        let native = self.builder.create_block();
        let exact = self.cold_block();
        let done = self.builder.create_block();
        let destination_was_dirty = self.block_dirty_vector_registers[usize::from(fields.rd)];
        self.builder.ins().brif(direct, native, &[], exact, &[]);
        self.builder.switch_to_block(native);
        let input_ty = if source_width == 32 {
            types::F32
        } else {
            types::F64
        };
        let input = self.builder.ins().bitcast(input_ty, bitcast_flags(), bits);
        let (result, result_ty) = match conversion {
            FloatConversion::SingleToDouble => {
                (self.builder.ins().fpromote(types::F64, input), types::I64)
            }
            FloatConversion::DoubleToSingle => {
                (self.builder.ins().fdemote(types::F32, input), types::I32)
            }
        };
        let result = self
            .builder
            .ins()
            .bitcast(result_ty, bitcast_flags(), result);
        let result = self.builder.ins().uextend(types::I128, result);
        let result = self.vector_as(result, types::I8X16);
        self.write_vector(fields.rd, result)?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(exact);
        self.block_dirty_vector_registers[usize::from(fields.rd)] = destination_was_dirty;
        self.emit_typed_fp_cold(source, instruction, fields, flags)?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        self.block_dirty_vector_registers[usize::from(fields.rd)] = true;
        Ok(())
    }

    fn emit_common_integer_to_float(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &mut LazyFlags,
    ) -> Result<(), DirectJitError> {
        debug_assert_eq!(
            fp_lowering_disposition(instruction),
            FpLoweringDisposition::GuardedNative
        );
        let source_ty = if fields.size & 2 != 0 {
            types::I64
        } else {
            types::I32
        };
        let destination_ty = if fields.opc == 0 {
            types::F32
        } else {
            types::F64
        };
        let value = self.read_register(fields.rn, false)?;
        let value = if source_ty == types::I32 {
            self.builder.ins().ireduce(types::I32, value)
        } else {
            value
        };
        let native = self.builder.create_block();
        let exact = self.cold_block();
        let done = self.builder.create_block();
        let destination_was_dirty = self.block_dirty_vector_registers[usize::from(fields.rd)];
        self.builder
            .ins()
            .brif(self.native_fp_enabled, native, &[], exact, &[]);
        self.builder.switch_to_block(native);
        let result = if matches!(instruction, Instruction::SignedIntToFloat(_)) {
            self.builder.ins().fcvt_from_sint(destination_ty, value)
        } else {
            self.builder.ins().fcvt_from_uint(destination_ty, value)
        };
        let integer_ty = if destination_ty == types::F32 {
            types::I32
        } else {
            types::I64
        };
        let result = self
            .builder
            .ins()
            .bitcast(integer_ty, bitcast_flags(), result);
        let result = self.builder.ins().uextend(types::I128, result);
        let result = self.vector_as(result, types::I8X16);
        self.write_vector(fields.rd, result)?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(exact);
        self.block_dirty_vector_registers[usize::from(fields.rd)] = destination_was_dirty;
        self.emit_typed_fp_cold(source, instruction, fields, flags)?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        self.block_dirty_vector_registers[usize::from(fields.rd)] = true;
        Ok(())
    }

    fn emit_common_float_to_integer(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &mut LazyFlags,
    ) -> Result<(), DirectJitError> {
        debug_assert_eq!(
            fp_lowering_disposition(instruction),
            FpLoweringDisposition::GuardedNative
        );
        let float_ty = if fields.opc == 0 {
            types::F32
        } else {
            types::F64
        };
        let integer_ty = if fields.size & 2 != 0 {
            types::I64
        } else {
            types::I32
        };
        let width = if float_ty == types::F32 { 32 } else { 64 };
        let bits = self.scalar_fp_bits(fields.rn, width)?;
        let value = self.builder.ins().bitcast(float_ty, bitcast_flags(), bits);
        let mut direct = self.fp_finite_or_zero(bits, width);
        direct = self.builder.ins().band(direct, self.native_fp_enabled);
        let unsigned = matches!(instruction, Instruction::FloatToUnsignedInt(_));
        let integer_bits = if integer_ty == types::I32 { 32 } else { 64 };
        let lower = if unsigned {
            0.0
        } else if integer_bits == 32 {
            -2147483648.0
        } else {
            -9223372036854775808.0
        };
        let upper = if unsigned {
            if integer_bits == 32 {
                4294967296.0
            } else {
                18446744073709551616.0
            }
        } else if integer_bits == 32 {
            2147483648.0
        } else {
            9223372036854775808.0
        };
        let lower = self.float_constant(float_ty, lower);
        let upper = self.float_constant(float_ty, upper);
        let above_lower = self
            .builder
            .ins()
            .fcmp(FloatCC::GreaterThanOrEqual, value, lower);
        let below_upper = self.builder.ins().fcmp(FloatCC::LessThan, value, upper);
        let in_range = self.builder.ins().band(above_lower, below_upper);
        direct = self.builder.ins().band(direct, in_range);
        let native = self.builder.create_block();
        let exact = self.cold_block();
        let done = self.builder.create_block();
        let destination_was_dirty = if fields.rd == 31 {
            false
        } else {
            self.block_dirty_registers[usize::from(fields.rd)]
        };
        self.builder.ins().brif(direct, native, &[], exact, &[]);
        self.builder.switch_to_block(native);
        let result = if unsigned {
            self.builder.ins().fcvt_to_uint_sat(integer_ty, value)
        } else {
            self.builder.ins().fcvt_to_sint_sat(integer_ty, value)
        };
        let result = if integer_ty == types::I32 {
            self.builder.ins().uextend(types::I64, result)
        } else {
            result
        };
        self.write_register(fields.rd, result)?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(exact);
        if fields.rd != 31 {
            self.block_dirty_registers[usize::from(fields.rd)] = destination_was_dirty;
        }
        self.emit_typed_fp_cold(source, instruction, fields, flags)?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        if fields.rd != 31 {
            self.block_dirty_registers[usize::from(fields.rd)] = true;
        }
        Ok(())
    }

    fn emit_common_vector_integer_to_float(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &mut LazyFlags,
    ) -> Result<(), DirectJitError> {
        let lane_bits = if fields.opc == 0 { 32 } else { 64 };
        let integer_ty = if lane_bits == 32 {
            types::I32X4
        } else {
            types::I64X2
        };
        let float_ty = if lane_bits == 32 {
            types::F32X4
        } else {
            types::F64X2
        };
        let signed = matches!(
            instruction,
            Instruction::VectorSignedIntToFloat(_) | Instruction::ScalarVectorSignedIntToFloat(_)
        );
        let scalar = matches!(
            instruction,
            Instruction::ScalarVectorSignedIntToFloat(_)
                | Instruction::ScalarVectorUnsignedIntToFloat(_)
        );
        let vector_bits = if scalar {
            lane_bits
        } else if fields.vector_128 {
            128
        } else {
            64
        };
        let value = self.read_vector(fields.rn)?;
        let value = self.mask_vector(value, vector_bits);
        let value = self.vector_as(value, integer_ty);
        let native = self.builder.create_block();
        let exact = self.cold_block();
        let done = self.builder.create_block();
        let destination_was_dirty = self.block_dirty_vector_registers[usize::from(fields.rd)];
        self.builder
            .ins()
            .brif(self.native_fp_enabled, native, &[], exact, &[]);
        self.builder.switch_to_block(native);
        let result = if signed {
            self.builder.ins().fcvt_from_sint(float_ty, value)
        } else {
            self.builder.ins().fcvt_from_uint(float_ty, value)
        };
        let result = self.vector_as(result, types::I8X16);
        let result = self.mask_vector(result, vector_bits);
        self.write_vector(fields.rd, result)?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(exact);
        self.block_dirty_vector_registers[usize::from(fields.rd)] = destination_was_dirty;
        self.emit_typed_fp_cold(source, instruction, fields, flags)?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        self.block_dirty_vector_registers[usize::from(fields.rd)] = true;
        Ok(())
    }

    fn float_constant(&mut self, ty: cranelift_codegen::ir::Type, value: f64) -> Value {
        if ty == types::F32 {
            self.builder
                .ins()
                .f32const(Ieee32::with_bits((value as f32).to_bits()))
        } else {
            self.builder
                .ins()
                .f64const(Ieee64::with_bits(value.to_bits()))
        }
    }

    fn scalar_fp_bits(&mut self, register: u8, width: u32) -> Result<Value, DirectJitError> {
        let value = self.read_vector_as(register, types::I128)?;
        Ok(self
            .builder
            .ins()
            .ireduce(if width == 32 { types::I32 } else { types::I64 }, value))
    }

    /// True for zero and finite normal values. NaNs, infinities and denormal
    /// inputs use the exact edge because their payload/status contracts differ
    /// between Arm and the host FP ISA.
    fn fp_finite_or_zero(&mut self, bits: Value, width: u32) -> Value {
        let (exponent_mask, magnitude_mask) = if width == 32 {
            (0x7f80_0000_u64, 0x7fff_ffff_u64)
        } else {
            (0x7ff0_0000_0000_0000, 0x7fff_ffff_ffff_ffff)
        };
        let exponent = self.builder.ins().band_imm_u(bits, exponent_mask as i64);
        let exponent_nonzero = self.builder.ins().icmp_imm_s(IntCC::NotEqual, exponent, 0);
        let exponent_finite =
            self.builder
                .ins()
                .icmp_imm_s(IntCC::NotEqual, exponent, exponent_mask as i64);
        let normal = self.builder.ins().band(exponent_nonzero, exponent_finite);
        let magnitude = self.builder.ins().band_imm_u(bits, magnitude_mask as i64);
        let zero = self.builder.ins().icmp_imm_s(IntCC::Equal, magnitude, 0);
        self.builder.ins().bor(zero, normal)
    }

    fn fp_vector_finite_or_zero(&mut self, value: Value, lane_bits: u32) -> Value {
        let (exponent_mask, fraction_mask) = if lane_bits == 32 {
            (0x7f80_0000_u64, 0x007f_ffff_u64)
        } else {
            (0x7ff0_0000_0000_0000, 0x000f_ffff_ffff_ffff)
        };
        let ty = self.builder.func.dfg.value_type(value);
        let exponent_mask = self.fp_vector_lane_constant(ty, lane_bits, exponent_mask);
        let fraction_mask = self.fp_vector_lane_constant(ty, lane_bits, fraction_mask);
        let zero = self.fp_vector_lane_constant(ty, lane_bits, 0);
        let exponent = self.builder.ins().band(value, exponent_mask);
        let fraction = self.builder.ins().band(value, fraction_mask);
        let exponent_zero = self.builder.ins().icmp(IntCC::Equal, exponent, zero);
        let exponent_ones = self
            .builder
            .ins()
            .icmp(IntCC::Equal, exponent, exponent_mask);
        let fraction_nonzero = self.builder.ins().icmp(IntCC::NotEqual, fraction, zero);
        let subnormal = self.builder.ins().band(exponent_zero, fraction_nonzero);
        let invalid = self.builder.ins().bor(exponent_ones, subnormal);
        let invalid = self.vector_as(invalid, types::I128);
        self.builder.ins().icmp_imm_s(IntCC::Equal, invalid, 0)
    }

    fn fp_vector_active_lanes_nonzero(
        &mut self,
        value: Value,
        lane_bits: u32,
        lanes: u8,
        magnitude_mask: u64,
    ) -> Value {
        let ty = self.builder.func.dfg.value_type(value);
        let magnitude_mask = self.fp_vector_lane_constant(ty, lane_bits, magnitude_mask);
        let zero = self.fp_vector_lane_constant(ty, lane_bits, 0);
        let magnitude = self.builder.ins().band(value, magnitude_mask);
        let zero_lanes = self.builder.ins().icmp(IntCC::Equal, magnitude, zero);
        let zero_lanes = self.vector_as(zero_lanes, types::I128);
        let active_bits = u32::from(lanes) * lane_bits;
        let active_mask = if active_bits == 128 {
            u128::MAX
        } else {
            (1_u128 << active_bits) - 1
        };
        let active_mask = self.vector_constant(active_mask);
        let active_mask = self.vector_as(active_mask, types::I128);
        let active_zero_lanes = self.builder.ins().band(zero_lanes, active_mask);
        self.builder
            .ins()
            .icmp_imm_s(IntCC::Equal, active_zero_lanes, 0)
    }

    fn fp_vector_lane_constant(
        &mut self,
        ty: cranelift_codegen::ir::Type,
        lane_bits: u32,
        lane_value: u64,
    ) -> Value {
        let mut bits = 0_u128;
        for offset in (0..128).step_by(lane_bits as usize) {
            bits |= u128::from(lane_value) << offset;
        }
        let value = self.vector_constant(bits);
        self.vector_as(value, ty)
    }

    fn write_vector(&mut self, index: u8, value: Value) -> Result<(), DirectJitError> {
        let variable = self.vector_registers[usize::from(index)].ok_or_else(|| {
            DirectJitError::internal(format!("direct JIT vector V{index} was not planned"))
        })?;
        self.builder.def_var(variable, value);
        self.block_dirty_vector_registers[usize::from(index)] = true;
        Ok(())
    }

    fn vector_as(&mut self, value: Value, ty: cranelift_codegen::ir::Type) -> Value {
        if self.builder.func.dfg.value_type(value) == ty {
            value
        } else {
            self.builder.ins().bitcast(ty, bitcast_flags(), value)
        }
    }

    fn read_vector_as(
        &mut self,
        index: u8,
        ty: cranelift_codegen::ir::Type,
    ) -> Result<Value, DirectJitError> {
        let value = self.read_vector(index)?;
        Ok(self.vector_as(value, ty))
    }

    fn vector_constant(&mut self, value: u128) -> Value {
        let constant = self
            .builder
            .func
            .dfg
            .constants
            .insert(Ieee128::with_bits(value).into());
        self.builder.ins().vconst(types::I8X16, constant)
    }

    fn shuffle_bytes(&mut self, first: Value, second: Value, mask: [u8; 16]) -> Value {
        let mask = self
            .builder
            .func
            .dfg
            .immediates
            .push(ConstantData::from(mask.as_slice()));
        self.builder.ins().shuffle(first, second, mask)
    }

    fn mask_vector(&mut self, value: Value, bits: u32) -> Value {
        if bits == 128 {
            return value;
        }
        let mask = self.vector_constant((1_u128 << bits) - 1);
        self.builder.ins().band(value, mask)
    }

    fn finish_vector(&mut self, value: Value, full_width: bool) -> Value {
        let value = self.vector_as(value, types::I8X16);
        if full_width {
            value
        } else {
            self.mask_vector(value, 64)
        }
    }

    fn emit_integer_vector(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let lane_bits = 8_u32 << fields.opc;
        let vector_ty = vector_type(integer_lane_type(lane_bits)?, lane_bits)?;
        let lhs = self.read_vector_as(fields.rn, vector_ty)?;
        let rhs = self.read_vector_as(fields.rm, vector_ty)?;
        let result = if fields.subtract {
            self.builder.ins().isub(lhs, rhs)
        } else {
            self.builder.ins().iadd(lhs, rhs)
        };
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_bitwise(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let first = self.read_vector(fields.rn)?;
        let second = self.read_vector(fields.rm)?;
        let result = match fields
            .bitwise_operation
            .expect("normalized SIMD bitwise operation")
        {
            BitwiseOperation::And => self.builder.ins().band(first, second),
            BitwiseOperation::BitClear => {
                let not_second = self.builder.ins().bnot(second);
                self.builder.ins().band(first, not_second)
            }
            BitwiseOperation::Or => self.builder.ins().bor(first, second),
            BitwiseOperation::OrNot => {
                let not_second = self.builder.ins().bnot(second);
                self.builder.ins().bor(first, not_second)
            }
            BitwiseOperation::ExclusiveOr => self.builder.ins().bxor(first, second),
            BitwiseOperation::Select => {
                let destination = self.read_vector(fields.rd)?;
                self.builder.ins().bitselect(destination, first, second)
            }
            BitwiseOperation::InsertIfTrue => {
                let destination = self.read_vector(fields.rd)?;
                self.builder.ins().bitselect(second, first, destination)
            }
            BitwiseOperation::InsertIfFalse => {
                let destination = self.read_vector(fields.rd)?;
                self.builder.ins().bitselect(second, destination, first)
            }
        };
        let result = self.mask_vector(result, if fields.vector_128 { 128 } else { 64 });
        self.write_vector(fields.rd, result)
    }

    fn emit_integer_compare(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let lane_bits = 8_u32 << fields.opc;
        let lane = integer_lane_type(lane_bits)?;
        let vector_ty = vector_type(lane, lane_bits)?;
        let lhs = self.read_vector_as(fields.rn, vector_ty)?;
        let zero = self.builder.ins().iconst(lane, 0);
        let zero = self.builder.ins().splat(vector_ty, zero);
        let rhs = if fields.compare_with_zero {
            zero
        } else {
            self.read_vector_as(fields.rm, vector_ty)?
        };
        let comparison = fields
            .integer_comparison
            .expect("normalized SIMD comparison");
        let result = match comparison {
            IntegerComparison::NonzeroBitTest => {
                let bits = self.builder.ins().band(lhs, rhs);
                self.builder.ins().icmp(IntCC::NotEqual, bits, zero)
            }
            comparison => {
                let condition = match comparison {
                    IntegerComparison::SignedGreaterThan => IntCC::SignedGreaterThan,
                    IntegerComparison::UnsignedGreaterThan => IntCC::UnsignedGreaterThan,
                    IntegerComparison::SignedGreaterThanOrEqual => IntCC::SignedGreaterThanOrEqual,
                    IntegerComparison::UnsignedGreaterThanOrEqual => {
                        IntCC::UnsignedGreaterThanOrEqual
                    }
                    IntegerComparison::SignedLessThan => IntCC::SignedLessThan,
                    IntegerComparison::SignedLessThanOrEqual => IntCC::SignedLessThanOrEqual,
                    IntegerComparison::Equal => IntCC::Equal,
                    IntegerComparison::NonzeroBitTest => unreachable!(),
                };
                self.builder.ins().icmp(condition, lhs, rhs)
            }
        };
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_integer_pairwise(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let lane_bits = 8_u32 << fields.opc;
        let lanes = (if fields.vector_128 { 128 } else { 64 }) / lane_bits;
        let lane_bytes = lane_bits / 8;
        let vector_ty = vector_type(integer_lane_type(lane_bits)?, lane_bits)?;
        let first = self.read_vector(fields.rn)?;
        let second = self.read_vector(fields.rm)?;
        let mut left_mask = [0_u8; 16];
        let mut right_mask = [0_u8; 16];
        for destination in 0..lanes {
            let (source_base, source_lane) = if destination < lanes / 2 {
                (0, destination * 2)
            } else {
                (16, (destination - lanes / 2) * 2)
            };
            for byte in 0..lane_bytes {
                let output = (destination * lane_bytes + byte) as usize;
                left_mask[output] = (source_base + source_lane * lane_bytes + byte) as u8;
                right_mask[output] = left_mask[output] + lane_bytes as u8;
            }
        }
        let left = self.shuffle_bytes(first, second, left_mask);
        let right = self.shuffle_bytes(first, second, right_mask);
        let left = self.vector_as(left, vector_ty);
        let right = self.vector_as(right, vector_ty);
        let operation = fields
            .pairwise_operation
            .expect("normalized pairwise operation");
        let result = self.select_pairwise_vector(left, right, operation);
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_integer_min_max(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let lane_bits = 8_u32 << fields.opc;
        let vector_ty = vector_type(integer_lane_type(lane_bits)?, lane_bits)?;
        let lhs = self.read_vector_as(fields.rn, vector_ty)?;
        let rhs = self.read_vector_as(fields.rm, vector_ty)?;
        let operation = fields
            .pairwise_operation
            .expect("normalized min/max operation");
        let result = self.select_pairwise_vector(lhs, rhs, operation);
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn select_pairwise_vector(
        &mut self,
        lhs: Value,
        rhs: Value,
        operation: PairwiseOperation,
    ) -> Value {
        match operation {
            PairwiseOperation::Add => self.builder.ins().iadd(lhs, rhs),
            operation => {
                let condition = match operation {
                    PairwiseOperation::SignedMaximum => IntCC::SignedGreaterThanOrEqual,
                    PairwiseOperation::SignedMinimum => IntCC::SignedLessThanOrEqual,
                    PairwiseOperation::UnsignedMaximum => IntCC::UnsignedGreaterThanOrEqual,
                    PairwiseOperation::UnsignedMinimum => IntCC::UnsignedLessThanOrEqual,
                    PairwiseOperation::Add => unreachable!(),
                };
                let mask = self.builder.ins().icmp(condition, lhs, rhs);
                self.builder.ins().bitselect(mask, lhs, rhs)
            }
        }
    }

    fn emit_permute(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let lane_bits = 8_u32 << fields.opc;
        let lane_count = (if fields.vector_128 { 128 } else { 64 }) / lane_bits;
        let lane_bytes = lane_bits / 8;
        let half = lane_count / 2;
        let first = self.read_vector(fields.rn)?;
        let second = self.read_vector(fields.rm)?;
        let operation = fields
            .permute_operation
            .expect("normalized SIMD permutation");
        let mut mask = [0_u8; 16];
        for destination in 0..lane_count {
            let (source_base, lane) = match operation {
                PermuteOperation::UnzipPrimary | PermuteOperation::UnzipSecondary => {
                    let odd = u32::from(matches!(operation, PermuteOperation::UnzipSecondary));
                    if destination < half {
                        (0, destination * 2 + odd)
                    } else {
                        (16, (destination - half) * 2 + odd)
                    }
                }
                PermuteOperation::TransposePrimary | PermuteOperation::TransposeSecondary => {
                    let odd = u32::from(matches!(operation, PermuteOperation::TransposeSecondary));
                    (
                        if destination & 1 == 0 { 0 } else { 16 },
                        (destination / 2) * 2 + odd,
                    )
                }
                PermuteOperation::ZipPrimary | PermuteOperation::ZipSecondary => {
                    let upper = u32::from(matches!(operation, PermuteOperation::ZipSecondary));
                    (
                        if destination & 1 == 0 { 0 } else { 16 },
                        destination / 2 + upper * half,
                    )
                }
            };
            for byte in 0..lane_bytes {
                mask[(destination * lane_bytes + byte) as usize] =
                    (source_base + lane * lane_bytes + byte) as u8;
            }
        }
        let result = self.shuffle_bytes(first, second, mask);
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_vector_extract(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let count = if fields.vector_128 { 16 } else { 8 };
        let first = self.read_vector(fields.rn)?;
        let second = self.read_vector(fields.rm)?;
        let mut mask = [0_u8; 16];
        for destination in 0..count {
            let source = destination + u32::from(fields.immediate_4);
            mask[destination as usize] = if source < count {
                source as u8
            } else {
                (16 + source - count) as u8
            };
        }
        let result = self.shuffle_bytes(first, second, mask);
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_narrow(
        &mut self,
        instruction: Instruction,
        fields: Operands,
    ) -> Result<(), DirectJitError> {
        let (destination_bits, shift) = if matches!(instruction, Instruction::ShiftRightNarrow(_)) {
            let high = u32::from(fields.shift_immediate >> 3);
            let destination = 8_u32 << (31 - high.leading_zeros());
            (
                destination,
                destination * 2 - u32::from(fields.shift_immediate),
            )
        } else {
            (8_u32 << fields.opc, 0)
        };
        let source_bits = destination_bits * 2;
        let lane_count = 128 / source_bits;
        let source_ty = vector_type(integer_lane_type(source_bits)?, source_bits)?;
        let mut source = self.read_vector_as(fields.rn, source_ty)?;
        if shift != 0 {
            source = self.builder.ins().ushr_imm_u(source, i64::from(shift));
        }
        let source = self.vector_as(source, types::I8X16);
        let source_bytes = source_bits / 8;
        let destination_bytes = destination_bits / 8;
        let mut packed_mask = [0_u8; 16];
        for lane in 0..lane_count {
            for byte in 0..destination_bytes {
                packed_mask[(lane * destination_bytes + byte) as usize] =
                    (lane * source_bytes + byte) as u8;
            }
        }
        let packed = self.shuffle_bytes(source, source, packed_mask);
        let result = if fields.vector_128 {
            let previous = self.read_vector(fields.rd)?;
            let mut upper_mask = [0_u8; 16];
            for byte in 0..8 {
                upper_mask[byte] = byte as u8;
                upper_mask[byte + 8] = (16 + byte) as u8;
            }
            self.shuffle_bytes(previous, packed, upper_mask)
        } else {
            self.mask_vector(packed, 64)
        };
        self.write_vector(fields.rd, result)
    }

    fn emit_immediate_shift(
        &mut self,
        instruction: Instruction,
        fields: Operands,
    ) -> Result<(), DirectJitError> {
        let immediate = u32::from(fields.shift_immediate);
        let high = immediate >> 3;
        let lane_bits = 8_u32 << (31 - high.leading_zeros());
        let right = matches!(
            instruction,
            Instruction::ScalarShiftRightImmediate(_) | Instruction::VectorShiftRightImmediate(_)
        );
        let scalar = matches!(
            instruction,
            Instruction::ScalarShiftRightImmediate(_) | Instruction::ScalarShiftLeftImmediate(_)
        );
        let shift = if right {
            2 * lane_bits - immediate
        } else {
            immediate - lane_bits
        };
        let lane = integer_lane_type(lane_bits)?;
        let vector_ty = vector_type(lane, lane_bits)?;
        let source = self.read_vector_as(fields.rn, vector_ty)?;
        let result = if right && shift == lane_bits && !fields.operation_bit {
            self.builder
                .ins()
                .sshr_imm_u(source, i64::from(lane_bits - 1))
        } else if right && shift == lane_bits {
            let zero = self.builder.ins().iconst(lane, 0);
            self.builder.ins().splat(vector_ty, zero)
        } else if right && !fields.operation_bit {
            self.builder.ins().sshr_imm_u(source, i64::from(shift))
        } else if right {
            self.builder.ins().ushr_imm_u(source, i64::from(shift))
        } else {
            self.builder.ins().ishl_imm_u(source, i64::from(shift))
        };
        let result = self.finish_vector(result, !scalar && fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_shift_left_long(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let immediate = u32::from(fields.shift_immediate);
        let high = immediate >> 3;
        let source_bits = 8_u32 << (31 - high.leading_zeros());
        let destination_bits = source_bits * 2;
        let shift = immediate - source_bits;
        let lane_count = 64 / source_bits;
        let source_lane = integer_lane_type(source_bits)?;
        let destination_lane = integer_lane_type(destination_bits)?;
        let source_ty = vector_type(source_lane, source_bits)?;
        let destination_ty = vector_type(destination_lane, destination_bits)?;
        let source = self.read_vector_as(fields.rn, source_ty)?;
        let zero = self.builder.ins().iconst(destination_lane, 0);
        let mut result = self.builder.ins().splat(destination_ty, zero);
        let first = if fields.vector_128 { lane_count } else { 0 };
        for index in 0..lane_count {
            let value = self
                .builder
                .ins()
                .extractlane(source, (first + index) as u8);
            let value = if fields.operation_bit {
                self.builder.ins().uextend(destination_lane, value)
            } else {
                self.builder.ins().sextend(destination_lane, value)
            };
            let value = if shift == 0 {
                value
            } else {
                self.builder.ins().ishl_imm_u(value, i64::from(shift))
            };
            result = self.builder.ins().insertlane(result, value, index as u8);
        }
        let result = self.vector_as(result, types::I8X16);
        self.write_vector(fields.rd, result)
    }

    fn emit_register_shift(
        &mut self,
        instruction: Instruction,
        fields: Operands,
    ) -> Result<(), DirectJitError> {
        let lane_bits = 8_u32 << fields.opc;
        let lane = integer_lane_type(lane_bits)?;
        let vector_ty = vector_type(lane, lane_bits)?;
        let values = self.read_vector_as(fields.rn, vector_ty)?;
        let mut distance = self.read_vector_as(fields.rm, vector_ty)?;
        let zero = self.builder.ins().iconst(lane, 0);
        let zero = self.builder.ins().splat(vector_ty, zero);
        if lane_bits > 8 {
            let low_byte = self.builder.ins().iconst(lane, 0xff);
            let low_byte = self.builder.ins().splat(vector_ty, low_byte);
            distance = self.builder.ins().band(distance, low_byte);
            distance = self
                .builder
                .ins()
                .ishl_imm_u(distance, i64::from(lane_bits - 8));
            distance = self
                .builder
                .ins()
                .sshr_imm_u(distance, i64::from(lane_bits - 8));
        }
        let nonnegative = self
            .builder
            .ins()
            .icmp(IntCC::SignedGreaterThanOrEqual, distance, zero);
        let negative = self.builder.ins().ineg(distance);
        let magnitude = self
            .builder
            .ins()
            .bitselect(nonnegative, distance, negative);
        let signed = matches!(instruction, Instruction::VectorSignedShiftRegister(_));
        let mut left = values;
        let mut right = values;
        let mut amount = 1_u32;
        while amount < lane_bits {
            let bit = self.builder.ins().iconst(lane, i64::from(amount));
            let bit = self.builder.ins().splat(vector_ty, bit);
            let selected = self.builder.ins().band(magnitude, bit);
            let selected = self.builder.ins().icmp(IntCC::NotEqual, selected, zero);
            let shifted_left = self.builder.ins().ishl_imm_u(left, i64::from(amount));
            let shifted_right = if signed {
                self.builder.ins().sshr_imm_u(right, i64::from(amount))
            } else {
                self.builder.ins().ushr_imm_u(right, i64::from(amount))
            };
            left = self.builder.ins().bitselect(selected, shifted_left, left);
            right = self.builder.ins().bitselect(selected, shifted_right, right);
            amount *= 2;
        }
        let width = self.builder.ins().iconst(lane, i64::from(lane_bits));
        let width = self.builder.ins().splat(vector_ty, width);
        let out = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, magnitude, width);
        let fill = if signed {
            self.builder
                .ins()
                .sshr_imm_u(values, i64::from(lane_bits - 1))
        } else {
            zero
        };
        left = self.builder.ins().bitselect(out, zero, left);
        right = self.builder.ins().bitselect(out, fill, right);
        let result = self.builder.ins().bitselect(nonnegative, left, right);
        let result = self.finish_vector(result, fields.vector_128);
        self.write_vector(fields.rd, result)
    }

    fn emit_add_across(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let lane_bits = 8_u32 << fields.opc;
        let lane_count = (if fields.vector_128 { 128 } else { 64 }) / lane_bits;
        let lane = integer_lane_type(lane_bits)?;
        let vector_ty = vector_type(lane, lane_bits)?;
        let mut value = self.read_vector_as(fields.rn, vector_ty)?;
        let lane_bytes = lane_bits / 8;
        let mut distance = lane_count / 2;
        while distance != 0 {
            let bytes = self.vector_as(value, types::I8X16);
            let mut mask = [0_u8; 16];
            for destination in 0..lane_count {
                let source = if destination < distance {
                    destination + distance
                } else {
                    destination
                };
                for byte in 0..lane_bytes {
                    mask[(destination * lane_bytes + byte) as usize] =
                        (source * lane_bytes + byte) as u8;
                }
            }
            let paired = self.shuffle_bytes(bytes, bytes, mask);
            let paired = self.vector_as(paired, vector_ty);
            value = self.builder.ins().iadd(value, paired);
            distance /= 2;
        }
        let result = self.builder.ins().extractlane(value, 0);
        let result = self.builder.ins().uextend(types::I128, result);
        let result = self.vector_as(result, types::I8X16);
        self.write_vector(fields.rd, result)
    }

    fn emit_float_immediate(
        &mut self,
        instruction: Instruction,
        fields: Operands,
    ) -> Result<(), DirectJitError> {
        let value = if matches!(instruction, Instruction::ScalarFloatImmediate(_)) {
            let (exponent, fraction) = match fields.opc {
                0 => (8, 23),
                1 => (11, 52),
                3 => (5, 10),
                _ => return Err(DirectJitError::invalid("invalid scalar FP immediate width")),
            };
            u128::from(expand_vfp_immediate(
                fields.fp_immediate_8,
                exponent,
                fraction,
            ))
        } else {
            let (lane, bits) = if fields.operation_bit {
                (expand_vfp_immediate(fields.immediate_8, 11, 52), 64)
            } else {
                (expand_vfp_immediate(fields.immediate_8, 8, 23), 32)
            };
            if bits == 64 {
                u128::from(lane) | (u128::from(lane) << 64)
            } else {
                let lane = u128::from(lane as u32);
                lane | lane << 32 | lane << 64 | lane << 96
            }
        };
        let value = self.vector_constant(value);
        let value = self.mask_vector(
            value,
            if fields.vector_128 || matches!(instruction, Instruction::ScalarFloatImmediate(_)) {
                128
            } else {
                64
            },
        );
        self.write_vector(fields.rd, value)
    }

    fn emit_move_to_general(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let vector = self.read_vector_as(fields.rn, types::I128)?;
        let (width, value) = match (fields.size & 2 != 0, fields.opc) {
            (false, 0) => (32, self.builder.ins().ireduce(types::I32, vector)),
            (false, 3) => (32, self.builder.ins().ireduce(types::I16, vector)),
            (true, 1) => (64, self.builder.ins().ireduce(types::I64, vector)),
            (true, 2) => {
                let value = self.builder.ins().ushr_imm_u(vector, 64);
                (64, self.builder.ins().ireduce(types::I64, value))
            }
            _ => return Err(DirectJitError::invalid("invalid FMOV general width")),
        };
        let value = cast_integer(&mut self.builder, value, types::I64, false);
        let _ = width;
        self.write_register(fields.rd, value)
    }

    fn emit_move_from_general(&mut self, fields: Operands) -> Result<(), DirectJitError> {
        let value = self.read_register(fields.rn, false)?;
        let general_64 = fields.size & 2 != 0;
        let value = if general_64 {
            value
        } else {
            self.builder.ins().ireduce(types::I32, value)
        };
        let value = cast_integer(&mut self.builder, value, types::I128, false);
        let value = match (general_64, fields.opc) {
            (false, 0) => value,
            (false, 3) => {
                let value = self.builder.ins().ireduce(types::I16, value);
                self.builder.ins().uextend(types::I128, value)
            }
            (true, 1) => value,
            (true, 2) => {
                let previous = self.read_vector_as(fields.rd, types::I128)?;
                let low = self.builder.ins().ireduce(types::I64, previous);
                let low = self.builder.ins().uextend(types::I128, low);
                let high = self.builder.ins().ishl_imm_u(value, 64);
                self.builder.ins().bor(low, high)
            }
            _ => return Err(DirectJitError::invalid("invalid FMOV general width")),
        };
        let value = self.vector_as(value, types::I8X16);
        self.write_vector(fields.rd, value)
    }

    fn emit_vector_memory(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        match instruction {
            Instruction::MemoryPair(_) => self.emit_vector_pair(source, fields, flags),
            Instruction::MemoryMultipleStructures(_)
            | Instruction::MemoryMultipleStructuresPostIndex(_) => {
                self.emit_multiple_structures(source, instruction, fields, flags)
            }
            Instruction::MemorySingleStructure(_)
            | Instruction::MemorySingleStructurePostIndex(_) => {
                self.emit_single_structure(source, instruction, fields, flags)
            }
            _ => {
                let size = simd_memory_access_size(fields.size, fields.opc)
                    .ok_or_else(|| DirectJitError::invalid("invalid SIMD transfer size"))?;
                let base = self.read_register(fields.rn, true)?;
                let offset = match instruction {
                    Instruction::MemoryUnsigned(_) => {
                        u64::from(fields.immediate_12) * size.bytes() as u64
                    }
                    Instruction::MemoryUnscaled(_)
                    | Instruction::MemoryPostIndex(_)
                    | Instruction::MemoryPreIndex(_) => {
                        sign_extend(u64::from(fields.immediate_9), 9) as u64
                    }
                    Instruction::MemoryRegister(_) => {
                        let offset = self.read_register(fields.rm, false)?;
                        let offset = match fields.option {
                            2 => {
                                let offset = self.builder.ins().ireduce(types::I32, offset);
                                self.builder.ins().uextend(types::I64, offset)
                            }
                            3 => offset,
                            6 => {
                                let offset = self.builder.ins().ireduce(types::I32, offset);
                                self.builder.ins().sextend(types::I64, offset)
                            }
                            7 => offset,
                            _ => {
                                return Err(DirectJitError::invalid(
                                    "invalid SIMD register-offset extension",
                                ));
                            }
                        };
                        let offset = if fields.scaled {
                            self.builder
                                .ins()
                                .ishl_imm_u(offset, i64::from(size.bytes().trailing_zeros()))
                        } else {
                            offset
                        };
                        let address = self.builder.ins().iadd(base, offset);
                        return self.emit_vector_transfer(source, fields, address, size, 0, flags);
                    }
                    _ => unreachable!(),
                };
                let address = if matches!(instruction, Instruction::MemoryPostIndex(_)) {
                    base
                } else {
                    self.builder.ins().iadd_imm_u(base, offset as i64)
                };
                self.emit_vector_transfer(source, fields, address, size, 0, flags)?;
                if matches!(
                    instruction,
                    Instruction::MemoryPostIndex(_) | Instruction::MemoryPreIndex(_)
                ) {
                    let updated = self.builder.ins().iadd_imm_u(base, offset as i64);
                    self.write_register_with_sp(fields.rn, true, updated)?;
                }
                Ok(())
            }
        }
    }

    fn emit_vector_transfer(
        &mut self,
        source: GuestVirtualAddress,
        fields: Operands,
        address: Value,
        size: MemoryAccessSize,
        element_index: u8,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let operation =
            MemoryOperation::new(size, MemoryOrdering::Relaxed).with_element_index(element_index);
        if fields.load {
            let value = if size == MemoryAccessSize::Quadword {
                self.memory_read_vector128(source, address, operation, flags)?
            } else {
                let value = self.memory_read(source, address, operation, flags)?;
                self.builder.ins().uextend(types::I128, value)
            };
            let value = self.vector_as(value, types::I8X16);
            self.write_vector(fields.rd, value)
        } else {
            if size == MemoryAccessSize::Quadword {
                let value = self.read_vector(fields.rd)?;
                return self.memory_write_vector128(source, address, value, operation, flags);
            }
            let value = self.read_vector_as(fields.rd, types::I128)?;
            let value = reduce_integer(&mut self.builder, value, size);
            self.memory_write(source, address, value, operation, flags)
        }
    }

    fn emit_vector_pair(
        &mut self,
        source: GuestVirtualAddress,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let size = simd_pair_access_size(fields.size)
            .ok_or_else(|| DirectJitError::invalid("invalid SIMD pair size"))?;
        let base = self.read_register(fields.rn, true)?;
        let offset = sign_extend(u64::from(fields.immediate_7), 7) * size.bytes() as i64;
        let updated = self.builder.ins().iadd_imm_s(base, offset);
        let first = if matches!(fields.mode, 2 | 3) {
            updated
        } else {
            base
        };
        let second = self.builder.ins().iadd_imm_u(first, size.bytes() as i64);
        if fields.load {
            let first_operation = MemoryOperation::new(size, MemoryOrdering::Relaxed);
            let second_operation = first_operation.with_element_index(1);
            let (first_value, second_value) = if size == MemoryAccessSize::Quadword {
                (
                    self.memory_read_vector128(source, first, first_operation, flags)?,
                    self.memory_read_vector128(source, second, second_operation, flags)?,
                )
            } else {
                let first_value = self.memory_read(source, first, first_operation, flags)?;
                let second_value = self.memory_read(source, second, second_operation, flags)?;
                (
                    self.builder.ins().uextend(types::I128, first_value),
                    self.builder.ins().uextend(types::I128, second_value),
                )
            };
            let first_value = self.vector_as(first_value, types::I8X16);
            let second_value = self.vector_as(second_value, types::I8X16);
            self.write_vector(fields.rd, first_value)?;
            self.write_vector(fields.rt2, second_value)?;
        } else {
            for (element_index, (register, address)) in [(fields.rd, first), (fields.rt2, second)]
                .into_iter()
                .enumerate()
            {
                let mut transfer = fields;
                transfer.rd = register;
                self.emit_vector_transfer(
                    source,
                    transfer,
                    address,
                    size,
                    element_index as u8,
                    flags,
                )?;
            }
        }
        if matches!(fields.mode, 1 | 3) {
            self.write_register_with_sp(fields.rn, true, updated)?;
        }
        Ok(())
    }

    fn emit_multiple_structures(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let shape = simd_multiple_structure_shape(fields)
            .ok_or_else(|| DirectJitError::invalid("invalid SIMD multiple-structure shape"))?;
        let vector_size = if shape.vector_bytes == 16 {
            MemoryAccessSize::Quadword
        } else {
            MemoryAccessSize::Doubleword
        };
        let base = self.read_register(fields.rn, true)?;
        let mut displacement = 0_u8;
        if shape.structure_registers == 1 {
            for repetition in 0..shape.repetitions {
                let address = self.builder.ins().iadd_imm_u(base, i64::from(displacement));
                let mut transfer = fields;
                transfer.rd = fields.rd.wrapping_add(repetition) & 31;
                self.emit_vector_transfer(
                    source,
                    transfer,
                    address,
                    vector_size,
                    repetition,
                    flags,
                )?;
                displacement += shape.vector_bytes;
            }
        } else {
            let lane_bits = (shape.element_size.bytes() * 8) as u32;
            let lane_ty = integer_lane_type(lane_bits)?;
            let vector_ty = vector_type(lane_ty, lane_bits)?;
            let mut element_index = 0_u8;
            for lane in 0..shape.elements_per_register {
                for register_offset in 0..shape.structure_registers {
                    let address = self.builder.ins().iadd_imm_u(base, i64::from(displacement));
                    let register = fields.rd.wrapping_add(register_offset) & 31;
                    if fields.load {
                        let value = self.memory_read(
                            source,
                            address,
                            MemoryOperation::new(shape.element_size, MemoryOrdering::Relaxed)
                                .with_element_index(element_index),
                            flags,
                        )?;
                        let previous = self.read_vector_as(register, vector_ty)?;
                        let value = cast_integer(&mut self.builder, value, lane_ty, false);
                        let result = self.builder.ins().insertlane(previous, value, lane);
                        let result = self.vector_as(result, types::I8X16);
                        self.write_vector(register, result)?;
                    } else {
                        let vector = self.read_vector_as(register, vector_ty)?;
                        let value = self.builder.ins().extractlane(vector, lane);
                        self.memory_write(
                            source,
                            address,
                            value,
                            MemoryOperation::new(shape.element_size, MemoryOrdering::Relaxed)
                                .with_element_index(element_index),
                            flags,
                        )?;
                    }
                    displacement += shape.element_size.bytes() as u8;
                    element_index += 1;
                }
            }
            if !fields.vector_128 && fields.load {
                for register_offset in 0..shape.structure_registers {
                    let register = fields.rd.wrapping_add(register_offset) & 31;
                    let value = self.read_vector(register)?;
                    let value = self.mask_vector(value, 64);
                    self.write_vector(register, value)?;
                }
            }
        }
        if matches!(
            instruction,
            Instruction::MemoryMultipleStructuresPostIndex(_)
        ) {
            let offset = if fields.rm == 31 {
                self.builder
                    .ins()
                    .iconst(types::I64, i64::from(shape.immediate_post_index))
            } else {
                self.read_register(fields.rm, false)?
            };
            let updated = self.builder.ins().iadd(base, offset);
            self.write_register_with_sp(fields.rn, true, updated)?;
        }
        Ok(())
    }

    fn emit_single_structure(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<(), DirectJitError> {
        let shape = simd_single_structure_shape(fields)
            .ok_or_else(|| DirectJitError::invalid("invalid SIMD single-structure shape"))?;
        let base = self.read_register(fields.rn, true)?;
        let lane_bits = (shape.element_size.bytes() * 8) as u32;
        let lane_ty = integer_lane_type(lane_bits)?;
        let vector_ty = vector_type(lane_ty, lane_bits)?;
        let mut displacement = 0_u8;
        for register_offset in 0..shape.structure_registers {
            let address = self.builder.ins().iadd_imm_u(base, i64::from(displacement));
            let register = fields.rd.wrapping_add(register_offset) & 31;
            match shape.mode {
                SimdMemoryMode::Lane(lane) if fields.load => {
                    let value = self.memory_read(
                        source,
                        address,
                        MemoryOperation::new(shape.element_size, MemoryOrdering::Relaxed)
                            .with_element_index(register_offset),
                        flags,
                    )?;
                    let previous = self.read_vector_as(register, vector_ty)?;
                    let value = cast_integer(&mut self.builder, value, lane_ty, false);
                    let result = self.builder.ins().insertlane(previous, value, lane);
                    let result = self.vector_as(result, types::I8X16);
                    self.write_vector(register, result)?;
                }
                SimdMemoryMode::Lane(lane) => {
                    let vector = self.read_vector_as(register, vector_ty)?;
                    let value = self.builder.ins().extractlane(vector, lane);
                    self.memory_write(
                        source,
                        address,
                        value,
                        MemoryOperation::new(shape.element_size, MemoryOrdering::Relaxed)
                            .with_element_index(register_offset),
                        flags,
                    )?;
                }
                SimdMemoryMode::Replicate => {
                    let value = self.memory_read(
                        source,
                        address,
                        MemoryOperation::new(shape.element_size, MemoryOrdering::Relaxed)
                            .with_element_index(register_offset),
                        flags,
                    )?;
                    let value = cast_integer(&mut self.builder, value, lane_ty, false);
                    let value = self.builder.ins().splat(vector_ty, value);
                    let value = self.finish_vector(value, fields.vector_128);
                    self.write_vector(register, value)?;
                }
                SimdMemoryMode::Multiple => {
                    return Err(DirectJitError::internal(
                        "single-structure shape selected multiple mode",
                    ));
                }
            }
            displacement += shape.element_size.bytes() as u8;
        }
        if matches!(instruction, Instruction::MemorySingleStructurePostIndex(_)) {
            let offset = if fields.rm == 31 {
                self.builder
                    .ins()
                    .iconst(types::I64, i64::from(shape.immediate_post_index))
            } else {
                self.read_register(fields.rm, false)?
            };
            let updated = self.builder.ins().iadd(base, offset);
            self.write_register_with_sp(fields.rn, true, updated)?;
        }
        Ok(())
    }

    fn emit_typed_fp_cold(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        fields: Operands,
        flags: &mut LazyFlags,
    ) -> Result<(), DirectJitError> {
        let constant = |translator: &mut Self, value: u64| {
            translator.builder.ins().iconst(types::I64, value as i64)
        };
        let vector_parts =
            |translator: &mut Self, index: u8| -> Result<[Value; 2], DirectJitError> {
                let value = translator.read_vector_as(index, types::I128)?;
                let low = translator.builder.ins().ireduce(types::I64, value);
                let high = translator.builder.ins().ushr_imm_u(value, 64);
                let high = translator.builder.ins().ireduce(types::I64, high);
                Ok([low, high])
            };
        let first = if matches!(
            instruction,
            Instruction::SignedIntToFloat(_) | Instruction::UnsignedIntToFloat(_)
        ) {
            [constant(self, 0), constant(self, 0)]
        } else {
            vector_parts(self, fields.rn)?
        };
        let second = if matches!(
            instruction,
            Instruction::VectorFloatDivide(_)
                | Instruction::VectorFloatMultiplyElement(_)
                | Instruction::ScalarFloatDivide(_)
                | Instruction::ScalarFloatAdd(_)
                | Instruction::ScalarFloatMultiply(_)
                | Instruction::ScalarFloatFusedMultiplyAdd(_)
                | Instruction::CompareRegister(_)
                | Instruction::ConditionalCompare(_)
        ) {
            vector_parts(self, fields.rm)?
        } else {
            [constant(self, 0), constant(self, 0)]
        };
        let mut extra = Vec::new();
        let (function, operands, shape, output): (usize, Vec<Value>, u64, ColdFpOutput) =
            match instruction {
                Instruction::SignedIntToFloat(_) | Instruction::UnsignedIntToFloat(_) => {
                    let input = self.read_register(fields.rn, false)?;
                    let unsigned =
                        u64::from(matches!(instruction, Instruction::UnsignedIntToFloat(_)));
                    let shape =
                        unsigned | (u64::from(fields.opc) << 8) | (u64::from(fields.size) << 16);
                    (
                        slow::fp_integer_to_float as *const () as usize,
                        vec![input],
                        shape,
                        ColdFpOutput::Vector,
                    )
                }
                Instruction::FloatToSignedInt(_) | Instruction::FloatToUnsignedInt(_) => {
                    let rounding = match fields
                        .float_to_integer_rounding
                        .expect("normalized FP conversion rounding")
                    {
                        FloatToIntegerRounding::NearestEven => 0,
                        FloatToIntegerRounding::NearestAway => 1,
                        FloatToIntegerRounding::TowardPositive => 2,
                        FloatToIntegerRounding::TowardNegative => 3,
                        FloatToIntegerRounding::TowardZero => 4,
                    };
                    let fraction = fields.fixed_point_fraction_bits.unwrap_or(u8::MAX);
                    let shape =
                        u64::from(matches!(instruction, Instruction::FloatToUnsignedInt(_)))
                            | (u64::from(fields.opc) << 8)
                            | (u64::from(fields.size) << 16)
                            | (rounding << 24)
                            | (u64::from(fraction) << 32);
                    (
                        slow::fp_float_to_integer as *const () as usize,
                        vec![first[0]],
                        shape,
                        ColdFpOutput::General,
                    )
                }
                Instruction::VectorSignedIntToFloat(_)
                | Instruction::VectorUnsignedIntToFloat(_)
                | Instruction::ScalarVectorSignedIntToFloat(_)
                | Instruction::ScalarVectorUnsignedIntToFloat(_)
                | Instruction::ScalarFloatConvert(_)
                | Instruction::ScalarFloatRound(_)
                | Instruction::ScalarFloatSquareRoot(_) => {
                    let kind = match instruction {
                        Instruction::VectorSignedIntToFloat(_) => {
                            slow::FpUnaryKind::VectorSignedIntToFloat
                        }
                        Instruction::VectorUnsignedIntToFloat(_) => {
                            slow::FpUnaryKind::VectorUnsignedIntToFloat
                        }
                        Instruction::ScalarVectorSignedIntToFloat(_) => {
                            slow::FpUnaryKind::ScalarVectorSignedIntToFloat
                        }
                        Instruction::ScalarVectorUnsignedIntToFloat(_) => {
                            slow::FpUnaryKind::ScalarVectorUnsignedIntToFloat
                        }
                        Instruction::ScalarFloatConvert(_) => {
                            match fields.float_conversion.expect("normalized FP conversion") {
                                FloatConversion::SingleToDouble => {
                                    slow::FpUnaryKind::ConvertSingleDouble
                                }
                                FloatConversion::DoubleToSingle => {
                                    slow::FpUnaryKind::ConvertDoubleSingle
                                }
                            }
                        }
                        Instruction::ScalarFloatRound(_) => match fields
                            .float_round_operation
                            .expect("normalized FP round")
                        {
                            FloatRoundOperation::NearestEven => slow::FpUnaryKind::RoundNearestEven,
                            FloatRoundOperation::TowardPositive => slow::FpUnaryKind::RoundPositive,
                            FloatRoundOperation::TowardNegative => slow::FpUnaryKind::RoundNegative,
                            FloatRoundOperation::TowardZero => slow::FpUnaryKind::RoundZero,
                            FloatRoundOperation::NearestAway => slow::FpUnaryKind::RoundNearestAway,
                            FloatRoundOperation::Exact => slow::FpUnaryKind::RoundExact,
                            FloatRoundOperation::CurrentMode => slow::FpUnaryKind::RoundCurrent,
                        },
                        Instruction::ScalarFloatSquareRoot(_) => slow::FpUnaryKind::SquareRoot,
                        _ => unreachable!(),
                    };
                    let shape = kind as u64
                        | (u64::from(fields.opc) << 8)
                        | (u64::from(fields.vector_128) << 16);
                    (
                        slow::fp_unary as *const () as usize,
                        first.to_vec(),
                        shape,
                        ColdFpOutput::Vector,
                    )
                }
                Instruction::VectorFloatDivide(_)
                | Instruction::VectorFloatMultiplyElement(_)
                | Instruction::ScalarFloatDivide(_)
                | Instruction::ScalarFloatAdd(_)
                | Instruction::ScalarFloatMultiply(_) => {
                    let kind = match instruction {
                        Instruction::VectorFloatDivide(_) => slow::FpBinaryKind::VectorDivide,
                        Instruction::VectorFloatMultiplyElement(_) => {
                            slow::FpBinaryKind::VectorMultiplyElement
                        }
                        Instruction::ScalarFloatDivide(_) => slow::FpBinaryKind::Divide,
                        Instruction::ScalarFloatAdd(_) => {
                            match fields.float_add_operation.expect("normalized FP add") {
                                FloatAddOperation::Add => slow::FpBinaryKind::Add,
                                FloatAddOperation::Subtract => slow::FpBinaryKind::Subtract,
                            }
                        }
                        Instruction::ScalarFloatMultiply(_) => match fields
                            .float_multiply_operation
                            .expect("normalized FP multiply")
                        {
                            FloatMultiplyOperation::Multiply => slow::FpBinaryKind::Multiply,
                            FloatMultiplyOperation::NegatedMultiply => {
                                slow::FpBinaryKind::NegatedMultiply
                            }
                        },
                        _ => unreachable!(),
                    };
                    let shape = kind as u64
                        | (u64::from(fields.opc) << 8)
                        | (u64::from(fields.vector_128) << 16)
                        | (u64::from(fields.fp_element_lane) << 24);
                    (
                        slow::fp_binary as *const () as usize,
                        vec![first[0], first[1], second[0], second[1]],
                        shape,
                        ColdFpOutput::Vector,
                    )
                }
                Instruction::ScalarFloatFusedMultiplyAdd(_) => {
                    let third = vector_parts(self, fields.ra)?;
                    let kind = match fields
                        .float_fused_multiply_operation
                        .expect("normalized FP fused operation")
                    {
                        FloatFusedMultiplyOperation::MultiplyAdd => slow::FpFusedKind::MultiplyAdd,
                        FloatFusedMultiplyOperation::MultiplySubtract => {
                            slow::FpFusedKind::MultiplySubtract
                        }
                        FloatFusedMultiplyOperation::NegatedMultiplyAdd => {
                            slow::FpFusedKind::NegatedMultiplyAdd
                        }
                        FloatFusedMultiplyOperation::NegatedMultiplySubtract => {
                            slow::FpFusedKind::NegatedMultiplySubtract
                        }
                    };
                    let shape = kind as u64 | (u64::from(fields.opc) << 8);
                    (
                        slow::fp_fused as *const () as usize,
                        vec![first[0], second[0], third[0]],
                        shape,
                        ColdFpOutput::Vector,
                    )
                }
                Instruction::CompareRegister(_)
                | Instruction::CompareZero(_)
                | Instruction::ConditionalCompare(_) => {
                    let kind = match instruction {
                        Instruction::CompareRegister(_) => slow::FpCompareKind::Register,
                        Instruction::CompareZero(_) => slow::FpCompareKind::Zero,
                        Instruction::ConditionalCompare(_) => slow::FpCompareKind::Conditional,
                        _ => unreachable!(),
                    };
                    let shape = kind as u64
                        | (u64::from(fields.opc) << 8)
                        | (u64::from(fields.signaling_compare) << 16)
                        | (u64::from(fields.condition) << 24)
                        | (u64::from(fields.nzcv_immediate) << 32);
                    let packed = self.packed_flags(flags);
                    extra.push(self.builder.ins().uextend(types::I64, packed));
                    (
                        slow::fp_compare as *const () as usize,
                        vec![first[0], second[0]],
                        shape,
                        ColdFpOutput::Flags,
                    )
                }
                _ => {
                    return Err(DirectJitError::unsupported(
                        "A64 FP instruction lacks an exact typed boundary",
                    ));
                }
            };
        let packed_result =
            self.call_typed_fp_cold(function, &operands, &extra, shape, source, flags)?;
        match output {
            ColdFpOutput::Vector => {
                let result =
                    self.load_context(types::I64, offset_of!(NativeContext, slow_result_low))?;
                let high =
                    self.load_context(types::I64, offset_of!(NativeContext, slow_result_high))?;
                let low = self.builder.ins().uextend(types::I128, result);
                let high = self.builder.ins().uextend(types::I128, high);
                let high = self.builder.ins().ishl_imm_u(high, 64);
                let value = self.builder.ins().bor(low, high);
                let value = self.vector_as(value, types::I8X16);
                self.write_vector(fields.rd, value)?;
            }
            ColdFpOutput::General if fields.rd != 31 => {
                let result =
                    self.load_context(types::I64, offset_of!(NativeContext, slow_result_low))?;
                self.write_register(fields.rd, result)?;
            }
            ColdFpOutput::General => {}
            ColdFpOutput::Flags => {
                let packed = self.builder.ins().ushr_imm_u(packed_result, 32);
                let packed = self.builder.ins().ireduce(types::I32, packed);
                *flags = LazyFlags::Packed(packed);
            }
        }
        Ok(())
    }

    fn call_typed_fp_cold(
        &mut self,
        function: usize,
        operands: &[Value],
        extra: &[Value],
        shape: u64,
        source: GuestVirtualAddress,
        flags: &LazyFlags,
    ) -> Result<Value, DirectJitError> {
        self.publish_fpsr_state();
        let shape = self.builder.ins().iconst(types::I64, shape as i64);
        let boundary = self.cold_calls.fp(operands.len(), extra.len())?;
        let callee = self.builder.ins().iconst(types::I64, boundary as i64);
        let target = self.builder.ins().iconst(types::I64, function as i64);
        let mut signature = Signature::new(self.call_conv);
        signature.params.push(AbiParam::new(types::I64));
        signature.params.push(AbiParam::new(types::I64));
        for argument in operands.iter().chain(extra).chain([&shape]) {
            signature
                .params
                .push(AbiParam::new(self.builder.func.dfg.value_type(*argument)));
        }
        signature.returns.push(AbiParam::new(types::I32));
        let signature = self.builder.import_signature(signature);
        let mut call_arguments = Vec::with_capacity(operands.len() + extra.len() + 3);
        call_arguments.push(self.context);
        call_arguments.push(target);
        call_arguments.extend_from_slice(operands);
        call_arguments.extend_from_slice(extra);
        call_arguments.push(shape);
        let call = self
            .builder
            .ins()
            .call_indirect(signature, callee, &call_arguments);
        let status = self.builder.inst_results(call)[0];
        let success = self.builder.create_block();
        let failed = self.cold_block();
        let succeeded = self.builder.ins().icmp_imm_s(IntCC::Equal, status, 0);
        self.builder
            .ins()
            .brif(succeeded, success, &[], failed, &[]);
        self.builder.switch_to_block(failed);
        self.commit_state(source, flags)?;
        self.dispatch_fp_failure(status, source);
        self.builder.switch_to_block(success);
        let packed = self.load_context(types::I64, offset_of!(NativeContext, slow_result_flags))?;
        let fpsr = self.builder.ins().ireduce(types::I32, packed);
        self.builder.def_var(self.fpsr_state, fpsr);
        self.block_dirty_fpsr = true;
        Ok(packed)
    }
}

#[derive(Clone, Copy)]
enum ColdFpOutput {
    Vector,
    General,
    Flags,
}

fn bitcast_flags() -> MemFlagsData {
    MemFlagsData::new().with_endianness(Endianness::Little)
}

fn integer_lane_type(bits: u32) -> Result<cranelift_codegen::ir::Type, DirectJitError> {
    match bits {
        8 => Ok(types::I8),
        16 => Ok(types::I16),
        32 => Ok(types::I32),
        64 => Ok(types::I64),
        _ => Err(DirectJitError::invalid("invalid SIMD lane width")),
    }
}

fn vector_type(
    lane: cranelift_codegen::ir::Type,
    lane_bits: u32,
) -> Result<cranelift_codegen::ir::Type, DirectJitError> {
    lane.by(128 / lane_bits)
        .ok_or_else(|| DirectJitError::unsupported("host CLIF lacks required SIMD shape"))
}

fn cast_integer(
    builder: &mut cranelift_frontend::FunctionBuilder<'_>,
    value: Value,
    ty: cranelift_codegen::ir::Type,
    signed: bool,
) -> Value {
    let from = builder.func.dfg.value_type(value);
    if from == ty {
        value
    } else if from.bits() > ty.bits() {
        builder.ins().ireduce(ty, value)
    } else if signed {
        builder.ins().sextend(ty, value)
    } else {
        builder.ins().uextend(ty, value)
    }
}

fn reduce_integer(
    builder: &mut cranelift_frontend::FunctionBuilder<'_>,
    value: Value,
    size: MemoryAccessSize,
) -> Value {
    let ty = match size {
        MemoryAccessSize::Byte => types::I8,
        MemoryAccessSize::Halfword => types::I16,
        MemoryAccessSize::Word => types::I32,
        MemoryAccessSize::Doubleword => types::I64,
        MemoryAccessSize::Quadword => types::I128,
    };
    if builder.func.dfg.value_type(value) == ty {
        value
    } else {
        builder.ins().ireduce(ty, value)
    }
}

fn scalar_width(opc: u8) -> Result<u32, DirectJitError> {
    match opc {
        0 => Ok(32),
        1 => Ok(64),
        3 => Ok(16),
        _ => Err(DirectJitError::invalid("invalid scalar FP width")),
    }
}

fn expand_modified_immediate(
    cmode: u8,
    immediate: u8,
    operation_bit: bool,
) -> Result<u64, DirectJitError> {
    let immediate = u64::from(immediate);
    let value = match cmode {
        0..=7 => {
            let lane = immediate << ((cmode >> 1) * 8);
            lane | lane << 32
        }
        8..=11 => {
            let lane = immediate << (((cmode >> 1) & 1) * 8);
            lane | lane << 16 | lane << 32 | lane << 48
        }
        12 => {
            let lane = immediate << 8 | 0xff;
            lane | lane << 32
        }
        13 => {
            let lane = immediate << 16 | 0xffff;
            lane | lane << 32
        }
        14 if !operation_bit => immediate * 0x0101_0101_0101_0101,
        14 => {
            let mut result = 0;
            for bit in 0..8 {
                if immediate & (1 << bit) != 0 {
                    result |= 0xff << (bit * 8);
                }
            }
            result
        }
        _ => return Err(DirectJitError::invalid("invalid SIMD modified immediate")),
    };
    Ok(if operation_bit && cmode != 14 {
        !value
    } else {
        value
    })
}

fn expand_vfp_immediate(immediate: u8, exponent_bits: u32, fraction_bits: u32) -> u64 {
    let sign = u64::from(immediate >> 7);
    let control = u64::from((immediate >> 6) & 1);
    let tail = u64::from((immediate >> 4) & 3);
    let fraction = u64::from(immediate & 0xf);
    let sign_shift = exponent_bits + fraction_bits;
    let repeated = if control == 0 {
        0
    } else {
        (1_u64 << (exponent_bits - 3)) - 1
    };
    sign << sign_shift
        | (control ^ 1) << (sign_shift - 1)
        | repeated << (fraction_bits + 2)
        | tail << fraction_bits
        | fraction << (fraction_bits - 4)
}

const fn sign_extend(value: u64, bits: u8) -> i64 {
    ((value << (64 - bits)) as i64) >> (64 - bits)
}
