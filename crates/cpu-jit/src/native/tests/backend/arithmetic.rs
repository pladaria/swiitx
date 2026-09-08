use super::*;
use nixe_cpu::semantics::{
    arithmetic::{add_with_carry, subtract_with_carry},
    bits::BitWidth,
};

#[test]
fn arithmetic_recipes_use_final_ssa_operand_locations() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for allocator in ["single_pass", "backtracking"] {
            for width in [32, 64] {
                for kind in 0..4 {
                    for count in [4, 31] {
                        let operands = operands(count);
                        let mut function = ir::Function::new();
                        let sig = function.import_signature(signature(&operands));
                        let block = function.dfg.make_block();
                        function.layout.append_block(block);
                        let mut cursor = FuncCursor::new(&mut function).at_bottom(block);
                        let inst = cursor.ins().nixe_entry(sig, 10);
                        let mut values = cursor.func.dfg.inst_results(inst).to_vec();
                        let ty = if width == 32 { types::I32 } else { types::I64 };
                        let (lhs, rhs) = if width == 32 {
                            (
                                cursor.ins().ireduce(ty, values[0]),
                                cursor.ins().ireduce(ty, values[2]),
                            )
                        } else {
                            (values[0], values[2])
                        };
                        let carry = cursor.ins().ireduce(types::I8, values[4]);
                        let result = match kind {
                            0 => cursor.ins().iadd(lhs, rhs),
                            1 => cursor.ins().isub(lhs, rhs),
                            2 => {
                                let sum = cursor.ins().iadd(lhs, rhs);
                                let carry = cursor.ins().uextend(ty, carry);
                                cursor.ins().iadd(sum, carry)
                            }
                            _ => {
                                let difference = cursor.ins().isub(lhs, rhs);
                                let borrow = cursor.ins().bxor_imm_u(carry, 1);
                                let borrow = cursor.ins().uextend(ty, borrow);
                                cursor.ins().isub(difference, borrow)
                            }
                        };
                        // W results zero the upper half of the architectural X.
                        values[6] = if width == 32 {
                            cursor.ins().uextend(types::I64, result)
                        } else {
                            result
                        };
                        let recipe_start = values.len();
                        values.extend([lhs, rhs, carry, result]);
                        cursor.ins().nixe_exit(20, &values);
                        let code = compile(abi, allocator, function);
                        let input = boundary(abi, &code, 10);
                        let output = boundary(abi, &code, 20);
                        // X3 is overwritten before any read. Do not demand an
                        // input that optimization is entitled to eliminate.
                        let live_inputs: Vec<_> = operands
                            .iter()
                            .copied()
                            .filter(|(value, _)| *value != GuestValue::General(3))
                            .collect();
                        let ingress = EntryContract {
                            abi,
                            live_in: live(&live_inputs),
                            bindings: input.bindings(&live_inputs).unwrap(),
                            nzcv: NzcvLocation::Packed(
                                input.location(operands.len(), types::I32).unwrap(),
                            ),
                        };
                        ingress.validate().unwrap();
                        let mut state = exit(abi, &output, &operands, 2, false);
                        let lhs = output.location(recipe_start, ty).unwrap();
                        let rhs = output.location(recipe_start + 1, ty).unwrap();
                        let carry = output.location(recipe_start + 2, types::I8).unwrap();
                        let result = output.location(recipe_start + 3, ty).unwrap();
                        state.nzcv = NzcvLocation::Deferred(match kind {
                            0 => LazyFlags::Add {
                                lhs,
                                rhs,
                                result,
                                width,
                            },
                            1 => LazyFlags::Subtract {
                                lhs,
                                rhs,
                                result,
                                width,
                            },
                            2 => LazyFlags::AddCarry {
                                lhs,
                                rhs,
                                carry,
                                result,
                                width,
                            },
                            _ => LazyFlags::SubtractCarry {
                                lhs,
                                rhs,
                                carry,
                                result,
                                width,
                            },
                        });
                        state.dirty_live = state.live;
                        // This fragment performs no FP operation, but the
                        // gateway must still restore the caller environment.
                        state.host_fpsr_pending = false;
                        if !canonical::native(abi) {
                            continue;
                        }
                        let frame_extent = extent(&code);
                        let unit = publish_unlinked(abi, code, ingress, state);
                        let bit_width = BitWidth::new(width).unwrap();
                        let sign = 1u128 << (width - 1);
                        for (a, b, c) in [
                            (0, 0, false),
                            (sign - 1, 0, true),
                            (bit_width.mask(), 1, true),
                            (sign, 1, false),
                        ] {
                            let mut seed = 119;
                            let mut actual = A64State::default();
                            for value in actual.general_register_storage_mut() {
                                *value = next(&mut seed);
                            }
                            for value in actual.vector_register_storage_mut() {
                                *value =
                                    u128::from(next(&mut seed)) << 64 | u128::from(next(&mut seed));
                            }
                            actual.general_register_storage_mut()[0] = a as u64;
                            actual.general_register_storage_mut()[1] = b as u64;
                            actual.general_register_storage_mut()[2] = u64::from(c);
                            actual.set_fpsr(1 << 27);
                            actual.set_nzcv(Nzcv::from_bits(0xf0000000));
                            let mut expected = actual.clone();
                            let result = if kind % 2 == 0 {
                                add_with_carry(a, b, kind == 2 && c, bit_width)
                            } else {
                                subtract_with_carry(a, b, kind == 1 || c, bit_width)
                            };
                            expected.general_register_storage_mut()[3] = result.result as u64;
                            expected.set_nzcv(Nzcv::from_bits(
                                ((result.result >> (width - 1)) as u32) << 31
                                    | u32::from(result.result == 0) << 30
                                    | u32::from(result.carry_out) << 29
                                    | u32::from(result.overflow) << 28,
                            ));
                            expected.set_pc(0x12345678);
                            run_unlinked(&unit, &mut actual, frame_extent);
                            assert_eq!(
                                actual, expected,
                                "{abi:?}/{allocator}/width={width}/kind={kind}/count={count}/a={a:x}/b={b:x}/c={c}"
                            );
                        }
                        unit.shutdown();
                    }
                }
            }
        }
    }
}
