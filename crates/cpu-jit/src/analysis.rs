//! Shared A64 use/def and observation-aware liveness for both JIT tiers.
//! Values are whole X/SP and V registers; partial writes read the preserved
//! destination before defining the new value. W and scalar SIMD writes that
//! zero the upper bits define the whole register. NZCV is tracked per bit.
//! PC is supplied by the instruction/exit identity, not treated as a cached SSA
//! register. Exclusive-monitor and memory state remain canonical runtime state.

use crate::fp_policy::fp_lowering_disposition;
use nixe_cpu::decode::a64::{A64Instruction, control, fp_simd, integer, memory, system};
use nixe_cpu::semantics::a64::{
    SimdMemoryMode, simd_multiple_structure_shape, simd_single_structure_shape,
};

// Effects mirror the existing family emitters and shared Arm semantics:
// https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions
// https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions

const GENERAL_REGISTER_COUNT: usize = 31;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IntegerRegisterSet {
    pub x: [bool; GENERAL_REGISTER_COUNT],
    pub sp: bool,
}

/// Architectural flag bits, in NZCV nibble order.
pub const N: u8 = 8;
pub const Z: u8 = 4;
pub const C: u8 = 2;
pub const V: u8 = 1;
pub const NZCV: u8 = N | Z | C | V;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StateSet {
    pub integer: IntegerRegisterSet,
    pub vector: [bool; 32],
    pub nzcv: u8,
    pub fpcr: bool,
    pub fpsr: bool,
    pub tpidr_el0: bool,
    pub tpidrro_el0: bool,
}

impl StateSet {
    pub const ALL: Self = Self {
        integer: IntegerRegisterSet {
            x: [true; 31],
            sp: true,
        },
        vector: [true; 32],
        nzcv: NZCV,
        fpcr: true,
        fpsr: true,
        tpidr_el0: true,
        tpidrro_el0: true,
    };

    pub fn union(mut self, other: Self) -> Self {
        for (a, b) in self.integer.x.iter_mut().zip(other.integer.x) {
            *a |= b;
        }
        self.integer.sp |= other.integer.sp;
        for (a, b) in self.vector.iter_mut().zip(other.vector) {
            *a |= b;
        }
        self.nzcv |= other.nzcv;
        self.fpcr |= other.fpcr;
        self.fpsr |= other.fpsr;
        self.tpidr_el0 |= other.tpidr_el0;
        self.tpidrro_el0 |= other.tpidrro_el0;
        self
    }

    pub fn without(mut self, other: Self) -> Self {
        for (a, b) in self.integer.x.iter_mut().zip(other.integer.x) {
            *a &= !b;
        }
        self.integer.sp &= !other.integer.sp;
        for (a, b) in self.vector.iter_mut().zip(other.vector) {
            *a &= !b;
        }
        self.nzcv &= !other.nzcv;
        self.fpcr &= !other.fpcr;
        self.fpsr &= !other.fpsr;
        self.tpidr_el0 &= !other.tpidr_el0;
        self.tpidrro_el0 &= !other.tpidrro_el0;
        self
    }

    pub fn intersection(self, other: Self) -> Self {
        self.without(Self::ALL.without(other))
    }
    pub fn is_empty(self) -> bool {
        self == Self::default()
    }
}

/// State consumed/defined by the instruction and observable on its side exits.
/// A typed helper's architectural arguments/results use the same reads/writes;
/// a nonretry fault observes the PRE-instruction state, before any destination
/// write. Runtime exits after successful execution observe the POST-state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InstructionEffects {
    pub reads: StateSet,
    pub writes: StateSet,
    pub observe_before: StateSet,
    pub observe_after: StateSet,
}

impl InstructionEffects {
    pub fn live_before(self, live_after: StateSet) -> StateSet {
        live_after
            .union(self.observe_after)
            .without(self.writes)
            .union(self.reads)
            .union(self.observe_before)
    }
}

/// Which flags the architectural condition actually consumes (AL/NV: none).
/// Matches nixe_cpu::semantics::conditions::evaluate_a64.
pub const fn condition_flags(condition: u8) -> u8 {
    match (condition & 15) >> 1 {
        0 => Z,
        1 => C,
        2 => N,
        3 => V,
        4 => C | Z,
        5 => N | V,
        6 => N | Z | V,
        _ => 0,
    }
}

pub fn instruction_effects(instruction: A64Instruction) -> InstructionEffects {
    let mut effect = InstructionEffects::default();
    let (read, write) = (&mut effect.reads, &mut effect.writes);
    match instruction {
        A64Instruction::Integer(inst) => {
            register_access_integer(inst, &mut read.integer, &mut write.integer);
            let f = inst.operands();
            match inst {
                integer::Instruction::AddSubImmediate(_)
                | integer::Instruction::AddSubExtended(_)
                | integer::Instruction::AddSubShifted(_)
                | integer::Instruction::AddSubCarry(_) => {
                    if f.set_flags {
                        write.nzcv = NZCV;
                    }
                    if matches!(inst, integer::Instruction::AddSubCarry(_)) {
                        read.nzcv = C;
                    }
                }
                integer::Instruction::LogicalImmediate(_)
                | integer::Instruction::LogicalShifted(_) => {
                    if f.subtract && f.set_flags {
                        write.nzcv = NZCV;
                    }
                }
                integer::Instruction::ConditionalCompareImmediate(_)
                | integer::Instruction::ConditionalCompareRegister(_) => {
                    read.nzcv = condition_flags(f.condition);
                    write.nzcv = NZCV;
                }
                integer::Instruction::ConditionalSelect(_) => {
                    read.nzcv = condition_flags(f.condition)
                }
                _ => {}
            }
        }
        A64Instruction::Memory(inst) => {
            register_access_memory(inst, &mut read.integer, &mut write.integer);
            effect.observe_before = StateSet::ALL;
        }
        A64Instruction::FpSimd(inst) => {
            register_access_fp_simd_general(inst, &mut read.integer, &mut write.integer);
            register_access_fp_simd_vector(inst, &mut read.vector, &mut write.vector);
            let f = inst.operands();
            if fp_lowering_disposition(inst).accesses_status() {
                read.fpcr = true;
                read.fpsr = true;
                write.fpsr = true;
                // A trapped exact FP helper must reconstruct the prefault state.
                effect.observe_before = StateSet::ALL;
            }
            match inst {
                fp_simd::Instruction::CompareRegister(_) | fp_simd::Instruction::CompareZero(_) => {
                    write.nzcv = NZCV
                }
                fp_simd::Instruction::ConditionalCompare(_) => {
                    read.nzcv = condition_flags(f.condition);
                    write.nzcv = NZCV;
                }
                fp_simd::Instruction::ScalarFloatConditionalSelect(_) => {
                    read.nzcv = condition_flags(f.condition)
                }
                fp_simd::Instruction::MemoryUnsigned(_)
                | fp_simd::Instruction::MemoryUnscaled(_)
                | fp_simd::Instruction::MemoryPostIndex(_)
                | fp_simd::Instruction::MemoryPreIndex(_)
                | fp_simd::Instruction::MemoryRegister(_)
                | fp_simd::Instruction::MemoryPair(_)
                | fp_simd::Instruction::MemoryMultipleStructures(_)
                | fp_simd::Instruction::MemoryMultipleStructuresPostIndex(_)
                | fp_simd::Instruction::MemorySingleStructure(_)
                | fp_simd::Instruction::MemorySingleStructurePostIndex(_) => {
                    effect.observe_before = StateSet::ALL
                }
                _ => {}
            }
        }
        A64Instruction::System(inst) => {
            register_access_system(inst, &mut read.integer, &mut write.integer);
            let f = inst.operands();
            match inst {
                system::Instruction::ReadRegister(_) => match f.system_key {
                    0xd53b_4200 => read.nzcv = NZCV,
                    0xd53b_4400 => read.fpcr = true,
                    0xd53b_4420 => read.fpsr = true,
                    0xd53b_d040 => read.tpidr_el0 = true,
                    0xd53b_d060 => read.tpidrro_el0 = true,
                    // Runtime-register helpers use a canonical failure exit.
                    // Unknown system operands must likewise retain precise state.
                    _ => effect.observe_before = StateSet::ALL,
                },
                system::Instruction::WriteRegister(_) => match f.system_key {
                    0xd51b_4200 => write.nzcv = NZCV,
                    0xd51b_4400 => {
                        write.fpcr = true;
                        effect.observe_after = StateSet::ALL;
                    }
                    0xd51b_4420 => write.fpsr = true,
                    0xd51b_d040 => write.tpidr_el0 = true,
                    _ => effect.observe_before = StateSet::ALL,
                },
                // Runtime hints/cache maintenance may schedule, invalidate or
                // reject the operation; use the architectural state at this PC.
                system::Instruction::Hint(_) | system::Instruction::System(_) => {
                    if !matches!(inst, system::Instruction::Hint(_))
                        || !matches!(f.hint, 0 | 32 | 34 | 36 | 38)
                    {
                        effect.observe_before = StateSet::ALL;
                    }
                }
                system::Instruction::Barrier(_) | system::Instruction::ClearExclusive(_) => {
                    // The current typed call boundary has a canonical failure
                    // edge even when the operation normally cannot fail.
                    effect.observe_before = StateSet::ALL;
                }
            }
        }
        A64Instruction::Control(inst) => {
            let f = inst.operands();
            match inst {
                control::Instruction::BranchLinkImmediate(_) => {
                    mark_write(&mut read.integer, &mut write.integer, 30, false)
                }
                control::Instruction::BranchRegister(_) => {
                    mark_read(&mut read.integer, f.rn, false);
                    if f.branch_register_key == 0xd63f_0000 {
                        mark_write(&mut read.integer, &mut write.integer, 30, false);
                    }
                }
                control::Instruction::CompareBranch(_) | control::Instruction::TestBranch(_) => {
                    mark_read(&mut read.integer, f.rd, false)
                }
                control::Instruction::ConditionalBranch(_) => {
                    read.nzcv = condition_flags(f.condition)
                }
                control::Instruction::SupervisorCall(_) | control::Instruction::Breakpoint(_) => {
                    effect.observe_before = StateSet::ALL
                }
                control::Instruction::Nop(_) | control::Instruction::BranchImmediate(_) => {}
            }
        }
        A64Instruction::RecognizedUnsupported { .. } => effect.observe_before = StateSet::ALL,
    }
    effect
}

/// Summary in execution order, not the union of reads: a write can kill the
/// incoming value before a later use. Side observations cannot be killed by
/// later writes. This is analysis data, not another instruction representation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BlockEffects {
    pub reads: StateSet,
    pub writes: StateSet,
    pub live_in: StateSet,
}

impl BlockEffects {
    pub fn push(&mut self, effects: InstructionEffects) {
        self.live_in = self.live_in.union(
            effects
                .reads
                .union(effects.observe_before)
                .without(self.writes),
        );
        self.writes = self.writes.union(effects.writes);
        self.live_in = self
            .live_in
            .union(effects.observe_after.without(self.writes));
        self.reads = self.reads.union(effects.reads);
    }

    pub fn live_before(self, live_out: StateSet) -> StateSet {
        self.live_in.union(live_out.without(self.writes))
    }
}

/// Successor indexes refer to this invocation's block slice. An external edge
/// supplies its actual observation/target contract in exit_live; unresolved or
/// canonical exits use ALL. Multiple selected entries share this same solution.
pub struct FlowBlock<'a> {
    pub effects: BlockEffects,
    pub successors: &'a [usize],
    pub exit_live: StateSet,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BlockLiveness {
    pub live_in: StateSet,
    pub live_out: StateSet,
}

pub fn liveness(blocks: &[FlowBlock<'_>]) -> Vec<BlockLiveness> {
    let mut result = vec![BlockLiveness::default(); blocks.len()];
    loop {
        let mut changed = false;
        for (index, block) in blocks.iter().enumerate().rev() {
            let mut live_out = block.exit_live;
            for &successor in block.successors {
                live_out = live_out.union(result[successor].live_in);
            }
            let next = BlockLiveness {
                live_in: block.effects.live_before(live_out),
                live_out,
            };
            changed |= next != result[index];
            result[index] = next;
        }
        if !changed {
            return result;
        }
    }
}

fn register_access_memory(
    instruction: memory::Instruction,
    accessed: &mut IntegerRegisterSet,
    dirty: &mut IntegerRegisterSet,
) {
    use nixe_cpu::semantics::a64::{ScalarTransfer, pair_transfer, scalar_transfer};

    let fields = instruction.operands();
    if !matches!(instruction, memory::Instruction::Literal(_)) {
        mark_read(accessed, fields.rn, true);
    }
    if matches!(instruction, memory::Instruction::Register(_)) {
        mark_read(accessed, fields.rm, false);
    }
    match instruction {
        memory::Instruction::Literal(_) => mark_write(accessed, dirty, fields.rt, false),
        memory::Instruction::Unsigned(_)
        | memory::Instruction::Unscaled(_)
        | memory::Instruction::PostIndex(_)
        | memory::Instruction::PreIndex(_)
        | memory::Instruction::Register(_) => {
            match scalar_transfer(
                fields.opc,
                nixe_cpu::semantics::a64::memory_size(fields.size),
            ) {
                Some(ScalarTransfer::Store) => mark_read(accessed, fields.rt, false),
                Some(ScalarTransfer::Load(_)) => mark_write(accessed, dirty, fields.rt, false),
                None => {}
            }
            if matches!(
                instruction,
                memory::Instruction::PostIndex(_) | memory::Instruction::PreIndex(_)
            ) {
                mark_write(accessed, dirty, fields.rn, true);
            }
        }
        memory::Instruction::Pair(_) => {
            if pair_transfer(fields.size, fields.load).is_some() {
                if fields.load {
                    mark_write(accessed, dirty, fields.rt, false);
                    mark_write(accessed, dirty, fields.rt2, false);
                } else {
                    mark_read(accessed, fields.rt, false);
                    mark_read(accessed, fields.rt2, false);
                }
                if matches!(fields.mode, 1 | 3) {
                    mark_write(accessed, dirty, fields.rn, true);
                }
            }
        }
        memory::Instruction::LoadAcquire(_) | memory::Instruction::LoadExclusive(_) => {
            mark_write(accessed, dirty, fields.rt, false);
        }
        memory::Instruction::LoadExclusivePair(_) => {
            mark_write(accessed, dirty, fields.rt, false);
            mark_write(accessed, dirty, fields.rt2, false);
        }
        memory::Instruction::StoreRelease(_) => mark_read(accessed, fields.rt, false),
        memory::Instruction::StoreExclusive(_) => {
            mark_read(accessed, fields.rt, false);
            mark_write(accessed, dirty, fields.rm, false);
        }
        memory::Instruction::StoreExclusivePair(_) => {
            mark_read(accessed, fields.rt, false);
            mark_read(accessed, fields.rt2, false);
            mark_write(accessed, dirty, fields.rm, false);
        }
        memory::Instruction::AtomicReadModifyWrite(_) => {
            mark_read(accessed, fields.rm, false);
            mark_write(accessed, dirty, fields.rt, false);
        }
        memory::Instruction::CompareAndSwap(_) => {
            mark_read(accessed, fields.rm, false);
            mark_read(accessed, fields.rt, false);
            mark_write(accessed, dirty, fields.rm, false);
        }
        memory::Instruction::CompareAndSwapPair(_) => {
            mark_read(accessed, fields.rm, false);
            mark_read(accessed, fields.rm.wrapping_add(1), false);
            mark_read(accessed, fields.rt, false);
            mark_read(accessed, fields.rt.wrapping_add(1), false);
            mark_write(accessed, dirty, fields.rm, false);
            mark_write(accessed, dirty, fields.rm.wrapping_add(1), false);
        }
    }
}

fn register_access_system(
    instruction: system::Instruction,
    accessed: &mut IntegerRegisterSet,
    dirty: &mut IntegerRegisterSet,
) {
    let fields = instruction.operands();
    match instruction {
        system::Instruction::ReadRegister(_) => mark_write(accessed, dirty, fields.rt, false),
        system::Instruction::WriteRegister(_) => mark_read(accessed, fields.rt, false),
        system::Instruction::System(_) if fields.system_key != 0xd508_7500 => {
            mark_read(accessed, fields.rt, false);
        }
        system::Instruction::Hint(_)
        | system::Instruction::Barrier(_)
        | system::Instruction::ClearExclusive(_)
        | system::Instruction::System(_) => {}
    }
}

fn register_access_fp_simd_general(
    instruction: fp_simd::Instruction,
    accessed: &mut IntegerRegisterSet,
    dirty: &mut IntegerRegisterSet,
) {
    let fields = instruction.operands();
    match instruction {
        fp_simd::Instruction::DuplicateGeneral(_)
        | fp_simd::Instruction::InsertGeneral(_)
        | fp_simd::Instruction::MoveFromGeneral(_)
        | fp_simd::Instruction::SignedIntToFloat(_)
        | fp_simd::Instruction::UnsignedIntToFloat(_) => {
            mark_read(accessed, fields.rn, false);
        }
        fp_simd::Instruction::UnsignedMoveToGeneral(_)
        | fp_simd::Instruction::MoveToGeneral(_)
        | fp_simd::Instruction::FloatToSignedInt(_)
        | fp_simd::Instruction::FloatToUnsignedInt(_) => {
            mark_write(accessed, dirty, fields.rd, false);
        }
        fp_simd::Instruction::MemoryUnsigned(_)
        | fp_simd::Instruction::MemoryUnscaled(_)
        | fp_simd::Instruction::MemoryPostIndex(_)
        | fp_simd::Instruction::MemoryPreIndex(_)
        | fp_simd::Instruction::MemoryRegister(_)
        | fp_simd::Instruction::MemoryPair(_)
        | fp_simd::Instruction::MemoryMultipleStructures(_)
        | fp_simd::Instruction::MemoryMultipleStructuresPostIndex(_)
        | fp_simd::Instruction::MemorySingleStructure(_)
        | fp_simd::Instruction::MemorySingleStructurePostIndex(_) => {
            mark_read(accessed, fields.rn, true);
            if matches!(instruction, fp_simd::Instruction::MemoryRegister(_))
                || (matches!(
                    instruction,
                    fp_simd::Instruction::MemoryMultipleStructuresPostIndex(_)
                        | fp_simd::Instruction::MemorySingleStructurePostIndex(_)
                ) && fields.rm != 31)
            {
                mark_read(accessed, fields.rm, false);
            }
            if matches!(
                instruction,
                fp_simd::Instruction::MemoryPostIndex(_)
                    | fp_simd::Instruction::MemoryPreIndex(_)
                    | fp_simd::Instruction::MemoryMultipleStructuresPostIndex(_)
                    | fp_simd::Instruction::MemorySingleStructurePostIndex(_)
            ) || (matches!(instruction, fp_simd::Instruction::MemoryPair(_))
                && matches!(fields.mode, 1 | 3))
            {
                mark_write(accessed, dirty, fields.rn, true);
            }
        }
        _ => {}
    }
}

fn register_access_fp_simd_vector(
    instruction: fp_simd::Instruction,
    accessed: &mut [bool; 32],
    dirty: &mut [bool; 32],
) {
    let fields = instruction.operands();
    let read = |accessed: &mut [bool; 32], index: u8| {
        accessed[usize::from(index)] = true;
    };
    let write = |_accessed: &mut [bool; 32], dirty: &mut [bool; 32], index: u8| {
        dirty[usize::from(index)] = true;
    };
    match instruction {
        fp_simd::Instruction::UnsignedMoveToGeneral(_)
        | fp_simd::Instruction::MoveToGeneral(_)
        | fp_simd::Instruction::FloatToSignedInt(_)
        | fp_simd::Instruction::FloatToUnsignedInt(_)
        | fp_simd::Instruction::CompareZero(_) => read(accessed, fields.rn),
        fp_simd::Instruction::CompareRegister(_) | fp_simd::Instruction::ConditionalCompare(_) => {
            read(accessed, fields.rn);
            read(accessed, fields.rm);
        }
        fp_simd::Instruction::MemoryUnsigned(_)
        | fp_simd::Instruction::MemoryUnscaled(_)
        | fp_simd::Instruction::MemoryPostIndex(_)
        | fp_simd::Instruction::MemoryPreIndex(_)
        | fp_simd::Instruction::MemoryRegister(_) => {
            if fields.load {
                write(accessed, dirty, fields.rd);
            } else {
                read(accessed, fields.rd);
            }
        }
        fp_simd::Instruction::MemoryPair(_) => {
            for register in [fields.rd, fields.rt2] {
                if fields.load {
                    write(accessed, dirty, register);
                } else {
                    read(accessed, register);
                }
            }
        }
        fp_simd::Instruction::MemoryMultipleStructures(_)
        | fp_simd::Instruction::MemoryMultipleStructuresPostIndex(_) => {
            let shape = simd_multiple_structure_shape(fields)
                .expect("allocation validated the SIMD multiple-structure shape");
            for index in 0..shape.register_count() {
                let register = fields.rd.wrapping_add(index) & 31;
                if fields.load {
                    if shape.structure_registers > 1 {
                        read(accessed, register);
                    }
                    write(accessed, dirty, register);
                } else {
                    read(accessed, register);
                }
            }
        }
        fp_simd::Instruction::MemorySingleStructure(_)
        | fp_simd::Instruction::MemorySingleStructurePostIndex(_) => {
            let shape = simd_single_structure_shape(fields)
                .expect("allocation validated the SIMD single-structure shape");
            for index in 0..shape.register_count() {
                let register = fields.rd.wrapping_add(index) & 31;
                if !fields.load || matches!(shape.mode, SimdMemoryMode::Lane(_)) {
                    read(accessed, register);
                }
                if fields.load {
                    write(accessed, dirty, register);
                }
            }
        }
        fp_simd::Instruction::MoveFromGeneral(_)
        | fp_simd::Instruction::DuplicateGeneral(_)
        | fp_simd::Instruction::ModifiedImmediate(_)
        | fp_simd::Instruction::ScalarFloatImmediate(_)
        | fp_simd::Instruction::VectorFloatImmediate(_) => {
            if matches!(instruction, fp_simd::Instruction::ModifiedImmediate(_))
                && fields.cmode <= 11
                && fields.cmode & 1 != 0
                || matches!(instruction, fp_simd::Instruction::MoveFromGeneral(_))
                    && fields.size & 2 != 0
                    && fields.opc == 2
            {
                read(accessed, fields.rd);
            }
            write(accessed, dirty, fields.rd);
        }
        fp_simd::Instruction::InsertElement(_) => {
            read(accessed, fields.rn);
            read(accessed, fields.rd);
            write(accessed, dirty, fields.rd);
        }
        fp_simd::Instruction::InsertGeneral(_)
        | fp_simd::Instruction::ShiftRightNarrow(_)
        | fp_simd::Instruction::ExtractNarrow(_) => {
            if !matches!(instruction, fp_simd::Instruction::InsertGeneral(_)) {
                read(accessed, fields.rn);
            }
            if !matches!(instruction, fp_simd::Instruction::InsertGeneral(_)) && fields.vector_128 {
                read(accessed, fields.rd);
            }
            if matches!(instruction, fp_simd::Instruction::InsertGeneral(_)) {
                read(accessed, fields.rd);
            }
            write(accessed, dirty, fields.rd);
        }
        fp_simd::Instruction::Bitwise(_) => {
            read(accessed, fields.rn);
            read(accessed, fields.rm);
            if matches!(
                fields.bitwise_operation,
                Some(
                    fp_simd::BitwiseOperation::Select
                        | fp_simd::BitwiseOperation::InsertIfTrue
                        | fp_simd::BitwiseOperation::InsertIfFalse
                )
            ) {
                read(accessed, fields.rd);
            }
            write(accessed, dirty, fields.rd);
        }
        fp_simd::Instruction::Integer(_)
        | fp_simd::Instruction::IntegerCompare(_)
        | fp_simd::Instruction::IntegerPairwise(_)
        | fp_simd::Instruction::IntegerMinMax(_)
        | fp_simd::Instruction::PermuteTwoSource(_)
        | fp_simd::Instruction::Extract(_)
        | fp_simd::Instruction::VectorSignedShiftRegister(_)
        | fp_simd::Instruction::VectorUnsignedShiftRegister(_)
        | fp_simd::Instruction::VectorFloatDivide(_)
        | fp_simd::Instruction::VectorFloatMultiplyElement(_)
        | fp_simd::Instruction::ScalarFloatDivide(_)
        | fp_simd::Instruction::ScalarFloatAdd(_)
        | fp_simd::Instruction::ScalarFloatMultiply(_)
        | fp_simd::Instruction::ScalarFloatConditionalSelect(_) => {
            read(accessed, fields.rn);
            read(accessed, fields.rm);
            write(accessed, dirty, fields.rd);
        }
        fp_simd::Instruction::ScalarFloatFusedMultiplyAdd(_) => {
            read(accessed, fields.rn);
            read(accessed, fields.rm);
            read(accessed, fields.ra);
            write(accessed, dirty, fields.rd);
        }
        fp_simd::Instruction::ScalarMove(_)
        | fp_simd::Instruction::ScalarAbsolute(_)
        | fp_simd::Instruction::ScalarNegate(_)
        | fp_simd::Instruction::VectorFloatAbsolute(_)
        | fp_simd::Instruction::VectorFloatNegate(_)
        | fp_simd::Instruction::DuplicateElement(_)
        | fp_simd::Instruction::ScalarShiftRightImmediate(_)
        | fp_simd::Instruction::VectorShiftRightImmediate(_)
        | fp_simd::Instruction::ScalarShiftLeftImmediate(_)
        | fp_simd::Instruction::VectorShiftLeftImmediate(_)
        | fp_simd::Instruction::ShiftLeftLong(_)
        | fp_simd::Instruction::CountBits(_)
        | fp_simd::Instruction::AddAcrossVector(_)
        | fp_simd::Instruction::VectorSignedIntToFloat(_)
        | fp_simd::Instruction::VectorUnsignedIntToFloat(_)
        | fp_simd::Instruction::ScalarVectorSignedIntToFloat(_)
        | fp_simd::Instruction::ScalarVectorUnsignedIntToFloat(_)
        | fp_simd::Instruction::ScalarFloatConvert(_)
        | fp_simd::Instruction::ScalarFloatRound(_)
        | fp_simd::Instruction::ScalarFloatSquareRoot(_) => {
            read(accessed, fields.rn);
            write(accessed, dirty, fields.rd);
        }
        fp_simd::Instruction::SignedIntToFloat(_) | fp_simd::Instruction::UnsignedIntToFloat(_) => {
            write(accessed, dirty, fields.rd);
        }
    }
}

fn register_access_integer(
    instruction: integer::Instruction,
    accessed: &mut IntegerRegisterSet,
    dirty: &mut IntegerRegisterSet,
) {
    let fields = instruction.operands();
    match instruction {
        integer::Instruction::MoveWide(_) => {
            let opcode = u8::from(fields.subtract) * 2 + u8::from(fields.set_flags);
            if opcode == 3 {
                mark_read(accessed, fields.rd, false);
            }
            mark_write(accessed, dirty, fields.rd, false);
        }
        integer::Instruction::AddSubImmediate(_) => {
            mark_read(accessed, fields.rn, true);
            mark_write(accessed, dirty, fields.rd, !fields.set_flags);
        }
        integer::Instruction::AddSubExtended(_) => {
            mark_read(accessed, fields.rn, true);
            mark_read(accessed, fields.rm, false);
            mark_write(accessed, dirty, fields.rd, !fields.set_flags);
        }
        integer::Instruction::AddSubShifted(_) | integer::Instruction::AddSubCarry(_) => {
            mark_read(accessed, fields.rn, false);
            mark_read(accessed, fields.rm, false);
            mark_write(accessed, dirty, fields.rd, false);
        }
        integer::Instruction::LogicalImmediate(_) => {
            mark_read(accessed, fields.rn, false);
            mark_write(accessed, dirty, fields.rd, false);
        }
        integer::Instruction::LogicalShifted(_)
        | integer::Instruction::Extract(_)
        | integer::Instruction::TwoSource(_)
        | integer::Instruction::ConditionalSelect(_) => {
            mark_read(accessed, fields.rn, false);
            mark_read(accessed, fields.rm, false);
            mark_write(accessed, dirty, fields.rd, false);
        }
        integer::Instruction::Bitfield(_) => {
            mark_read(accessed, fields.rn, false);
            if u8::from(fields.subtract) * 2 + u8::from(fields.set_flags) == 1 {
                mark_read(accessed, fields.rd, false);
            }
            mark_write(accessed, dirty, fields.rd, false);
        }
        integer::Instruction::ConditionalCompareRegister(_) => {
            mark_read(accessed, fields.rn, false);
            mark_read(accessed, fields.rm, false);
        }
        integer::Instruction::ConditionalCompareImmediate(_) => {
            mark_read(accessed, fields.rn, false);
        }
        integer::Instruction::ThreeSource(_) => {
            mark_read(accessed, fields.rn, false);
            mark_read(accessed, fields.rm, false);
            if !matches!(fields.opcode_3, 2 | 6) {
                mark_read(accessed, fields.ra, false);
            }
            mark_write(accessed, dirty, fields.rd, false);
        }
        integer::Instruction::OneSource(_) => {
            mark_read(accessed, fields.rn, false);
            mark_write(accessed, dirty, fields.rd, false);
        }
        integer::Instruction::Adr(_) | integer::Instruction::Adrp(_) => {
            mark_write(accessed, dirty, fields.rd, false);
        }
    }
}

fn mark_read(accessed: &mut IntegerRegisterSet, index: u8, register31_is_sp: bool) {
    if index == 31 {
        accessed.sp |= register31_is_sp;
    } else {
        accessed.x[usize::from(index)] = true;
    }
}

fn mark_write(
    _accessed: &mut IntegerRegisterSet,
    dirty: &mut IntegerRegisterSet,
    index: u8,
    register31_is_sp: bool,
) {
    if index == 31 {
        dirty.sp |= register31_is_sp;
    } else {
        let slot = usize::from(index);
        dirty.x[slot] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_cpu::decode::{self, DecodeResult};
    use nixe_cpu::location::{InstructionEncoding, LocationDescriptor};
    use nixe_cpu::platform::{PlatformDecoder, TargetPlatform};
    use nixe_memory::GuestVirtualAddress;

    fn instruction(bits: u32) -> A64Instruction {
        let encoding = InstructionEncoding::from_u32(bits);
        let platform = TargetPlatform::Switch1;
        let location =
            LocationDescriptor::new(GuestVirtualAddress::new(0x1000), platform.profile_id());
        let DecodeResult::Decoded(decoded) =
            decode::decode(PlatformDecoder::new(platform), location, encoding)
        else {
            panic!("invalid fixture {bits:08x}")
        };
        decode::a64::normalize(&decoded.instruction, encoding)
    }
    fn effects(bits: u32) -> InstructionEffects {
        instruction_effects(instruction(bits))
    }
    fn block(bits: &[u32]) -> BlockEffects {
        let mut block = BlockEffects::default();
        for &bits in bits {
            block.push(effects(bits));
        }
        block
    }
    fn x(index: usize) -> StateSet {
        let mut state = StateSet::default();
        state.integer.x[index] = true;
        state
    }

    #[test]
    fn full_and_partial_integer_writes_have_distinct_inputs() {
        let movz_w0 = effects(0x5280_0020);
        assert_eq!(movz_w0.writes, x(0));
        assert!(
            movz_w0.reads.is_empty(),
            "W writes kill the complete X value"
        );
        let movk_x0 = effects(0xf280_0020);
        assert_eq!(movk_x0.reads, x(0));
        assert_eq!(movk_x0.writes, x(0));
        let sequence = block(&[0x5280_0020, 0xf280_0020]);
        assert!(
            !sequence.live_before(x(0)).integer.x[0],
            "MOVZ defines the preserved part before MOVK"
        );
        assert!(block(&[0xf280_0020]).live_before(x(0)).integer.x[0]);
        assert!(effects(0xd280_003f).writes.is_empty(), "XZR is not state");
    }

    #[test]
    fn vector_lane_inserts_read_only_the_preserved_destination() {
        let insert = effects(0x4e18_1c20); // INS V0.D[1], X1
        assert!(insert.reads.integer.x[1]);
        assert!(insert.reads.vector[0]);
        assert!(!insert.reads.vector[1], "the source is X1, not V1");
        assert!(insert.writes.vector[0]);
        let fmov = effects(0x1e27_0020); // FMOV S0, W1 clears the upper V bits
        assert!(fmov.writes.vector[0]);
        assert!(!fmov.reads.vector[0]);
        let and = effects(0x4e22_1c20); // AND V0.16B, V1.16B, V2.16B
        assert!(!and.reads.vector[0]);
        assert!(and.reads.vector[1] && and.reads.vector[2]);
    }

    #[test]
    fn flags_track_only_consumed_bits_and_survive_infrastructure() {
        assert_eq!(effects(0x5400_0000).reads.nzcv, Z); // B.EQ
        assert_eq!(effects(0xba02_0020).reads.nzcv, C); // ADCS
        assert_eq!(effects(0xba02_0020).writes.nzcv, NZCV);
        assert_eq!(condition_flags(14), 0);
        assert_eq!(condition_flags(15), 0);
        assert_eq!(condition_flags(12), N | Z | V);
        for condition in 0..16 {
            let decoded = nixe_cpu::semantics::conditions::Condition::from_encoding(condition);
            for bit in [N, Z, C, V] {
                let changes = (0..16_u8).any(|flags| {
                    nixe_cpu::semantics::conditions::evaluate_a64(decoded, u32::from(flags) << 28)
                        != nixe_cpu::semantics::conditions::evaluate_a64(
                            decoded,
                            u32::from(flags ^ bit) << 28,
                        )
                });
                assert_eq!(condition_flags(condition) & bit != 0, changes);
            }
        }
        let live = StateSet {
            nzcv: C | V,
            ..StateSet::default()
        };
        // Poll/bridge arithmetic changes HOST flags, not guest NZCV; it has no
        // architectural NZCV def and must preserve the live producer.
        assert_eq!(InstructionEffects::default().live_before(live).nzcv, C | V);
        assert_eq!(effects(0xab02_0020).live_before(live).nzcv, 0); // ADDS replaces them
    }

    #[test]
    fn faults_observe_the_pre_write_state_and_helpers_keep_status() {
        let load = effects(0xf940_0020); // LDR X0, [X1]
        assert!(load.writes.integer.x[0]);
        assert!(load.live_before(StateSet::default()).integer.x[0]);
        assert_eq!(load.observe_before, StateSet::ALL);
        assert!(
            !block(&[0xd280_0020, 0xf940_0020]).live_in.integer.x[0],
            "a preceding write supplies the prefault value"
        );
        assert_eq!(effects(0xd53b_e040).observe_before, StateSet::ALL); // runtime counter helper
        let svc = effects(0xd400_0001);
        assert_eq!(svc.live_before(StateSet::default()), StateSet::ALL);
        let fadd = effects(0x1e22_2820);
        assert!(fadd.reads.fpcr && fadd.reads.fpsr && fadd.writes.fpsr);
        let msr_fpsr = effects(0xd51b_4420);
        assert!(
            !msr_fpsr.reads.fpsr,
            "replacement does not consume old sticky status"
        );
        assert!(msr_fpsr.writes.fpsr);
    }

    #[test]
    fn liveness_solves_diamonds_loops_and_independent_entries() {
        let write = block(&[0xd280_0020]); // MOVZ X0, #1
        let empty = BlockEffects::default();
        let blocks = [
            FlowBlock {
                effects: empty,
                successors: &[1, 2],
                exit_live: StateSet::default(),
            },
            FlowBlock {
                effects: write,
                successors: &[3],
                exit_live: StateSet::default(),
            },
            FlowBlock {
                effects: empty,
                successors: &[3],
                exit_live: StateSet::default(),
            },
            FlowBlock {
                effects: empty,
                successors: &[],
                exit_live: x(0),
            },
        ];
        let live = liveness(&blocks);
        assert!(live[0].live_in.integer.x[0], "the bypass path preserves X0");
        assert!(
            !live[1].live_in.integer.x[0],
            "the writing entry kills old X0"
        );
        assert!(live[2].live_in.integer.x[0]);
        let loop_block = [FlowBlock {
            effects: block(&[0x9100_0400]),
            successors: &[0],
            exit_live: x(0),
        }]; // ADD X0, X0, #1
        assert!(liveness(&loop_block)[0].live_in.integer.x[0]);
    }

    #[test]
    fn every_ready_decoder_fixture_uses_the_shared_classifier() {
        for pattern in decode::a64::patterns() {
            if let Some(fixture) = pattern.regression_fixture {
                let bits = fixture.encoding.bits();
                // Allocation constraints may reject the catalog's base word;
                // decode first, exactly as the production frontend does.
                let platform = TargetPlatform::Switch1;
                let location = LocationDescriptor::new(
                    GuestVirtualAddress::new(0x1000),
                    platform.profile_id(),
                );
                if let DecodeResult::Decoded(decoded) = decode::decode(
                    PlatformDecoder::new(platform),
                    location,
                    InstructionEncoding::from_u32(bits),
                ) {
                    instruction_effects(decode::a64::normalize(
                        &decoded.instruction,
                        decoded.encoding,
                    ));
                }
            }
        }
    }
}
