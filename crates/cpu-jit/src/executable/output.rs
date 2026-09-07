//! Owned final backend output. No JITModule, recompilation, context-borrowed
//! symbol indices, or pre-allocation state-map guesses cross this boundary.

use super::Error;
use crate::{abi::HostAbi, native::AllocatedBoundary};
use cranelift_codegen::{
    CompiledCode, FinalizedRelocTarget, MachTrap, binemit::Reloc, ir, nixe::StateMap,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    Local(u32),
    User { namespace: u32, index: u32 },
    LibCall(ir::LibCall),
    Known(ir::KnownSymbol),
}

#[derive(Clone, Debug)]
pub(crate) struct Relocation {
    pub offset: u32,
    pub kind: Reloc,
    pub target: Target,
    pub addend: i64,
}

pub(crate) struct Metadata {
    pub abi: HostAbi,
    pub frame_extent: u32,
    pub entries: Box<[(ir::Block, u32)]>,
    pub states: Box<[StateMap]>,
    pub faults: Box<[StateMap]>,
    pub traps: Box<[MachTrap]>,
    pub relocations: Box<[Relocation]>,
}
impl Metadata {
    pub fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of_val(&*self.entries)
            + std::mem::size_of_val(&*self.states)
            + std::mem::size_of_val(&*self.faults)
            + std::mem::size_of_val(&*self.traps)
            + std::mem::size_of_val(&*self.relocations)
            + self
                .states
                .iter()
                .chain(self.faults.iter())
                .map(|map| {
                    map.values.capacity()
                        * std::mem::size_of::<cranelift_codegen::nixe::LocatedValue>()
                })
                .sum::<usize>()
    }
}

pub(crate) struct Output {
    pub bytes: Box<[u8]>,
    pub alignment: usize,
    pub metadata: Metadata,
}
impl Output {
    pub fn from_backend(
        abi: HostAbi,
        mut code: CompiledCode,
        function: &ir::Function,
    ) -> Result<Self, Error> {
        if !code.buffer.user_stack_maps().is_empty() || !code.buffer.unwind_info.is_empty() {
            return Err(Error::Output(
                "unexpected host stack/unwind metadata in Nixe output".into(),
            ));
        }
        for map in code
            .buffer
            .nixe_states
            .iter()
            .chain(code.buffer.nixe_faults.iter())
        {
            AllocatedBoundary::new(abi, &code, map)
                .map_err(|error| Error::Output(error.to_string()))?;
        }
        let frame_extent = code
            .buffer
            .frame_layout()
            .and_then(|frame| frame.nixe_frame_size)
            .ok_or_else(|| {
                Error::Output("backend output does not use the Nixe frame ABI".into())
            })?;
        if !(crate::abi::TRANSFER_BYTES..=crate::abi::SPILL_BYTES).contains(&frame_extent) {
            return Err(Error::Output(
                "backend frame exceeds the spill arena".into(),
            ));
        }
        let relocations = code
            .buffer
            .relocs()
            .iter()
            .map(|reloc| {
                let target = match &reloc.target {
                    FinalizedRelocTarget::Func(offset) => Target::Local(*offset),
                    FinalizedRelocTarget::ExternalName(ir::ExternalName::User(reference)) => {
                        let name = function
                            .params
                            .user_named_funcs()
                            .get(*reference)
                            .ok_or_else(|| {
                                Error::Output("unresolved backend user symbol index".into())
                            })?;
                        Target::User {
                            namespace: name.namespace,
                            index: name.index,
                        }
                    }
                    FinalizedRelocTarget::ExternalName(ir::ExternalName::LibCall(call)) => {
                        Target::LibCall(*call)
                    }
                    FinalizedRelocTarget::ExternalName(ir::ExternalName::KnownSymbol(symbol)) => {
                        Target::Known(*symbol)
                    }
                    other => {
                        return Err(Error::Output(format!(
                            "unsupported backend relocation target: {other:?}"
                        )));
                    }
                };
                Ok(Relocation {
                    offset: reloc.offset,
                    kind: reloc.kind,
                    target,
                    addend: reloc.addend,
                })
            })
            .collect::<Result<Box<[_]>, Error>>()?;
        let bytes: Box<[u8]> = code.code_buffer().into();
        if code
            .buffer
            .nixe_entries
            .iter()
            .any(|(_, offset)| *offset as usize >= bytes.len())
        {
            return Err(Error::Output("entry label outside emitted code".into()));
        }
        Ok(Self {
            bytes,
            alignment: code.buffer.alignment as usize,
            metadata: Metadata {
                abi,
                frame_extent,
                entries: std::mem::take(&mut code.buffer.nixe_entries).into_boxed_slice(),
                states: std::mem::take(&mut code.buffer.nixe_states).into_boxed_slice(),
                faults: std::mem::take(&mut code.buffer.nixe_faults).into_boxed_slice(),
                traps: code.buffer.traps().into(),
                relocations,
            },
        })
    }

    /// Modify only owned staging bytes, using eventual RX addresses. No partial
    /// relocation result is reachable on failure. External owners must retain
    /// the resolved symbols through publication and execution.
    pub fn relocate(
        &mut self,
        base: usize,
        mut resolve: impl FnMut(&Target) -> Option<usize>,
    ) -> Result<(), Error> {
        for reloc in &self.metadata.relocations {
            let fail = |detail| Error::Relocation {
                offset: reloc.offset,
                kind: reloc.kind,
                detail,
            };
            let target = match reloc.target {
                Target::Local(offset) if (offset as usize) < self.bytes.len() => {
                    base.checked_add(offset as usize)
                }
                Target::Local(_) => return Err(fail("local target outside code")),
                _ => resolve(&reloc.target),
            }
            .ok_or_else(|| fail("unresolved or overflowing target"))?;
            let target = (target as i128) + i128::from(reloc.addend);
            let target =
                usize::try_from(target).map_err(|_| fail("target/addend address overflow"))?;
            let at = base
                .checked_add(reloc.offset as usize)
                .ok_or_else(|| fail("relocation address overflow"))?;
            use Reloc::*;
            let width = match (self.metadata.abi, reloc.kind) {
                (_, Abs8) => 8,
                (_, Abs4)
                | (HostAbi::X86_64, X86PCRel4 | X86CallPCRel4)
                | (HostAbi::Aarch64, Arm64Call | Aarch64AdrPrelPgHi21 | Aarch64AddAbsLo12Nc) => 4,
                _ => return Err(fail("unsupported relocation for this host ABI")),
            };
            let start = reloc.offset as usize;
            let end = start
                .checked_add(width)
                .ok_or_else(|| fail("relocation extent overflow"))?;
            let bytes = self
                .bytes
                .get_mut(start..end)
                .ok_or_else(|| fail("relocation outside code"))?;
            let delta = target as i128 - at as i128;
            match reloc.kind {
                Abs8 => bytes.copy_from_slice(&(target as u64).to_le_bytes()),
                Abs4 => bytes.copy_from_slice(
                    &u32::try_from(target)
                        .map_err(|_| fail("absolute address exceeds 32 bits"))?
                        .to_le_bytes(),
                ),
                X86PCRel4 | X86CallPCRel4 => bytes.copy_from_slice(
                    &i32::try_from(delta)
                        .map_err(|_| fail("PC-relative target requires an island"))?
                        .to_le_bytes(),
                ),
                kind => {
                    // AArch64 relocation encoding/ranges:
                    // https://github.com/ARM-software/abi-aa/blob/main/aaelf64/aaelf64.rst#static-aarch64-relocations
                    if at & 3 != 0 {
                        return Err(fail("unaligned AArch64 instruction"));
                    }
                    let instruction = u32::from_le_bytes(bytes.try_into().unwrap());
                    let patched = match kind {
                        Arm64Call => {
                            if instruction & 0xfc00_0000 != 0x9400_0000 {
                                return Err(fail("Arm64Call does not name BL"));
                            }
                            if target & 3 != 0 || !(-(1i128 << 27)..(1i128 << 27)).contains(&delta)
                            {
                                return Err(fail("unaligned call or target requires an island"));
                            }
                            (instruction & 0xfc00_0000) | (((delta >> 2) as u32) & 0x03ff_ffff)
                        }
                        Aarch64AdrPrelPgHi21 => {
                            if instruction & 0x9f00_0000 != 0x9000_0000 {
                                return Err(fail("page relocation does not name ADRP"));
                            }
                            let pages = ((target & !4095) as i128 - (at & !4095) as i128) >> 12;
                            if !(-(1i128 << 20)..(1i128 << 20)).contains(&pages) {
                                return Err(fail("ADRP target exceeds signed 21 pages"));
                            }
                            let immediate = pages as u32 & 0x1f_ffff;
                            (instruction & !0x60ff_ffe0)
                                | ((immediate & 3) << 29)
                                | ((immediate >> 2) << 5)
                        }
                        Aarch64AddAbsLo12Nc => {
                            if instruction & 0xffc0_0000 != 0x9100_0000 {
                                return Err(fail("low relocation does not name unshifted ADD X"));
                            }
                            (instruction & !0x003f_fc00) | ((target as u32 & 4095) << 10)
                        }
                        _ => unreachable!(),
                    };
                    bytes.copy_from_slice(&patched.to_le_bytes());
                }
            }
        }
        Ok(())
    }
}
