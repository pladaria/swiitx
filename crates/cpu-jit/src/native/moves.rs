//! Flag-transparent host moves. Register encodings match HostAbi and the fork's
//! final maps. Only 32/64-bit scalar and 128-bit vector boundary values occur.

use crate::abi::{HostAbi, RegisterClass, TRANSFER_BYTES, ValueLocation};
use RegisterClass::{Integer, Vector};
use ValueLocation::{Constant, Register, Spill};

// Separate from cycle saves. Constants use this slot to populate a SIMD input
// without borrowing any allocator-visible SIMD register.
pub(super) const SCRATCH_SLOT: u32 = TRANSFER_BYTES - 16;

#[derive(Clone, Copy)]
pub(super) struct Copy {
    pub source: ValueLocation,
    pub destination: ValueLocation,
    pub bytes: u8,
}

pub(super) struct Emitter {
    pub(super) abi: HostAbi,
    code: Vec<u8>,
}

impl Emitter {
    pub fn new(abi: HostAbi) -> Self {
        Self {
            abi,
            code: Vec::new(),
        }
    }
    pub fn finish(self) -> Vec<u8> {
        self.code
    }
    pub(super) fn code_byte(&mut self, byte: u8) {
        self.code.push(byte);
    }

    pub fn jump_register(&mut self, register: u8) {
        if self.abi == HostAbi::X86_64 {
            self.x64(&[], false, &[0xff], 4, register, None);
        } else {
            self.word(0xd61f0000 | (u32::from(register) << 5));
        }
    }

    pub fn copy(&mut self, copy: Copy) {
        let Copy {
            source,
            destination,
            bytes,
        } = copy;
        debug_assert!(matches!(bytes, 4 | 8 | 16));
        match (source, destination) {
            (
                Register {
                    class: sc,
                    index: s,
                },
                Register {
                    class: dc,
                    index: d,
                },
            ) => {
                self.register_move(sc, s, dc, d, bytes);
            }
            (Register { class, index }, Spill { offset, .. }) => {
                self.memory(false, class, index, offset, bytes)
            }
            (Spill { offset, .. }, Register { class, index }) => {
                self.memory(true, class, index, offset, bytes)
            }
            (
                Constant(value),
                Register {
                    class: Integer,
                    index,
                },
            ) => self.constant(index, value as u64, bytes),
            (
                Constant(value),
                Register {
                    class: Vector,
                    index,
                },
            ) => {
                self.copy(Copy {
                    source: Constant(value),
                    destination: Spill {
                        offset: SCRATCH_SLOT,
                        bytes,
                    },
                    bytes,
                });
                self.memory(true, Vector, index, SCRATCH_SLOT, bytes);
            }
            (source @ (Constant(_) | Spill { .. }), Spill { offset, .. }) => {
                let scratch = self.abi.reserved().link_scratch[0];
                let part = bytes.min(8);
                for delta in (0..bytes).step_by(usize::from(part)) {
                    match source {
                        Constant(value) => {
                            self.constant(scratch, (value >> (delta * 8)) as u64, part)
                        }
                        Spill { offset, .. } => {
                            self.memory(true, Integer, scratch, offset + u32::from(delta), part)
                        }
                        _ => unreachable!(),
                    }
                    self.memory(false, Integer, scratch, offset + u32::from(delta), part);
                }
            }
            (_, Constant(_)) => unreachable!("validated ingress cannot be a constant"),
        }
    }

    // x86-64 SSE2/MOV and AArch64 MOV/ORR/FMOV/LDR/STR do not change host
    // condition flags or the FP environment. There are no arithmetic helpers.
    fn register_move(&mut self, sc: RegisterClass, s: u8, dc: RegisterClass, d: u8, bytes: u8) {
        if self.abi == HostAbi::X86_64 {
            match (sc, dc) {
                (Integer, Integer) => self.x64(&[], bytes == 8, &[0x89], s, d, None),
                (Vector, Vector) => self.x64(&[0xf3], false, &[0x0f, 0x6f], d, s, None),
                (Integer, Vector) => self.x64(&[0x66], bytes == 8, &[0x0f, 0x6e], d, s, None),
                (Vector, Integer) => self.x64(&[0x66], bytes == 8, &[0x0f, 0x7e], s, d, None),
            }
        } else {
            let word = match (sc, dc) {
                (Integer, Integer) => {
                    (if bytes == 8 { 0xaa0003e0 } else { 0x2a0003e0 }) | (u32::from(s) << 16)
                }
                (Vector, Vector) => 0x4ea01c00 | (u32::from(s) << 16) | (u32::from(s) << 5),
                (Integer, Vector) => {
                    (if bytes == 8 { 0x9e670000 } else { 0x1e270000 }) | (u32::from(s) << 5)
                }
                (Vector, Integer) => {
                    (if bytes == 8 { 0x9e660000 } else { 0x1e260000 }) | (u32::from(s) << 5)
                }
            } | u32::from(d);
            self.word(word);
        }
    }

    pub fn memory(
        &mut self,
        load: bool,
        class: RegisterClass,
        register: u8,
        offset: u32,
        bytes: u8,
    ) {
        let frame = self.abi.reserved().frame;
        self.memory_at(load, class, register, frame, offset, bytes);
    }

    pub fn memory_at(
        &mut self,
        load: bool,
        class: RegisterClass,
        register: u8,
        base_register: u8,
        offset: u32,
        bytes: u8,
    ) {
        let frame = base_register;
        if self.abi == HostAbi::X86_64 {
            match (class, bytes, load) {
                (Integer, 4 | 8, _) => self.x64(
                    &[],
                    bytes == 8,
                    &[if load { 0x8b } else { 0x89 }],
                    register,
                    frame,
                    Some(offset),
                ),
                (Vector, 16, _) => self.x64(
                    &[0xf3],
                    false,
                    &[0x0f, if load { 0x6f } else { 0x7f }],
                    register,
                    frame,
                    Some(offset),
                ),
                (Vector, 8, true) => {
                    self.x64(&[0xf3], false, &[0x0f, 0x7e], register, frame, Some(offset))
                }
                (Vector, 8, false) => {
                    self.x64(&[0x66], false, &[0x0f, 0xd6], register, frame, Some(offset))
                }
                (Vector, 4, _) => self.x64(
                    &[0x66],
                    false,
                    &[0x0f, if load { 0x6e } else { 0x7e }],
                    register,
                    frame,
                    Some(offset),
                ),
                _ => unreachable!("unsupported boundary width"),
            }
        } else {
            let base = match (class, bytes) {
                (Integer, 4) => 0xb9000000,
                (Integer, 8) => 0xf9000000,
                (Vector, 4) => 0xbd000000,
                (Vector, 8) => 0xfd000000,
                (Vector, 16) => 0x3d800000,
                _ => unreachable!("unsupported boundary width"),
            };
            debug_assert_eq!(offset % u32::from(bytes), 0);
            debug_assert!(offset / u32::from(bytes) < 4096);
            self.word(
                base | (u32::from(load) << 22)
                    | ((offset / u32::from(bytes)) << 10)
                    | (u32::from(frame) << 5)
                    | u32::from(register),
            );
        }
    }

    pub fn constant(&mut self, register: u8, value: u64, bytes: u8) {
        if self.abi == HostAbi::X86_64 {
            let rex = 0x40 | (u8::from(bytes == 8) << 3) | (register >> 3);
            if rex != 0x40 {
                self.code.push(rex);
            }
            self.code.push(0xb8 | (register & 7));
            self.code
                .extend_from_slice(&value.to_le_bytes()[..usize::from(bytes)]);
        } else {
            let width = if bytes == 8 { 0x80000000 } else { 0 };
            self.word(width | 0x52800000 | ((value as u32 & 0xffff) << 5) | u32::from(register));
            for half in 1..bytes / 2 {
                let immediate = ((value >> (half * 16)) & 0xffff) as u32;
                if immediate != 0 {
                    self.word(
                        width
                            | 0x72800000
                            | (u32::from(half) << 21)
                            | (immediate << 5)
                            | u32::from(register),
                    );
                }
            }
        }
    }

    pub(super) fn word(&mut self, word: u32) {
        self.code.extend_from_slice(&word.to_le_bytes());
    }

    /// Merge packed NZCV in `value` with canonical NZCV addressed by `pointer`.
    /// For a partial mask the pointer becomes the old value; caller reloads it
    /// before storing. A full mask leaves the pointer intact. Only x86-64 host
    /// flags are clobbered by this operation.
    pub fn merge_nzcv(&mut self, value: u8, pointer: u8, bits: u8) {
        // With all four bits dirty, old canonical NZCV is not an input.
        if bits == crate::analysis::NZCV {
            if self.abi == HostAbi::X86_64 {
                self.x64(&[], false, &[0x81], 4, value, None);
                self.code.extend_from_slice(&0xf0000000u32.to_le_bytes());
            } else {
                self.word(0x33000000 | (27 << 10) | (31 << 5) | u32::from(value));
            }
            return;
        }
        self.memory_at(true, Integer, pointer, pointer, 0, 4);
        let mask = u32::from(bits) << 28;
        if self.abi == HostAbi::X86_64 {
            self.x64(&[], false, &[0x81], 4, value, None); // AND value, mask
            self.code.extend_from_slice(&mask.to_le_bytes());
            self.x64(&[], false, &[0x81], 4, pointer, None);
            self.code
                .extend_from_slice(&(!mask & 0xf0000000).to_le_bytes());
            self.x64(&[], false, &[0x09], pointer, value, None); // OR value, old
        } else {
            // BFC Wd,#lsb,#width is BFM Wd,WZR,#(-lsb mod 32),#(width-1).
            // Clear bit ranges without borrowing a third register for a mask.
            for register in [value, pointer] {
                self.word(0x33000000 | (27 << 10) | (31 << 5) | u32::from(register));
            }
            for bit in 0..4 {
                let register = if bits & (1 << bit) != 0 {
                    pointer
                } else {
                    value
                };
                self.word(0x33000000 | ((32 - (28 + bit)) << 16) | (31 << 5) | u32::from(register));
            }
            self.word(
                0x2a000000
                    | (u32::from(pointer) << 16)
                    | (u32::from(value) << 5)
                    | u32::from(value),
            );
        }
    }

    pub(super) fn x64(
        &mut self,
        prefix: &[u8],
        wide: bool,
        opcode: &[u8],
        reg: u8,
        rm: u8,
        offset: Option<u32>,
    ) {
        self.code.extend_from_slice(prefix);
        let rex = 0x40 | (u8::from(wide) << 3) | ((reg >> 3) << 2) | (rm >> 3);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code.extend_from_slice(opcode);
        let mode = match offset {
            None => 0xc0,
            Some(0) => 0,
            Some(1..=127) => 0x40,
            Some(_) => 0x80,
        };
        self.code.push(mode | ((reg & 7) << 3) | (rm & 7));
        match offset {
            None | Some(0) => {}
            Some(offset @ 1..=127) => self.code.push(offset as u8),
            Some(offset) => self.code.extend_from_slice(&offset.to_le_bytes()),
        }
    }
}
