use super::*;
use crate::abi::HostAbi;
use cranelift_codegen::{
    Context,
    control::ControlPlane,
    cursor::{Cursor, FuncCursor},
    ir::{self, InstBuilder, MemFlagsData, types},
    isa::{self, CallConv},
    settings::{self, Configurable},
};
use output::{Metadata, Relocation};
use std::os::fd::AsRawFd;

fn host() -> HostAbi {
    if cfg!(target_arch = "x86_64") {
        HostAbi::X86_64
    } else {
        HostAbi::Aarch64
    }
}

fn output(abi: HostAbi, bytes: Vec<u8>) -> Output {
    Output {
        bytes: bytes.into_boxed_slice(),
        alignment: 16,
        metadata: Metadata {
            abi,
            frame_extent: crate::abi::TRANSFER_BYTES,
            entries: Box::new([]),
            states: Box::new([]),
            faults: Box::new([]),
            traps: Box::new([]),
            relocations: Box::new([]),
        },
    }
}

fn return_value(value: u8) -> Output {
    let bytes = match host() {
        HostAbi::X86_64 => vec![0xf3, 0x0f, 0x1e, 0xfa, 0xb8, value, 0, 0, 0, 0xc3],
        HostAbi::Aarch64 => [
            0xd503245f_u32,
            0x52800000 | (u32::from(value) << 5),
            0xd65f03c0,
        ]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect(),
    };
    output(host(), bytes)
}

fn permissions(address: usize) -> String {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
    for line in maps.lines() {
        let mut fields = line.split_whitespace();
        let range = fields.next().unwrap();
        let (start, end) = range.split_once('-').unwrap();
        if (usize::from_str_radix(start, 16).unwrap()..usize::from_str_radix(end, 16).unwrap())
            .contains(&address)
        {
            return fields.next().unwrap().to_owned();
        }
    }
    panic!("address {address:#x} is not mapped");
}

unsafe fn execute(code: &Installed) -> u32 {
    // This fixture is a host System-ABI leaf, not a native guest unit. Its
    // immutable lease stays alive through the call; no JITModule is involved.
    let function: unsafe extern "C" fn() -> u32 =
        unsafe { std::mem::transmute(code.allocation.address()) };
    unsafe { function() }
}

#[test]
fn synthetic_code_executes_from_rx_with_the_write_view_closed() {
    let cache = Cache::new().unwrap();
    let initial = cache.usage().unwrap();
    let code = cache
        .install(return_value(42), Tier::Lcq, |_| None)
        .unwrap();
    assert_eq!(unsafe { execute(&code) }, 42);
    assert!(permissions(code.allocation.address()).starts_with("r-x"));
    let rw = cache
        .lock()
        .unwrap()
        .backing
        .as_ref()
        .unwrap()
        .rw
        .as_ref()
        .unwrap()
        .base
        .as_ptr() as usize;
    assert!(permissions(rw).starts_with("---"));
    assert_eq!(cache.usage().unwrap().committed, SEGMENT_BYTES);
    assert!(cache.usage().unwrap().metadata > initial.metadata);
    for line in std::fs::read_to_string("/proc/self/maps")
        .unwrap()
        .lines()
        .filter(|line| line.contains("memfd:nixe-jit"))
    {
        let mode = line.split_whitespace().nth(1).unwrap();
        assert!(!(mode.contains('w') && mode.contains('x')), "{line}");
    }
    drop(code);
    // Free spans do not pretend already-committed backing has been released.
    assert_eq!(cache.usage().unwrap().committed, SEGMENT_BYTES);
    assert!(unsafe { cache.decommit_empty(0) }.unwrap());
    assert_eq!(cache.usage().unwrap(), initial);
}

#[test]
fn individual_spans_reuse_and_empty_segments_release_real_backing() {
    let cache = Cache::new().unwrap();
    let first = cache.install(return_value(1), Tier::Lcq, |_| None).unwrap();
    let address = first.allocation.address();
    let generation = first.allocation.generation;
    let neighbor = cache.install(return_value(2), Tier::Lcq, |_| None).unwrap();
    assert!(!unsafe { cache.decommit_empty(0) }.unwrap());
    drop(first);
    let replacement = cache.install(return_value(3), Tier::Lcq, |_| None).unwrap();
    assert_eq!(replacement.allocation.address(), address);
    assert_eq!(unsafe { execute(&replacement) }, 3);
    assert_eq!(unsafe { execute(&neighbor) }, 2);
    drop((replacement, neighbor));
    let blocks = || {
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        assert_eq!(
            unsafe {
                libc::fstat(
                    cache
                        .lock()
                        .unwrap()
                        .backing
                        .as_ref()
                        .unwrap()
                        .fd
                        .as_raw_fd(),
                    stat.as_mut_ptr(),
                )
            },
            0
        );
        unsafe { stat.assume_init().st_blocks }
    };
    assert!(blocks() > 0);
    assert!(unsafe { cache.decommit_empty(0) }.unwrap());
    assert_eq!(blocks(), 0);
    assert!(permissions(address).starts_with("---"));
    let reused = cache.install(return_value(4), Tier::Lcq, |_| None).unwrap();
    assert_eq!(reused.allocation.address(), address);
    assert_ne!(reused.allocation.generation, generation);
    let expected = return_value(4);
    assert_eq!(
        unsafe { std::slice::from_raw_parts(address as *const u8, expected.bytes.len()) },
        &*expected.bytes
    );
    assert_eq!(unsafe { execute(&reused) }, 4);
}

#[test]
fn best_fit_then_lowest_address_and_coalescing_match_policy() {
    let cache = Cache::new().unwrap();
    let a = cache.allocate(80, 16, Tier::Lcq).unwrap();
    let b = cache.allocate(16, 16, Tier::Lcq).unwrap();
    let c = cache.allocate(48, 16, Tier::Lcq).unwrap();
    let d = cache.allocate(16, 16, Tier::Lcq).unwrap();
    let e = cache.allocate(80, 16, Tier::Lcq).unwrap();
    let f = cache.allocate(16, 16, Tier::Lcq).unwrap();
    let (low, small) = (a.address(), c.address());
    drop((a, c, e));
    let fit = cache.allocate(40, 16, Tier::Lcq).unwrap();
    assert_eq!(fit.address(), small);
    let tie = cache.allocate(60, 16, Tier::Lcq).unwrap();
    assert_eq!(tie.address(), low);
    drop((b, d, f, fit, tie));
    let state = cache.lock().unwrap();
    assert_eq!(
        (
            state.segments[0].bump,
            state.segments[0].free_len,
            state.segments[0].live
        ),
        (0, 0, 0)
    );
}

#[test]
fn alignment_padding_and_fragmentation_are_reusable() {
    let cache = Cache::new().unwrap();
    let a = cache.allocate(3, 1, Tier::Lcq).unwrap();
    let b = cache.allocate(19, 4096, Tier::Lcq).unwrap();
    assert_eq!(b.address() % 4096, 0);
    let gap = cache.allocate(16, 16, Tier::Lcq).unwrap();
    assert!(a.address() < gap.address() && gap.address() < b.address());
    drop((a, b, gap));
    assert_eq!(cache.lock().unwrap().segments[0].bump, 0);
    assert!(cache.allocate(1, 3, Tier::Lcq).is_err());
    let mut malformed = return_value(1);
    malformed.alignment = 3;
    assert!(cache.install(malformed, Tier::Lcq, |_| None).is_err());
}

#[test]
fn tiers_do_not_share_live_segments_but_borrow_empty_ones() {
    let cache = Cache::new().unwrap();
    let lcq = cache.allocate(16, 16, Tier::Lcq).unwrap();
    let hcq = cache.allocate(16, 16, Tier::Hcq).unwrap();
    assert_ne!(lcq.segment, hcq.segment);
    let first_segment = lcq.segment;
    drop((lcq, hcq));
    let borrowed = cache.allocate(16, 16, Tier::Hcq).unwrap();
    assert_eq!(borrowed.segment, first_segment);
}

#[test]
fn reservation_bounds_and_islands_exclude_unallocatable_bytes() {
    let cache = Cache::new().unwrap();
    assert_eq!(
        (SEGMENTS - 1) * SEGMENT_BYTES + segment_size(127),
        WINDOW_BYTES
    );
    assert_eq!(segment_size(127), 15 * MIB);
    let base = cache.executable_base();
    assert_eq!(cache.segment_for_pc(base - 1), None);
    assert_eq!(cache.segment_for_pc(base), Some(0));
    assert_eq!(cache.segment_for_pc(base + WINDOW_BYTES - 1), Some(127));
    assert_eq!(cache.segment_for_pc(base + WINDOW_BYTES), None);
    assert_eq!(cache.segment_for_pc(usize::MAX), None);
    let whole = cache
        .allocate(SEGMENT_BYTES - ISLAND_BYTES, 16, Tier::Lcq)
        .unwrap();
    let next = cache.allocate(16, 16, Tier::Lcq).unwrap();
    assert_eq!((whole.segment, next.segment), (0, 1));
    assert!(whole.address() + whole.len() <= base + SEGMENT_BYTES - ISLAND_BYTES);
    assert!(cache.allocate(SEGMENT_BYTES, 16, Tier::Lcq).is_err());
}

#[test]
fn budget_charges_overlap_and_preserves_lcq_capacity() {
    let usage = Usage {
        committed: 600 * MIB,
        metadata: 8 * MIB,
    };
    assert!(usage.check(1, Tier::Hcq).is_err());
    assert!(usage.check(LCQ_RESERVE, Tier::Lcq).is_ok());
    assert!(usage.check(LCQ_RESERVE + 1, Tier::Lcq).is_err());
    assert!(usage.needs_reclamation());
    assert!(
        Usage {
            committed: SOFT_BYTES,
            metadata: 0
        }
        .check(1, Tier::Hcq)
        .is_err()
    );
    let cache = Cache::new().unwrap();
    let initial = cache.usage().unwrap();
    let old = cache.charge_metadata(1024, Tier::Lcq).unwrap();
    let new = cache.charge_metadata(2048, Tier::Lcq).unwrap();
    assert_eq!(cache.usage().unwrap().metadata, initial.metadata + 3072);
    drop(old);
    assert_eq!(cache.usage().unwrap().metadata, initial.metadata + 2048);
    drop(new);
    assert_eq!(cache.usage().unwrap(), initial);
}

#[test]
fn relocation_failure_returns_the_unpublished_span_and_metadata() {
    let cache = Cache::new().unwrap();
    let warmup = cache.allocate(16, 16, Tier::Lcq).unwrap();
    let address = warmup.address();
    drop(warmup);
    let initial = cache.usage().unwrap();
    let mut bad = output(host(), vec![0; 16]);
    bad.metadata.relocations = vec![Relocation {
        offset: 0,
        kind: Reloc::Abs8,
        target: Target::User {
            namespace: 1,
            index: 2,
        },
        addend: 0,
    }]
    .into_boxed_slice();
    assert!(matches!(
        cache.install(bad, Tier::Lcq, |_| None),
        Err(Error::Relocation { .. })
    ));
    assert_eq!(cache.usage().unwrap(), initial);
    let code = cache.install(return_value(5), Tier::Lcq, |_| None).unwrap();
    assert_eq!(code.allocation.address(), address);
    assert_eq!(unsafe { execute(&code) }, 5);
}

#[test]
fn local_absolute_relocations_use_rx_not_rw_addresses() {
    let cache = Cache::new().unwrap();
    let mut data = output(host(), vec![0; 16]);
    data.metadata.relocations = vec![Relocation {
        offset: 0,
        kind: Reloc::Abs8,
        target: Target::Local(8),
        addend: 3,
    }]
    .into_boxed_slice();
    let code = cache
        .install(data, Tier::Lcq, |_| {
            panic!("local target must not call resolver")
        })
        .unwrap();
    let stored = unsafe { (code.allocation.address() as *const u64).read_unaligned() };
    assert_eq!(stored as usize, code.allocation.address() + 11);
}

#[test]
fn relocation_ranges_alignment_and_instruction_fields_are_checked() {
    let mut x86 = output(HostAbi::X86_64, vec![0; 8]);
    x86.metadata.relocations = vec![Relocation {
        offset: 0,
        kind: Reloc::X86CallPCRel4,
        target: Target::User {
            namespace: 1,
            index: 2,
        },
        addend: -4,
    }]
    .into_boxed_slice();
    x86.relocate(0x1000, |_| Some(0x2000)).unwrap();
    assert_eq!(
        i32::from_le_bytes(x86.bytes[..4].try_into().unwrap()),
        0xffc
    );
    assert!(x86.relocate(0x1000, |_| Some(usize::MAX)).is_err());
    x86.metadata.relocations[0].kind = Reloc::X86GOTPCRel4;
    assert!(x86.relocate(0x1000, |_| Some(0x2000)).is_err());
    let mut arm = output(HostAbi::Aarch64, 0x97ff_ffffu32.to_le_bytes().to_vec());
    arm.metadata.relocations = vec![Relocation {
        offset: 0,
        kind: Reloc::Arm64Call,
        target: Target::User {
            namespace: 0,
            index: 0,
        },
        addend: 0,
    }]
    .into_boxed_slice();
    arm.relocate(0x1000, |_| Some(0x1004)).unwrap();
    assert_eq!(
        u32::from_le_bytes(arm.bytes[..4].try_into().unwrap()),
        0x94000001
    );
    assert!(arm.relocate(0x1000, |_| Some(0x1002)).is_err());
    assert!(arm.relocate(0x1000, |_| Some(0x1000 + (1 << 27))).is_err());
    arm.metadata.relocations[0].offset = 2;
    assert!(arm.relocate(0x1000, |_| Some(0x1004)).is_err());
}

#[test]
fn adrp_handles_the_full_signed_four_gibibyte_range_and_replaces_immediates() {
    let mut arm = output(
        HostAbi::Aarch64,
        [0x90000000u32, 0x91000000]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect(),
    );
    arm.metadata.relocations = vec![
        Relocation {
            offset: 0,
            kind: Reloc::Aarch64AdrPrelPgHi21,
            target: Target::User {
                namespace: 0,
                index: 0,
            },
            addend: 0,
        },
        Relocation {
            offset: 4,
            kind: Reloc::Aarch64AddAbsLo12Nc,
            target: Target::User {
                namespace: 0,
                index: 0,
            },
            addend: 0,
        },
    ]
    .into_boxed_slice();
    let base = 1usize << 33;
    for target in [base + (1 << 32) - 1, base - (1 << 32), base + 0xabc] {
        arm.relocate(base, |_| Some(target)).unwrap();
        let page = u32::from_le_bytes(arm.bytes[..4].try_into().unwrap());
        let immediate = ((page >> 29) & 3) | (((page >> 5) & 0x7ffff) << 2);
        let signed = ((immediate << 11) as i32) >> 11;
        assert_eq!(
            (base as i128) + ((signed as i128) << 12),
            (target & !4095) as i128
        );
        let low = u32::from_le_bytes(arm.bytes[4..].try_into().unwrap());
        assert_eq!((low >> 10) & 4095, target as u32 & 4095);
    }
    assert!(arm.relocate(base, |_| Some(base + (1 << 32))).is_err());
}

#[test]
fn backend_output_survives_context_reset_with_exact_labels_and_fault_maps() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for allocator in ["single_pass", "backtracking"] {
            let mut function = ir::Function::new();
            let block = function.dfg.make_block();
            function.layout.append_block(block);
            let signature = function.import_signature(ir::Signature::new(CallConv::SystemV));
            let mut cursor = FuncCursor::new(&mut function).at_bottom(block);
            cursor.ins().nixe_entry(signature, 10);
            let address = cursor.ins().get_pinned_reg(types::I64);
            let name = cursor
                .func
                .declare_imported_user_function(ir::UserExternalName::new(42, 7));
            let external = cursor.func.import_function(ir::ExtFuncData {
                name: ir::ExternalName::user(name),
                signature,
                colocated: false,
                patchable: false,
            });
            let external_address = cursor.ins().func_addr(types::I64, external);
            cursor
                .ins()
                .store(MemFlagsData::new(), external_address, address, 16);
            cursor.ins().nixe_fault_start(30, &[]);
            let value = cursor
                .ins()
                .load(types::I64, MemFlagsData::new(), address, 0);
            cursor.ins().store(MemFlagsData::new(), value, address, 8);
            cursor.ins().nixe_fault_end(30, &[]);
            cursor.ins().nixe_exit(20, &[]);
            cranelift_codegen::nixe::set_entries(&mut function, &[block]).unwrap();
            let mut flags = settings::builder();
            for (name, value) in [
                ("enable_nixe_abi", "true"),
                ("enable_pinned_reg", "true"),
                ("regalloc_algorithm", allocator),
                ("opt_level", "none"),
            ] {
                flags.set(name, value).unwrap();
            }
            let triple = if abi == HostAbi::X86_64 {
                "x86_64-unknown-linux-gnu"
            } else {
                "aarch64-unknown-linux-gnu"
            };
            let isa = isa::lookup(triple.parse().unwrap())
                .unwrap()
                .finish(settings::Flags::new(flags))
                .unwrap();
            let mut context = Context::for_function(function);
            context
                .compile(&*isa, &mut ControlPlane::default())
                .unwrap();
            let code = context.take_compiled_code().unwrap();
            let expected = code.code_buffer().to_vec();
            let offsets = code.buffer.nixe_entries.clone();
            let faults = code.buffer.nixe_faults.clone();
            let owned = Output::from_backend(abi, code, &context.func).unwrap();
            context.clear();
            assert_eq!(&*owned.bytes, &expected);
            assert_eq!(&*owned.metadata.entries, &offsets);
            assert_eq!(owned.metadata.faults.len(), 2);
            assert_eq!(&*owned.metadata.faults, &faults);
            assert!(!owned.metadata.states.is_empty());
            assert!(!owned.metadata.relocations.is_empty());
            assert!(
                owned
                    .metadata
                    .relocations
                    .iter()
                    .any(|relocation| relocation.target
                        == Target::User {
                            namespace: 42,
                            index: 7
                        })
            );
            assert!(owned.metadata.frame_extent <= crate::abi::SPILL_BYTES);
            let cache = Cache::new().unwrap();
            let symbol = cache.executable_base() + 0x12340;
            let installed = cache
                .install(owned, Tier::Lcq, |target| {
                    (*target
                        == Target::User {
                            namespace: 42,
                            index: 7,
                        })
                    .then_some(symbol)
                })
                .unwrap();
            assert_eq!(&*installed.metadata.entries, &offsets);
            assert_eq!(&*installed.metadata.faults, &faults);
            assert!(permissions(installed.allocation.address()).starts_with("r-x"));
        }
    }
}

#[test]
fn span_churn_never_overlaps_live_allocations_and_returns_to_empty() {
    let cache = Cache::new().unwrap();
    let mut slots: [Option<Allocation>; 32] = std::array::from_fn(|_| None);
    let mut random = 1u64;
    for _ in 0..2000 {
        random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
        let index = (random >> 32) as usize % slots.len();
        if slots[index].take().is_some() {
            continue;
        }
        let length = (random as usize % 1024) + 1;
        let alignment = 1 << (4 + (random >> 48) as usize % 9);
        let allocation = cache.allocate(length, alignment, Tier::Lcq).unwrap();
        assert_eq!(allocation.address() % alignment, 0);
        for other in slots.iter().flatten() {
            assert!(
                allocation.address() + allocation.len() <= other.address()
                    || other.address() + other.len() <= allocation.address()
            );
        }
        slots[index] = Some(allocation);
    }
    drop(slots);
    let state = cache.lock().unwrap();
    assert!(
        state
            .segments
            .iter()
            .all(|segment| segment.live == 0 && segment.bump == 0 && segment.free_len == 0)
    );
}

#[test]
fn hard_limit_rejection_does_not_commit_a_segment_or_consume_reserve() {
    let cache = Cache::new().unwrap();
    let initial = cache.usage().unwrap();
    // Exercise the accounting authority without physically allocating 640 MiB.
    let charge = cache
        .charge_metadata(HARD_BYTES - initial.total(), Tier::Lcq)
        .unwrap();
    assert!(cache.install(return_value(1), Tier::Lcq, |_| None).is_err());
    assert_eq!(cache.usage().unwrap().committed, 0);
    assert_eq!(cache.usage().unwrap().total(), HARD_BYTES);
    drop(charge);
    assert_eq!(cache.usage().unwrap(), initial);
}

#[test]
fn code_is_coherent_when_published_and_reused_on_another_thread() {
    let cache = Cache::new().unwrap();
    let (send, receive) = std::sync::mpsc::sync_channel::<Installed>(0);
    let (done, acknowledged) = std::sync::mpsc::sync_channel(0);
    std::thread::scope(|scope| {
        scope.spawn(move || {
            for expected in 0..32 {
                let code = receive.recv().unwrap();
                assert_eq!(unsafe { execute(&code) }, expected);
                drop(code);
                done.send(()).unwrap();
            }
        });
        for value in 0..32 {
            let code = cache
                .install(return_value(value), Tier::Lcq, |_| None)
                .unwrap();
            send.send(code).unwrap();
            acknowledged.recv().unwrap();
        }
    });
}
