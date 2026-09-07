//! Shared tiered-JIT identities and native state contracts. Publication and
//! reclamation belong to the runtime, not these immutable values. Addresses
//! are meaningful only while the caller protects the owning execution epoch.

use crate::analysis::StateSet;
pub use crate::fp_env::UnsupportedFpControl;
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::{CpuProfileId, ProcessCpuContext};
use nixe_cpu::state::a64::{A64State, Nzcv};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::{MaybeUninit, offset_of};
use std::num::{NonZeroU64, NonZeroUsize};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LazyFlags<Value> {
    Canonical(Value),
    Packed(Value),
    Add {
        lhs: Value,
        rhs: Value,
        result: Value,
        width: u8,
    },
    Subtract {
        lhs: Value,
        rhs: Value,
        result: Value,
        width: u8,
    },
    AddCarry {
        lhs: Value,
        rhs: Value,
        carry: Value,
        result: Value,
        width: u8,
    },
    SubtractCarry {
        lhs: Value,
        rhs: Value,
        carry: Value,
        result: Value,
        width: u8,
    },
    Logical {
        result: Value,
        width: u8,
    },
    Conditional {
        predicate: Value,
        when_true: Box<LazyFlags<Value>>,
        /// The instruction's four-bit NZCV literal, not packed bits 31:28.
        when_false: u32,
    },
}

impl<Value> LazyFlags<Value> {
    pub const fn dirty(&self) -> bool {
        !matches!(self, Self::Canonical(_))
    }
}

/// Unspecialized code reads the current FPCR; specialized code is usable only
/// with this exact FPCR. Mapping/content generations are deliberately not keys.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FpSpecialization {
    Dynamic,
    Exact(u32),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BlockKey {
    pub address_space: AddressSpaceId,
    pub pc: GuestVirtualAddress,
    pub profile: CpuProfileId,
    pub platform: TargetPlatform,
    pub fp: FpSpecialization,
}

impl BlockKey {
    pub const fn new(
        cpu: ProcessCpuContext,
        pc: GuestVirtualAddress,
        fp: FpSpecialization,
    ) -> Option<Self> {
        if pc.get() & 3 != 0 {
            return None;
        }
        Some(Self {
            address_space: cpu.address_space_id(),
            pc,
            profile: cpu.profile_id(),
            platform: cpu.platform(),
            fp,
        })
    }

    pub const fn at(self, pc: GuestVirtualAddress) -> Option<Self> {
        if pc.get() & 3 != 0 {
            return None;
        }
        Some(Self { pc, ..self })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InstructionKey(BlockKey);
impl InstructionKey {
    pub const fn new(key: BlockKey) -> Option<Self> {
        if key.pc.get() & 3 != 0 {
            None
        } else {
            Some(Self(key))
        }
    }
    pub const fn block_key(self) -> BlockKey {
        self.0
    }
}

/// Identity allocation is serialized by the owning runtime. On exhaustion the
/// caller disables HCQ; required LCQ/lifecycle allocation reports this error.
pub trait Identity: Copy {
    const NAME: &'static str;
    fn from_nonzero(value: NonZeroU64) -> Self;
}
macro_rules! identities {
    ($($name:ident),+ $(,)?) => {$ (
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(NonZeroU64);
        impl $name {
            pub const fn new(value: u64) -> Option<Self> { match NonZeroU64::new(value) { Some(value) => Some(Self(value)), None => None } }
            pub const fn get(self) -> u64 { self.0.get() }
        }
        impl Identity for $name { const NAME: &'static str = stringify!($name); fn from_nonzero(value: NonZeroU64) -> Self { Self(value) } }
    )+};
}
identities!(
    CodeUnitId,
    CodeVersion,
    ReachabilityVersion,
    HcqFamilyId,
    FamilyVersion,
    SegmentGeneration,
    ExecutionEpoch,
    AdmissionEpoch,
    MaintenanceSequence,
    DispatchGeneration,
    AdmissionSnapshotSequence,
    SampleSequence
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IdentityExhausted(pub &'static str);
impl std::fmt::Display for IdentityExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JIT {} identity space exhausted", self.0)
    }
}
impl std::error::Error for IdentityExhausted {}

pub struct CheckedCounter<T: Identity> {
    last: u64,
    marker: PhantomData<T>,
}
impl<T: Identity> Default for CheckedCounter<T> {
    fn default() -> Self {
        Self {
            last: 0,
            marker: PhantomData,
        }
    }
}
impl<T: Identity> CheckedCounter<T> {
    /// Reserve the exact identities for one cold, indivisible metadata update.
    /// Failure consumes nothing; the returned iterator does not borrow state.
    pub(crate) fn take_ids(
        &mut self,
        count: usize,
    ) -> Result<impl Iterator<Item = T> + use<T>, IdentityExhausted> {
        let end = self
            .last
            .checked_add(u64::try_from(count).map_err(|_| IdentityExhausted(T::NAME))?)
            .ok_or(IdentityExhausted(T::NAME))?;
        let range = self.last..end;
        self.last = end;
        Ok(range.map(|value| T::from_nonzero(NonZeroU64::new(value + 1).unwrap())))
    }
    #[cfg(test)]
    pub(crate) fn exhausted() -> Self {
        Self {
            last: u64::MAX,
            marker: PhantomData,
        }
    }

    pub fn next_id(&mut self) -> Result<T, IdentityExhausted> {
        let next = self.last.checked_add(1).ok_or(IdentityExhausted(T::NAME))?;
        self.last = next;
        Ok(T::from_nonzero(NonZeroU64::new(next).unwrap()))
    }
}

/// Local state-map index is scoped by the globally non-reused CodeVersion.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExitSiteKey {
    pub source: CodeVersion,
    pub state_map: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublishedEntry {
    pub unit: CodeUnitId,
    pub version: CodeVersion,
    pub canonical: NonZeroUsize,
    pub fast: NonZeroUsize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HcqEntry {
    pub entry: PublishedEntry,
    pub family: HcqFamilyId,
    pub family_version: FamilyVersion,
}

/// Publish ONE immutable instance, not separate atomics for these fields. The
/// preferred entry is derived, so it cannot disagree with the resident tiers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DispatchPayload {
    reachability: ReachabilityVersion,
    lcq: Option<PublishedEntry>,
    hcq: Option<HcqEntry>,
}
impl DispatchPayload {
    pub const fn new(
        reachability: ReachabilityVersion,
        lcq: Option<PublishedEntry>,
        hcq: Option<HcqEntry>,
    ) -> Self {
        Self {
            reachability,
            lcq,
            hcq,
        }
    }
    pub const fn reachability(&self) -> ReachabilityVersion {
        self.reachability
    }
    pub const fn lcq(&self) -> Option<PublishedEntry> {
        self.lcq
    }
    pub const fn hcq(&self) -> Option<HcqEntry> {
        self.hcq
    }
    pub const fn preferred(&self) -> Option<PublishedEntry> {
        match self.hcq {
            Some(hcq) => Some(hcq.entry),
            None => self.lcq,
        }
    }
}

/// Version of the NativeFrame/register/entry layout recorded by CodeUnits.
pub const NATIVE_ABI_VERSION: u32 = 1;
pub const TRANSFER_BYTES: u32 = 2048;
pub const SPILL_BYTES: u32 = 16384;
pub const SAMPLE_INTERVAL: i64 = 4096;

/// Stable codes written to NativeFrame.exit_reason. The storage is u32 so an
/// invalid native result can be diagnosed without constructing an invalid Rust enum.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeExitReason {
    None = 0,
    Dispatch = 1,
    BudgetExhausted = 2,
    Control = 3,
    Architectural = 4,
    Unsupported = 5,
    DataFault = 6,
    Scheduled = 7,
    Internal = 8,
    Reconcile = 9,
}

impl TryFrom<u32> for NativeExitReason {
    type Error = u32;
    fn try_from(value: u32) -> Result<Self, u32> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Dispatch),
            2 => Ok(Self::BudgetExhausted),
            3 => Ok(Self::Control),
            4 => Ok(Self::Architectural),
            5 => Ok(Self::Unsupported),
            6 => Ok(Self::DataFault),
            7 => Ok(Self::Scheduled),
            8 => Ok(Self::Internal),
            9 => Ok(Self::Reconcile),
            unknown => Err(unknown),
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct PollBudget {
    pub sample_remaining: i64,
    pub slice_remaining: i64,
    pub armed_span: i64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BudgetError {
    InvalidSample,
    ExhaustedSlice,
    InvalidDeadline,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PollOutcome {
    pub sample: bool,
    pub exhausted: bool,
}
impl PollBudget {
    pub fn new(sample_remaining: i64, slice_remaining: i64) -> Result<Self, BudgetError> {
        if !(1..=SAMPLE_INTERVAL).contains(&sample_remaining) {
            return Err(BudgetError::InvalidSample);
        }
        if slice_remaining <= 0 {
            return Err(BudgetError::ExhaustedSlice);
        }
        Ok(Self {
            sample_remaining,
            slice_remaining,
            armed_span: sample_remaining.min(slice_remaining),
        })
    }
    /// Reconcile the pinned counter at a cold poll or canonical exit. Negative
    /// deadlines carry overshoot; even forced exits charge both balances.
    pub fn reconcile(&mut self, remaining: i64, forced: bool) -> Result<PollOutcome, BudgetError> {
        let spent = self
            .armed_span
            .checked_sub(remaining)
            .filter(|spent| *spent >= 0)
            .ok_or(BudgetError::InvalidDeadline)?;
        let sample = self
            .sample_remaining
            .checked_sub(spent)
            .ok_or(BudgetError::InvalidDeadline)?;
        let slice = self
            .slice_remaining
            .checked_sub(spent)
            .ok_or(BudgetError::InvalidDeadline)?;
        self.sample_remaining = if sample <= 0 {
            let phase = sample.rem_euclid(SAMPLE_INTERVAL);
            if phase == 0 { SAMPLE_INTERVAL } else { phase }
        } else {
            sample
        };
        self.slice_remaining = slice;
        self.armed_span = if slice > 0 {
            self.sample_remaining.min(slice)
        } else {
            0
        };
        Ok(PollOutcome {
            sample: sample <= 0 && !forced,
            exhausted: slice <= 0,
        })
    }
}

/// Shared ownership record used by the existing FP environment implementation
/// and the new gateway. No second host/guest FP protocol is introduced.
#[repr(C)]
#[derive(Default)]
pub struct HostFpState {
    pub(crate) saved_control: u64,
    pub(crate) saved_status: u64,
    pub(crate) active: u32,
    pub(crate) saved: u32,
    pub(crate) suspended: u32,
}

/// These point into the canonical A64State; never a duplicate register image.
#[repr(C)]
pub struct CanonicalState {
    pub state: *mut A64State,
    pub x: *mut u64,
    pub sp: *mut u64,
    pub vector: *mut u128,
    pub pc: *mut u64,
    pub nzcv: *mut Nzcv,
    pub fpcr: *mut u32,
    pub fpsr: *mut u32,
    pub tpidr_el0: *mut u64,
    pub tpidrro_el0: *mut u64,
}
impl CanonicalState {
    fn new(state: &mut A64State) -> Self {
        Self {
            state,
            x: state.general_register_storage_mut().as_mut_ptr(),
            sp: state.stack_pointer_storage_mut(),
            vector: state.vector_register_storage_mut().as_mut_ptr(),
            pc: state.program_counter_storage_mut(),
            nzcv: state.nzcv_storage_mut(),
            fpcr: state.fpcr_storage_mut(),
            fpsr: state.fpsr_storage_mut(),
            tpidr_el0: state.tpidr_el0_storage_mut(),
            tpidrro_el0: state.tpidrro_el0_storage_mut(),
        }
    }
}

/// The spill arena starts at offset zero, matching the tested Cranelift fork.
/// Only allocated/initialized slots may be read. Host services stay behind a
/// caller-owned pointer; generated guest instructions never interpret it.
#[repr(C, align(64))]
pub struct NativeFrame<'a> {
    pub spill: [MaybeUninit<u8>; SPILL_BYTES as usize],
    pub canonical: CanonicalState,
    pub budget: PollBudget,
    pub host_fp: HostFpState,
    pub execution_epoch: u64, // zero only when inactive
    pub admission_epoch: u64,
    pub runtime: *mut c_void,
    pub exit_pc: u64,
    pub exit_source_version: u64,
    pub exit_state_map: u32,
    pub exit_reason: u32,
    /// Invocation-local assembly continuation, installed by the native gateway.
    /// Never an inter-unit link or a host return address used by guest units.
    pub(crate) gateway_exit: usize,
    state_borrow: PhantomData<&'a mut A64State>,
}
impl<'a> NativeFrame<'a> {
    pub fn new(state: &'a mut A64State, budget: PollBudget) -> Self {
        Self {
            spill: [MaybeUninit::uninit(); SPILL_BYTES as usize],
            canonical: CanonicalState::new(state),
            budget,
            host_fp: HostFpState::default(),
            execution_epoch: 0,
            admission_epoch: 0,
            runtime: std::ptr::null_mut(),
            exit_pc: 0,
            exit_source_version: 0,
            exit_state_map: 0,
            exit_reason: 0,
            gateway_exit: 0,
            state_borrow: PhantomData,
        }
    }
}
const _: () = assert!(offset_of!(NativeFrame<'static>, spill) == 0);
const _: () = assert!(offset_of!(NativeFrame<'static>, canonical) == SPILL_BYTES as usize);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostAbi {
    X86_64,
    Aarch64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReservedRegisters {
    pub frame: u8,
    pub poll: u8,
    pub arena: u8,
    pub link_scratch: &'static [u8],
}
impl HostAbi {
    /// Architectural register encodings, not virtual-register or allocation IDs.
    pub const fn reserved(self) -> ReservedRegisters {
        match self {
            Self::X86_64 => ReservedRegisters {
                frame: 15,
                poll: 14,
                arena: 13,
                link_scratch: &[11],
            },
            Self::Aarch64 => ReservedRegisters {
                frame: 21,
                poll: 20,
                arena: 19,
                link_scratch: &[16, 17],
            },
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegisterClass {
    Integer,
    Vector,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueLocation {
    Register {
        class: RegisterClass,
        index: u8,
    },
    /// Absolute byte offset from the pinned NativeFrame register.
    Spill {
        offset: u32,
        bytes: u8,
    },
    Constant(u128),
}
impl ValueLocation {
    pub fn valid(self, abi: HostAbi, bytes: u8) -> bool {
        match self {
            Self::Register {
                class: RegisterClass::Integer,
                index,
            } => {
                let reserved = abi.reserved();
                matches!(bytes, 1 | 2 | 4 | 8)
                    && index != reserved.frame
                    && index != reserved.poll
                    && index != reserved.arena
                    && !reserved.link_scratch.contains(&index)
                    && match abi {
                        // Architectural register numbers; r4=RSP, r5=RBP.
                        HostAbi::X86_64 => index < 16 && !matches!(index, 4 | 5),
                        HostAbi::Aarch64 => index < 29 && index != 18,
                    }
            }
            Self::Register {
                class: RegisterClass::Vector,
                index,
            } => {
                matches!(bytes, 1 | 2 | 4 | 8 | 16)
                    && index < if abi == HostAbi::X86_64 { 16 } else { 32 }
            }
            Self::Spill {
                offset,
                bytes: extent,
            } => {
                extent == bytes
                    && matches!(bytes, 1 | 2 | 4 | 8 | 16)
                    && offset >= TRANSFER_BYTES
                    && offset % u32::from(bytes) == 0
                    && offset
                        .checked_add(u32::from(bytes))
                        .is_some_and(|end| end <= SPILL_BYTES)
            }
            Self::Constant(value) => match bytes {
                1 => value <= u128::from(u8::MAX),
                2 => value <= u128::from(u16::MAX),
                4 => value <= u128::from(u32::MAX),
                8 => value <= u128::from(u64::MAX),
                16 => true,
                _ => false,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuestValue {
    General(u8),
    Sp,
    Vector(u8),
    Fpcr,
    Fpsr,
    TpidrEl0,
    TpidrroEl0,
}
impl GuestValue {
    pub fn state(self) -> Option<StateSet> {
        let mut state = StateSet::default();
        match self {
            Self::General(index) if index < 31 => state.integer.x[usize::from(index)] = true,
            Self::Vector(index) if index < 32 => state.vector[usize::from(index)] = true,
            Self::Sp => state.integer.sp = true,
            Self::Fpcr => state.fpcr = true,
            Self::Fpsr => state.fpsr = true,
            Self::TpidrEl0 => state.tpidr_el0 = true,
            Self::TpidrroEl0 => state.tpidrro_el0 = true,
            _ => return None,
        }
        Some(state)
    }
    pub const fn bytes(self) -> u8 {
        match self {
            Self::Vector(_) => 16,
            Self::Fpcr | Self::Fpsr => 4,
            _ => 8,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValueBinding {
    pub value: GuestValue,
    pub location: ValueLocation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NzcvLocation {
    /// Canonical NZCV is still authoritative (no dirty lazy producer).
    Canonical,
    Packed(ValueLocation),
    /// Host condition flags; bridges/poll arithmetic must preserve the live bits.
    Host {
        carry_inverted: bool,
    },
    Deferred(LazyFlags<ValueLocation>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CanonicalEntryContract {
    pub live_in: StateSet,
}

/// Physical fast ingress. Canonical ingress has no physical input requirements.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EntryContract {
    pub live_in: StateSet,
    pub abi: HostAbi,
    pub bindings: Box<[ValueBinding]>,
    pub nzcv: NzcvLocation,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExitStateMap {
    pub site: ExitSiteKey,
    pub abi: HostAbi,
    /// Include clean live values too: helper saves and fast bridges need their
    /// physical locations even when canonical exit need not store them.
    pub live: StateSet,
    pub dirty_live: StateSet,
    pub bindings: Box<[ValueBinding]>,
    pub nzcv: NzcvLocation,
    /// OR pending host status with the mapped/canonical software FPSR.
    pub host_fpsr_pending: bool,
}

/// Canonical ingress needs only live_in. Fast ingress additionally needs the
/// physical bindings. Boundary emission supplies these from FINAL allocation.
impl EntryContract {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.live_in.nzcv != 0 && self.nzcv == NzcvLocation::Canonical {
            return Err("fast ingress needs a physical NZCV contract");
        }
        validate_bindings(self.abi, self.live_in, &self.bindings, false)?;
        for (index, binding) in self.bindings.iter().enumerate() {
            if matches!(binding.location, ValueLocation::Constant(_)) {
                return Err("fast inputs need register or spill locations");
            }
            if self.bindings[..index]
                .iter()
                .any(|other| locations_overlap(binding.location, other.location))
            {
                return Err("distinct fast inputs overlap");
            }
        }
        if self.live_in.nzcv != 0
            && matches!(
                self.nzcv,
                NzcvLocation::Packed(ValueLocation::Constant(_)) | NzcvLocation::Deferred(_)
            )
        {
            return Err(
                "fast NZCV input must be packed or host flags, not a source producer recipe",
            );
        }
        if self.live_in.nzcv != 0
            && let NzcvLocation::Packed(location) = self.nzcv
            && self
                .bindings
                .iter()
                .any(|binding| locations_overlap(location, binding.location))
        {
            return Err("fast NZCV input overlaps another input");
        }
        validate_nzcv(self.abi, self.live_in.nzcv, &self.nzcv)
    }
}
impl ExitStateMap {
    /// Infrastructure clobbers host flags, not architectural NZCV. Materialize
    /// or preserve only this intersection before emitting a clobbering bridge,
    /// poll or lookup; deferred SSA recipes do not depend on host flags.
    pub fn flags_to_preserve(&self, host_clobbers: u8) -> u8 {
        if matches!(self.nzcv, NzcvLocation::Host { .. }) {
            self.live.nzcv & host_clobbers
        } else {
            0
        }
    }
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.dirty_live.without(self.live).is_empty() {
            return Err("dirty live state is not a subset of live state");
        }
        if self.host_fpsr_pending && !self.dirty_live.fpsr {
            return Err("pending host FPSR is not accounted as dirty");
        }
        if self.dirty_live.nzcv != 0 && self.nzcv == NzcvLocation::Canonical {
            return Err("dirty NZCV cannot remain canonical");
        }
        validate_bindings(self.abi, self.live, &self.bindings, self.host_fpsr_pending)?;
        validate_nzcv(self.abi, self.live.nzcv, &self.nzcv)
    }
}

fn validate_bindings(
    abi: HostAbi,
    mut required: StateSet,
    bindings: &[ValueBinding],
    host_fpsr: bool,
) -> Result<(), &'static str> {
    required.nzcv = 0;
    let mut found = StateSet::default();
    for binding in bindings {
        let state = binding.value.state().ok_or("invalid guest register")?;
        if !found.intersection(state).is_empty() {
            return Err("duplicate guest binding");
        }
        if !binding.location.valid(abi, binding.value.bytes()) {
            return Err("invalid or reserved physical location");
        }
        found = found.union(state);
    }
    if host_fpsr {
        found.fpsr = true;
    }
    if found != required {
        return Err("physical map does not exactly cover required state");
    }
    Ok(())
}

fn validate_nzcv(abi: HostAbi, bits: u8, location: &NzcvLocation) -> Result<(), &'static str> {
    if bits & !crate::analysis::NZCV != 0 {
        return Err("invalid NZCV mask");
    }
    if let NzcvLocation::Packed(value) = location
        && !value.valid(abi, 4)
    {
        return Err("invalid packed NZCV location");
    }
    if let NzcvLocation::Deferred(recipe) = location {
        validate_recipe(abi, recipe)?;
    }
    Ok(())
}

pub(crate) fn locations_overlap(a: ValueLocation, b: ValueLocation) -> bool {
    match (a, b) {
        (
            ValueLocation::Register {
                class: ac,
                index: ai,
            },
            ValueLocation::Register {
                class: bc,
                index: bi,
            },
        ) => ac == bc && ai == bi,
        (
            ValueLocation::Spill {
                offset: a,
                bytes: an,
            },
            ValueLocation::Spill {
                offset: b,
                bytes: bn,
            },
        ) => a < b + u32::from(bn) && b < a + u32::from(an),
        _ => false,
    }
}

fn validate_recipe(abi: HostAbi, recipe: &LazyFlags<ValueLocation>) -> Result<(), &'static str> {
    let check = |value: ValueLocation, bytes| {
        if value.valid(abi, bytes) {
            Ok(())
        } else {
            Err("invalid lazy NZCV operand")
        }
    };
    match recipe {
        LazyFlags::Canonical(value) | LazyFlags::Packed(value) => check(*value, 4),
        LazyFlags::Add {
            lhs,
            rhs,
            result,
            width,
        }
        | LazyFlags::Subtract {
            lhs,
            rhs,
            result,
            width,
        } => {
            if !matches!(width, 32 | 64) {
                return Err("invalid lazy NZCV width");
            }
            for value in [lhs, rhs, result] {
                check(*value, width / 8)?;
            }
            Ok(())
        }
        LazyFlags::AddCarry {
            lhs,
            rhs,
            result,
            carry,
            width,
        }
        | LazyFlags::SubtractCarry {
            lhs,
            rhs,
            result,
            carry,
            width,
        } => {
            if !matches!(width, 32 | 64) {
                return Err("invalid lazy NZCV width");
            }
            for value in [lhs, rhs, result] {
                check(*value, width / 8)?;
            }
            check(*carry, 1)
        }
        LazyFlags::Logical { result, width } => {
            if !matches!(width, 32 | 64) {
                return Err("invalid lazy NZCV width");
            }
            check(*result, width / 8)
        }
        LazyFlags::Conditional {
            predicate,
            when_true,
            when_false,
        } => {
            check(*predicate, 1)?;
            if when_false & !0xf != 0 {
                return Err("invalid conditional NZCV literal");
            }
            validate_recipe(abi, when_true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn identities_never_wrap_or_reuse_zero() {
        let mut counter = CheckedCounter::<CodeVersion>::default();
        assert_eq!(counter.next_id().unwrap().get(), 1);
        counter.last = u64::MAX - 1;
        assert_eq!(counter.next_id().unwrap().get(), u64::MAX);
        assert_eq!(counter.next_id(), Err(IdentityExhausted("CodeVersion")));
        assert_eq!(counter.next_id(), Err(IdentityExhausted("CodeVersion")));
        assert!(CodeVersion::new(0).is_none());
    }

    #[test]
    fn keys_distinguish_semantics_but_not_mapping_generations() {
        let cpu = ProcessCpuContext::new(TargetPlatform::Switch1, AddressSpaceId::new(1));
        let key = BlockKey::new(
            cpu,
            GuestVirtualAddress::new(0x1000),
            FpSpecialization::Dynamic,
        )
        .unwrap();
        assert_ne!(
            key,
            BlockKey {
                fp: FpSpecialization::Exact(0),
                ..key
            }
        );
        assert_ne!(
            key,
            BlockKey {
                profile: CpuProfileId::new(17),
                ..key
            }
        );
        assert_ne!(
            key,
            BlockKey {
                address_space: AddressSpaceId::new(2),
                ..key
            }
        );
        assert!(key.at(GuestVirtualAddress::new(0x1002)).is_none());
        assert_ne!(
            InstructionKey::new(key),
            InstructionKey::new(key.at(GuestVirtualAddress::new(0x1004)).unwrap())
        );
    }

    #[test]
    fn dispatch_preference_is_a_coherent_immutable_choice() {
        let lcq = PublishedEntry {
            unit: CodeUnitId::new(1).unwrap(),
            version: CodeVersion::new(1).unwrap(),
            canonical: NonZeroUsize::new(0x1000).unwrap(),
            fast: NonZeroUsize::new(0x1010).unwrap(),
        };
        let hcq = HcqEntry {
            entry: PublishedEntry {
                unit: CodeUnitId::new(2).unwrap(),
                version: CodeVersion::new(2).unwrap(),
                canonical: NonZeroUsize::new(0x2000).unwrap(),
                fast: NonZeroUsize::new(0x2010).unwrap(),
            },
            family: HcqFamilyId::new(1).unwrap(),
            family_version: FamilyVersion::new(1).unwrap(),
        };
        let reachability = ReachabilityVersion::new(1).unwrap();
        assert_eq!(
            DispatchPayload::new(reachability, None, None).preferred(),
            None
        );
        assert_eq!(
            DispatchPayload::new(reachability, Some(lcq), None).preferred(),
            Some(lcq)
        );
        let payload = DispatchPayload::new(reachability, Some(lcq), Some(hcq));
        assert_eq!(payload.preferred(), Some(hcq.entry));
        assert_eq!(payload.lcq(), Some(lcq));
        assert_eq!(payload.hcq(), Some(hcq));
        assert_eq!(payload.reachability(), reachability);
    }

    #[test]
    fn frame_matches_fork_spill_partition_and_borrows_canonical_storage() {
        assert_eq!(align_of::<NativeFrame<'_>>(), 64);
        assert_eq!(offset_of!(NativeFrame<'_>, spill), 0);
        assert_eq!(offset_of!(NativeFrame<'_>, canonical), 16384);
        assert_eq!(size_of::<NativeFrame<'_>>() % 64, 0);
        let mut state = A64State::default();
        let original = state.general_register_storage_mut().as_mut_ptr();
        let frame = NativeFrame::new(&mut state, PollBudget::new(4096, 123).unwrap());
        assert_eq!(frame.canonical.x, original);
        assert_eq!(frame.spill.as_ptr() as usize % 64, 0);
        assert_eq!(frame.spill.len() - TRANSFER_BYTES as usize, 14336);
        assert_eq!(
            NativeExitReason::try_from(frame.exit_reason),
            Ok(NativeExitReason::None)
        );
        assert_eq!(NativeExitReason::try_from(u32::MAX), Err(u32::MAX));
    }

    #[test]
    fn poll_budget_charges_overshoot_and_forced_transitions() {
        let mut budget = PollBudget::new(5, 30).unwrap();
        assert_eq!(
            budget.reconcile(-3, false).unwrap(),
            PollOutcome {
                sample: true,
                exhausted: false
            }
        );
        assert_eq!(budget.sample_remaining, 4093);
        assert_eq!(budget.slice_remaining, 22);
        assert_eq!(budget.armed_span, 22);
        assert!(budget.reconcile(-2, true).unwrap().exhausted);
        assert_eq!(budget.slice_remaining, -2);
        let mut budget = PollBudget::new(1, 10000).unwrap();
        assert!(!budget.reconcile(-8192, true).unwrap().sample);
        assert_eq!(budget.sample_remaining, 4096);
        assert_eq!(budget.slice_remaining, 1807);
        assert_eq!(
            PollBudget::new(1, 0).unwrap_err(),
            BudgetError::ExhaustedSlice
        );
    }

    #[test]
    fn physical_maps_reject_reserved_registers_missing_state_and_aliasing() {
        for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
            let pins: &[u8] = if abi == HostAbi::X86_64 {
                &[4, 5, 11, 13, 14, 15]
            } else {
                &[16, 17, 18, 19, 20, 21, 29, 30, 31]
            };
            for &index in pins {
                assert!(
                    !ValueLocation::Register {
                        class: RegisterClass::Integer,
                        index
                    }
                    .valid(abi, 8)
                );
            }
            assert!(
                !ValueLocation::Spill {
                    offset: 2040,
                    bytes: 8
                }
                .valid(abi, 8)
            );
            assert!(
                ValueLocation::Spill {
                    offset: 16368,
                    bytes: 16
                }
                .valid(abi, 16)
            );
            assert!(
                !ValueLocation::Spill {
                    offset: 16384,
                    bytes: 8
                }
                .valid(abi, 8)
            );
            let mut required = StateSet::default();
            required.integer.x[0] = true;
            let binding = ValueBinding {
                value: GuestValue::General(0),
                location: ValueLocation::Register {
                    class: RegisterClass::Integer,
                    index: 0,
                },
            };
            let mut entry = EntryContract {
                live_in: required,
                abi,
                bindings: vec![binding].into_boxed_slice(),
                nzcv: NzcvLocation::Canonical,
            };
            assert!(entry.validate().is_ok());
            entry.live_in.nzcv = crate::analysis::NZCV;
            entry.nzcv = NzcvLocation::Packed(binding.location);
            assert_eq!(
                entry.validate(),
                Err("fast NZCV input overlaps another input")
            );
            entry.live_in.nzcv = 0;
            entry.nzcv = NzcvLocation::Canonical;
            entry.live_in.integer.x[1] = true;
            assert!(entry.validate().is_err());
            entry.bindings = vec![
                binding,
                ValueBinding {
                    value: GuestValue::General(1),
                    ..binding
                },
            ]
            .into_boxed_slice();
            assert_eq!(entry.validate(), Err("distinct fast inputs overlap"));
        }
    }

    #[test]
    fn exit_maps_preserve_lazy_flags_and_host_sticky_status() {
        let required = StateSet {
            nzcv: crate::analysis::C,
            fpsr: true,
            ..StateSet::default()
        };
        let mut exit = ExitStateMap {
            site: ExitSiteKey {
                source: CodeVersion::new(3).unwrap(),
                state_map: 0,
            },
            abi: HostAbi::X86_64,
            live: required,
            dirty_live: required,
            bindings: Box::new([]),
            nzcv: NzcvLocation::Canonical,
            host_fpsr_pending: true,
        };
        assert!(exit.validate().is_err());
        exit.nzcv = NzcvLocation::Deferred(LazyFlags::Subtract {
            lhs: ValueLocation::Constant(8),
            rhs: ValueLocation::Constant(3),
            result: ValueLocation::Constant(5),
            width: 64,
        });
        assert!(exit.validate().is_ok());
        exit.live.integer.x[7] = true;
        exit.bindings = vec![ValueBinding {
            value: GuestValue::General(7),
            location: ValueLocation::Constant(42),
        }]
        .into_boxed_slice();
        assert!(
            exit.validate().is_ok(),
            "clean live locations are needed by fast bridges too"
        );
        assert!(
            !exit.dirty_live.integer.x[7],
            "canonical exit must not store the clean value"
        );
        exit.nzcv = NzcvLocation::Host {
            carry_inverted: true,
        };
        assert_eq!(
            exit.flags_to_preserve(crate::analysis::NZCV),
            crate::analysis::C
        );
        assert_eq!(exit.flags_to_preserve(crate::analysis::Z), 0);
        exit.nzcv = NzcvLocation::Deferred(LazyFlags::Conditional {
            predicate: ValueLocation::Spill {
                offset: TRANSFER_BYTES,
                bytes: 1,
            },
            when_true: Box::new(LazyFlags::Logical {
                result: ValueLocation::Constant(0),
                width: 32,
            }),
            when_false: 15,
        });
        assert!(exit.validate().is_ok());
        assert_eq!(exit.flags_to_preserve(crate::analysis::NZCV), 0);
        exit.nzcv = NzcvLocation::Packed(ValueLocation::Register {
            class: RegisterClass::Integer,
            index: 15,
        });
        assert!(exit.validate().is_err());
    }
}
