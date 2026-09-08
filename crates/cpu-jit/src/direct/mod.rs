mod compiler;
mod fp_env;
mod lookup;
mod region;
mod slow;

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;

use nixe_cpu::decode::{self, DecodeResult};
use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::exclusive::ExclusiveMonitorState;
use nixe_cpu::execution::{
    ArchitecturalTimer, BoundMemory, ControlRequest, CpuControl, CpuExit, CpuFault, CpuFaultKind,
    ExecutionReport, MemoryBinding, RunRequest, SchedulerRequest, VcpuEventState,
};
use nixe_cpu::location::{InstructionEncoding, LocationDescriptor};
use nixe_cpu::memory::{CodePageDependency, CodePageSpan, CpuMemory, DataAccessFault};
use nixe_cpu::memory::{DataAccessKind, DirectFaultResolution, MemoryAccessSize};
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register, A64State, Nzcv};
use nixe_cpu_direct_memory::{
    CapturedFault, FaultDisposition, NativeFaultRegion, NativeFaultRegistry, NativeFaultSite,
    NativeInvocation, NativeMemoryAccessKind, WorkerFaultContext,
};
use nixe_memory::{
    CpuMemoryBackend, DirectAddressSpaceView, GuestPhysicalPageId, GuestVirtualAddress,
    MemoryInvalidation, MemoryInvalidationCursor, MemoryInvalidationError, MemoryInvalidationKind,
    MemoryInvalidationSource,
};

use self::compiler::{
    CompilerRuntime, DirectCompiler, HCQ_COMPILER_POLICY, LCQ_COMPILER_POLICY, NativeGateway,
    Promotion,
};
use self::lookup::{EntryState, NativeLookupNode, NativeLookupSlot, RegionLookup};
use self::region::{
    HCQ_MAX_REGION_INSTRUCTIONS, LCQ_MAX_REGION_INSTRUCTIONS, RegionKey, discover_region,
};
use crate::abi::NativeExitReason;
use crate::fp_policy::{NATIVE_FPCR_MASK, native_fpcr_supported};

const DEFAULT_MAX_NATIVE_CODE_BYTES: usize = 1024 * 1024 * 1024;
const MAX_NATIVE_FAULT_REGIONS: usize = 262_144;
const HCQ_PENDING_PER_WORKER: usize = 64;

const fn hcq_worker_count(logical_processors: usize) -> usize {
    let workers = logical_processors.saturating_sub(6) / 3;
    if workers < 1 {
        1
    } else if workers > 4 {
        4
    } else {
        workers
    }
}

const EXIT_NONE: u32 = NativeExitReason::None as u32;
const EXIT_DISPATCH: u32 = NativeExitReason::Dispatch as u32;
const EXIT_CONTROL: u32 = NativeExitReason::Control as u32;
const EXIT_ARCHITECTURAL: u32 = NativeExitReason::Architectural as u32;
const EXIT_UNSUPPORTED: u32 = NativeExitReason::Unsupported as u32;
const EXIT_DATA_FAULT: u32 = NativeExitReason::DataFault as u32;
const EXIT_SCHEDULED: u32 = NativeExitReason::Scheduled as u32;
const EXIT_INTERNAL: u32 = NativeExitReason::Internal as u32;
const EXIT_RECONCILE: u32 = NativeExitReason::Reconcile as u32;

const COARSE_PROGRESS: u64 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectJitErrorKind {
    InvalidGuestCode,
    Unsupported,
    Capacity,
    Shutdown,
    Internal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectJitError {
    kind: DirectJitErrorKind,
    detail: Box<str>,
}

impl DirectJitError {
    fn invalid(detail: impl Into<Box<str>>) -> Self {
        Self {
            kind: DirectJitErrorKind::InvalidGuestCode,
            detail: detail.into(),
        }
    }

    fn unsupported(detail: impl Into<Box<str>>) -> Self {
        Self {
            kind: DirectJitErrorKind::Unsupported,
            detail: detail.into(),
        }
    }

    fn capacity(detail: impl Into<Box<str>>) -> Self {
        Self {
            kind: DirectJitErrorKind::Capacity,
            detail: detail.into(),
        }
    }

    fn internal(detail: impl Into<Box<str>>) -> Self {
        Self {
            kind: DirectJitErrorKind::Internal,
            detail: detail.into(),
        }
    }

    fn shutdown() -> Self {
        Self {
            kind: DirectJitErrorKind::Shutdown,
            detail: "direct JIT process is shut down".into(),
        }
    }
}

impl fmt::Display for DirectJitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for DirectJitError {}

#[derive(Clone, Debug, Eq, PartialEq)]
enum DirectExit {
    Dispatch {
        pc: GuestVirtualAddress,
        progress: u64,
    },
    Control {
        pc: GuestVirtualAddress,
        progress: u64,
    },
    Architectural {
        pc: GuestVirtualAddress,
        detail: u32,
        progress: u64,
    },
    Unsupported {
        pc: GuestVirtualAddress,
        progress: u64,
    },
    DataFault {
        pc: GuestVirtualAddress,
        fault: DataAccessFault,
        progress: u64,
    },
    Scheduled {
        pc: GuestVirtualAddress,
        request: SchedulerRequest,
        progress: u64,
    },
    LoaderReturn {
        pc: GuestVirtualAddress,
        result_code: u64,
        progress: u64,
    },
    Internal {
        pc: GuestVirtualAddress,
        progress: u64,
    },
    Reconcile,
}

#[repr(C)]
struct NativeContext {
    state: *mut A64State,
    x: *mut u64,
    vector: *mut u128,
    sp: *mut u64,
    pc: *mut u64,
    nzcv: *mut Nzcv,
    fpcr: *mut u32,
    fpsr: *mut u32,
    tpidr_el0: *mut u64,
    tpidrro_el0: *mut u64,
    memory: *const dyn CpuMemory,
    exclusive: *mut ExclusiveMonitorState,
    timer: *const dyn ArchitecturalTimer,
    events: *const VcpuEventState,
    address_space: u64,
    direct_base: usize,
    direct_size: usize,
    loader_return: u64,
    control_pending: usize,
    synchronization_counter: usize,
    exit_pc: u64,
    exit_kind: u32,
    exit_detail: u32,
    slow_status: u32,
    slow_result_low: u64,
    slow_result_high: u64,
    slow_result_flags: u64,
    // Ephemeral native-invocation cache; canonical FPCR remains in `state`.
    guest_fpcr: u32,
    native_fp_enabled: u32,
    host_fp: fp_env::HostFpState,
    native_lookup: *const NativeLookupSlot,
    invalidation_signal: *const AtomicU64,
    invalidation_cursor: u64,
    process_pending: *const AtomicU32,
    hcq_scheduler: *const HcqScheduler,
    data_fault: Option<DataAccessFault>,
    direct_fault_error: Option<Box<str>>,
}

impl NativeContext {
    #[allow(clippy::too_many_arguments)]
    fn new(
        state: &mut A64State,
        memory: &dyn CpuMemory,
        exclusive: &mut ExclusiveMonitorState,
        timer: &dyn ArchitecturalTimer,
        events: &VcpuEventState,
        address_space: nixe_memory::AddressSpaceId,
        loader_return: Option<GuestVirtualAddress>,
        control: &CpuControl,
        native_lookup: *const NativeLookupSlot,
        process_pending: &AtomicU32,
        hcq_scheduler: &HcqScheduler,
    ) -> Self {
        let direct = memory.direct_address_space_view(address_space);
        let state_pointer = std::ptr::from_mut(&mut *state);
        let x = state.general_register_storage_mut().as_mut_ptr();
        let vector = state.vector_register_storage_mut().as_mut_ptr();
        let sp = std::ptr::from_mut(state.stack_pointer_storage_mut());
        let pc = std::ptr::from_mut(state.program_counter_storage_mut());
        let nzcv = std::ptr::from_mut(state.nzcv_storage_mut());
        let guest_fpcr = state.fpcr();
        let fpcr = std::ptr::from_mut(state.fpcr_storage_mut());
        let fpsr = std::ptr::from_mut(state.fpsr_storage_mut());
        let tpidr_el0 = std::ptr::from_mut(state.tpidr_el0_storage_mut());
        let tpidrro_el0 = std::ptr::from_mut(state.tpidrro_el0_storage_mut());
        let invalidation_signal = memory.invalidation_signal();
        // Native code and every slow callback complete before `run` returns.
        // Erasing these borrow lifetimes keeps the generated ABI pointer-only;
        // no pointer escapes the stack-owned native context.
        let memory = unsafe { std::mem::transmute::<&dyn CpuMemory, *const dyn CpuMemory>(memory) };
        let timer = unsafe {
            std::mem::transmute::<&dyn ArchitecturalTimer, *const dyn ArchitecturalTimer>(timer)
        };
        Self {
            state: state_pointer,
            x,
            vector,
            sp,
            pc,
            nzcv,
            fpcr,
            fpsr,
            tpidr_el0,
            tpidrro_el0,
            memory,
            exclusive,
            timer,
            events,
            address_space: address_space.get(),
            direct_base: direct.map_or(0, |view| view.base),
            direct_size: direct.map_or(0, |view| view.address_space_size),
            loader_return: loader_return.map_or(u64::MAX, GuestVirtualAddress::get),
            control_pending: control.pending_word_address(),
            synchronization_counter: control.synchronization_counter_address(),
            exit_pc: 0,
            exit_kind: EXIT_NONE,
            exit_detail: 0,
            slow_status: 0,
            slow_result_low: 0,
            slow_result_high: 0,
            slow_result_flags: 0,
            guest_fpcr,
            native_fp_enabled: u32::from(native_fpcr_supported(guest_fpcr)),
            host_fp: fp_env::HostFpState::default(),
            native_lookup,
            invalidation_signal,
            invalidation_cursor: MemoryInvalidationCursor::INITIAL.get(),
            process_pending,
            hcq_scheduler,
            data_fault: None,
            direct_fault_error: None,
        }
    }

    fn exit(&mut self) -> Result<DirectExit, DirectJitError> {
        if let Some(detail) = self.direct_fault_error.take() {
            return Err(DirectJitError::internal(detail));
        }
        let pc = GuestVirtualAddress::new(self.exit_pc);
        match self.exit_kind {
            EXIT_DISPATCH => Ok(DirectExit::Dispatch {
                pc,
                progress: COARSE_PROGRESS,
            }),
            EXIT_CONTROL => Ok(DirectExit::Control { pc, progress: 0 }),
            EXIT_ARCHITECTURAL => Ok(DirectExit::Architectural {
                pc,
                detail: self.exit_detail,
                progress: COARSE_PROGRESS,
            }),
            EXIT_UNSUPPORTED => Ok(DirectExit::Unsupported { pc, progress: 0 }),
            EXIT_DATA_FAULT => Ok(DirectExit::DataFault {
                pc,
                fault: self.data_fault.clone().ok_or_else(|| {
                    DirectJitError::internal("direct JIT data-fault exit has no fault")
                })?,
                progress: COARSE_PROGRESS,
            }),
            EXIT_SCHEDULED => {
                let request = match self.exit_detail {
                    0 => SchedulerRequest::Yield,
                    1 => SchedulerRequest::WaitForEvent,
                    2 => SchedulerRequest::WaitForInterrupt,
                    3 => SchedulerRequest::SendEvent,
                    detail => {
                        return Err(DirectJitError::internal(format!(
                            "direct JIT returned unknown scheduler request {detail}"
                        )));
                    }
                };
                Ok(DirectExit::Scheduled {
                    pc,
                    request,
                    progress: COARSE_PROGRESS,
                })
            }
            EXIT_INTERNAL => Ok(DirectExit::Internal { pc, progress: 0 }),
            EXIT_RECONCILE => Ok(DirectExit::Reconcile),
            kind => Err(DirectJitError::internal(format!(
                "direct JIT returned unknown native exit {kind}"
            ))),
        }
    }
}

struct PublishedRegion {
    key: RegionKey,
    #[cfg(test)]
    entry: usize,
    #[cfg(test)]
    native_bytes: usize,
    #[cfg(test)]
    clif_instructions: usize,
    #[cfg(test)]
    deferred_register_loads: usize,
    #[cfg(test)]
    exit_tail_count: usize,
    dependencies: Box<[CodePageDependency]>,
    mapping_dependencies: Box<[CodePageSpan]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BoundMemoryBackend {
    address_space: nixe_memory::AddressSpaceId,
    end_exclusive: GuestVirtualAddress,
    backend: CpuMemoryBackend,
    direct: Option<DirectAddressSpaceView>,
}

#[derive(Clone, Copy)]
struct HcqRequest {
    key: RegionKey,
    generation: u64,
    invalidation_cursor: MemoryInvalidationCursor,
}

#[derive(Default)]
struct HcqStackState {
    pending: Vec<HcqRequest>,
    capacity: usize,
    shutdown: bool,
}

struct HcqScheduler {
    state: Mutex<HcqStackState>,
    available: Condvar,
    closed: AtomicBool,
}

impl HcqScheduler {
    fn new() -> Self {
        Self {
            state: Mutex::new(HcqStackState::default()),
            available: Condvar::new(),
            closed: AtomicBool::new(false),
        }
    }

    fn configure(&self, worker_count: usize) {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .capacity = HCQ_PENDING_PER_WORKER * worker_count;
    }

    fn push(&self, request: HcqRequest) -> bool {
        if self.closed.load(Ordering::Acquire) {
            return false;
        }
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if self.closed.load(Ordering::Acquire)
            || state.shutdown
            || state.pending.len() == state.capacity
        {
            return false;
        }
        state.pending.push(request);
        drop(state);
        self.available.notify_one();
        true
    }

    fn pop(&self) -> Option<HcqRequest> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if state.shutdown || self.closed.load(Ordering::Acquire) {
                return None;
            }
            if let Some(request) = state.pending.pop() {
                return Some(request);
            }
            state = self
                .available
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn close(&self) -> Vec<HcqRequest> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.closed.store(true, Ordering::Release);
        let pending = std::mem::take(&mut state.pending);
        drop(state);
        self.available.notify_all();
        pending
    }

    fn shutdown(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.shutdown = true;
        state.pending.clear();
        drop(state);
        self.available.notify_all();
    }
}

unsafe extern "C" fn request_hcq(
    scheduler: *const HcqScheduler,
    node: *const NativeLookupNode,
    invalidation_cursor: u64,
) {
    let scheduler = unsafe { &*scheduler };
    let node = unsafe { &*node };
    if scheduler.closed.load(Ordering::Acquire) {
        node.disable_promotion();
        return;
    }
    let Some(generation) = node.try_queue_hcq() else {
        return;
    };
    let request = HcqRequest {
        key: node.key(),
        generation,
        invalidation_cursor: MemoryInvalidationCursor::new(invalidation_cursor),
    };
    if !scheduler.push(request) {
        node.restore_lcq(generation);
    }
}

struct ProcessState {
    lookup: Arc<RegionLookup>,
    regions: HashMap<RegionKey, Arc<PublishedRegion>>,
    physical_dependencies: HashMap<GuestPhysicalPageId, HashSet<RegionKey>>,
    retired: Vec<Arc<PublishedRegion>>,
    invalidation_cursor: MemoryInvalidationCursor,
    invalidations: Vec<MemoryInvalidation>,
    native_bytes: usize,
    memory_backend: Option<BoundMemoryBackend>,
    memory_owner: Option<Arc<dyn BoundMemory>>,
    failure: Option<DirectJitError>,
}

impl ProcessState {
    #[cfg(test)]
    fn region_for(&self, key: RegionKey) -> Option<&Arc<PublishedRegion>> {
        self.regions.get(&key)
    }
}

pub struct JitProcess {
    cpu: ProcessCpuContext,
    max_native_code_bytes: usize,
    pending: AtomicU32,
    fault_registry: Arc<NativeFaultRegistry>,
    runtime: CompilerRuntime,
    published: Condvar,
    hcq: Arc<HcqScheduler>,
    hcq_started: AtomicBool,
    hcq_workers: Mutex<Vec<JoinHandle<()>>>,
    state: Mutex<ProcessState>,
}

impl JitProcess {
    pub fn new(cpu: ProcessCpuContext) -> Result<Self, DirectJitError> {
        crate::native::check_host().map_err(DirectJitError::unsupported)?;
        let fault_registry = Arc::new(
            NativeFaultRegistry::with_capacity(MAX_NATIVE_FAULT_REGIONS)
                .map_err(|error| DirectJitError::unsupported(error.to_string()))?,
        );
        let runtime = CompilerRuntime::new()?;
        Ok(Self {
            cpu,
            max_native_code_bytes: DEFAULT_MAX_NATIVE_CODE_BYTES,
            pending: AtomicU32::new(0),
            fault_registry,
            runtime,
            published: Condvar::new(),
            hcq: Arc::new(HcqScheduler::new()),
            hcq_started: AtomicBool::new(false),
            hcq_workers: Mutex::new(Vec::new()),
            state: Mutex::new(ProcessState {
                lookup: Arc::new(RegionLookup::new()),
                regions: HashMap::new(),
                physical_dependencies: HashMap::new(),
                retired: Vec::new(),
                invalidation_cursor: MemoryInvalidationCursor::INITIAL,
                invalidations: Vec::new(),
                native_bytes: 0,
                memory_backend: None,
                memory_owner: None,
                failure: None,
            }),
        })
    }

    pub fn bind_memory(&self, binding: MemoryBinding<'_>) -> Result<(), DirectJitError> {
        self.bind_memory_inner(binding, None)
    }

    pub fn bind_owned_memory(
        self: &Arc<Self>,
        binding: MemoryBinding<'_>,
        owner: Arc<dyn BoundMemory>,
    ) -> Result<(), DirectJitError> {
        if !std::ptr::eq(binding.memory, owner.as_ref()) {
            return Err(DirectJitError::invalid(
                "JIT memory owner differs from the bound memory",
            ));
        }
        self.bind_memory_inner(binding, Some(owner))?;
        self.start_hcq_workers()
    }

    fn bind_memory_inner(
        &self,
        binding: MemoryBinding<'_>,
        owner: Option<Arc<dyn BoundMemory>>,
    ) -> Result<(), DirectJitError> {
        if binding.address_space != self.cpu.address_space_id() {
            return Err(DirectJitError::invalid(
                "JIT memory binding uses a different process address space",
            ));
        }
        let backend = binding.memory.cpu_memory_backend(binding.address_space);
        let direct = match backend {
            CpuMemoryBackend::Checked => None,
            CpuMemoryBackend::LinuxDirect => {
                nixe_cpu_direct_memory::install()
                    .map_err(|error| DirectJitError::unsupported(error.to_string()))?;
                let view = binding
                    .memory
                    .direct_address_space_view(binding.address_space)
                    .ok_or_else(|| {
                        DirectJitError::internal(
                            "LinuxDirect memory binding has no direct address-space view",
                        )
                    })?;
                if view.address_space_size as u64 != binding.end_exclusive.get() {
                    return Err(DirectJitError::invalid(
                        "direct arena size differs from the process address space",
                    ));
                }
                Some(view)
            }
        };
        let requested = BoundMemoryBackend {
            address_space: binding.address_space,
            end_exclusive: binding.end_exclusive,
            backend,
            direct,
        };
        {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(bound) = state.memory_backend {
                if bound != requested {
                    return Err(DirectJitError::invalid(
                        "JIT process memory backend is immutable after binding",
                    ));
                }
                if let Some(owner) = owner {
                    match &state.memory_owner {
                        Some(bound_owner) if Arc::ptr_eq(bound_owner, &owner) => {}
                        Some(_) => {
                            return Err(DirectJitError::invalid(
                                "JIT process memory owner is immutable after binding",
                            ));
                        }
                        None => state.memory_owner = Some(owner),
                    }
                }
            } else {
                state.memory_backend = Some(requested);
                state.memory_owner = owner;
            }
        }
        Ok(())
    }

    fn bound_memory_backend(&self) -> Result<BoundMemoryBackend, DirectJitError> {
        Ok(self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .memory_backend
            .unwrap_or(BoundMemoryBackend {
                address_space: self.cpu.address_space_id(),
                end_exclusive: GuestVirtualAddress::new(0),
                backend: CpuMemoryBackend::Checked,
                direct: None,
            }))
    }

    fn start_hcq_workers(self: &Arc<Self>) -> Result<(), DirectJitError> {
        if self
            .hcq_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(());
        }
        let worker_count = hcq_worker_count(
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
        );
        self.hcq.configure(worker_count);
        let (memory, backend) = {
            let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let memory = state.memory_owner.as_ref().cloned().ok_or_else(|| {
                DirectJitError::internal("HCQ workers require process-owned memory")
            })?;
            let backend = state
                .memory_backend
                .ok_or_else(|| DirectJitError::internal("HCQ workers require bound memory"))?
                .backend;
            (memory, backend)
        };
        let mut workers = self
            .hcq_workers
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for index in 0..worker_count {
            let process = Arc::downgrade(self);
            let scheduler = Arc::clone(&self.hcq);
            let memory = Arc::clone(&memory);
            let runtime = self.runtime.addresses();
            let worker = std::thread::Builder::new()
                .name(format!("nixe-hcq-{index}"))
                .spawn(move || {
                    let mut compiler = match DirectCompiler::new(HCQ_COMPILER_POLICY, runtime) {
                        Ok(compiler) => compiler,
                        Err(error) => {
                            if let Some(process) = process.upgrade() {
                                process.fail_background(error);
                            }
                            return;
                        }
                    };
                    if let Err(error) = compiler.bind_memory_backend(backend) {
                        if let Some(process) = process.upgrade() {
                            process.fail_background(error);
                        }
                        return;
                    }
                    while let Some(request) = scheduler.pop() {
                        let Some(process) = process.upgrade() else {
                            return;
                        };
                        if let Err(error) =
                            process.compile_hcq(&mut compiler, memory.as_ref(), request)
                        {
                            if error.kind == DirectJitErrorKind::Capacity {
                                process.close_hcq_after_capacity(request);
                            } else {
                                process.fail_background(error);
                                return;
                            }
                        }
                    }
                })
                .map_err(|error| {
                    self.hcq.shutdown();
                    DirectJitError::unsupported(format!(
                        "could not start HCQ compilation worker: {error}"
                    ))
                })?;
            workers.push(worker);
        }
        Ok(())
    }

    fn close_hcq_after_capacity(&self, current: HcqRequest) {
        let pending = self.hcq.close();
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        for request in std::iter::once(current).chain(pending) {
            if let Some(node) = state.lookup.get(request.key) {
                node.restore_lcq_without_promotion(request.generation);
            }
        }
        drop(state);
        self.published.notify_all();
    }

    fn compile_hcq(
        &self,
        compiler: &mut DirectCompiler,
        memory: &dyn CpuMemory,
        request: HcqRequest,
    ) -> Result<(), DirectJitError> {
        let (lookup, node) = {
            let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let lookup = Arc::clone(&state.lookup);
            let Some(node) = lookup.get(request.key) else {
                return Ok(());
            };
            (lookup, node)
        };
        if node.generation() != request.generation
            || node.state() != EntryState::HcqQueued
            || node.entry() == 0
        {
            return Ok(());
        }
        let location = LocationDescriptor::new(request.key.start, self.cpu.profile_id());
        let region = match discover_region(
            self.cpu,
            memory,
            location,
            HCQ_MAX_REGION_INSTRUCTIONS,
            |_| false,
        ) {
            Ok(region) => region,
            Err(error) => {
                let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                reconcile_invalidations(&mut state, &lookup, memory, request.key.address_space)?;
                if node.generation() != request.generation
                    || node.state() != EntryState::HcqQueued
                    || node.entry() == 0
                {
                    return Ok(());
                }
                return Err(error);
            }
        };
        let entry_cells = region
            .external_exits
            .iter()
            .filter_map(|exit| exit.target.map(|target| request.key.at(target)))
            .map(|target| lookup.get_or_create(target).entry_address())
            .collect::<Vec<_>>();
        let mut compiled = compiler.compile(&region, &entry_cells, None)?;
        let fault_metadata = native_fault_metadata(&mut compiled)?;
        let entry = compiled.entry;
        let published = Arc::new(PublishedRegion {
            key: request.key,
            #[cfg(test)]
            entry: compiled.entry,
            #[cfg(test)]
            native_bytes: compiled.native_bytes,
            #[cfg(test)]
            clif_instructions: compiled.clif_instructions,
            #[cfg(test)]
            deferred_register_loads: compiled.deferred_register_loads,
            #[cfg(test)]
            exit_tail_count: compiled.exit_tail_count,
            dependencies: region.dependencies,
            mapping_dependencies: region.mapping_dependencies,
        });
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.native_bytes = state
            .native_bytes
            .checked_add(compiled.native_bytes)
            .ok_or_else(|| DirectJitError::capacity("direct JIT code size overflows"))?;
        if state.native_bytes > self.max_native_code_bytes {
            return Err(DirectJitError::capacity(format!(
                "direct JIT code arena exhausted: used={} limit={}",
                state.native_bytes, self.max_native_code_bytes,
            )));
        }
        reconcile_invalidations(&mut state, &lookup, memory, request.key.address_space)?;
        let invalidated = invalidations_since(
            memory,
            request.invalidation_cursor,
            request.key.address_space,
            &mut state.invalidations,
        )?;
        if self.pending.load(Ordering::Acquire) != 0 {
            return Ok(());
        }
        if node.generation() != request.generation
            || node.state() != EntryState::HcqQueued
            || node.entry() == 0
        {
            return Ok(());
        }
        if invalidated.affects_published(&published) {
            node.restore_lcq(request.generation);
            self.published.notify_all();
            return Ok(());
        }
        if let Some(metadata) = fault_metadata {
            self.fault_registry
                .publish(metadata)
                .map_err(|error| DirectJitError::internal(error.to_string()))?;
        }
        for dependency in &published.dependencies {
            state
                .physical_dependencies
                .entry(dependency.page)
                .or_default()
                .insert(request.key);
        }
        let previous = state.regions.insert(request.key, Arc::clone(&published));
        if !node.publish_hcq(request.generation, entry) {
            if let Some(previous) = previous {
                state.regions.insert(request.key, previous);
            } else {
                state.regions.remove(&request.key);
            }
            return Ok(());
        }
        if let Some(previous) = previous {
            state.retired.push(previous);
        }
        drop(state);
        self.published.notify_all();
        log::debug!(
            "direct JIT replaced LCQ with HCQ: address-space={:#x} pc={:#018x}",
            request.key.address_space.get(),
            request.key.start.get(),
        );
        Ok(())
    }

    fn fail_background(&self, error: DirectJitError) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.failure.is_none() {
            state.failure = Some(error);
        }
        self.pending.store(1, Ordering::Release);
        drop(state);
        self.published.notify_all();
    }

    /// Reconciles invalidations and returns an already-published entry without
    /// compiling. Callers use this after acquiring the short native-execution
    /// lease; a missing entry means that a transition invalidated the region
    /// between compilation and lease acquisition, so compilation is retried
    /// outside the lease.
    fn published_entry_for(
        &self,
        memory: &(impl CpuMemory + ?Sized),
        location: LocationDescriptor,
    ) -> Result<Option<(NativeGateway, usize, u64)>, DirectJitError> {
        let key = RegionKey::new(self.cpu, location);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let lookup = Arc::clone(&state.lookup);
        reconcile_invalidations(&mut state, &lookup, memory, key.address_space)?;
        if let Some(error) = &state.failure {
            return Err(error.clone());
        }
        if self.pending.load(Ordering::Acquire) != 0 {
            return Err(DirectJitError::shutdown());
        }
        Ok(lookup.get(key).and_then(|node| {
            let entry = node.entry();
            (entry != 0).then(|| {
                (
                    self.runtime.gateway(),
                    entry,
                    state.invalidation_cursor.get(),
                )
            })
        }))
    }

    fn reconcile(&self, memory: &(impl CpuMemory + ?Sized)) -> Result<(), DirectJitError> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let lookup = Arc::clone(&state.lookup);
        reconcile_invalidations(&mut state, &lookup, memory, self.cpu.address_space_id())?;
        Ok(())
    }

    fn native_lookup(&self) -> *const NativeLookupSlot {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .lookup
            .native_base()
    }

    pub fn shutdown(&self) {
        self.pending.store(1, Ordering::Release);
        self.hcq.shutdown();
        let current = std::thread::current().id();
        let workers = std::mem::take(
            &mut *self
                .hcq_workers
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
        );
        for worker in workers {
            if worker.thread().id() != current {
                let _ = worker.join();
            }
        }
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let lookup = Arc::clone(&state.lookup);
        let keys: Vec<_> = state.regions.keys().copied().collect();
        for key in keys {
            retire_region(&mut state, &lookup, key);
        }
        self.published.notify_all();
    }

    pub fn synchronize_address_space(&self, memory: &dyn CpuMemory) -> Result<(), DirectJitError> {
        self.reconcile(memory)
    }
}

impl Drop for JitProcess {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn native_fault_metadata(
    compiled: &mut compiler::CompiledRegion,
) -> Result<Option<Arc<NativeFaultRegion>>, DirectJitError> {
    if compiled.fault_sites.is_empty() {
        return Ok(None);
    }
    let native_end = compiled
        .entry
        .checked_add(compiled.native_bytes)
        .ok_or_else(|| DirectJitError::internal("compiled native region address overflows"))?;
    let sites = std::mem::take(&mut compiled.fault_sites)
        .into_vec()
        .into_iter()
        .map(|site| {
            let native_start = compiled
                .entry
                .checked_add(site.native_start as usize)
                .ok_or_else(|| DirectJitError::internal("native fault-site start overflows"))?;
            let native_end = compiled
                .entry
                .checked_add(site.native_end as usize)
                .ok_or_else(|| DirectJitError::internal("native fault-site end overflows"))?;
            Ok(NativeFaultSite {
                native_start,
                native_end,
                access: site.access,
            })
        })
        .collect::<Result<Vec<_>, DirectJitError>>()?;
    Ok(Some(Arc::new(NativeFaultRegion {
        native_start: compiled.entry,
        native_end,
        sites: Arc::from(sites),
    })))
}

#[derive(Default)]
struct InvalidationSummary {
    all: bool,
    physical_pages: HashSet<GuestPhysicalPageId>,
    mapping_ranges: Vec<(GuestVirtualAddress, u64)>,
}

impl InvalidationSummary {
    fn affects_published(&self, region: &PublishedRegion) -> bool {
        self.all
            || self.mapping_ranges.iter().any(|&(start, size)| {
                region
                    .mapping_dependencies
                    .iter()
                    .any(|span| mapping_range_overlaps(*span, start, size))
            })
            || region
                .dependencies
                .iter()
                .any(|dependency| self.physical_pages.contains(&dependency.page))
    }
}

fn invalidations_since(
    memory: &(impl MemoryInvalidationSource + ?Sized),
    cursor: MemoryInvalidationCursor,
    address_space: nixe_memory::AddressSpaceId,
    records: &mut Vec<MemoryInvalidation>,
) -> Result<InvalidationSummary, DirectJitError> {
    records.clear();
    match memory.read_invalidations_since(cursor, records) {
        Ok(_) => Ok(summarize_invalidations(records, address_space)),
        Err(MemoryInvalidationError::HistoryLost { .. }) => Ok(InvalidationSummary {
            all: true,
            ..InvalidationSummary::default()
        }),
        Err(error) => Err(DirectJitError::internal(format!(
            "direct JIT could not revalidate compiled code: {error}"
        ))),
    }
}

fn summarize_invalidations(
    records: &[MemoryInvalidation],
    address_space: nixe_memory::AddressSpaceId,
) -> InvalidationSummary {
    let mut summary = InvalidationSummary::default();
    for record in records {
        match record.kind {
            MemoryInvalidationKind::Mapping {
                address_space: changed,
                start,
                size,
            } if changed == address_space => summary.mapping_ranges.push((start, size)),
            MemoryInvalidationKind::InstructionCache {
                address_space: changed,
            } if changed == address_space => summary.all = true,
            MemoryInvalidationKind::ExecutableContent { first, second } => {
                summary.physical_pages.insert(first);
                summary.physical_pages.extend(second);
            }
            MemoryInvalidationKind::Mapping { .. }
            | MemoryInvalidationKind::InstructionCache { .. } => {}
        }
    }
    summary
}

fn reconcile_invalidations(
    state: &mut ProcessState,
    lookup: &RegionLookup,
    memory: &(impl MemoryInvalidationSource + ?Sized),
    address_space: nixe_memory::AddressSpaceId,
) -> Result<InvalidationSummary, DirectJitError> {
    let mut records = std::mem::take(&mut state.invalidations);
    records.clear();
    let cursor = match memory.read_invalidations_since(state.invalidation_cursor, &mut records) {
        Ok(cursor) => cursor,
        Err(MemoryInvalidationError::HistoryLost { latest, .. }) => {
            state.invalidation_cursor = latest;
            state.invalidations = records;
            let keys: Vec<_> = state.regions.keys().copied().collect();
            for key in keys {
                retire_region(state, lookup, key);
            }
            return Ok(InvalidationSummary {
                all: true,
                ..InvalidationSummary::default()
            });
        }
        Err(error) => {
            state.invalidations = records;
            return Err(DirectJitError::internal(format!(
                "direct JIT could not consume memory invalidations: {error}"
            )));
        }
    };
    state.invalidation_cursor = cursor;

    let summary = summarize_invalidations(&records, address_space);

    let mut keys: HashSet<_> = if summary.all {
        state.regions.keys().copied().collect()
    } else {
        summary
            .physical_pages
            .iter()
            .filter_map(|page| state.physical_dependencies.get(page))
            .flatten()
            .copied()
            .collect()
    };
    if !summary.all && !summary.mapping_ranges.is_empty() {
        keys.extend(
            state
                .regions
                .values()
                .chain(&state.retired)
                .filter_map(|region| {
                    summary
                        .mapping_ranges
                        .iter()
                        .any(|&(start, size)| {
                            region
                                .mapping_dependencies
                                .iter()
                                .any(|span| mapping_range_overlaps(*span, start, size))
                        })
                        .then_some(region.key)
                }),
        );
    }
    for key in keys {
        retire_region(state, lookup, key);
    }
    state.invalidations = records;
    Ok(summary)
}

fn mapping_range_overlaps(span: CodePageSpan, start: GuestVirtualAddress, size: u64) -> bool {
    if size == 0 {
        return false;
    }
    let changed_start = u128::from(start.get());
    let changed_end = (changed_start + u128::from(size)).min(1_u128 << 64);
    let span_start = u128::from(span.start.get());
    let span_end = span
        .end_exclusive
        .map_or(1_u128 << 64, |end| u128::from(end.get()));
    changed_start < span_end && span_start < changed_end
}

fn retire_region(state: &mut ProcessState, lookup: &RegionLookup, key: RegionKey) {
    let Some(region) = state.regions.remove(&key) else {
        return;
    };
    if let Some(node) = lookup.get(key) {
        node.invalidate();
    }
    state.retired.push(region);
}

pub struct JitThread {
    control: CpuControl,
    compiler: Mutex<Option<Box<DirectCompiler>>>,
    exclusive: Mutex<ExclusiveMonitorState>,
    fault_context: Mutex<Option<WorkerFaultContext>>,
    #[cfg(test)]
    events: VcpuEventState,
}

impl Default for JitThread {
    fn default() -> Self {
        Self::new()
    }
}

struct NativeRunRequest<'a> {
    memory: &'a dyn CpuMemory,
    state: &'a mut A64State,
    loader_return: Option<GuestVirtualAddress>,
    timer: &'a dyn ArchitecturalTimer,
    events: &'a VcpuEventState,
}

impl JitThread {
    pub fn new() -> Self {
        Self {
            control: CpuControl::default(),
            compiler: Mutex::new(None),
            exclusive: Mutex::new(ExclusiveMonitorState::default()),
            fault_context: Mutex::new(None),
            #[cfg(test)]
            events: VcpuEventState::default(),
        }
    }

    fn entry_for(
        &self,
        process: &JitProcess,
        memory: &(impl CpuMemory + ?Sized),
        location: LocationDescriptor,
    ) -> Result<(NativeGateway, usize, u64), DirectJitError> {
        let key = RegionKey::new(process.cpu, location);
        let lookup = Arc::clone(
            &process
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .lookup,
        );
        let node = lookup.get_or_create(key);
        loop {
            let mut state = process.state.lock().unwrap_or_else(PoisonError::into_inner);
            reconcile_invalidations(&mut state, &lookup, memory, key.address_space)?;
            if let Some(error) = &state.failure {
                return Err(error.clone());
            }
            if process.pending.load(Ordering::Acquire) != 0 {
                return Err(DirectJitError::shutdown());
            }
            let entry = node.entry();
            if entry != 0 {
                return Ok((
                    process.runtime.gateway(),
                    entry,
                    state.invalidation_cursor.get(),
                ));
            }
            match node.state() {
                EntryState::Empty => {
                    let Some(generation) = node.try_begin_lcq() else {
                        continue;
                    };
                    let cursor = state.invalidation_cursor;
                    drop(state);
                    let result = self.compile_lcq(
                        process,
                        memory,
                        Arc::clone(&node),
                        Arc::clone(&lookup),
                        generation,
                        cursor,
                    );
                    if result.is_err() {
                        node.abort_lcq(generation);
                        process.published.notify_all();
                    }
                    return result;
                }
                EntryState::CompilingLcq => {
                    drop(
                        process
                            .published
                            .wait(state)
                            .unwrap_or_else(PoisonError::into_inner),
                    );
                }
                EntryState::Lcq | EntryState::HcqQueued | EntryState::Hcq => {
                    return Err(DirectJitError::internal(
                        "published JIT entry state has a null canonical cell",
                    ));
                }
            }
        }
    }

    fn compile_lcq(
        &self,
        process: &JitProcess,
        memory: &(impl CpuMemory + ?Sized),
        node: Arc<lookup::NativeLookupNode>,
        lookup: Arc<RegionLookup>,
        generation: u64,
        mut compile_cursor: MemoryInvalidationCursor,
    ) -> Result<(NativeGateway, usize, u64), DirectJitError> {
        let key = node.key();
        let location = LocationDescriptor::new(key.start, process.cpu.profile_id());
        let backend = process.bound_memory_backend()?.backend;
        loop {
            let region = discover_region(
                process.cpu,
                memory,
                location,
                LCQ_MAX_REGION_INSTRUCTIONS,
                |pc| lookup.native_entry_lock_free(key.at(pc)) != 0,
            )?;
            let entry_cells = region
                .external_exits
                .iter()
                .filter_map(|exit| exit.target.map(|target| key.at(target)))
                .map(|target| lookup.get_or_create(target).entry_address())
                .collect::<Vec<_>>();
            let mut compiled = {
                let mut compiler = self.compiler.lock().unwrap_or_else(PoisonError::into_inner);
                if compiler.is_none() {
                    *compiler = Some(Box::new(DirectCompiler::new(
                        LCQ_COMPILER_POLICY,
                        process.runtime.addresses(),
                    )?));
                }
                let compiler = compiler
                    .as_deref_mut()
                    .expect("LCQ compiler was initialized");
                compiler.bind_memory_backend(backend)?;
                let promotion = process
                    .hcq_started
                    .load(Ordering::Acquire)
                    .then_some(Promotion {
                        hotness_address: node.hotness_address(),
                        node_address: Arc::as_ptr(&node).addr(),
                    });
                compiler.compile(&region, &entry_cells, promotion)?
            };
            let fault_metadata = native_fault_metadata(&mut compiled)?;
            let entry = compiled.entry;
            let published = Arc::new(PublishedRegion {
                key,
                #[cfg(test)]
                entry: compiled.entry,
                #[cfg(test)]
                native_bytes: compiled.native_bytes,
                #[cfg(test)]
                clif_instructions: compiled.clif_instructions,
                #[cfg(test)]
                deferred_register_loads: compiled.deferred_register_loads,
                #[cfg(test)]
                exit_tail_count: compiled.exit_tail_count,
                dependencies: region.dependencies,
                mapping_dependencies: region.mapping_dependencies,
            });

            let mut state = process.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.native_bytes = state
                .native_bytes
                .checked_add(compiled.native_bytes)
                .ok_or_else(|| DirectJitError::capacity("direct JIT code size overflows"))?;
            if state.native_bytes > process.max_native_code_bytes {
                return Err(DirectJitError::capacity(format!(
                    "direct JIT code arena exhausted: used={} limit={}",
                    state.native_bytes, process.max_native_code_bytes,
                )));
            }
            reconcile_invalidations(&mut state, &lookup, memory, key.address_space)?;
            let invalidated = invalidations_since(
                memory,
                compile_cursor,
                key.address_space,
                &mut state.invalidations,
            )?;
            if process.pending.load(Ordering::Acquire) != 0 {
                return Err(DirectJitError::shutdown());
            }
            if node.generation() != generation || node.state() != EntryState::CompilingLcq {
                drop(state);
                process.published.notify_all();
                return self.entry_for(process, memory, location);
            }
            if invalidated.affects_published(&published) {
                compile_cursor = state.invalidation_cursor;
                continue;
            }
            if let Some(metadata) = fault_metadata {
                process
                    .fault_registry
                    .publish(metadata)
                    .map_err(|error| DirectJitError::internal(error.to_string()))?;
            }
            for dependency in &published.dependencies {
                state
                    .physical_dependencies
                    .entry(dependency.page)
                    .or_default()
                    .insert(key);
            }
            let previous = state.regions.insert(key, Arc::clone(&published));
            assert!(
                previous.is_none(),
                "a native region generation is published once"
            );
            if !node.publish_lcq(generation, entry) {
                state.regions.remove(&key);
                return Err(DirectJitError::internal(
                    "LCQ publication lost canonical entry ownership",
                ));
            }
            let cursor = state.invalidation_cursor.get();
            drop(state);
            process.published.notify_all();
            return Ok((process.runtime.gateway(), entry, cursor));
        }
    }

    #[cfg(test)]
    fn request_preempt(&self) {
        self.control.request(ControlRequest::Preempt);
    }

    #[must_use]
    pub fn control(&self) -> CpuControl {
        self.control.clone()
    }

    pub fn clear_local_exclusive_reservation(&mut self) {
        *self
            .exclusive
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner) = Default::default();
    }

    pub fn synchronize_address_space(
        &mut self,
        process: &JitProcess,
        binding: MemoryBinding<'_>,
    ) -> Result<(), CpuFault> {
        process
            .bind_memory(binding)
            .map_err(|error| jit_fault(error, 0, &A64State::default()))?;
        process
            .synchronize_address_space(binding.memory)
            .map_err(|error| jit_fault(error, 0, &A64State::default()))?;
        let backend = process
            .bound_memory_backend()
            .map_err(|error| jit_fault(error, 0, &A64State::default()))?;
        if backend.backend == CpuMemoryBackend::LinuxDirect {
            let context = self
                .fault_context
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner);
            if context.is_none() {
                *context = Some(WorkerFaultContext::register().map_err(|error| {
                    jit_fault(
                        DirectJitError::unsupported(error.to_string()),
                        0,
                        &A64State::default(),
                    )
                })?);
            }
        }
        self.control
            .acknowledge_invalidation(binding.invalidation_cursor.get());
        Ok(())
    }

    pub fn prepare_shutdown(
        &mut self,
        process: &JitProcess,
        binding: MemoryBinding<'_>,
    ) -> Result<(), CpuFault> {
        self.synchronize_address_space(process, binding)?;
        self.clear_local_exclusive_reservation();
        self.fault_context
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        Ok(())
    }

    pub fn run_slice(
        &mut self,
        process: &JitProcess,
        mut request: RunRequest<'_>,
    ) -> Result<ExecutionReport, CpuFault> {
        let backend = process
            .bound_memory_backend()
            .map_err(|error| jit_fault(error, 0, request.state))?;
        if backend.backend == CpuMemoryBackend::LinuxDirect
            && !request
                .memory_lease
                .as_ref()
                .is_some_and(|lease| lease.authorizes(request.memory))
        {
            return Err(jit_fault(
                DirectJitError::invalid(
                    "LinuxDirect JIT execution requires its live mapping lease",
                ),
                0,
                request.state,
            ));
        }
        // The caller lease proves the public binding, but retaining it while
        // discovering and compiling a region can stall unrelated GPU or
        // mapping transitions for the complete compile. Native execution
        // acquires its own fresh lease after the compiled entry has been
        // revalidated against pending invalidations.
        if backend.backend == CpuMemoryBackend::LinuxDirect {
            drop(request.memory_lease.take());
        }
        let mut executed = 0;
        loop {
            let pending_interrupts = request.events.take_pending_interrupts();
            if pending_interrupts != 0 {
                return Ok(report(
                    executed,
                    CpuExit::PendingEvent {
                        mask: pending_interrupts,
                    },
                    request.state,
                ));
            }
            let exit = self
                .run_with_runtime(
                    process,
                    NativeRunRequest {
                        memory: request.memory,
                        state: request.state,
                        loader_return: request.loader_return,
                        timer: request.timer,
                        events: &request.events,
                    },
                )
                .map_err(|error| jit_fault(error, executed, request.state))?;
            let completed = direct_exit_progress(&exit);
            executed = executed.saturating_add(completed);
            let source = |pc| LocationDescriptor::new(pc, process.cpu.profile_id());
            let stop = match exit {
                DirectExit::Dispatch { .. } => continue,
                DirectExit::Control { .. } => {
                    if let Some(control) = self.control.take_pending() {
                        self.control.acknowledge(control);
                        if !control.contains(ControlRequest::Preempt) {
                            continue;
                        }
                    }
                    CpuExit::Safepoint
                }
                DirectExit::Architectural { pc, detail, .. } => {
                    let class = detail >> 24;
                    let immediate = detail & 0x00ff_ffff;
                    match class {
                        1 => CpuExit::SupervisorCall {
                            source: source(pc),
                            immediate,
                        },
                        2 => CpuExit::ArchitecturalException {
                            source: source(pc),
                            kind: ExceptionKind::Breakpoint,
                            syndrome: Some(u64::from(immediate)),
                        },
                        6 => CpuExit::ArchitecturalException {
                            source: source(pc),
                            kind: ExceptionKind::FloatingPoint,
                            syndrome: Some(u64::from(immediate)),
                        },
                        _ => {
                            return Err(internal_fault(
                                format!("native exit has unknown architectural class {class}"),
                                executed,
                                request.state,
                            ));
                        }
                    }
                }
                DirectExit::Unsupported { pc, .. } => {
                    unsupported_exit(process, request.memory, pc, executed, request.state)?
                }
                DirectExit::DataFault { pc, fault, .. } => CpuExit::DataFault {
                    source: source(pc),
                    fault,
                },
                DirectExit::Scheduled { pc, request, .. } => CpuExit::Scheduled {
                    source: source(pc),
                    request,
                },
                DirectExit::LoaderReturn {
                    pc, result_code, ..
                } => CpuExit::LoaderReturn {
                    source: source(pc),
                    result_code,
                },
                DirectExit::Internal { .. } => {
                    return Err(internal_fault(
                        "native execution returned an internal exit",
                        executed,
                        request.state,
                    ));
                }
                DirectExit::Reconcile => unreachable!("reconciliation is handled internally"),
            };
            return Ok(report(executed, stop, request.state));
        }
    }

    #[cfg(test)]
    fn run(
        &self,
        process: &JitProcess,
        memory: &dyn CpuMemory,
        state: &mut A64State,
    ) -> Result<DirectExit, DirectJitError> {
        loop {
            let exit = self.run_with_runtime(
                process,
                NativeRunRequest {
                    memory,
                    state,
                    loader_return: None,
                    timer: &ZeroTimer,
                    events: &self.events,
                },
            )?;
            if !matches!(exit, DirectExit::Dispatch { .. }) {
                return Ok(exit);
            }
        }
    }

    fn run_with_runtime(
        &self,
        process: &JitProcess,
        request: NativeRunRequest<'_>,
    ) -> Result<DirectExit, DirectJitError> {
        let NativeRunRequest {
            memory,
            state,
            loader_return,
            timer,
            events,
        } = request;
        let backend = process.bound_memory_backend()?;
        let actual_backend = memory.cpu_memory_backend(process.cpu.address_space_id());
        let actual_direct = memory.direct_address_space_view(process.cpu.address_space_id());
        if actual_backend != backend.backend || actual_direct != backend.direct {
            return Err(DirectJitError::invalid(
                "JIT execution memory differs from its immutable process binding",
            ));
        }
        let mut fault_context = self
            .fault_context
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if backend.backend == CpuMemoryBackend::LinuxDirect && fault_context.is_none() {
            return Err(DirectJitError::internal(
                "direct JIT worker entered native code without a fault context",
            ));
        }
        let mut exclusive = self
            .exclusive
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let native_lookup = process.native_lookup();
        let mut context = NativeContext::new(
            state,
            memory,
            &mut exclusive,
            timer,
            events,
            process.cpu.address_space_id(),
            loader_return,
            &self.control,
            native_lookup,
            &process.pending,
            &process.hcq,
        );
        loop {
            let pc = GuestVirtualAddress::new(unsafe { *context.pc });
            if pc.get() == context.loader_return {
                return Ok(DirectExit::LoaderReturn {
                    pc,
                    result_code: state.read_x(A64Register::General(
                        A64GeneralRegister::new(0).expect("valid result register"),
                    )),
                    progress: 0,
                });
            }
            let location = LocationDescriptor::new(pc, process.cpu.profile_id());
            let compiled_entry = self.entry_for(process, memory, location)?;
            let native_lease = if backend.backend == CpuMemoryBackend::LinuxDirect {
                let lease = memory.acquire_execution_lease().ok_or_else(|| {
                    DirectJitError::invalid(
                        "LinuxDirect memory cannot acquire its native execution lease",
                    )
                })?;
                if !lease.authorizes(memory) {
                    return Err(DirectJitError::invalid(
                        "LinuxDirect memory acquired a lease from another owner",
                    ));
                }
                Some(lease)
            } else {
                None
            };
            let (gateway, entry, invalidation_cursor) =
                if backend.backend == CpuMemoryBackend::LinuxDirect {
                    let Some(entry) = process.published_entry_for(memory, location)? else {
                        drop(native_lease);
                        continue;
                    };
                    entry
                } else {
                    compiled_entry
                };
            context.invalidation_cursor = invalidation_cursor;
            if let Some(arena) = backend.direct {
                let worker = fault_context
                    .as_mut()
                    .expect("direct backend retains a registered fault context");
                let gateway = unsafe {
                    std::mem::transmute::<NativeGateway, nixe_cpu_direct_memory::NativeGateway>(
                        gateway,
                    )
                };
                let invocation = unsafe {
                    fp_env::begin(&mut context);
                    worker.invoke(
                        arena,
                        &process.fault_registry,
                        dispatch_direct_fault,
                        std::ptr::from_mut(&mut context).cast(),
                        NativeInvocation {
                            gateway,
                            context: std::ptr::from_mut(&mut context).cast(),
                            entry,
                        },
                    )
                };
                unsafe { fp_env::finish(&mut context) };
                invocation.map_err(|error| DirectJitError::internal(error.to_string()))?;
            } else {
                unsafe {
                    fp_env::begin(&mut context);
                    gateway(&mut context, entry);
                    fp_env::finish(&mut context);
                }
            }
            drop(native_lease);
            let exit = context.exit()?;
            if matches!(exit, DirectExit::Reconcile) {
                process.reconcile(memory)?;
                continue;
            }
            return Ok(exit);
        }
    }
}

unsafe extern "C" fn dispatch_direct_fault(
    opaque: *mut c_void,
    fault: *mut CapturedFault,
) -> FaultDisposition {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        dispatch_direct_fault_inner(&mut *opaque.cast::<NativeContext>(), &*fault)
    }))
    .unwrap_or(FaultDisposition::Fatal)
}

unsafe fn dispatch_direct_fault_inner(
    context: &mut NativeContext,
    fault: &CapturedFault,
) -> FaultDisposition {
    let site = fault.site();
    if site.access.address_space.get() != context.address_space {
        return FaultDisposition::Fatal;
    }
    let fault_address = fault.fault_address();
    let Some(arena_end) = context.direct_base.checked_add(context.direct_size) else {
        return FaultDisposition::Fatal;
    };
    if fault_address < context.direct_base || fault_address >= arena_end {
        return FaultDisposition::Fatal;
    }
    let Ok(guest_address) = u64::try_from(fault_address - context.direct_base) else {
        return FaultDisposition::Fatal;
    };
    let Some(size) = direct_access_size(site.access.size) else {
        return FaultDisposition::Fatal;
    };
    let kind = match site.access.kind {
        NativeMemoryAccessKind::Read => DataAccessKind::Read,
        NativeMemoryAccessKind::Write => DataAccessKind::Write,
    };
    let memory = unsafe { &*context.memory };
    let address = GuestVirtualAddress::new(guest_address);
    match memory.resolve_direct_fault(site.access.address_space, address, size, kind) {
        DirectFaultResolution::Retry => FaultDisposition::Retry,
        DirectFaultResolution::Fault(data_fault) => {
            unsafe { &mut *context.state }.set_pc(site.access.guest_pc.get());
            context.exit_pc = site.access.guest_pc.get();
            context.exit_kind = EXIT_DATA_FAULT;
            context.exit_detail = 0;
            context.data_fault = Some(data_fault);
            FaultDisposition::Escape
        }
        DirectFaultResolution::Fatal(detail) => {
            unsafe { &mut *context.state }.set_pc(site.access.guest_pc.get());
            context.exit_pc = site.access.guest_pc.get();
            context.exit_kind = EXIT_INTERNAL;
            context.direct_fault_error = Some(detail);
            FaultDisposition::Escape
        }
    }
}

const fn direct_access_size(bytes: u8) -> Option<MemoryAccessSize> {
    match bytes {
        1 => Some(MemoryAccessSize::Byte),
        2 => Some(MemoryAccessSize::Halfword),
        4 => Some(MemoryAccessSize::Word),
        8 => Some(MemoryAccessSize::Doubleword),
        16 => Some(MemoryAccessSize::Quadword),
        _ => None,
    }
}

fn direct_exit_progress(exit: &DirectExit) -> u64 {
    match exit {
        DirectExit::Dispatch { progress, .. }
        | DirectExit::Control { progress, .. }
        | DirectExit::Architectural { progress, .. }
        | DirectExit::Unsupported { progress, .. }
        | DirectExit::DataFault { progress, .. }
        | DirectExit::Scheduled { progress, .. }
        | DirectExit::Internal { progress, .. }
        | DirectExit::LoaderReturn { progress, .. } => *progress,
        DirectExit::Reconcile => 0,
    }
}

fn report(progress: u64, stop: CpuExit, state: &A64State) -> ExecutionReport {
    let context = (!matches!(stop, CpuExit::DataFault { .. })).then(|| state.register_context());
    ExecutionReport {
        progress,
        stop,
        context,
    }
}

fn jit_fault(error: DirectJitError, progress: u64, state: &A64State) -> CpuFault {
    let kind = match error.kind {
        DirectJitErrorKind::InvalidGuestCode => CpuFaultKind::InvalidRequest,
        DirectJitErrorKind::Unsupported
        | DirectJitErrorKind::Capacity
        | DirectJitErrorKind::Shutdown => CpuFaultKind::Unavailable,
        DirectJitErrorKind::Internal => CpuFaultKind::Internal,
    };
    CpuFault {
        backend: "jit",
        kind,
        progress,
        message: error.detail,
        context: Box::new(state.register_context()),
    }
}

fn internal_fault(message: impl Into<Box<str>>, progress: u64, state: &A64State) -> CpuFault {
    CpuFault {
        backend: "jit",
        kind: CpuFaultKind::Internal,
        progress,
        message: message.into(),
        context: Box::new(state.register_context()),
    }
}

fn unsupported_exit(
    process: &JitProcess,
    memory: &dyn CpuMemory,
    pc: GuestVirtualAddress,
    progress: u64,
    state: &A64State,
) -> Result<CpuExit, CpuFault> {
    let source = LocationDescriptor::new(pc, process.cpu.profile_id());
    let fetched = memory
        .fetch32(process.cpu.address_space_id(), pc)
        .map_err(|fault| internal_fault(fault.to_string(), progress, state))?;
    let encoding = InstructionEncoding::from_u32(fetched.bits);
    match decode::decode(process.cpu.decoder(), source, encoding) {
        DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => {
            Ok(CpuExit::UnsupportedSemantics {
                source,
                encoding,
                disassembly: decode::disassemble(&decoded.instruction).to_string().into(),
                coverage_id: decoded.instruction.coverage_id(),
            })
        }
        DecodeResult::Unallocated { reason, .. } => Err(internal_fault(
            format!("native unsupported exit decoded as unallocated: {reason}"),
            progress,
            state,
        )),
        DecodeResult::Reserved { name, reason, .. } => Err(internal_fault(
            format!("native unsupported exit decoded as reserved {name}: {reason}"),
            progress,
            state,
        )),
    }
}

#[cfg(test)]
struct ZeroTimer;

#[cfg(test)]
impl ArchitecturalTimer for ZeroTimer {
    fn snapshot(&self) -> nixe_cpu::execution::TimerSnapshot {
        nixe_cpu::execution::TimerSnapshot {
            counter: 0,
            frequency: 0,
        }
    }
}

#[cfg(test)]
mod tests;
