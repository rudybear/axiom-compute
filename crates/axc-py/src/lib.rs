// The pyo3 0.22 `#[pyfunction]` / `#[pymethods]` macros expand `?` through a
// `PyErr::from` that clippy flags as a useless conversion. The lint targets the
// generated code, not ours, so it is allowed crate-wide.
#![allow(clippy::useless_conversion)]

//! PyO3 bindings for AXIOM-Compute's CUDA↔Vulkan zero-copy interop — M4.1 (Phase 1c).
//!
//! This native module (`_axiom_compute`) does ONLY the Vulkan side: it compiles a
//! `.axc` source, builds a UUID-matched `VulkanContext` (external-interop), prepares
//! a `KernelHandle`, allocates EXPORTABLE DEDICATED shared buffers + the shared
//! TIMELINE semaphore, and exposes the export fds / dedicated bits / sizes to the
//! Python shim. The CUDA IMPORT (`cudaImportExternalMemory/Semaphore`,
//! `Signal/WaitExternalSemaphoresAsync`) is done in Python (cuda-python / ctypes) —
//! the fd is the clean ABI boundary, and Python is the single fd owner (G-3).
//!
//! ## Design choices honored here
//! - G-2: `dispatch_zero_copy` binds the exportable buffers directly as SSBOs — a
//!   NEW staging-free submit path in axc-runtime, NOT resident.rs upload/readback.
//! - G-3: `memory_fds()` / `semaphore_fd()` hand RAW fds to Python via into_raw_fd;
//!   Rust does not Drop-close them.
//! - G-7: the monotone (V1,V2) counter lives on the Python `Kernel`; this layer just
//!   receives `wait_value`/`signal_value` per call.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use axc_runtime::{VulkanContext, KernelHandle, SharedBufferSet, ExternalTimelineSemaphore,
                  ExternalBinarySemaphore, KernelMetadata, DispatchError};

/// Convert a runtime `DispatchError` into a Python exception.
fn dispatch_err(e: DispatchError) -> PyErr {
    // Fail-closed interop conditions map to ZeroCopyUnavailable on the Python side;
    // we raise a RuntimeError subclass-friendly message that the shim re-wraps.
    PyRuntimeError::new_err(format!("axiom-compute: {e}"))
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A compiled, UUID-matched kernel ready for zero-copy dispatch.
#[pyclass]
struct CompiledKernel {
    ctx: Arc<VulkanContext>,
    handle: KernelHandle,
    meta: KernelMetadata,
}

#[pymethods]
impl CompiledKernel {
    /// The selected Vulkan device's `deviceUUID` (16 raw bytes), for cross-checking
    /// against torch.cuda on the Python side.
    fn device_uuid(&self) -> Vec<u8> {
        self.ctx
            .external_caps()
            .map(|c| c.device_uuid.to_vec())
            .unwrap_or_default()
    }

    /// The kernel's `@workgroup` size `(x, y, z)`.
    fn workgroup_size(&self) -> (u32, u32, u32) {
        let w = self.meta.workgroup_size;
        (w[0], w[1], w[2])
    }

    /// The number of buffer bindings (SSBOs) the kernel expects.
    fn buffer_count(&self) -> usize {
        self.meta.binding_plan.buffers.len()
    }

    /// Total push-constant bytes the kernel expects.
    fn push_constant_bytes(&self) -> u32 {
        self.meta.binding_plan.push_constant_total_bytes
    }

    /// Allocate the exportable DEDICATED shared buffers + the shared semaphore.
    ///
    /// `sizes` is one byte-size per binding (binding order). When `use_binary_fallback`
    /// is true (G-6 spike-failed path), a binary semaphore pair is created.
    fn allocate_shared_buffers(
        &self,
        sizes: Vec<u64>,
        use_binary_fallback: bool,
    ) -> PyResult<SharedBuffers> {
        if sizes.len() != self.buffer_count() {
            return Err(PyValueError::new_err(format!(
                "expected {} buffer sizes, got {}",
                self.buffer_count(),
                sizes.len()
            )));
        }
        let set = self
            .ctx
            .allocate_shared_buffers(&self.handle, &sizes, use_binary_fallback)
            .map_err(dispatch_err)?;
        Ok(SharedBuffers { set: Some(set) })
    }

    /// G-2 staging-free zero-copy dispatch: wait `wait_value`@COMPUTE_SHADER,
    /// signal `signal_value`. No host copy, no fence on the data path.
    fn dispatch_zero_copy(
        &self,
        bufs: &SharedBuffers,
        workgroups: (u32, u32, u32),
        push_constants: Vec<u8>,
        wait_value: u64,
        signal_value: u64,
    ) -> PyResult<()> {
        let set = bufs
            .set
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("shared buffers already torn down"))?;
        self.ctx
            .dispatch_zero_copy(
                &self.handle,
                set,
                workgroups,
                &push_constants,
                wait_value,
                signal_value,
            )
            .map_err(dispatch_err)
    }

    /// G-4 drain: block until the external timeline reaches >= `value` (or
    /// device_wait_idle on the binary fallback) before any free at teardown.
    fn drain(&self, bufs: &SharedBuffers, value: u64) -> PyResult<()> {
        let set = bufs
            .set
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("shared buffers already torn down"))?;
        self.ctx.drain_external_timeline(set, value).map_err(dispatch_err)
    }

    /// G-4 step (5): free the Vulkan side of the shared buffers. Idempotent: a second
    /// call is a no-op (the set was already consumed). The Python side calls this
    /// LAST, after draining + destroying the CUDA imports.
    fn teardown_shared_buffers(&self, bufs: &mut SharedBuffers) {
        if let Some(set) = bufs.set.take() {
            self.ctx.teardown_shared_buffers(set);
        }
    }
}

/// The exportable shared buffer set + shared semaphore handed to Python (G-3).
#[pyclass]
struct SharedBuffers {
    // `Option` so teardown is idempotent (consumed once).
    set: Option<SharedBufferSet>,
}

#[pymethods]
impl SharedBuffers {
    /// G-3: take the raw OPAQUE_FDs for every buffer (ownership → Python).
    fn memory_fds(&mut self) -> PyResult<Vec<i32>> {
        self.set
            .as_mut()
            .map(|s| s.take_memory_fds_raw())
            .ok_or_else(|| PyRuntimeError::new_err("shared buffers already torn down"))
    }

    /// Per-buffer allocation sizes (binding order).
    fn sizes(&self) -> PyResult<Vec<u64>> {
        self.set
            .as_ref()
            .map(|s| s.sizes())
            .ok_or_else(|| PyRuntimeError::new_err("shared buffers already torn down"))
    }

    /// G-1: per-buffer dedication bits (all true) — Python sets
    /// cudaExternalMemoryDedicated to MATCH.
    fn dedicated(&self) -> PyResult<Vec<bool>> {
        self.set
            .as_ref()
            .map(|s| s.dedicated())
            .ok_or_else(|| PyRuntimeError::new_err("shared buffers already torn down"))
    }

    /// Whether the timeline-over-fd path is active (vs the binary fallback, G-6).
    fn timeline_supported(&self) -> PyResult<bool> {
        self.set
            .as_ref()
            .map(|s| s.timeline_supported())
            .ok_or_else(|| PyRuntimeError::new_err("shared buffers already torn down"))
    }

    /// G-3: take the raw timeline-semaphore fd (ownership → Python), or -1 on the
    /// binary fallback.
    fn semaphore_fd(&mut self) -> PyResult<i32> {
        self.set
            .as_mut()
            .map(|s| s.take_semaphore_fd_raw())
            .ok_or_else(|| PyRuntimeError::new_err("shared buffers already torn down"))
    }

    /// G-6 fallback: take the two binary-semaphore fds (A, B).
    fn binary_semaphore_fds(&mut self) -> PyResult<(i32, i32)> {
        self.set
            .as_mut()
            .map(|s| s.take_binary_semaphore_fds_raw())
            .ok_or_else(|| PyRuntimeError::new_err("shared buffers already torn down"))
    }

    /// AT-2109 (G-2): number of transient staging buffers allocated on the zero-copy
    /// path — MUST be 0.
    fn staging_buffers_allocated(&self) -> PyResult<u32> {
        self.set
            .as_ref()
            .map(|s| s.staging_buffers_allocated)
            .ok_or_else(|| PyRuntimeError::new_err("shared buffers already torn down"))
    }

    /// AT-2109 (G-2): number of vkCmdCopyBuffer records on the zero-copy path —
    /// MUST be 0.
    fn copy_buffer_records(&self) -> PyResult<u32> {
        self.set
            .as_ref()
            .map(|s| s.copy_buffer_records)
            .ok_or_else(|| PyRuntimeError::new_err("shared buffers already torn down"))
    }
}

/// A standalone exportable timeline semaphore for the AT-2110 interop spike.
#[pyclass]
struct SpikeTimelineSemaphore {
    ctx: Arc<VulkanContext>,
    sem: Option<ExternalTimelineSemaphore>,
}

#[pymethods]
impl SpikeTimelineSemaphore {
    /// G-3: take the raw OPAQUE_FD (ownership → Python).
    fn fd(&mut self) -> PyResult<i32> {
        self.sem
            .as_mut()
            .map(|s| s.take_fd_raw())
            .ok_or_else(|| PyRuntimeError::new_err("spike semaphore already torn down"))
    }

    /// Vulkan-side relay: wait timeline >= `wait_value`, then signal `signal_value`.
    fn relay(&self, wait_value: u64, signal_value: u64) -> PyResult<()> {
        let sem = self
            .sem
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("spike semaphore already torn down"))?;
        self.ctx
            .spike_timeline_relay(sem, wait_value, signal_value)
            .map_err(dispatch_err)
    }

    /// Destroy the Vulkan semaphore (after the spike completes).
    fn teardown(&mut self) {
        if let Some(sem) = self.sem.take() {
            self.ctx.teardown_spike_timeline(sem);
        }
    }
}

/// A standalone exportable binary semaphore pair for the AT-2110 fallback spike.
#[pyclass]
struct SpikeBinaryPair {
    ctx: Arc<VulkanContext>,
    pair: Option<(ExternalBinarySemaphore, ExternalBinarySemaphore)>,
}

#[pymethods]
impl SpikeBinaryPair {
    /// G-3: take the two raw OPAQUE_FDs (A, B) (ownership → Python).
    fn fds(&mut self) -> PyResult<(i32, i32)> {
        match self.pair.as_mut() {
            Some((a, b)) => Ok((a.take_fd_raw(), b.take_fd_raw())),
            None => Err(PyRuntimeError::new_err("spike pair already torn down")),
        }
    }

    /// Destroy both Vulkan semaphores (after the spike completes).
    fn teardown(&mut self) {
        if let Some(pair) = self.pair.take() {
            self.ctx.teardown_spike_binary_pair(pair);
        }
    }
}

/// Compile a `.axc` source and build a UUID-matched zero-copy kernel.
///
/// `device_uuid` is the CUDA device UUID (16 raw bytes) the Vulkan device MUST
/// match (FAIL-CLOSED). Raises on no UUID match or missing external-fd extensions.
#[pyfunction]
fn compile_kernel(source: &str, device_uuid: Vec<u8>) -> PyResult<CompiledKernel> {
    if device_uuid.len() != 16 {
        return Err(PyValueError::new_err(format!(
            "device_uuid must be 16 raw bytes, got {}",
            device_uuid.len()
        )));
    }
    let mut uuid = [0u8; 16];
    uuid.copy_from_slice(&device_uuid);

    let (spirv_bytes, meta) = axc_driver::compile_source_with_meta(source)
        .map_err(|e| PyRuntimeError::new_err(format!("axiom-compute compile error: {e}")))?;
    let words = bytes_to_words(&spirv_bytes);

    let ctx = VulkanContext::new_for_external_interop(uuid).map_err(dispatch_err)?;

    let handle = ctx
        .prepare_kernel_checked(
            &words,
            &meta.binding_plan,
            meta.binding_plan.push_constant_total_bytes,
            &meta.entry_point,
            meta.coopmat.as_ref(),
            &meta.kernel_name,
            meta.shared_memory_bytes,
        )
        .map_err(dispatch_err)?;

    Ok(CompiledKernel {
        ctx: Arc::new(ctx),
        handle,
        meta,
    })
}

/// Build a standalone exportable timeline semaphore for the AT-2110 spike.
#[pyfunction]
fn make_spike_timeline_semaphore(device_uuid: Vec<u8>) -> PyResult<SpikeTimelineSemaphore> {
    let ctx = build_interop_ctx(device_uuid)?;
    let sem = ctx.create_spike_timeline_semaphore(0).map_err(dispatch_err)?;
    Ok(SpikeTimelineSemaphore { ctx: Arc::new(ctx), sem: Some(sem) })
}

/// Build a standalone exportable binary semaphore pair for the AT-2110 fallback spike.
#[pyfunction]
fn make_spike_binary_pair(device_uuid: Vec<u8>) -> PyResult<SpikeBinaryPair> {
    let ctx = build_interop_ctx(device_uuid)?;
    let pair = ctx.create_spike_binary_semaphore_pair().map_err(dispatch_err)?;
    Ok(SpikeBinaryPair { ctx: Arc::new(ctx), pair: Some(pair) })
}

fn build_interop_ctx(device_uuid: Vec<u8>) -> PyResult<VulkanContext> {
    if device_uuid.len() != 16 {
        return Err(PyValueError::new_err("device_uuid must be 16 raw bytes"));
    }
    let mut uuid = [0u8; 16];
    uuid.copy_from_slice(&device_uuid);
    VulkanContext::new_for_external_interop(uuid).map_err(dispatch_err)
}

/// The native module imported by the `axiom_compute` Python package.
#[pymodule]
fn _axiom_compute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompiledKernel>()?;
    m.add_class::<SharedBuffers>()?;
    m.add_class::<SpikeTimelineSemaphore>()?;
    m.add_class::<SpikeBinaryPair>()?;
    m.add_function(wrap_pyfunction!(compile_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(make_spike_timeline_semaphore, m)?)?;
    m.add_function(wrap_pyfunction!(make_spike_binary_pair, m)?)?;
    Ok(())
}
