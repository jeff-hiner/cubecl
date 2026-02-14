use super::storage::{WgpuResource, WgpuStorage};
use crate::AutoCompiler;
use crate::schedule::{BindingsResource, ScheduleTask, ScheduledWgpuBackend};
use alloc::sync::Arc;
use cubecl_common::{
    backtrace::BackTrace,
    bytes::Bytes,
    profile::{ProfileDuration, TimingMethod},
    stream_id::StreamId,
};
#[cfg(feature = "spirv")]
use cubecl_common::cache::{Cache, CacheOption};
use cubecl_core::{
    MemoryConfiguration, WgpuCompilationOptions,
    future::DynFut,
    prelude::*,
    server::{
        Allocation, AllocationDescriptor, Binding, Bindings, CopyDescriptor, ExecutionError,
        IoError, LaunchError, ProfileError, ProfilingToken, ServerCommunication, ServerUtilities,
    },
};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    compiler::{CompilationError, CubeTask},
    config::GlobalConfig,
    logging::ServerLogger,
    memory_management::{MemoryAllocationMode, offset_handles},
    server::ComputeServer,
    storage::BindingResource,
    stream::scheduler::{SchedulerMultiStream, SchedulerMultiStreamOptions, SchedulerStrategy},
};
#[cfg(feature = "spirv")]
use cubecl_runtime::kernel::Visibility;
use hashbrown::HashMap;
use wgpu::ComputePipeline;

/// Persistent GPU pipeline cache for shader compilation across runs.
///
/// Wraps a [`wgpu::PipelineCache`] and its disk path so compiled shaders
/// can be saved on shutdown and reloaded on next startup.
pub(crate) struct PipelineCacheState {
    /// The wgpu pipeline cache handle.
    pub(crate) cache: wgpu::PipelineCache,
    /// Path where the cache blob is persisted.
    save_path: std::path::PathBuf,
}

impl std::fmt::Debug for PipelineCacheState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCacheState")
            .field("save_path", &self.save_path)
            .finish_non_exhaustive()
    }
}

impl PipelineCacheState {
    /// Create a new pipeline cache state from a wgpu cache and its save path.
    pub(crate) fn new(cache: wgpu::PipelineCache, save_path: std::path::PathBuf) -> Self {
        Self { cache, save_path }
    }
}

impl PipelineCacheState {
    /// Write the current cache contents to disk.
    ///
    /// This is called eagerly after each new pipeline is compiled, since the
    /// global server static is never dropped on process exit.
    pub(crate) fn save(&self) {
        if let Some(data) = self.cache.get_data() {
            if let Some(parent) = self.save_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            // Atomic write: write to temp file then rename.
            let tmp = self.save_path.with_extension("tmp");
            if std::fs::write(&tmp, &data).is_ok() {
                let _ = std::fs::rename(&tmp, &self.save_path);
            }
        }
    }
}

/// Cached SPIR-V compilation output, following the `PtxCacheEntry` pattern from `cubecl-cuda`.
///
/// Stores everything needed to recreate a compute pipeline without re-running
/// the CubeCL IR optimizer or SPIR-V code generator.
#[cfg(feature = "spirv")]
#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub(crate) struct SpirvCacheEntry {
    /// Shader entrypoint function name.
    pub(crate) entrypoint_name: String,
    /// Workgroup dimensions `(x, y, z)`.
    pub(crate) cube_dim: (u32, u32, u32),
    /// Assembled SPIR-V words.
    pub(crate) spirv: Vec<u32>,
    /// Per-binding visibility (read vs read-write).
    pub(crate) binding_visibilities: Vec<Visibility>,
    /// Whether the kernel uses a metadata buffer.
    pub(crate) has_metadata: bool,
    /// Number of scalar parameter buffers.
    pub(crate) num_scalars: usize,
}

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer {
    pub(crate) device: wgpu::Device,
    pipelines: HashMap<KernelId, Arc<ComputePipeline>>,
    scheduler: SchedulerMultiStream<ScheduledWgpuBackend>,
    pub compilation_options: WgpuCompilationOptions,
    pub(crate) backend: wgpu::Backend,
    pub(crate) utilities: Arc<ServerUtilities<Self>>,
    pub(crate) pipeline_cache: Option<PipelineCacheState>,
    #[cfg(feature = "spirv")]
    spirv_cache: Option<Cache<String, SpirvCacheEntry>>,
}

impl ServerCommunication for WgpuServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl WgpuServer {
    /// Create a new server.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        compilation_options: WgpuCompilationOptions,
        device: wgpu::Device,
        queue: wgpu::Queue,
        tasks_max: usize,
        backend: wgpu::Backend,
        timing_method: TimingMethod,
        utilities: ServerUtilities<Self>,
        pipeline_cache: Option<PipelineCacheState>,
    ) -> Self {
        let backend_scheduler = ScheduledWgpuBackend::new(
            device.clone(),
            queue.clone(),
            memory_properties,
            memory_config,
            timing_method,
            tasks_max,
            utilities.logger.clone(),
        );

        let config = GlobalConfig::get();
        let max_streams = config.streaming.max_streams;

        Self {
            compilation_options,
            device,
            pipelines: HashMap::new(),
            scheduler: SchedulerMultiStream::new(
                utilities.logger.clone(),
                backend_scheduler,
                SchedulerMultiStreamOptions {
                    max_streams,
                    max_tasks: tasks_max,
                    strategy: SchedulerStrategy::Interleave,
                },
            ),
            backend,
            utilities: Arc::new(utilities),
            pipeline_cache,
            #[cfg(feature = "spirv")]
            spirv_cache: config.compilation.cache.as_ref().map(|cache_config| {
                Cache::new(
                    "spirv",
                    CacheOption::default()
                        .name("wgpu-spirv")
                        .root(cache_config.root()),
                )
            }),
        }
    }

    fn prepare_bindings(&mut self, bindings: Bindings) -> BindingsResource {
        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let resources = bindings
            .buffers
            .iter()
            .map(|b| {
                let stream = self.scheduler.stream(&b.stream);
                stream.mem_manage.get_resource(b.clone()).unwrap()
            })
            .collect::<Vec<_>>();

        BindingsResource {
            resources,
            metadata: bindings.metadata,
            scalars: bindings.scalars,
        }
    }

    fn pipeline(
        &mut self,
        kernel: <Self as ComputeServer>::Kernel,
        mode: ExecutionMode,
    ) -> Result<Arc<ComputePipeline>, CompilationError> {
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        if let Some(pipeline) = self.pipelines.get(&kernel_id) {
            return Ok(pipeline.clone());
        }

        // Check SPIR-V disk cache before running the full compilation pipeline.
        #[cfg(feature = "spirv")]
        let stable_name = if let Some(cache) = &self.spirv_cache {
            let name = kernel_id.stable_format();
            if let Some(entry) = cache.get(&name) {
                log::trace!("Using SPIR-V cache");
                let pipeline = self.create_pipeline_from_spirv_cache(entry)?;
                self.pipelines.insert(kernel_id, pipeline.clone());
                if let Some(cache) = &self.pipeline_cache {
                    cache.save();
                }
                return Ok(pipeline);
            }
            Some(name)
        } else {
            None
        };

        let mut compiler = compiler(self.backend);
        let mut compile = compiler.compile(self, kernel, mode)?;

        if self.scheduler.logger.compilation_activated() {
            compile.debug_info = Some(DebugInformation::new(
                compiler.lang_tag(),
                kernel_id.clone(),
            ));
        }
        self.scheduler.logger.log_compilation(&compile);

        // Extract cache data before create_pipeline consumes the compiled kernel.
        #[cfg(feature = "spirv")]
        let spirv_cache_data = if self.spirv_cache.is_some() && stable_name.is_some() {
            if let Some(crate::AutoRepresentation::SpirV(repr)) = &compile.repr {
                Some(SpirvCacheEntry {
                    entrypoint_name: compile.entrypoint_name.clone(),
                    cube_dim: (compile.cube_dim.x, compile.cube_dim.y, compile.cube_dim.z),
                    spirv: repr.assemble(),
                    binding_visibilities: repr.bindings.iter().map(|b| b.visibility).collect(),
                    has_metadata: repr.has_metadata,
                    num_scalars: repr.scalars.len(),
                })
            } else {
                None
            }
        } else {
            None
        };

        // /!\ Do not delete the following commented code.
        // This is useful while working on the metal compiler.
        // Also the errors are printed nicely which is not the case when this is the runtime
        // that does it.
        // println!("SOURCE:\n{}", compile.source);
        // {
        //     // Write shader in metal file then compile it for error
        //     std::fs::write("shader.metal", &compile.source).expect("should write to file");
        //     let _status = std::process::Command::new("xcrun")
        //         .args(vec![
        //             "-sdk",
        //             "macosx",
        //             "metal",
        //             "-o",
        //             "shader.ir",
        //             "-c",
        //             "shader.metal",
        //         ])
        //         .status()
        //         .expect("should launch the command");
        //     // std::process::exit(status.code().unwrap());
        // }
        let pipeline = self.create_pipeline(compile, mode)?;

        // Insert into SPIR-V disk cache after successful compilation.
        #[cfg(feature = "spirv")]
        if let Some(cache) = &mut self.spirv_cache {
            if let (Some(name), Some(entry)) = (stable_name, spirv_cache_data) {
                if let Err(err) = cache.insert(name, entry) {
                    log::warn!("Unable to save SPIR-V cache: {err:?}");
                }
            }
        }

        self.pipelines.insert(kernel_id.clone(), pipeline.clone());

        if let Some(cache) = &self.pipeline_cache {
            cache.save();
        }

        Ok(pipeline)
    }
}

impl ComputeServer for WgpuServer {
    type Kernel = Box<dyn CubeTask<AutoCompiler>>;
    type Storage = WgpuStorage;
    type Info = wgpu::Backend;

    fn logger(&self) -> Arc<ServerLogger> {
        self.scheduler.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(&mut self, _sizes: &[usize], _stream_id: StreamId) -> Result<Vec<Bytes>, IoError> {
        // TODO: Check if using a staging buffer is useful here.
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        })
    }

    fn create(
        &mut self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        let align = self.device.limits().min_storage_buffer_offset_alignment as usize;
        let strides = descriptors
            .iter()
            .map(|desc| contiguous_strides(desc.shape))
            .collect::<Vec<_>>();
        let sizes = descriptors
            .iter()
            .map(|desc| desc.shape.iter().product::<usize>() * desc.elem_size)
            .collect::<Vec<_>>();
        let total_size = sizes
            .iter()
            .map(|it| it.next_multiple_of(align))
            .sum::<usize>();

        let stream = self.scheduler.stream(&stream_id);
        let mem_handle = stream.empty(total_size as u64, stream_id)?;
        let handles = offset_handles(mem_handle, &sizes, align);

        Ok(handles
            .into_iter()
            .zip(strides)
            .map(|(handle, strides)| Allocation::new(handle, strides))
            .collect())
    }

    fn read<'a>(
        &mut self,
        descriptors: Vec<CopyDescriptor<'a>>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        let mut streams = vec![stream_id];
        let mut resources = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            if contiguous_strides(desc.shape) != desc.strides {
                return Box::pin(async {
                    Err(IoError::UnsupportedStrides {
                        backtrace: BackTrace::capture(),
                    })
                });
            }
            if !streams.contains(&desc.binding.stream) {
                streams.push(desc.binding.stream);
            }
            let stream = self.scheduler.stream(&desc.binding.stream);
            let resource = match stream.mem_manage.get_resource(desc.binding) {
                Ok(val) => val,
                Err(err) => return Box::pin(async move { Err(err) }),
            };
            resources.push((resource, desc.shape.to_vec(), desc.elem_size));
        }

        self.scheduler.execute_streams(streams);
        let stream = self.scheduler.stream(&stream_id);
        stream.read_resources(resources)
    }

    fn write(
        &mut self,
        descriptors: Vec<(CopyDescriptor<'_>, Bytes)>,
        stream_id: StreamId,
    ) -> Result<(), IoError> {
        for (desc, data) in descriptors {
            if contiguous_strides(desc.shape) != desc.strides {
                return Err(IoError::UnsupportedStrides {
                    backtrace: BackTrace::capture(),
                });
            }

            let stream = self.scheduler.stream(&desc.binding.stream);
            let resource = stream.mem_manage.get_resource(desc.binding.clone())?;
            let task = ScheduleTask::Write {
                data,
                buffer: resource,
            };

            self.scheduler.register(stream_id, task, [].into_iter());
        }

        Ok(())
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> BindingResource<WgpuResource> {
        let mut streams = vec![stream_id];
        if binding.stream != stream_id {
            streams.push(binding.stream);
        }
        self.scheduler.execute_streams(streams);
        let stream = self.scheduler.stream(&binding.stream);
        let resource = stream.mem_manage.get_resource(binding.clone()).unwrap();
        BindingResource::new(binding, resource)
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), LaunchError> {
        let pipeline = self.pipeline(kernel, mode)?;
        let buffers = bindings.buffers.clone();
        let resources = self.prepare_bindings(bindings);
        let task = ScheduleTask::Execute {
            pipeline,
            count,
            resources,
        };

        self.scheduler.register(stream_id, task, buffers.iter());

        Ok(())
    }

    fn flush(&mut self, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.flush()
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ExecutionError>> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.sync()
    }

    fn start_profile(&mut self, stream_id: StreamId) -> ProfilingToken {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.start_profile()
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.end_profile(token)
    }

    fn memory_usage(
        &mut self,
        stream_id: StreamId,
    ) -> cubecl_runtime::memory_management::MemoryUsage {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.memory_usage()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.memory_cleanup(true);
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.mode(mode);
    }
}

fn compiler(backend: wgpu::Backend) -> AutoCompiler {
    match backend {
        #[cfg(feature = "spirv")]
        wgpu::Backend::Vulkan => AutoCompiler::SpirV(Default::default()),
        #[cfg(feature = "msl")]
        wgpu::Backend::Metal => AutoCompiler::Msl(Default::default()),
        _ => AutoCompiler::Wgsl(Default::default()),
    }
}

pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
