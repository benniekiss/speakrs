pub(crate) mod embedding;
pub(crate) mod segmentation;

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
use std::{ffi::CStr, path::Path, sync::OnceLock};
use std::{
    fmt::{self, Display},
    path::PathBuf,
};

pub use embedding::EmbeddingModel;
#[cfg(any(
    feature = "coreml",
    feature = "cuda",
    feature = "migraphx",
    feature = "webgpu"
))]
use ort::ep;
use ort::session::builder::SessionBuilder;
pub use segmentation::{
    FRAME_DURATION_SECONDS,
    FRAME_STEP_SECONDS,
    SEGMENTATION_WINDOW_SECONDS,
    SegmentationError,
    SegmentationModel,
};

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
static ORT_RUNTIME_INIT: OnceLock<Result<(), OrtRuntimeError>> = OnceLock::new();

/// Which backend and acceleration to use for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutionMode {
    /// CPU-only via ORT (portable, slowest)
    Cpu,
    /// ONNX Runtime CoreML execution provider with a ~1s step
    #[cfg_attr(docsrs, doc(cfg(feature = "coreml")))]
    CoreMl,
    /// NVIDIA GPU with concurrent fused seg+emb via crossbeam
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    Cuda,
    /// AMD GPU via ONNX Runtime's MIGraphX execution provider
    #[cfg_attr(docsrs, doc(cfg(feature = "migraphx")))]
    MiGraphX,
    /// Cross-platform GPU acceleration via ONNX Runtime's WebGPU provider
    #[cfg_attr(docsrs, doc(cfg(feature = "webgpu")))]
    WebGpu,
}

impl Display for ExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val = match self {
            Self::Cpu => "cpu",
            Self::CoreMl => "coreml",
            Self::Cuda => "cuda",
            Self::MiGraphX => "migraphx",
            Self::WebGpu => "webgpu",
        };

        write!(f, "{val}")
    }
}

impl ExecutionMode {
    pub(crate) fn uses_concurrent_pipeline(self) -> bool {
        matches!(self, Self::CoreMl | Self::Cuda | Self::MiGraphX)
    }

    pub(crate) fn embedding_intra_threads(self) -> usize {
        if self.uses_concurrent_pipeline() {
            1
        } else {
            available_threads()
        }
    }

    pub(crate) fn validate(self) -> Result<(), ExecutionModeError> {
        match self {
            Self::Cpu => Ok(()),
            Self::CoreMl if cfg!(feature = "coreml") => Ok(()),
            Self::MiGraphX if cfg!(feature = "migraphx") => Ok(()),
            Self::Cuda if cfg!(feature = "cuda") => Ok(()),
            Self::WebGpu if cfg!(feature = "webgpu") => Ok(()),
            _ => Err(ExecutionModeError {
                mode: self,
                feature: self.to_string(),
            }),
        }
    }
}

pub(crate) fn available_threads() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

/// Errors that can occur while loading a model or initializing ONNX Runtime
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelLoadError {
    /// Requested execution mode is not supported by this build
    #[error(transparent)]
    UnsupportedExecutionMode(#[from] ExecutionModeError),
    /// ONNX Runtime could not be prepared for this process
    #[error(transparent)]
    Runtime(#[from] OrtRuntimeError),
    /// ONNX Runtime returned an error after initialization completed
    #[error(transparent)]
    Ort(#[from] ort::Error),
    /// A required native model asset is missing for the selected execution mode
    #[error("{mode} requires native asset `{path}`")]
    MissingNativeAsset {
        /// The execution mode that requires the asset
        mode: ExecutionMode,
        /// The missing compiled CoreML bundle path
        path: PathBuf,
    },
    /// A required native model asset exists but failed to load
    #[error("{mode} failed to load native asset `{path}`: {message}")]
    NativeAssetLoad {
        /// The execution mode that requires the asset
        mode: ExecutionMode,
        /// The compiled CoreML bundle path that failed to load
        path: PathBuf,
        /// The backend load error
        message: String,
    },
}

/// Errors that can occur while preparing the process-wide ONNX Runtime environment
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum OrtRuntimeError {
    /// Dynamic runtime discovery or validation failed before `ort` could initialize
    #[error(transparent)]
    Dynamic(#[from] DynamicRuntimeError),
    /// `ort::init_from` failed after runtime validation succeeded
    #[error("failed to initialize ONNX Runtime: {message}")]
    Initialization {
        /// The initialization error returned by `ort`
        message: String,
    },
}

/// Errors from locating or validating the dynamic ONNX Runtime library
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum DynamicRuntimeError {
    /// No candidate runtime library was found
    #[error(
        "missing ONNX Runtime dynamic library `{library_name}`; set `ORT_DYLIB_PATH` or place it next to the test/binary\nsearched: {searched}"
    )]
    Missing {
        /// The platform-specific dynamic library filename
        library_name: &'static str,
        /// The candidate paths checked before giving up
        searched: String,
    },
    /// Loading the requested runtime library failed
    #[error("failed to load ONNX Runtime dynamic library at `{path}`: {message}")]
    Load {
        /// The path that failed to load
        path: PathBuf,
        /// The dynamic loader error
        message: String,
    },
    /// The requested runtime library does not export `OrtGetApiBase`
    #[error("ONNX Runtime dynamic library at `{path}` does not export `OrtGetApiBase`")]
    MissingApiBase {
        /// The path that was missing the required symbol
        path: PathBuf,
    },
    /// The requested runtime library returned a null API base pointer
    #[error("ONNX Runtime dynamic library at `{path}` returned a null `OrtApiBase`")]
    NullApiBase {
        /// The path that returned a null API pointer
        path: PathBuf,
    },
    /// The requested runtime library is older than the `ort` crate expects
    #[error(
        "ONNX Runtime dynamic library at `{path}` is too old; expected >= 1.{required_minor}.x, got `{found_version}`"
    )]
    IncompatibleVersion {
        /// The incompatible runtime library path
        path: PathBuf,
        /// The minimum ONNX Runtime minor version required by `ort`
        required_minor: u32,
        /// The version reported by the discovered runtime library
        found_version: String,
    },
}

/// Errors from requesting an execution mode that is not supported in the current build
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
#[error("{mode} requires the `{feature}` Cargo feature")]
pub struct ExecutionModeError {
    mode: ExecutionMode,
    feature: String,
}

impl From<ExecutionModeError> for ort::Error {
    fn from(error: ExecutionModeError) -> Self {
        ort::Error::new(error.to_string())
    }
}

/// Map an execution mode to ORT execution providers
///
/// CoreML modes register ONNX Runtime's CoreML execution provider. ONNX Runtime
/// automatically falls back to its CPU provider for unsupported graph nodes.
pub fn with_execution_mode(
    builder: SessionBuilder,
    mode: ExecutionMode,
) -> Result<SessionBuilder, ort::Error> {
    with_execution_mode_options(builder, mode, false)
}

pub(crate) fn with_execution_mode_options(
    builder: SessionBuilder,
    mode: ExecutionMode,
    cuda_graph: bool,
) -> Result<SessionBuilder, ort::Error> {
    mode.validate()?;

    #[cfg(not(feature = "cuda"))]
    let _ = cuda_graph;

    match mode {
        ExecutionMode::Cpu => Ok(builder),
        #[cfg(feature = "coreml")]
        ExecutionMode::CoreMl => Ok(builder.with_execution_providers([ep::CoreML::default()
            .with_model_format(ep::coreml::ModelFormat::MLProgram)
            .with_static_input_shapes(true)
            .build()
            .error_on_failure()])?),
        #[cfg(feature = "cuda")]
        ExecutionMode::Cuda => Ok(builder.with_execution_providers([ep::CUDA::default()
            .with_tf32(true)
            .with_prefer_nhwc(true)
            .with_cuda_graph(cuda_graph)
            .build()
            .error_on_failure()])?),
        #[cfg(feature = "migraphx")]
        ExecutionMode::MiGraphX => Ok(builder
            .with_execution_providers([ep::MIGraphX::default().build().error_on_failure()])?),
        #[cfg(feature = "webgpu")]
        ExecutionMode::WebGpu => {
            Ok(builder
                .with_execution_providers([ep::WebGPU::default().build().error_on_failure()])?)
        },

        #[cfg(not(all(
            feature = "coreml",
            feature = "cuda",
            feature = "migraphx",
            feature = "webgpu"
        )))]
        _ => {
            unreachable!("mode validation failed without the `{mode}` feature")
        },
    }
}

pub(crate) fn ensure_ort_ready() -> Result<(), ModelLoadError> {
    #[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
    {
        let init_result = ORT_RUNTIME_INIT.get_or_init(|| OrtRuntimeLoader::new().initialize());
        init_result.clone()?;
    }

    Ok(())
}

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
struct OrtRuntimeLoader {
    library_name: &'static str,
}

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
impl OrtRuntimeLoader {
    fn new() -> Self {
        Self {
            library_name: Self::default_library_name(),
        }
    }

    fn initialize(&self) -> Result<(), OrtRuntimeError> {
        let path = self.resolve_library_path()?;
        self.validate_library(&path)?;

        ort::init_from(&path)
            .map(|builder| {
                builder.commit();
            })
            .map_err(|error| OrtRuntimeError::Initialization {
                message: error.to_string(),
            })
    }

    fn resolve_library_path(&self) -> Result<PathBuf, DynamicRuntimeError> {
        if let Ok(path) = std::env::var("ORT_DYLIB_PATH")
            && !path.is_empty()
        {
            let path = PathBuf::from(path);
            return path.exists().then_some(path.clone()).ok_or_else(|| {
                DynamicRuntimeError::Missing {
                    library_name: self.library_name,
                    searched: path.display().to_string(),
                }
            });
        }

        let candidates = self.candidate_paths();
        candidates
            .iter()
            .find(|path| path.exists())
            .cloned()
            .ok_or_else(|| DynamicRuntimeError::Missing {
                library_name: self.library_name,
                searched: Self::format_paths(&candidates),
            })
    }

    fn candidate_paths(&self) -> Vec<PathBuf> {
        let mut candidates = Vec::new();

        if let Ok(exe) = std::env::current_exe()
            && let Some(exe_dir) = exe.parent()
        {
            candidates.push(exe_dir.join(self.library_name));
            if let Some(parent) = exe_dir.parent() {
                candidates.push(parent.join(self.library_name));
            }
        }

        if let Ok(cwd) = std::env::current_dir() {
            candidates.push(cwd.join(self.library_name));
            candidates.push(cwd.join("target/debug").join(self.library_name));
            candidates.push(cwd.join("target/debug/deps").join(self.library_name));
            candidates.push(cwd.join("target/release").join(self.library_name));
            candidates.push(cwd.join("target/release/deps").join(self.library_name));
        }

        dedup_paths(candidates)
    }

    fn validate_library(&self, path: &Path) -> Result<(), DynamicRuntimeError> {
        // safety: we only open the candidate runtime long enough to validate its exported API
        let library = unsafe { libloading::Library::new(path) }.map_err(|error| {
            DynamicRuntimeError::Load {
                path: path.to_path_buf(),
                message: error.to_string(),
            }
        })?;

        // safety: the library handle stays alive while the retrieved symbol is used below
        let get_api_base: libloading::Symbol<
            unsafe extern "C" fn() -> *const ort::sys::OrtApiBase,
        > = unsafe { library.get(b"OrtGetApiBase") }.map_err(|_| {
            DynamicRuntimeError::MissingApiBase {
                path: path.to_path_buf(),
            }
        })?;

        // safety: `OrtGetApiBase` has the stable ONNX Runtime entrypoint signature
        let api_base = unsafe { get_api_base() };
        if api_base.is_null() {
            return Err(DynamicRuntimeError::NullApiBase {
                path: path.to_path_buf(),
            });
        }

        // safety: the validated runtime exposes a process-stable version string pointer
        let version_ptr = unsafe { ((*api_base).GetVersionString)() };
        // safety: ONNX Runtime documents the version string as a null-terminated C string
        let version = unsafe { CStr::from_ptr(version_ptr) }
            .to_string_lossy()
            .into_owned();
        let minor = version
            .split('.')
            .nth(1)
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0);
        if minor < ort::MINOR_VERSION {
            return Err(DynamicRuntimeError::IncompatibleVersion {
                path: path.to_path_buf(),
                required_minor: ort::MINOR_VERSION,
                found_version: version,
            });
        }

        Ok(())
    }

    const fn default_library_name() -> &'static str {
        #[cfg(target_os = "windows")]
        {
            "onnxruntime.dll"
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            "libonnxruntime.so"
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            "libonnxruntime.dylib"
        }
    }

    fn format_paths(paths: &[PathBuf]) -> String {
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut unique = Vec::with_capacity(paths.len());
    for path in paths {
        if !unique.contains(&path) {
            unique.push(path);
        }
    }
    unique
}

#[cfg(test)]
mod tests {
    use super::ExecutionMode;
    #[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
    use super::{DynamicRuntimeError, OrtRuntimeError, ensure_ort_ready};

    #[test]
    fn embedding_threads_follow_pipeline_topology() {
        let available = super::available_threads();

        assert_eq!(ExecutionMode::Cpu.embedding_intra_threads(), available);
        assert_eq!(ExecutionMode::WebGpu.embedding_intra_threads(), available);
        assert_eq!(ExecutionMode::CoreMl.embedding_intra_threads(), 1);
        assert_eq!(ExecutionMode::Cuda.embedding_intra_threads(), 1);
        assert_eq!(ExecutionMode::MiGraphX.embedding_intra_threads(), 1);
    }

    #[cfg(not(feature = "coreml"))]
    #[test]
    fn coreml_modes_require_feature() {
        let error = ExecutionMode::CoreMl.validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "coreml requires the `coreml` Cargo feature"
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_modes_require_feature() {
        let error = ExecutionMode::Cuda.validate().unwrap_err();
        assert_eq!(error.to_string(), "cuda requires the `cuda` Cargo feature");
    }

    #[cfg(not(feature = "migraphx"))]
    #[test]
    fn migraphx_mode_requires_feature() {
        let error = ExecutionMode::MiGraphX.validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "migraphx requires the `migraphx` Cargo feature"
        );
    }

    #[cfg(not(feature = "webgpu"))]
    #[test]
    fn webgpu_mode_requires_feature() {
        let error = ExecutionMode::WebGpu.validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "webgpu requires the `webgpu` Cargo feature"
        );
    }

    #[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
    #[test]
    fn dynamic_runtime_preflight_fails_instead_of_hanging() {
        let original = std::env::var_os("ORT_DYLIB_PATH");
        let missing = std::env::temp_dir().join("missing-ort-runtime/libonnxruntime.dylib");
        // safety: this test mutates a process-global env var and restores it before returning
        unsafe {
            std::env::set_var("ORT_DYLIB_PATH", &missing);
        }

        let error = ensure_ort_ready().unwrap_err();
        assert!(matches!(
            error,
            super::ModelLoadError::Runtime(OrtRuntimeError::Dynamic(
                DynamicRuntimeError::Missing { .. }
            ))
        ));

        // safety: this test restores the original process-global env var before returning
        unsafe {
            match original {
                Some(value) => std::env::set_var("ORT_DYLIB_PATH", value),
                None => std::env::remove_var("ORT_DYLIB_PATH"),
            }
        }
    }
}
