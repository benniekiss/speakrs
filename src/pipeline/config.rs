use crate::pipeline::{AhcConfig, BinarizeConfig, VbxConfig};

/// Sliding window length for segmentation model input, in seconds
pub const SEGMENTATION_WINDOW_SECONDS: f64 = 10.0;
/// Sliding window step for segmentation, in seconds
pub const SEGMENTATION_STEP_SECONDS: f64 = 1.0;
/// Duration of each output frame from the segmentation model, in seconds
pub const FRAME_DURATION_SECONDS: f64 = 0.0619375;
/// Hop between consecutive output frames from the segmentation model, in seconds
pub const FRAME_STEP_SECONDS: f64 = 0.016875;

/// Minimum speaker activity (sum of weights) to run embedding inference.
/// Speakers below this threshold are skipped because their NaN embedding is filtered out later
pub(crate) const MIN_SPEAKER_ACTIVITY: f32 = 10.0;

/// How to map cluster assignments back to per-frame speaker activations
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ReconstructMethod {
    /// Standard top-K selection (pyannote-compatible)
    Standard,
    /// Temporal smoothing. If scores are within epsilon, keep the previous speaker.
    Smoothed {
        /// Score difference below which the previous frame's speaker is preferred
        epsilon: f32,
    },
}

/// Tunable parameters for the diarization pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Hysteresis binarization and min-duration filtering
    pub binarize: BinarizeConfig,
    /// Agglomerative hierarchical clustering settings
    pub ahc: AhcConfig,
    /// Variational Bayes HMM clustering settings
    pub vbx: VbxConfig,
    /// Maximum gap in seconds between segments to merge into one
    pub merge_gap: f64,
    /// Minimum speaker activity weight to keep a speaker in output
    pub speaker_keep_threshold: f64,
    /// Strategy for mapping clusters back to frame activations
    pub reconstruct_method: ReconstructMethod,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            binarize: BinarizeConfig::default(),
            ahc: AhcConfig::default(),
            vbx: VbxConfig::default(),
            merge_gap: 0.0,
            speaker_keep_threshold: 1e-7,
            reconstruct_method: ReconstructMethod::Smoothed { epsilon: 0.1 },
        }
    }
}
