use std::path::PathBuf;

use super::{
    OwnedDiarizationPipeline,
    config::{PipelineConfig, SEGMENTATION_STEP_SECONDS},
    queued::{QueueReceiver, QueueSender},
    types::PipelineError,
};
use crate::{
    clustering::plda::PldaTransform,
    inference::{ExecutionMode, embedding::EmbeddingModel, segmentation::SegmentationModel},
    models::ModelBundle,
    powerset::PowersetMapping,
};

/// Builder for constructing diarization pipelines
///
/// # Examples
///
/// ```no_run
/// use speakrs::{ExecutionMode, PipelineBuilder};
///
/// // minimal
/// let mut pipeline = PipelineBuilder::from_pretrained(ExecutionMode::Cpu)?.build()?;
///
/// // from local directory
/// let mut pipeline = PipelineBuilder::from_dir("./models", ExecutionMode::Cpu).build()?;
/// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
/// ```
pub struct PipelineBuilder {
    bundle: ModelBundle,
    mode: ExecutionMode,
    pipeline: Option<PipelineConfig>,
}

impl PipelineBuilder {
    /// Start from a local models directory
    pub fn from_dir(models_dir: impl Into<PathBuf>, mode: ExecutionMode) -> Self {
        Self {
            bundle: ModelBundle::from_dir(models_dir),
            mode,
            pipeline: None,
        }
    }

    /// Start from a pre-resolved [`ModelBundle`](ModelBundle)
    pub fn from_bundle(bundle: ModelBundle, mode: ExecutionMode) -> Self {
        Self {
            bundle,
            mode,
            pipeline: None,
        }
    }

    /// Download models from HuggingFace and start building
    #[cfg(feature = "online")]
    #[cfg_attr(docsrs, doc(cfg(feature = "online")))]
    pub fn from_pretrained(mode: ExecutionMode) -> Result<Self, PipelineError> {
        mode.validate()?;
        let bundle = ModelBundle::from_pretrained(mode)?;
        Ok(Self::from_bundle(bundle, mode))
    }

    /// Override pipeline config (thresholds, clustering)
    pub fn pipeline(mut self, config: PipelineConfig) -> Self {
        self.pipeline = Some(config);
        self
    }

    /// Build the owned pipeline
    pub fn build(self) -> Result<OwnedDiarizationPipeline, PipelineError> {
        self.mode.validate()?;

        let pipeline = self.pipeline.unwrap_or_default();
        let seg_model = SegmentationModel::with_mode(
            self.bundle.segmentation_path(),
            SEGMENTATION_STEP_SECONDS as f32,
            self.mode,
        )?;
        let emb_model = EmbeddingModel::with_mode(self.bundle.embedding_path(), self.mode)?;
        let plda = PldaTransform::from_dir(self.bundle.plda_dir())?;

        Ok(OwnedDiarizationPipeline {
            seg_model,
            emb_model,
            plda,
            powerset: PowersetMapping::new(3, 2),
            default_config: pipeline,
        })
    }

    /// Build and immediately convert to a background-processing queue
    pub fn build_queued(self) -> Result<(QueueSender, QueueReceiver), PipelineError> {
        let pipeline = self.build()?;
        Ok(pipeline.into_queued()?)
    }
}
