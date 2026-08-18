use ndarray::{Array2, Array3, s};
use tracing::{debug, trace};

use crate::inference::embedding::{
    BatchPhaseTiming, EmbeddingModel, MaskedEmbeddingInput, SplitTailInput,
};
use crate::pipeline::{
    MIN_SPEAKER_ACTIVITY, clean_masks, select_speaker_weights, write_speaker_mask_to_slice,
};
use crate::reconstruct::Reconstructor;

use super::{
    ChunkEmbeddings, ChunkLayout, DecodedSegmentations, EmbeddingPath, PendingEmbedding,
    PendingSplitEmbedding, PipelineError, SpeakerCountTrack,
};

impl DecodedSegmentations {
    pub(in crate::pipeline) fn nchunks(&self) -> usize {
        self.0.shape()[0]
    }

    pub(in crate::pipeline) fn num_speakers(&self) -> usize {
        if self.0.ndim() < 3 {
            return 0;
        }
        self.0.shape()[2]
    }

    pub(in crate::pipeline) fn speaker_count(&self, layout: &ChunkLayout) -> SpeakerCountTrack {
        let reconstructor = Reconstructor::new(self, &layout.start_frames, 0);
        reconstructor.speaker_count(layout.output_frames)
    }

    pub(in crate::pipeline) fn extract_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embedding_path: EmbeddingPath,
    ) -> Result<ChunkEmbeddings, PipelineError> {
        let num_chunks = self.0.shape()[0];
        let num_speakers = self.0.shape()[2];
        let mut embeddings = Array3::<f32>::from_elem((num_chunks, num_speakers, 256), f32::NAN);

        match embedding_path {
            EmbeddingPath::MultiMask => {
                self.extract_multi_mask_embeddings(audio, emb_model, layout, &mut embeddings)?
            },
            EmbeddingPath::Split => {
                self.extract_split_embeddings(audio, emb_model, layout, &mut embeddings)?
            },
            EmbeddingPath::Masked => {
                if emb_model.prefers_chunk_embedding_path() {
                    self.extract_chunk_embeddings(audio, emb_model, layout, &mut embeddings)?;
                } else {
                    self.extract_masked_embeddings(audio, emb_model, layout, &mut embeddings)?;
                }
            },
        }

        Ok(ChunkEmbeddings(embeddings))
    }

    fn extract_chunk_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        for chunk_idx in 0..self.0.shape()[0] {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);
            let chunk_embeddings =
                emb_model.embed_chunk_speakers(chunk_audio, chunk_segmentations, &clean_masks)?;
            embeddings
                .slice_mut(s![chunk_idx, .., ..])
                .assign(&chunk_embeddings);
        }

        Ok(())
    }

    fn extract_masked_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let mut storage = Array3Writer(embeddings);
        let mut pending = Vec::with_capacity(emb_model.primary_batch_size());

        for chunk_idx in 0..self.0.shape()[0] {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);

            for speaker_idx in 0..self.0.shape()[2] {
                let mask = chunk_segmentations.column(speaker_idx);
                let activity: f32 = mask.iter().sum();
                if activity < MIN_SPEAKER_ACTIVITY {
                    continue;
                }

                pending.push(PendingEmbedding {
                    chunk_idx,
                    speaker_idx,
                    audio: chunk_audio,
                    mask: mask.to_vec(),
                    clean_mask: clean_masks.column(speaker_idx).to_vec(),
                });
                if pending.len() == emb_model.primary_batch_size() {
                    flush_masked(emb_model, &pending, &mut storage)?;
                    pending.clear();
                }
            }
        }

        while !pending.is_empty() {
            let batch_len = emb_model.best_batch_len(pending.len());
            flush_masked(emb_model, &pending[..batch_len], &mut storage)?;
            pending.drain(..batch_len);
        }

        Ok(())
    }

    fn extract_split_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let batch_size = emb_model.split_primary_batch_size();
        let num_speakers = self.0.shape()[2];
        let min_num_samples = emb_model.min_num_samples();

        let mut storage = Array3Writer(embeddings);
        let mut pending: Vec<PendingSplitEmbedding> = Vec::with_capacity(batch_size);
        let mut fbanks: Vec<Array2<f32>> = Vec::new();
        let mut tail_batches = 0usize;
        let mut active_items = 0usize;

        for chunk_idx in 0..self.0.shape()[0] {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let fbank = emb_model.compute_chunk_fbank(chunk_audio)?;
            fbanks.push(fbank);
            let mut current_fbank_idx = fbanks.len() - 1;

            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_masks(&chunk_segmentations);

            for speaker_idx in 0..num_speakers {
                let Some(weights) = select_speaker_weights(
                    &chunk_segmentations,
                    &clean_masks,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                ) else {
                    continue;
                };
                active_items += 1;
                pending.push(PendingSplitEmbedding {
                    chunk_idx,
                    speaker_idx,
                    fbank_idx: current_fbank_idx,
                    weights,
                });
                if pending.len() == batch_size {
                    flush_split(emb_model, &pending, &fbanks, &mut storage)?;
                    tail_batches += 1;
                    pending.clear();

                    if speaker_idx + 1 < num_speakers {
                        let kept_fbank = fbanks.swap_remove(current_fbank_idx);
                        fbanks.clear();
                        fbanks.push(kept_fbank);
                        current_fbank_idx = 0;
                    } else {
                        fbanks.clear();
                    }
                }
            }
        }

        if !pending.is_empty() {
            flush_split(emb_model, &pending, &fbanks, &mut storage)?;
            tail_batches += 1;
        }

        debug!(
            batches = tail_batches,
            active_items,
            total_items = self.0.shape()[0] * num_speakers,
            "Split embeddings complete (fbank+tail streaming)"
        );

        Ok(())
    }

    fn extract_multi_mask_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let batch_size = emb_model.multi_mask_batch_size();
        let num_speakers = self.0.shape()[2];
        let num_chunks = self.0.shape()[0];
        let min_num_samples = emb_model.min_num_samples();

        let mut storage = Array3Writer(embeddings);
        let num_frames = self.0.shape()[1];
        let mut audio_buffer: Vec<&[f32]> = Vec::with_capacity(batch_size);
        let mut flat_masks = vec![0.0; batch_size * num_speakers * num_frames];
        let mut active_flags: Vec<bool> = Vec::with_capacity(batch_size * num_speakers);
        let mut chunk_indices: Vec<usize> = Vec::with_capacity(batch_size);
        let mut batches = 0usize;
        let mut timing = MultiMaskTiming::default();

        for chunk_idx in 0..num_chunks {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let mask_base = audio_buffer.len() * num_speakers;
            let mut any_active = false;

            for speaker_idx in 0..num_speakers {
                let mask_idx = mask_base + speaker_idx;
                let offset = mask_idx * num_frames;
                let active = write_speaker_mask_to_slice(
                    &chunk_segmentations,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                    &mut flat_masks[offset..offset + num_frames],
                );
                active_flags.push(active);
                any_active |= active;
            }

            if !any_active {
                active_flags.truncate(mask_base);
                continue;
            }

            audio_buffer.push(chunk_audio);
            chunk_indices.push(chunk_idx);

            if audio_buffer.len() == batch_size {
                let batch = MultiMaskBatch {
                    audio_slices: &audio_buffer,
                    flat_masks: &flat_masks,
                    mask_stride: num_frames,
                    active_flags: &active_flags,
                    chunk_indices: &chunk_indices,
                    num_speakers,
                };
                timing += flush_multi_mask_audio(emb_model, &batch, &mut storage)?;
                batches += 1;
                audio_buffer.clear();
                flat_masks.fill(0.0);
                active_flags.clear();
                chunk_indices.clear();
            }
        }

        if !audio_buffer.is_empty() {
            let batch = MultiMaskBatch {
                audio_slices: &audio_buffer,
                flat_masks: &flat_masks,
                mask_stride: num_frames,
                active_flags: &active_flags,
                chunk_indices: &chunk_indices,
                num_speakers,
            };
            timing += flush_multi_mask_audio(emb_model, &batch, &mut storage)?;
            batches += 1;
        }

        debug!(
            batches,
            total_chunks = num_chunks,
            "Multi-mask embeddings complete"
        );
        trace_multi_mask_timing(timing, batches, num_chunks);

        Ok(())
    }
}

pub(in crate::pipeline) trait EmbeddingStorage {
    fn store(&mut self, chunk_idx: usize, speaker_idx: usize, embedding: &[f32]);
}

fn store_row<S: EmbeddingStorage>(
    storage: &mut S,
    chunk_idx: usize,
    speaker_idx: usize,
    row: ndarray::ArrayView1<'_, f32>,
) {
    if let Some(values) = row.as_slice() {
        storage.store(chunk_idx, speaker_idx, values);
        return;
    }

    let values = row.to_vec();
    storage.store(chunk_idx, speaker_idx, &values);
}

/// Writes embeddings into a pre-allocated Array3 by (chunk, speaker) index
pub(in crate::pipeline) struct Array3Writer<'a>(pub &'a mut Array3<f32>);

impl EmbeddingStorage for Array3Writer<'_> {
    fn store(&mut self, chunk_idx: usize, speaker_idx: usize, embedding: &[f32]) {
        self.0
            .slice_mut(s![chunk_idx, speaker_idx, ..])
            .assign(&ndarray::ArrayView1::from(embedding));
    }
}

pub(in crate::pipeline) fn flush_masked<S: EmbeddingStorage>(
    emb_model: &mut EmbeddingModel,
    pending: &[PendingEmbedding<'_>],
    storage: &mut S,
) -> Result<(), PipelineError> {
    let batch_inputs: Vec<_> = pending
        .iter()
        .map(|item| MaskedEmbeddingInput {
            audio: item.audio,
            mask: &item.mask,
            clean_mask: Some(&item.clean_mask),
        })
        .collect();
    let batch_embeddings = emb_model.embed_batch(&batch_inputs)?;

    for (batch_idx, item) in pending.iter().enumerate() {
        store_row(
            storage,
            item.chunk_idx,
            item.speaker_idx,
            batch_embeddings.row(batch_idx),
        );
    }

    Ok(())
}

pub(in crate::pipeline) fn flush_split<S: EmbeddingStorage>(
    emb_model: &mut EmbeddingModel,
    pending: &[PendingSplitEmbedding],
    fbanks: &[Array2<f32>],
    storage: &mut S,
) -> Result<(), PipelineError> {
    let batch_inputs: Vec<_> = pending
        .iter()
        .map(|item| SplitTailInput {
            fbank: &fbanks[item.fbank_idx],
            weights: &item.weights,
        })
        .collect();
    let batch_embeddings = emb_model.embed_tail_batch_inputs(&batch_inputs)?;

    for (batch_idx, item) in pending.iter().enumerate() {
        store_row(
            storage,
            item.chunk_idx,
            item.speaker_idx,
            batch_embeddings.row(batch_idx),
        );
    }

    Ok(())
}

pub(in crate::pipeline) struct MultiMaskBatch<'a> {
    pub audio_slices: &'a [&'a [f32]],
    pub flat_masks: &'a [f32],
    pub mask_stride: usize,
    pub active_flags: &'a [bool],
    pub chunk_indices: &'a [usize],
    pub num_speakers: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub(in crate::pipeline) struct MultiMaskTiming {
    pub fbank: BatchPhaseTiming,
    pub tail: BatchPhaseTiming,
    pub storage: std::time::Duration,
}

impl std::ops::AddAssign for MultiMaskTiming {
    fn add_assign(&mut self, rhs: Self) {
        self.fbank += rhs.fbank;
        self.tail += rhs.tail;
        self.storage += rhs.storage;
    }
}

pub(in crate::pipeline) fn trace_multi_mask_timing(
    timing: MultiMaskTiming,
    batches: usize,
    chunks: usize,
) {
    trace!(
        batches,
        chunks,
        fbank_input_ms = timing.fbank.input.as_millis(),
        fbank_inference_ms = timing.fbank.inference.as_millis(),
        fbank_output_ms = timing.fbank.output.as_millis(),
        tail_input_ms = timing.tail.input.as_millis(),
        tail_inference_ms = timing.tail.inference.as_millis(),
        tail_output_ms = timing.tail.output.as_millis(),
        embedding_store_ms = timing.storage.as_millis(),
        "Multi-mask phase timing"
    );
}

pub(in crate::pipeline) fn flush_multi_mask_audio<S: EmbeddingStorage>(
    emb_model: &mut EmbeddingModel,
    batch: &MultiMaskBatch<'_>,
    storage: &mut S,
) -> Result<MultiMaskTiming, PipelineError> {
    debug_assert_eq!(batch.audio_slices.len(), batch.chunk_indices.len());
    let num_masks = batch.audio_slices.len() * batch.num_speakers;
    debug_assert_eq!(batch.active_flags.len(), num_masks);

    let (fbanks, fbank_timing) =
        emb_model.compute_chunk_fbanks_batch_profiled(batch.audio_slices)?;

    let fbank_refs: Vec<_> = fbanks.iter().collect();
    let mask_refs: Vec<&[f32]> = batch
        .flat_masks
        .chunks(batch.mask_stride)
        .take(num_masks)
        .collect();

    let (batch_embeddings, tail_timing) =
        emb_model.embed_multi_mask_batch_profiled(&fbank_refs, &mask_refs)?;

    let storage_start = std::time::Instant::now();
    for (fbank_idx, &chunk_idx) in batch.chunk_indices.iter().enumerate() {
        for speaker_idx in 0..batch.num_speakers {
            let mask_idx = fbank_idx * batch.num_speakers + speaker_idx;
            if !batch.active_flags[mask_idx] {
                continue;
            }
            store_row(
                storage,
                chunk_idx,
                speaker_idx,
                batch_embeddings.row(mask_idx),
            );
        }
    }
    let storage = storage_start.elapsed();

    Ok(MultiMaskTiming {
        fbank: fbank_timing,
        tail: tail_timing,
        storage,
    })
}
