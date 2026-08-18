use crossbeam_channel::Sender;
use ndarray::Array2;
use ort::value::TensorRef;
use tracing::debug;

use super::{PRIMARY_BATCH_SIZE, SegmentationError, SegmentationModel};
use crate::inference::segmentation::tensor::{SegmentationWindows, first_output, output_shape3};

impl SegmentationModel {
    /// Run segmentation on audio, streaming raw logits through a channel
    ///
    /// Same logic as `run()`, but sends each decoded window through `tx` as it's produced
    /// instead of collecting into a Vec. Returns total window count
    pub fn run_streaming(
        &mut self,
        audio: &[f32],
        tx: Sender<Array2<f32>>,
    ) -> Result<usize, SegmentationError> {
        self.run_with_sink(audio, |window| {
            tx.send(window)?;
            Ok(())
        })
    }

    fn run_with_sink(
        &mut self,
        audio: &[f32],
        mut emit: impl FnMut(Array2<f32>) -> Result<(), SegmentationError>,
    ) -> Result<usize, SegmentationError> {
        let windows = SegmentationWindows::collect(audio, self.window_samples, self.step_samples);
        let total_windows = windows.total_windows();
        if windows.is_empty() {
            return Ok(0);
        }

        let seg_start = std::time::Instant::now();
        let mut seg_infer_time = std::time::Duration::ZERO;
        let mut seg_batched = 0u32;
        let mut seg_single = 0u32;

        let has_batched = self.primary_batched_session.is_some();

        let mut next_idx = 0;
        while next_idx < total_windows {
            let remaining = total_windows - next_idx;

            if remaining > 1 && has_batched {
                let batch_len = remaining.min(PRIMARY_BATCH_SIZE);
                let batch: Vec<&[f32]> = (next_idx..next_idx + batch_len)
                    .map(|idx| windows.window(idx, "batched segmentation"))
                    .collect::<Result<_, _>>()?;

                let t = std::time::Instant::now();
                let results = self.run_batch(&batch)?;
                seg_infer_time += t.elapsed();
                seg_batched += 1;
                for result in results {
                    emit(result)?;
                }
                next_idx += batch_len;
                continue;
            }

            let t = std::time::Instant::now();
            let result = self.run_window(windows.window(next_idx, "single segmentation")?)?;
            seg_infer_time += t.elapsed();
            seg_single += 1;
            emit(result)?;
            next_idx += 1;
        }

        let total_seg = seg_start.elapsed();
        debug!(
            windows = total_windows,
            seg_batched,
            seg_single,
            seg_infer_ms = seg_infer_time.as_millis(),
            seg_total_ms = total_seg.as_millis(),
            seg_overhead_ms = (total_seg - seg_infer_time).as_millis(),
            "Segmentation thread profile"
        );

        Ok(total_windows)
    }

    /// Run segmentation on audio, returning raw logits per window
    ///
    /// Returns `Vec<Array2<f32>>` where each element is [frames, 7] logits
    pub fn run(&mut self, audio: &[f32]) -> Result<Vec<Array2<f32>>, ort::Error> {
        let mut results = Vec::new();
        self.run_with_sink(audio, |window| {
            results.push(window);
            Ok(())
        })
        .map_err(|error| ort::Error::new(error.to_string()))?;
        Ok(results)
    }

    fn run_window(&mut self, window: &[f32]) -> Result<Array2<f32>, SegmentationError> {
        self.input_buffer.fill(0.0);
        self.input_buffer
            .slice_mut(ndarray::s![0, 0, ..window.len()])
            .assign(&ndarray::ArrayView1::from(window));
        let input_tensor = TensorRef::from_array_view(self.input_buffer.view())?;

        let outputs = self.session.run(ort::inputs![input_tensor])?;
        let output = first_output(outputs.values(), "segmentation window output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;

        let (_batch, frames, classes) = output_shape3(shape, "segmentation window output")?;

        Array2::from_shape_vec((frames, classes), data.to_vec()).map_err(|error| {
            SegmentationError::MalformedOutput {
                context: "segmentation window output",
                message: format!("invalid output shape: {error}"),
            }
        })
    }

    fn run_batch(&mut self, windows: &[&[f32]]) -> Result<Vec<Array2<f32>>, SegmentationError> {
        if windows.len() > PRIMARY_BATCH_SIZE {
            return Err(SegmentationError::Invariant {
                context: "segmentation batch input",
                message: format!(
                    "received {} windows for batch size {PRIMARY_BATCH_SIZE}",
                    windows.len()
                ),
            });
        }
        if windows.len() < PRIMARY_BATCH_SIZE {
            self.primary_batch_input_buffer
                .slice_mut(ndarray::s![windows.len().., .., ..])
                .fill(0.0);
        }
        for (batch_idx, window) in windows.iter().enumerate() {
            let copy_len = window.len().min(self.window_samples);
            self.primary_batch_input_buffer
                .slice_mut(ndarray::s![batch_idx, 0, ..copy_len])
                .assign(&ndarray::ArrayView1::from(&window[..copy_len]));
            if copy_len < self.window_samples {
                self.primary_batch_input_buffer
                    .slice_mut(ndarray::s![batch_idx, 0, copy_len..])
                    .fill(0.0);
            }
        }
        let input_tensor = TensorRef::from_array_view(self.primary_batch_input_buffer.view())?;

        let outputs = self
            .primary_batched_session
            .as_mut()
            .ok_or_else(|| ort::Error::new("missing primary batched segmentation session"))?
            .run(ort::inputs![input_tensor])?;
        let output = first_output(outputs.values(), "segmentation batch output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;

        let (batch, frames, classes) = output_shape3(shape, "segmentation batch output")?;
        if batch < windows.len() {
            return Err(SegmentationError::MalformedOutput {
                context: "segmentation batch output",
                message: format!(
                    "model returned {batch} rows for {} input windows",
                    windows.len()
                ),
            });
        }
        let stride = frames * classes;
        let expected_len =
            batch
                .checked_mul(stride)
                .ok_or_else(|| SegmentationError::MalformedOutput {
                    context: "segmentation batch output",
                    message: format!("output shape {shape} exceeded addressable memory"),
                })?;
        if data.len() != expected_len {
            return Err(SegmentationError::MalformedOutput {
                context: "segmentation batch output",
                message: format!(
                    "shape {shape} expected {expected_len} values, got {}",
                    data.len()
                ),
            });
        }

        (0..windows.len())
            .map(|batch_idx| {
                let start = batch_idx * stride;
                Array2::from_shape_vec((frames, classes), data[start..start + stride].to_vec())
                    .map_err(|error| SegmentationError::MalformedOutput {
                        context: "segmentation batch output",
                        message: format!("invalid output shape: {error}"),
                    })
            })
            .collect::<Result<Vec<_>, _>>()
    }
}
