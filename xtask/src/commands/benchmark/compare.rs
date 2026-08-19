use std::time::Duration;

use color_eyre::eyre::{Result, bail};

use super::{runner::SingleRunOutput, *};

struct RunResult {
    name: String,
    mean_seconds: f64,
    min_seconds: f64,
    rttm: String,
    speakers: usize,
    segments: usize,
}

impl RunResult {
    fn new(name: &str, mean_seconds: f64, min_seconds: f64, rttm: String) -> Self {
        let mut speakers_set = std::collections::HashSet::new();
        let mut segments = 0;
        for line in rttm.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.first() == Some(&"SPEAKER") && parts.len() >= 8 {
                segments += 1;
                speakers_set.insert(parts[7].to_string());
            }
        }
        Self {
            name: name.to_string(),
            mean_seconds,
            min_seconds,
            rttm,
            speakers: speakers_set.len(),
            segments,
        }
    }
}

enum CompareOutcome {
    Completed(RunResult),
    Failed { name: String, reason: String },
}

struct CompareRunner {
    spec: CommandSpec,
}

impl CompareRunner {
    fn run_once(&self) -> Result<SingleRunOutput> {
        let mut command = self.spec.build_command();
        capture_benchmark_cmd(&mut command, Duration::from_secs(30 * 60))
    }
}

struct CompareRecorder<'a> {
    results: &'a mut Vec<CompareOutcome>,
    warmups: u32,
    runs: u32,
}

impl<'a> CompareRecorder<'a> {
    fn record(&mut self, name: &str, runner: &CompareRunner) {
        match run_compare_impl(name, self.warmups, self.runs, runner) {
            Ok(result) => {
                println!(
                    "  {name}: mean {:.2}s, min {:.2}s",
                    result.mean_seconds, result.min_seconds
                );
                self.results.push(CompareOutcome::Completed(result));
            },
            Err(err) => {
                println!("  {name}: FAILED ({err})");
                self.results.push(CompareOutcome::Failed {
                    name: name.to_string(),
                    reason: err.to_string(),
                });
            },
        }
    }
}

pub fn compare(source: &str, runs: u32, warmups: u32) -> Result<()> {
    let prepared_audio = prepare_audio(source)?;
    let wav = prepared_audio.wav_path();
    let wav_str = wav.to_string_lossy();
    let root = project_root();

    println!();
    println!("=== Building binaries ===");
    cargo_build_xtask(&["coreml".to_string()])?;

    let audio_seconds = wav_duration_seconds(wav)?;

    println!();
    println!("=== Running benchmarks ===");

    let speakrs_binary = root.join("target/release/xtask");

    let implementations: Vec<(&str, Vec<String>)> = vec![
        (
            "pyannote MPS",
            vec![
                "uv".into(),
                "run".into(),
                "--project".into(),
                "scripts/pyannote-bench".into(),
                "python".into(),
                "scripts/pyannote-bench/diarize.py".into(),
                "--device".into(),
                "mps".into(),
                wav_str.to_string(),
            ],
        ),
        (
            "speakrs CoreML",
            vec![
                speakrs_binary.to_string_lossy().into(),
                "diarize".into(),
                "--mode".into(),
                "coreml".into(),
                wav_str.to_string(),
            ],
        ),
    ];

    let mut results: Vec<CompareOutcome> = Vec::new();
    let mut compare_recorder = CompareRecorder {
        results: &mut results,
        warmups,
        runs,
    };

    for (name, command_args) in &implementations {
        let runner = CompareRunner {
            spec: CommandSpec::from_argv(command_args).current_dir(&root),
        };
        compare_recorder.record(name, &runner);
    }

    let ref_rttm = results.iter().find_map(|result| match result {
        CompareOutcome::Completed(run) if run.name == "pyannote MPS" => Some(run.rttm.clone()),
        _ => None,
    });

    let minutes = (audio_seconds / 60.0) as u32;
    let secs = audio_seconds % 60.0;
    println!();
    println!(
        "Audio: {minutes}:{secs:04.1} ({audio_seconds:.1}s)  |  Warmups: {warmups}  |  Runs: {runs}"
    );
    println!();

    let name_width = 22;
    println!(
        "{:<name_width$} {:>9} {:>9} {:>9} {:>9} {:>10}  Status",
        "Implementation", "Mean", "Min", "Speakers", "Segments", "Parity %"
    );
    println!("{}", "─".repeat(name_width + 56));

    for result in &results {
        match result {
            CompareOutcome::Completed(run) => {
                let mean_str = format!("{:.2}s", run.mean_seconds);
                let min_str = format!("{:.2}s", run.min_seconds);
                let speakers_str = if run.segments > 0 {
                    run.speakers.to_string()
                } else {
                    "—".to_string()
                };
                let segments_str = run.segments.to_string();
                let parity_str = match ref_rttm.as_ref() {
                    Some(_) if run.name == "pyannote MPS" => "(reference)".to_string(),
                    Some(reference) if run.segments > 0 => {
                        timeline_overlap_pct(reference, &run.rttm)
                            .map(|pct| format!("{pct:.1}%"))
                            .unwrap_or_else(|| "N/A".to_string())
                    },
                    _ => "N/A".to_string(),
                };

                println!(
                    "{:<name_width$} {:>9} {:>9} {:>9} {:>9} {:>10}  ok",
                    run.name, mean_str, min_str, speakers_str, segments_str, parity_str
                );
            },
            CompareOutcome::Failed { name, reason } => {
                println!(
                    "{:<name_width$} {:>9} {:>9} {:>9} {:>9} {:>10}  failed ({reason})",
                    name, "—", "—", "—", "—", "N/A"
                );
            },
        }
    }

    Ok(())
}

fn run_compare_impl(
    name: &str,
    warmups: u32,
    runs: u32,
    runner: &CompareRunner,
) -> Result<RunResult> {
    if runs == 0 {
        bail!("runs must be greater than 0");
    }

    for _ in 0..warmups {
        runner.run_once()?;
    }

    let mut measurements = Vec::with_capacity(runs as usize);
    let mut representative_rttm = None;
    for _ in 0..runs {
        let output = runner.run_once()?;
        if representative_rttm.is_none() {
            representative_rttm = Some(output.rttm);
        }
        measurements.push(output.elapsed_seconds);
    }

    Ok(summarize_compare_runs(
        name,
        &measurements,
        representative_rttm.unwrap_or_default(),
    ))
}

fn summarize_compare_runs(name: &str, measurements: &[f64], rttm: String) -> RunResult {
    let mean_seconds = measurements.iter().sum::<f64>() / measurements.len() as f64;
    let min_seconds = measurements.iter().copied().fold(f64::INFINITY, f64::min);
    RunResult::new(name, mean_seconds, min_seconds, rttm)
}

fn timeline_overlap_pct(ref_rttm: &str, test_rttm: &str) -> Option<f64> {
    let ref_intervals = parse_rttm_intervals(ref_rttm);
    let test_intervals = parse_rttm_intervals(test_rttm);

    if ref_intervals.is_empty() || test_intervals.is_empty() {
        return None;
    }

    let mut merged = test_intervals;
    merged.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut combined = vec![merged[0]];
    for &(start, end) in &merged[1..] {
        if let Some(last) = combined.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
            continue;
        }
        combined.push((start, end));
    }

    let ref_total: f64 = ref_intervals.iter().map(|(start, end)| end - start).sum();
    if ref_total <= 0.0 {
        return None;
    }

    let mut covered = 0.0;
    for &(ref_start, ref_end) in &ref_intervals {
        for &(merged_start, merged_end) in &combined {
            if merged_start >= ref_end {
                break;
            }
            if merged_end <= ref_start {
                continue;
            }
            covered += ref_end.min(merged_end) - ref_start.max(merged_start);
        }
    }

    Some(covered / ref_total * 100.0)
}

fn parse_rttm_intervals(rttm: &str) -> Vec<(f64, f64)> {
    rttm.lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.first() != Some(&"SPEAKER") || parts.len() < 5 {
                return None;
            }
            let start: f64 = parts[3].parse().ok()?;
            let duration: f64 = parts[4].parse().ok()?;
            Some((start, start + duration))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_compare_runs_uses_mean_and_min() {
        let result = summarize_compare_runs(
            "speakrs CoreML",
            &[6.0, 4.0, 5.0],
            "SPEAKER file1 1 0.000000 1.000000 <NA> <NA> spk1 <NA> <NA>\n".to_string(),
        );

        assert_eq!(result.name, "speakrs CoreML");
        assert_eq!(result.mean_seconds, 5.0);
        assert_eq!(result.min_seconds, 4.0);
        assert_eq!(result.segments, 1);
        assert_eq!(result.speakers, 1);
    }
}
