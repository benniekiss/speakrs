use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use color_eyre::eyre::Result;

use super::{
    BatchCommandRunner,
    DerAccumulation,
    DerImplResult,
    ImplType,
    PyannoteBatchSizes,
    validate::selected_implementations,
};

pub(super) type DerResults = (
    Vec<(&'static str, ImplType)>,
    HashMap<String, DerImplResult>,
);

pub(super) struct DerRunContext<'a> {
    pub root: &'a Path,
    pub run_dir: &'a Path,
    pub files: &'a [(PathBuf, PathBuf)],
    pub models_dir: &'a Path,
    pub impls: &'a [String],
    pub total_audio_seconds: f64,
    pub preflight_failures: &'a HashMap<String, String>,
    pub sleep_between: Option<Duration>,
    pub pyannote_batch_sizes: PyannoteBatchSizes,
}

pub(super) struct DerBenchEnv<'a> {
    root: &'a Path,
    models_dir: &'a Path,

    pyannote_batch_sizes: PyannoteBatchSizes,
}

impl<'a> DerBenchEnv<'a> {
    pub(super) fn new(
        root: &'a Path,
        models_dir: &'a Path,
        pyannote_batch_sizes: PyannoteBatchSizes,
    ) -> Self {
        Self {
            root,
            models_dir,
            pyannote_batch_sizes,
        }
    }

    fn speakrs_binary(&self) -> PathBuf {
        self.root.join("target/release/xtask")
    }

    pub(super) fn run_impl(
        &self,
        impl_type: &ImplType,
        wav_paths: &[&Path],
        timeout: Duration,
    ) -> Result<super::super::runner::BatchRunOutput> {
        match impl_type {
            ImplType::Speakrs(mode) => BatchCommandRunner::speakrs(
                &self.speakrs_binary(),
                mode,
                self.models_dir,
                wav_paths,
            )
            .run_with_retries(timeout),
            ImplType::Pyannote(device) => BatchCommandRunner::pyannote(
                self.root,
                device,
                wav_paths,
                self.pyannote_batch_sizes,
            )
            .run_with_retries(timeout),
        }
    }

    pub(super) fn skip_reason(&self, impl_type: &ImplType) -> Option<String> {
        match impl_type {
            ImplType::Pyannote(_) => {
                (!self.root.join("scripts/pyannote-bench/diarize.py").exists())
                    .then(|| "scripts/pyannote-bench/diarize.py not found".to_string())
            },
            ImplType::Speakrs(mode) => {
                #[cfg(not(target_os = "macos"))]
                if mode.starts_with("coreml") {
                    return Some("CoreML not available on this platform".to_string());
                }
                let _ = mode;
                None
            },
        }
    }
}

pub(crate) fn write_impl_result(
    run_dir: &Path,
    impl_name: &str,
    result: &DerImplResult,
    total_audio_seconds: f64,
) {
    let slug = impl_name.to_lowercase().replace(' ', "-");
    let payload = serde_json::json!({
        "implementation": impl_name,
        "status": result.status,
        "reason": result.reason,
        "der": result.der,
        "missed": result.missed,
        "false_alarm": result.false_alarm,
        "confusion": result.confusion,
        "time": result.time,
        "files": result.files,
        "total_audio_seconds": total_audio_seconds,
    });
    let _ = fs::write(
        run_dir.join(format!("{slug}.json")),
        serde_json::to_string_pretty(&payload).unwrap_or_default() + "\n",
    );
}

pub(super) fn run_der_implementations(ctx: &DerRunContext<'_>) -> Result<DerResults> {
    let env = DerBenchEnv::new(ctx.root, ctx.models_dir, ctx.pyannote_batch_sizes);
    let implementations = selected_implementations(ctx.impls);
    let wav_paths: Vec<&Path> = ctx.files.iter().map(|(wav, _)| wav.as_path()).collect();
    let mut all_results = HashMap::new();
    let batch_timeout = Duration::from_secs_f64((ctx.total_audio_seconds * 5.0).max(120.0));

    for (impl_name, impl_type) in &implementations {
        println!("Running {impl_name}...");

        if let Some(reason) = ctx.preflight_failures.get(*impl_name) {
            println!("  → skipped (preflight failed): {reason}");
            println!();
            let result = DerImplResult::failed(format!("preflight failed: {reason}"));
            write_impl_result(ctx.run_dir, impl_name, &result, ctx.total_audio_seconds);
            all_results.insert(impl_name.to_string(), result);
            continue;
        }

        if let Some(reason) = env.skip_reason(impl_type) {
            println!("  → skipped: {reason}");
            println!();
            let result = DerImplResult::skipped(reason);
            write_impl_result(ctx.run_dir, impl_name, &result, ctx.total_audio_seconds);
            all_results.insert(impl_name.to_string(), result);
            continue;
        }

        let benchmark_output = match env.run_impl(impl_type, &wav_paths, batch_timeout) {
            Ok(result) => result,
            Err(err) => {
                println!("  → failed: {err}");
                println!();
                let result = DerImplResult::failed(err.to_string());
                write_impl_result(ctx.run_dir, impl_name, &result, ctx.total_audio_seconds);
                all_results.insert(impl_name.to_string(), result);
                continue;
            },
        };

        let acc = DerAccumulation::compute(ctx.files, &benchmark_output.per_file_rttm)?;
        let (der_pct, miss_pct, fa_pct, conf_pct) = acc.der_percentages();
        let rtfx = ctx.total_audio_seconds / benchmark_output.total_seconds;
        if let Some(der) = der_pct {
            println!(
                "  → DER: {der:.1}%, Missed: {:.1}%, FA: {:.1}%, Confusion: {:.1}%, Time: {:.1}s, RTFx: {rtfx:.1}x",
                miss_pct.unwrap_or(0.0),
                fa_pct.unwrap_or(0.0),
                conf_pct.unwrap_or(0.0),
                benchmark_output.total_seconds,
            );
        } else {
            println!(
                "  → N/A, Time: {:.1}s, RTFx: {rtfx:.1}x",
                benchmark_output.total_seconds
            );
        }
        println!();

        let result = DerImplResult::completed(
            der_pct,
            miss_pct,
            fa_pct,
            conf_pct,
            benchmark_output.total_seconds,
            acc.file_count,
        );
        write_impl_result(ctx.run_dir, impl_name, &result, ctx.total_audio_seconds);
        all_results.insert(impl_name.to_string(), result);

        if let Some(delay) = ctx.sleep_between {
            println!(
                "  Sleeping {}s before next implementation...",
                delay.as_secs()
            );
            std::thread::sleep(delay);
        }
    }

    Ok((implementations, all_results))
}
