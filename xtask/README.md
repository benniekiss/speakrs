# xtask

Development CLI for speakrs. Two binaries:

- **`xtask`** -- local dev tasks (benchmarks, model management, comparisons)
- **`speakrs-bm`** -- GPU benchmark runner for dstack containers (requires `cuda` feature)

## Commands

| Command | Description |
|---------|-------------|
| `models` | Export and deploy ONNX models to HF |
| `fixtures` | Regenerate test fixtures via Python |
| `compare` | Diarization comparisons (run, rttm, accuracy) |
| `bench` | Local benchmarks (run, compare, der) |
| `dstack` | Remote GPU benchmarks via dstack |
| `dataset` | Download/upload benchmark datasets |
| `diarize` | Run speaker diarization on WAV files |
| `profile-ort-embedding` | Profile ORT embedding inference strategies |
| `profile-stages` | Profile pipeline stages |

## Local benchmarks

```bash
# single-file timing: speakrs vs pyannote
cargo xtask bench run path/to/file.wav

# multi-tool comparison on one file
cargo xtask bench compare path/to/file.wav

# DER evaluation on a dataset
cargo xtask bench der --dataset voxconverse-dev --impls speakrs,pyannote

# single-file benchmark
cargo xtask bench der --file path/to/audio.wav --rttm path/to/ref.rttm --impls scm,sk
```

### DER implementation aliases

`--impls` accepts comma-separated full names or aliases:

| Alias | Full name | Description |
|-------|-----------|-------------|
| `pmps` | `pyannote` | pyannote MPS |
| `pcpu` | `pyannote-cpu` | pyannote CPU |
| `pg` | `pyannote-cuda` | pyannote CUDA |
| `scm` | `coreml` | speakrs CoreML |
| `sg` | `cuda` | speakrs CUDA |
| `scpu` | `cpu` | speakrs CPU |

## Datasets

| ID | Alias | Name |
|----|-------|------|
| `voxconverse-dev` | `vd` | VoxConverse Dev |
| `voxconverse-test` | `vt` | VoxConverse Test |
| `ami-ihm` | `ai` | AMI IHM |
| `ami-sdm` | `as` | AMI SDM |
| `aishell4` | `a4` | AISHELL-4 |
| `earnings21` | `e21` | Earnings-21 |
| `alimeeting` | `ali` | AliMeeting |
| `ava-avd` | `ava` | AVA-AVD |
| `icsi` | | ICSI |

## GPU implementations (speakrs-bm)

The `speakrs-bm` binary uses the same `--impls` syntax with its GPU subset:

| CLI ID | Alias | Description |
|--------|-------|-------------|
| `speakrs` | `sg` | speakrs CUDA (fused, 1s step) |
| `pyannote` | `pg` | pyannote CUDA |
