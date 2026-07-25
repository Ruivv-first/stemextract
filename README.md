# stemextract

`stemextract` is a lightweight, typed Python library for extracting music stems
with Meta AI's pretrained HTDemucs model. It adds a stable one-line API,
cross-platform audio handling, accelerator selection, progress reporting, and
folder automation. It does not train or modify an AI model.

The PyPI distribution and Python import are both named `stemextract`.

> stemsep is an open-source engineering project that simplifies state-of-the-art audio source separation for music creators. The underlying deep learning separation model is developed by Meta AI (HTDemucs). This project only provides optimized user-facing encapsulation, workflow automation, and multi-format compatibility layers.

## Features

- One-line four-stem separation: vocals, drums, bass, and other
- Selective single-stem export
- Recursive folder processing with per-song output directories
- CUDA → Apple MPS → CPU automatic device selection
- Automatic CPU retry after CUDA or MPS out-of-memory failures
- WAV, FLAC, MP3, and M4A input and output
- Safe temporary conversion and atomic output replacement
- Progress bars for model inference, exports, and batches
- Automatic first-use HTDemucs weight download and standard PyTorch caching
- Clear exceptions for missing, unsupported, corrupted, or unwritable files

## Requirements

- Python 3.10 or newer
- [FFmpeg](https://ffmpeg.org/download.html) available on `PATH` for MP3/M4A
  input and output, plus fallback decoding
- Enough memory for HTDemucs. GPU acceleration is optional.

Install the library:

```bash
python -m pip install stemextract
```

For local development:

```bash
python -m pip install -e ".[dev]"
```

FFmpeg is a system dependency and is intentionally not bundled. Confirm it is
available with:

```bash
ffmpeg -version
```

## Quick start

```python
import stemextract

# Full stem separation
tracks = stemextract.extract("song.mp3", "output/song")

# Extract only vocals
vocals = stemextract.extract_vocals("song.m4a", "output/vocals")

# Batch extract an entire folder
result = stemextract.batch_extract("music", "separated")
print(f"{result.succeeded}/{result.total} songs completed")

# Save tracks with a custom lossless format
stemextract.save_tracks(tracks, "output/flac", format="flac")
```

The required public calls are:

```python
stemextract.extract(input_path, output_dir)
stemextract.extract_vocals(input_path, output_dir)
stemextract.batch_extract(input_folder, output_folder)
stemextract.save_tracks(tracks, output_dir, format="wav")
```

`extract()` and the selective extraction functions save their files and return
the selected channel-first `torch.Tensor` objects. This makes it possible to
save the same separation again in another supported format without repeating
inference.

## Complete local file example

The following example separates `/Users/mac/Desktop/1test.wav` and writes the
four WAV stems to `/Users/mac/Desktop/1test_stems`. It is also available as
[`examples/extract_local_file.py`](examples/extract_local_file.py).

```python
from pathlib import Path

import stemextract


INPUT_FILE = Path("/Users/mac/Desktop/1test.wav")
OUTPUT_DIR = Path("/Users/mac/Desktop/1test_stems")


def main() -> None:
    if not INPUT_FILE.is_file():
        raise SystemExit(f"Input audio file does not exist: {INPUT_FILE}")

    try:
        tracks = stemextract.extract(
            INPUT_FILE,
            OUTPUT_DIR,
            device="auto",
            progress=True,
        )
    except stemextract.StemExtractError as exc:
        raise SystemExit(f"Stem separation failed: {exc}") from exc

    print("Separation completed successfully:")
    for stem_name in tracks:
        print(f"  {stem_name}: {OUTPUT_DIR / f'{stem_name}.wav'}")


if __name__ == "__main__":
    main()
```

Run the example from a source checkout with:

```bash
python examples/extract_local_file.py
```

The output directory will contain `vocals.wav`, `drums.wav`, `bass.wav`, and
`other.wav`.

## Selective stems

HTDemucs produces four sources in one inference pass. Selective calls export
only the requested source:

```python
stemextract.extract_vocals("song.wav", "stems/vocals")
stemextract.extract_drums("song.wav", "stems/drums")
stemextract.extract_bass("song.wav", "stems/bass")
stemextract.extract_other("song.wav", "stems/other")

# Dynamic selection
stemextract.extract_stem("song.wav", "stems/selected", "drums")
```

Selective export saves disk space, but does not make HTDemucs infer only one
source—the pretrained model always computes its complete source set.

## Formats and output layout

WAV is the default output format. `format="flac"`, `"mp3"`, and `"m4a"` are
also supported. WAV and FLAC are written as 24-bit PCM. Compressed exports use
FFmpeg (320 kbps MP3 or 256 kbps AAC/M4A).

A single-song extraction writes stem files directly into the requested folder:

```text
output/song/
├── vocals.wav
├── drums.wav
├── bass.wav
└── other.wav
```

Batch mode creates one directory per song and preserves relative subfolders.
If files such as `song.mp3` and `song.flac` would collide, their output
directories receive format suffixes.

## Device and memory behavior

By default, stemextract chooses CUDA when available, then Apple MPS, then CPU:

```python
stemextract.extract("song.wav", "stems", device="auto")
stemextract.extract("song.wav", "stems", device="cpu")
```

Requesting an unavailable accelerator falls back to CPU. If accelerator
inference runs out of memory, stemextract releases the device cache and retries
once on CPU. For constrained systems, reducing Demucs' split segment can also
help:

```python
stemextract.extract("song.wav", "stems", segment=5)
```

## Error handling

All expected library errors inherit from `stemextract.StemExtractError`:

```python
try:
    stemextract.extract("possibly-broken.mp3", "stems")
except stemextract.UnsupportedFormatError as exc:
    print(f"Unsupported file: {exc}")
except stemextract.AudioDecodeError as exc:
    print(f"Unreadable audio: {exc}")
except stemextract.StemExtractError as exc:
    print(f"Separation failed: {exc}")
```

Batch processing continues past individual failures by default. Inspect
`result.failures`, or pass `continue_on_error=False` to stop at the first
failure.

## Model download and cache

No weights are bundled in `stemextract`. On the first separation, the upstream
`demucs` package downloads the official `htdemucs` weights. PyTorch stores
them in its standard hub/checkpoint cache, so later processes reuse them.
Within one Python process, stemextract also reuses loaded separator instances.

## Credits, license, and responsible use

HTDemucs and Demucs were created by Meta AI. The Demucs source code is:

> Copyright (c) Meta Platforms, Inc. and affiliates.

Demucs is distributed under the MIT License. The complete upstream notice is
preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and the upstream
project is available at
[facebookresearch/demucs](https://github.com/facebookresearch/demucs).
stemextract's own source code is also released under the MIT License; see
[LICENSE](LICENSE).

Relevant papers:

- Simon Rouard, Francisco Massa, and Alexandre Défossez, “Hybrid Transformers
  for Music Source Separation,” ICASSP 2023.
- Alexandre Défossez, “Hybrid Spectrogram and Waveform Source Separation,”
  ISMIR Workshop on Music Source Separation, 2021.

Model weights are downloaded from the upstream Demucs distribution and are not
relicensed or redistributed by stemextract. Review upstream model and training-data
terms for your use case. Only process audio that you have the right to use.

## Development checks

```bash
python -m pytest
ruff check .
mypy stemextract
python -m build
python -m twine check dist/*
```
