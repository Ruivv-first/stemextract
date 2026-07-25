"""Safe multi-format audio decoding, encoding, and track export."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

from .utils import (
    AudioDecodeError,
    AudioEncodeError,
    InvalidInputError,
    OutputError,
    PathLike,
    audio_format,
    ensure_output_directory,
    iter_progress,
    normalize_output_format,
    require_ffmpeg,
    require_python_module,
    safe_filename,
    validate_audio_file,
)

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class LoadedAudio:
    """Decoded channel-first floating-point audio."""

    waveform: "torch.Tensor"
    sample_rate: int


def _read_with_soundfile(path: Path) -> LoadedAudio:
    soundfile = require_python_module("soundfile")
    torch = require_python_module("torch")
    try:
        samples, sample_rate = soundfile.read(
            str(path), dtype="float32", always_2d=True
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise AudioDecodeError(
            f"Could not decode '{path.name}' with libsndfile: {exc}"
        ) from exc

    if samples.shape[0] == 0:
        raise AudioDecodeError(f"Audio file contains no samples: {path}")
    waveform = torch.from_numpy(samples.T.copy())
    if not bool(torch.isfinite(waveform).all()):
        raise AudioDecodeError(f"Audio file contains invalid sample values: {path}")
    return LoadedAudio(waveform=waveform, sample_rate=int(sample_rate))


def _decode_with_ffmpeg(path: Path) -> LoadedAudio:
    ffmpeg = require_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="stemextract-decode-") as temporary:
        decoded = Path(temporary) / "decoded.wav"
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "pcm_f32le",
            str(decoded),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise AudioDecodeError(
                f"FFmpeg could not be started while decoding '{path.name}': {exc}"
            ) from exc
        if result.returncode != 0 or not decoded.is_file():
            detail = result.stderr.strip() or "unknown FFmpeg decoding error"
            raise AudioDecodeError(
                f"Could not decode '{path.name}'. The file may be corrupted or "
                f"may not contain an audio stream. FFmpeg: {detail}"
            )
        return _read_with_soundfile(decoded)


def load_audio(input_path: PathLike) -> LoadedAudio:
    """Decode WAV, FLAC, MP3, or M4A audio into a channel-first tensor.

    WAV and FLAC use libsndfile first. Compressed formats, and files rejected
    by libsndfile, are decoded through FFmpeg into a temporary lossless stream.
    """

    path = validate_audio_file(input_path)
    if audio_format(path) in {"mp3", "m4a"}:
        return _decode_with_ffmpeg(path)
    try:
        return _read_with_soundfile(path)
    except AudioDecodeError:
        return _decode_with_ffmpeg(path)


def _frames_channels(track: Any) -> Any:
    numpy = require_python_module("numpy")
    torch = require_python_module("torch")

    if isinstance(track, torch.Tensor):
        data = track.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        try:
            data = numpy.asarray(track, dtype=numpy.float32)
        except (TypeError, ValueError) as exc:
            raise AudioEncodeError(
                f"Track data cannot be converted to floating-point audio: {exc}"
            ) from exc

    if data.ndim == 1:
        data = data[:, None]
    elif data.ndim == 2:
        # Demucs tensors are channel-first. Also accept conventional
        # frame-first arrays when their final dimension clearly holds channels.
        if data.shape[0] <= 8:
            data = data.T
        elif data.shape[1] > 8:
            raise AudioEncodeError(
                "Two-dimensional audio must be shaped as (channels, frames) "
                "or (frames, channels), with at most 8 channels."
            )
    else:
        raise AudioEncodeError(
            f"Track data must be 1D or 2D, but got shape {data.shape!r}."
        )

    if data.shape[0] == 0:
        raise AudioEncodeError("Track data contains no audio frames.")
    if data.shape[1] < 1 or data.shape[1] > 8:
        raise AudioEncodeError(
            f"Track data has an unsupported channel count: {data.shape[1]}."
        )
    if not bool(numpy.isfinite(data).all()):
        raise AudioEncodeError("Track data contains NaN or infinite sample values.")
    return numpy.clip(data, -1.0, 1.0)


def _check_output_target(path: Path, *, overwrite: bool) -> None:
    if path.exists() and path.is_dir():
        raise OutputError(f"Output file path is a directory: {path}")
    if path.exists() and not overwrite:
        raise OutputError(
            f"Output file already exists: {path}. Set overwrite=True to replace it."
        )


def _write_lossless(
    track: Any,
    output_path: Path,
    *,
    sample_rate: int,
    overwrite: bool,
) -> Path:
    soundfile = require_python_module("soundfile")
    _check_output_target(output_path, overwrite=overwrite)
    data = _frames_channels(track)
    subtype = "PCM_24"
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}-",
            suffix=output_path.suffix,
            dir=str(output_path.parent),
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        soundfile.write(
            str(temporary),
            data,
            sample_rate,
            format=output_path.suffix.lstrip(".").upper(),
            subtype=subtype,
        )
        os.replace(str(temporary), str(output_path))
    except (OSError, RuntimeError, ValueError) as exc:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
        raise AudioEncodeError(f"Could not write '{output_path.name}': {exc}") from exc
    return output_path


def _write_compressed(
    track: Any,
    output_path: Path,
    *,
    sample_rate: int,
    overwrite: bool,
) -> Path:
    ffmpeg = require_ffmpeg()
    _check_output_target(output_path, overwrite=overwrite)
    output_format = output_path.suffix.lower().lstrip(".")
    codec_args = (
        ["-c:a", "libmp3lame", "-b:a", "320k"]
        if output_format == "mp3"
        else ["-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart"]
    )

    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{output_path.stem}-",
            dir=str(output_path.parent),
        ) as temporary:
            temp_dir = Path(temporary)
            intermediate = temp_dir / "source.wav"
            encoded = temp_dir / f"encoded.{output_format}"
            _write_lossless(
                track,
                intermediate,
                sample_rate=sample_rate,
                overwrite=True,
            )
            command = [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(intermediate),
                *codec_args,
                str(encoded),
            ]
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or not encoded.is_file():
                detail = result.stderr.strip() or "unknown FFmpeg encoding error"
                raise AudioEncodeError(
                    f"Could not encode '{output_path.name}'. FFmpeg: {detail}"
                )
            os.replace(str(encoded), str(output_path))
    except AudioEncodeError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise AudioEncodeError(f"Could not write '{output_path.name}': {exc}") from exc
    return output_path


def save_audio(
    track: Any,
    output_path: PathLike,
    *,
    sample_rate: int = 44_100,
    overwrite: bool = True,
) -> Path:
    """Atomically write one track based on its file extension."""

    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise InvalidInputError(
            f"sample_rate must be a positive integer, got {sample_rate!r}."
        )
    try:
        path = Path(output_path).expanduser()
    except (TypeError, ValueError) as exc:
        raise OutputError(f"Invalid output file: {output_path!r}.") from exc
    ensure_output_directory(path.parent)
    output_format = normalize_output_format(path.suffix)
    if output_format in {"wav", "flac"}:
        return _write_lossless(
            track, path, sample_rate=sample_rate, overwrite=overwrite
        )
    return _write_compressed(track, path, sample_rate=sample_rate, overwrite=overwrite)


def save_tracks(
    tracks: Mapping[str, Any],
    output_dir: PathLike,
    format: str = "wav",
    *,
    sample_rate: int = 44_100,
    overwrite: bool = True,
    progress: bool = True,
) -> Dict[str, Path]:
    """Save named tracks to one directory.

    Args:
        tracks: Mapping from stem name to a channel-first tensor or numeric
            array. Demucs outputs can be passed directly.
        output_dir: Destination directory.
        format: ``"wav"``, ``"flac"``, ``"mp3"``, or ``"m4a"``.
        sample_rate: Audio sample rate. HTDemucs uses 44.1 kHz.
        overwrite: Whether existing files may be atomically replaced.
        progress: Display a track-writing progress bar.

    Returns:
        A mapping from each stem name to its written file path.
    """

    if not isinstance(tracks, Mapping) or not tracks:
        raise InvalidInputError("tracks must be a non-empty mapping of name to audio.")
    output_format = normalize_output_format(format)
    directory = ensure_output_directory(output_dir)
    prepared = []
    used_names: set[str] = set()
    for name, track in tracks.items():
        safe_name = safe_filename(str(name))
        folded = safe_name.casefold()
        if folded in used_names:
            raise OutputError(
                f"Multiple track names resolve to the same safe filename: {name!r}."
            )
        used_names.add(folded)
        prepared.append((str(name), track, directory / f"{safe_name}.{output_format}"))
    if not overwrite:
        for _, _, destination in prepared:
            _check_output_target(destination, overwrite=False)

    written: Dict[str, Path] = {}
    for name, track, destination in iter_progress(
        prepared,
        total=len(prepared),
        description="Saving stems",
        enabled=progress,
        unit="stem",
    ):
        written[name] = save_audio(
            track,
            destination,
            sample_rate=sample_rate,
            overwrite=overwrite,
        )
    return written


def convert_audio(
    input_path: PathLike,
    output_path: PathLike,
    *,
    overwrite: bool = True,
) -> Path:
    """Decode and safely convert one supported audio file."""

    loaded = load_audio(input_path)
    return save_audio(
        loaded.waveform,
        output_path,
        sample_rate=loaded.sample_rate,
        overwrite=overwrite,
    )
