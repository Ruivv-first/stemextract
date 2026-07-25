"""HTDemucs loading and inference.

Model credit and licensing
--------------------------
HTDemucs and the Demucs inference implementation were developed by Meta AI.
Demucs is Copyright (c) Meta Platforms, Inc. and affiliates and is available
under the MIT License. stemextract does not retrain, modify, or bundle the model;
it loads the official ``htdemucs`` pretrained weights through the Demucs
package. See README.md and THIRD_PARTY_NOTICES.md for full attribution.
"""

from __future__ import annotations

import threading
import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Tuple

from . import audio_io
from .utils import (
    STEMS,
    InvalidInputError,
    ModelLoadError,
    PathLike,
    SeparationError,
    detect_device,
    is_out_of_memory_error,
    release_device_memory,
    require_python_module,
    validate_audio_file,
)

if TYPE_CHECKING:
    import torch

MODEL_NAME = "htdemucs"
MODEL_SAMPLE_RATE = 44_100

_separator_cache: Optional[Tuple[Any, threading.RLock]] = None
_cache_lock = threading.RLock()


def _create_separator() -> Any:
    try:
        demucs_api = require_python_module("demucs.api", "demucs")
        return demucs_api.Separator(
            model=MODEL_NAME,
            device="cpu",
            shifts=1,
            overlap=0.25,
            split=True,
            segment=None,
            jobs=0,
            progress=False,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise ModelLoadError(
            "Could not load the pretrained HTDemucs model. On first use, "
            "Demucs downloads and caches the official model weights "
            "automatically. Check your internet connection, available disk "
            f"space, and PyTorch installation. Original error: {exc}"
        ) from exc


def _get_separator() -> Tuple[Any, threading.RLock]:
    global _separator_cache
    with _cache_lock:
        if _separator_cache is None:
            _separator_cache = (_create_separator(), threading.RLock())
        return _separator_cache


def clear_model_cache() -> None:
    """Clear stemextract's in-process model cache.

    This does not remove the pretrained weights from PyTorch's on-disk cache.
    """

    global _separator_cache
    with _cache_lock:
        _separator_cache = None


def _validate_inference_options(
    *, shifts: int, overlap: float, segment: Optional[int]
) -> None:
    if not isinstance(shifts, int) or shifts < 0:
        raise InvalidInputError(f"shifts must be a non-negative integer: {shifts!r}")
    if not isinstance(overlap, (int, float)) or not 0.0 <= float(overlap) < 1.0:
        raise InvalidInputError(
            f"overlap must be at least 0 and less than 1: {overlap!r}"
        )
    if segment is not None and (not isinstance(segment, int) or segment <= 0):
        raise InvalidInputError(
            f"segment must be a positive integer number of seconds or None: {segment!r}"
        )


def _separate_on_device(
    waveform: "torch.Tensor",
    sample_rate: int,
    *,
    device: str,
    progress: bool,
    shifts: int,
    overlap: float,
    segment: Optional[int],
) -> Dict[str, "torch.Tensor"]:
    separator, separator_lock = _get_separator()
    # Demucs normalizes its input tensor in place. Clone to keep retries safe.
    working = waveform.clone()
    with separator_lock:
        separator.update_parameter(
            device=device,
            shifts=shifts,
            overlap=overlap,
            split=True,
            segment=segment,
            jobs=0,
            progress=progress,
        )
        _, separated = separator.separate_tensor(working, sr=sample_rate)

    missing = [stem for stem in STEMS if stem not in separated]
    if missing:
        raise SeparationError(
            "HTDemucs returned an incomplete separation. Missing stems: "
            + ", ".join(missing)
        )
    return {stem: separated[stem] for stem in STEMS}


def separate(
    input_path: PathLike,
    *,
    stems: Optional[Iterable[str]] = None,
    device: Optional[str] = None,
    progress: bool = True,
    shifts: int = 1,
    overlap: float = 0.25,
    segment: Optional[int] = None,
) -> Dict[str, "torch.Tensor"]:
    """Separate an audio file with the official pretrained HTDemucs model.

    The full four-source model inference is performed even when only selected
    stems are returned; selective export avoids unnecessary output files, not
    model computation.
    """

    _validate_inference_options(shifts=shifts, overlap=overlap, segment=segment)
    path = validate_audio_file(input_path)
    if isinstance(stems, str):
        requested = (stems,)
    else:
        requested = tuple(STEMS if stems is None else stems)
    if not requested:
        raise InvalidInputError("At least one stem must be requested.")
    invalid_types = [stem for stem in requested if not isinstance(stem, str)]
    if invalid_types:
        raise InvalidInputError("Every requested stem name must be a string.")
    unknown = [stem for stem in requested if stem not in STEMS]
    if unknown:
        raise InvalidInputError(
            f"Unknown stem(s): {', '.join(unknown)}. Choose from: {', '.join(STEMS)}."
        )
    if len(set(requested)) != len(requested):
        raise InvalidInputError("The requested stem list contains duplicates.")

    selected_device = detect_device(device)
    loaded = audio_io.load_audio(path)
    try:
        tracks = _separate_on_device(
            loaded.waveform,
            loaded.sample_rate,
            device=selected_device,
            progress=progress,
            shifts=shifts,
            overlap=overlap,
            segment=segment,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        if selected_device != "cpu" and is_out_of_memory_error(exc):
            warnings.warn(
                f"{selected_device.upper()} ran out of memory; retrying HTDemucs "
                "separation on CPU. This will be slower.",
                RuntimeWarning,
                stacklevel=2,
            )
            release_device_memory(selected_device)
            try:
                tracks = _separate_on_device(
                    loaded.waveform,
                    loaded.sample_rate,
                    device="cpu",
                    progress=progress,
                    shifts=shifts,
                    overlap=overlap,
                    segment=segment,
                )
            except KeyboardInterrupt:
                raise
            except Exception as fallback_exc:
                raise SeparationError(
                    f"HTDemucs failed on {selected_device.upper()} due to "
                    "insufficient memory, and the CPU retry also failed. "
                    "Try setting a smaller segment value. "
                    f"CPU error: {fallback_exc}"
                ) from fallback_exc
        elif isinstance(exc, (ModelLoadError, SeparationError)):
            raise
        else:
            raise SeparationError(
                f"HTDemucs could not separate '{path.name}': {exc}"
            ) from exc
    return {stem: tracks[stem] for stem in requested}


def _extract(
    input_path: PathLike,
    output_dir: PathLike,
    *,
    stems: Iterable[str],
    format: str,
    device: Optional[str],
    progress: bool,
    overwrite: bool,
    shifts: int,
    overlap: float,
    segment: Optional[int],
) -> Dict[str, "torch.Tensor"]:
    tracks = separate(
        input_path,
        stems=stems,
        device=device,
        progress=progress,
        shifts=shifts,
        overlap=overlap,
        segment=segment,
    )
    audio_io.save_tracks(
        tracks,
        output_dir,
        format=format,
        sample_rate=MODEL_SAMPLE_RATE,
        overwrite=overwrite,
        progress=progress,
    )
    return tracks


def extract(
    input_path: PathLike,
    output_dir: PathLike,
    *,
    format: str = "wav",
    device: Optional[str] = None,
    progress: bool = True,
    overwrite: bool = True,
    shifts: int = 1,
    overlap: float = 0.25,
    segment: Optional[int] = None,
) -> Dict[str, "torch.Tensor"]:
    """Separate and save vocals, drums, bass, and other."""

    return _extract(
        input_path,
        output_dir,
        stems=STEMS,
        format=format,
        device=device,
        progress=progress,
        overwrite=overwrite,
        shifts=shifts,
        overlap=overlap,
        segment=segment,
    )


def extract_stem(
    input_path: PathLike,
    output_dir: PathLike,
    stem: str,
    *,
    format: str = "wav",
    device: Optional[str] = None,
    progress: bool = True,
    overwrite: bool = True,
    shifts: int = 1,
    overlap: float = 0.25,
    segment: Optional[int] = None,
) -> Dict[str, "torch.Tensor"]:
    """Separate a song and export exactly one selected stem."""

    return _extract(
        input_path,
        output_dir,
        stems=(stem,),
        format=format,
        device=device,
        progress=progress,
        overwrite=overwrite,
        shifts=shifts,
        overlap=overlap,
        segment=segment,
    )


def extract_vocals(
    input_path: PathLike,
    output_dir: PathLike,
    *,
    format: str = "wav",
    device: Optional[str] = None,
    progress: bool = True,
    overwrite: bool = True,
    shifts: int = 1,
    overlap: float = 0.25,
    segment: Optional[int] = None,
) -> Dict[str, "torch.Tensor"]:
    """Separate a song and export only its vocals."""

    return extract_stem(
        input_path,
        output_dir,
        "vocals",
        format=format,
        device=device,
        progress=progress,
        overwrite=overwrite,
        shifts=shifts,
        overlap=overlap,
        segment=segment,
    )


def extract_drums(
    input_path: PathLike,
    output_dir: PathLike,
    *,
    format: str = "wav",
    device: Optional[str] = None,
    progress: bool = True,
    overwrite: bool = True,
    shifts: int = 1,
    overlap: float = 0.25,
    segment: Optional[int] = None,
) -> Dict[str, "torch.Tensor"]:
    """Separate a song and export only its drums."""

    return extract_stem(
        input_path,
        output_dir,
        "drums",
        format=format,
        device=device,
        progress=progress,
        overwrite=overwrite,
        shifts=shifts,
        overlap=overlap,
        segment=segment,
    )


def extract_bass(
    input_path: PathLike,
    output_dir: PathLike,
    *,
    format: str = "wav",
    device: Optional[str] = None,
    progress: bool = True,
    overwrite: bool = True,
    shifts: int = 1,
    overlap: float = 0.25,
    segment: Optional[int] = None,
) -> Dict[str, "torch.Tensor"]:
    """Separate a song and export only its bass."""

    return extract_stem(
        input_path,
        output_dir,
        "bass",
        format=format,
        device=device,
        progress=progress,
        overwrite=overwrite,
        shifts=shifts,
        overlap=overlap,
        segment=segment,
    )


def extract_other(
    input_path: PathLike,
    output_dir: PathLike,
    *,
    format: str = "wav",
    device: Optional[str] = None,
    progress: bool = True,
    overwrite: bool = True,
    shifts: int = 1,
    overlap: float = 0.25,
    segment: Optional[int] = None,
) -> Dict[str, "torch.Tensor"]:
    """Separate a song and export only its other/accompaniment stem."""

    return extract_stem(
        input_path,
        output_dir,
        "other",
        format=format,
        device=device,
        progress=progress,
        overwrite=overwrite,
        shifts=shifts,
        overlap=overlap,
        segment=segment,
    )
