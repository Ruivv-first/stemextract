"""Shared utilities for paths, progress reporting, and device selection."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import re
import shutil
import warnings
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

PathLike = Union[str, Path]
T = TypeVar("T")

SUPPORTED_AUDIO_FORMATS = frozenset({"mp3", "m4a", "flac", "wav"})
SUPPORTED_OUTPUT_FORMATS = SUPPORTED_AUDIO_FORMATS
STEMS = ("vocals", "drums", "bass", "other")


class StemExtractError(RuntimeError):
    """Base class for all expected stemextract errors."""


class InvalidInputError(StemExtractError):
    """Raised when an input path or value is invalid."""


class UnsupportedFormatError(StemExtractError):
    """Raised when an audio format is not supported."""


class DependencyError(StemExtractError):
    """Raised when a required Python package or executable is unavailable."""


class AudioDecodeError(StemExtractError):
    """Raised when an input audio stream cannot be decoded safely."""


class AudioEncodeError(StemExtractError):
    """Raised when an output audio stream cannot be encoded safely."""


class ModelLoadError(StemExtractError):
    """Raised when HTDemucs or its pretrained weights cannot be loaded."""


class SeparationError(StemExtractError):
    """Raised when HTDemucs inference fails."""


class OutputError(StemExtractError):
    """Raised when an output directory or file cannot be written."""


class BatchProcessingError(StemExtractError):
    """Raised when batch processing cannot continue."""


def audio_format(path: PathLike) -> str:
    """Return a normalized extension without a leading dot."""

    try:
        return Path(path).suffix.lower().lstrip(".")
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"Invalid path: {path!r}.") from exc


def validate_audio_file(path: PathLike) -> Path:
    """Validate and return a single supported audio file path."""

    try:
        candidate = Path(path).expanduser()
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"Invalid input path: {path!r}.") from exc

    if not candidate.exists():
        raise InvalidInputError(f"Audio file does not exist: {candidate}")
    if not candidate.is_file():
        raise InvalidInputError(f"Expected an audio file, but got: {candidate}")

    suffix = audio_format(candidate)
    if suffix not in SUPPORTED_AUDIO_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_FORMATS))
        shown = f".{suffix}" if suffix else "no extension"
        raise UnsupportedFormatError(
            f"Unsupported audio format ({shown}) for '{candidate.name}'. "
            f"Supported formats: {supported}."
        )
    return candidate


def validate_input_folder(path: PathLike) -> Path:
    """Validate and return an input directory."""

    try:
        candidate = Path(path).expanduser()
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"Invalid input folder: {path!r}.") from exc
    if not candidate.exists():
        raise InvalidInputError(f"Input folder does not exist: {candidate}")
    if not candidate.is_dir():
        raise InvalidInputError(f"Expected a folder, but got: {candidate}")
    return candidate


def ensure_output_directory(path: PathLike) -> Path:
    """Create an output directory and return it."""

    try:
        directory = Path(path).expanduser()
    except (TypeError, ValueError) as exc:
        raise OutputError(f"Invalid output directory: {path!r}.") from exc

    if directory.exists() and not directory.is_dir():
        raise OutputError(f"Output path exists but is not a directory: {directory}")
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError(
            f"Could not create output directory '{directory}': {exc}"
        ) from exc
    return directory


def list_audio_files(folder: PathLike, *, recursive: bool = True) -> List[Path]:
    """Return supported audio files in deterministic order."""

    root = validate_input_folder(folder)
    iterator = root.rglob("*") if recursive else root.glob("*")
    files = [
        path
        for path in iterator
        if path.is_file() and audio_format(path) in SUPPORTED_AUDIO_FORMATS
    ]
    files.sort(key=lambda path: path.relative_to(root).as_posix().casefold())
    if not files:
        supported = ", ".join(f".{item}" for item in sorted(SUPPORTED_AUDIO_FORMATS))
        raise InvalidInputError(
            f"No supported audio files were found in '{root}' (expected {supported})."
        )
    return files


def normalize_output_format(value: str) -> str:
    """Validate an output format name."""

    if not isinstance(value, str):
        raise UnsupportedFormatError(
            f"Output format must be a string, not {type(value).__name__}."
        )
    normalized = value.strip().lower().lstrip(".")
    if normalized not in SUPPORTED_OUTPUT_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_FORMATS))
        raise UnsupportedFormatError(
            f"Unsupported output format '{value}'. Supported formats: {supported}."
        )
    return normalized


def safe_filename(value: str, *, max_length: int = 180) -> str:
    """Return a cross-platform safe file name component."""

    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value)).strip(" .")
    if not cleaned:
        raise OutputError(f"Cannot create a safe output name from {value!r}.")
    # These names are reserved on Windows, even when an extension is present.
    if cleaned.split(".", 1)[0].upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }:
        cleaned = f"_{cleaned}"
    if max_length < 16:
        raise ValueError("max_length must be at least 16.")
    if len(cleaned) > max_length:
        digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned[: max_length - 9]}-{digest}"
    return cleaned


def require_python_module(name: str, install_name: Optional[str] = None) -> Any:
    """Import a dependency and raise a concise installation error if missing."""

    try:
        return importlib.import_module(name)
    except ImportError as exc:
        package = install_name or name
        raise DependencyError(
            f"Required dependency '{package}' is not installed. "
            "Install stemextract with `python -m pip install stemextract`."
        ) from exc


def require_ffmpeg() -> str:
    """Return the FFmpeg executable or raise an actionable error."""

    executable = shutil.which("ffmpeg")
    if executable is None:
        raise DependencyError(
            "FFmpeg is required to decode MP3/M4A files and encode MP3/M4A "
            "outputs. Install FFmpeg and make sure `ffmpeg` is on PATH."
        )
    return executable


def detect_device(preferred: Optional[str] = None) -> str:
    """Select CUDA, Apple MPS, or CPU in that order.

    Args:
        preferred: Optional explicit device (``"cuda"``, ``"mps"``,
            ``"cpu"``, or ``"auto"``). Explicit unavailable accelerators
            fall back to CPU so one-line extraction remains reliable.
    """

    torch = require_python_module("torch")
    if preferred is not None and not isinstance(preferred, str):
        raise InvalidInputError(
            f"device must be a string or None, not {type(preferred).__name__}."
        )
    requested = (preferred or "auto").strip().lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise InvalidInputError(
            f"Unknown device '{preferred}'. Choose auto, cuda, mps, or cpu."
        )

    cuda_available = bool(
        hasattr(torch, "cuda")
        and hasattr(torch.cuda, "is_available")
        and torch.cuda.is_available()
    )
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(
        mps_backend is not None
        and hasattr(mps_backend, "is_available")
        and mps_backend.is_available()
    )

    if requested == "cuda":
        if cuda_available:
            return "cuda"
        warnings.warn(
            "CUDA was requested but is unavailable; using CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    if requested == "mps":
        if mps_available:
            return "mps"
        warnings.warn(
            "Apple MPS was requested but is unavailable; using CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def is_out_of_memory_error(exc: BaseException) -> bool:
    """Return whether an exception is recognizably an accelerator OOM."""

    if isinstance(exc, MemoryError):
        return True
    message = str(exc).casefold()
    markers: Sequence[str] = (
        "out of memory",
        "cuda error: out of memory",
        "mps backend out of memory",
        "not enough memory",
    )
    return any(marker in message for marker in markers)


def release_device_memory(device: str) -> None:
    """Ask PyTorch to release cached accelerator memory, if possible."""

    with contextlib.suppress(ImportError, RuntimeError, AttributeError):
        torch = importlib.import_module("torch")
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps":
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()


def progress_iter(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    description: str,
    enabled: bool = True,
    unit: str = "item",
) -> Iterable[T]:
    """Wrap an iterable in tqdm when progress display is enabled."""

    if not enabled:
        return iterable
    try:
        tqdm_module = importlib.import_module("tqdm.auto")
    except ImportError:
        return iterable
    return cast(
        Iterable[T],
        tqdm_module.tqdm(
            iterable,
            total=total,
            desc=description,
            unit=unit,
            dynamic_ncols=True,
        ),
    )


def iter_progress(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    description: str,
    enabled: bool = True,
    unit: str = "item",
) -> Iterator[T]:
    """Yield progress-wrapped items with a stable iterator return type."""

    yield from progress_iter(
        iterable,
        total=total,
        description=description,
        enabled=enabled,
        unit=unit,
    )
