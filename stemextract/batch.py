"""Folder-level batch stem separation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .core import extract
from .utils import (
    BatchProcessingError,
    InvalidInputError,
    PathLike,
    audio_format,
    ensure_output_directory,
    iter_progress,
    list_audio_files,
    safe_filename,
    validate_input_folder,
)


@dataclass(frozen=True)
class BatchItemResult:
    """Outcome for one input file in a batch."""

    input_path: Path
    output_dir: Path
    success: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class BatchResult:
    """Summary returned by :func:`batch_extract`."""

    items: Tuple[BatchItemResult, ...]

    @property
    def total(self) -> int:
        """Number of discovered input files."""

        return len(self.items)

    @property
    def succeeded(self) -> int:
        """Number of successfully separated files."""

        return sum(item.success for item in self.items)

    @property
    def failed(self) -> int:
        """Number of failed files."""

        return self.total - self.succeeded

    @property
    def failures(self) -> Tuple[BatchItemResult, ...]:
        """Failed item details."""

        return tuple(item for item in self.items if not item.success)


def _destinations(input_root: Path, files: List[Path], output_root: Path) -> List[Path]:
    relative_bases = [
        (path.relative_to(input_root).parent, safe_filename(path.stem))
        for path in files
    ]
    collision_keys = [
        (parent.as_posix().casefold(), base.casefold())
        for parent, base in relative_bases
    ]
    counts = Counter(collision_keys)
    destinations: List[Path] = []
    for index, path in enumerate(files):
        relative_parent, base = relative_bases[index]
        collision_key = collision_keys[index]
        # Avoid collisions such as song.mp3 and song.flac in the same folder.
        directory_name = (
            f"{base}_{audio_format(path)}" if counts[collision_key] > 1 else base
        )
        destinations.append(output_root / relative_parent / directory_name)
    return destinations


def batch_extract(
    input_folder: PathLike,
    output_folder: PathLike,
    *,
    format: str = "wav",
    device: Optional[str] = None,
    recursive: bool = True,
    progress: bool = True,
    overwrite: bool = True,
    continue_on_error: bool = True,
    shifts: int = 1,
    overlap: float = 0.25,
    segment: Optional[int] = None,
) -> BatchResult:
    """Separate every supported song in a folder.

    Each song receives its own output directory. Relative subfolder structure
    is preserved when ``recursive=True``. Corrupted files are reported in the
    returned result and do not stop the batch unless ``continue_on_error`` is
    false.
    """

    input_root = validate_input_folder(input_folder)
    output_root = ensure_output_directory(output_folder)
    resolved_input = input_root.resolve()
    resolved_output = output_root.resolve()
    if resolved_input == resolved_output:
        raise BatchProcessingError(
            "The batch output folder must be different from the input folder."
        )

    files = list_audio_files(input_root, recursive=recursive)
    try:
        resolved_output.relative_to(resolved_input)
    except ValueError:
        pass
    else:
        files = [
            path for path in files if not path.resolve().is_relative_to(resolved_output)
        ]
        if not files:
            raise InvalidInputError(
                f"No input audio files were found outside the output folder "
                f"'{output_root}'."
            )
    destinations = _destinations(input_root, files, output_root)
    results: List[BatchItemResult] = []

    pairs = [(path, destinations[index]) for index, path in enumerate(files)]
    for input_path, destination in iter_progress(
        pairs,
        total=len(pairs),
        description="Separating files",
        enabled=progress,
        unit="file",
    ):
        try:
            extract(
                input_path,
                destination,
                format=format,
                device=device,
                progress=False,
                overwrite=overwrite,
                shifts=shifts,
                overlap=overlap,
                segment=segment,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            item = BatchItemResult(
                input_path=input_path,
                output_dir=destination,
                success=False,
                error=str(exc),
            )
            results.append(item)
            if not continue_on_error:
                raise BatchProcessingError(
                    f"Batch processing stopped at '{input_path.name}': {exc}"
                ) from exc
        else:
            results.append(
                BatchItemResult(
                    input_path=input_path,
                    output_dir=destination,
                    success=True,
                )
            )

    return BatchResult(items=tuple(results))
