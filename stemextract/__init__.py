"""A small, friendly Python interface for HTDemucs stem separation.

stemextract is an engineering wrapper around the pretrained HTDemucs model from
Meta AI. It does not contain model-training code or bundle model weights.
Demucs is Copyright (c) Meta Platforms, Inc. and affiliates and is distributed
under the MIT License. See ``THIRD_PARTY_NOTICES.md`` in the source
distribution for the complete notice.
"""

from __future__ import annotations

from .audio_io import save_tracks
from .batch import BatchItemResult, BatchResult, batch_extract
from .core import (
    extract,
    extract_bass,
    extract_drums,
    extract_other,
    extract_stem,
    extract_vocals,
)
from .utils import (
    AudioDecodeError,
    AudioEncodeError,
    BatchProcessingError,
    DependencyError,
    InvalidInputError,
    ModelLoadError,
    OutputError,
    SeparationError,
    StemExtractError,
    UnsupportedFormatError,
)

__all__ = [
    "AudioDecodeError",
    "AudioEncodeError",
    "BatchItemResult",
    "BatchProcessingError",
    "BatchResult",
    "DependencyError",
    "InvalidInputError",
    "ModelLoadError",
    "OutputError",
    "SeparationError",
    "StemExtractError",
    "UnsupportedFormatError",
    "batch_extract",
    "extract",
    "extract_bass",
    "extract_drums",
    "extract_other",
    "extract_stem",
    "extract_vocals",
    "save_tracks",
]

__version__ = "0.2.0"
