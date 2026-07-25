from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stemextract.utils import (
    InvalidInputError,
    UnsupportedFormatError,
    list_audio_files,
    normalize_output_format,
    safe_filename,
    validate_audio_file,
)


class PathUtilityTests(unittest.TestCase):
    def test_validate_audio_file_rejects_missing_file(self) -> None:
        with self.assertRaisesRegex(InvalidInputError, "does not exist"):
            validate_audio_file("/definitely/not/a/song.wav")

    def test_validate_audio_file_rejects_unsupported_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "song.ogg"
            path.touch()
            with self.assertRaisesRegex(UnsupportedFormatError, "Unsupported"):
                validate_audio_file(path)

    def test_list_audio_files_is_recursive_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "album").mkdir()
            (root / "b.wav").touch()
            (root / "album" / "A.MP3").touch()
            (root / "notes.txt").touch()
            result = list_audio_files(root)
            self.assertEqual(
                [path.relative_to(root).as_posix() for path in result],
                ["album/A.MP3", "b.wav"],
            )

    def test_normalize_output_format_accepts_dot(self) -> None:
        self.assertEqual(normalize_output_format(".FLAC"), "flac")

    def test_safe_filename_removes_path_metacharacters(self) -> None:
        self.assertEqual(safe_filename("../lead:vocals"), "_lead_vocals")


if __name__ == "__main__":
    unittest.main()
