from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from stemextract import batch
from stemextract.utils import BatchProcessingError


class BatchTests(unittest.TestCase):
    def test_batch_preserves_tree_and_avoids_same_stem_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            output = Path(temporary) / "output"
            album = root / "album"
            album.mkdir(parents=True)
            (album / "song.mp3").touch()
            (album / "song.flac").touch()
            destinations = []

            def fake_extract(input_path: Path, output_dir: Path, **kwargs: Any) -> None:
                del input_path, kwargs
                destinations.append(output_dir)

            with patch.object(batch, "extract", side_effect=fake_extract):
                result = batch.batch_extract(root, output, progress=False)

            self.assertEqual(result.total, 2)
            self.assertEqual(result.succeeded, 2)
            self.assertEqual(
                {path.relative_to(output).as_posix() for path in destinations},
                {"album/song_mp3", "album/song_flac"},
            )

    def test_batch_collects_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            root.mkdir()
            (root / "broken.wav").touch()
            with patch.object(batch, "extract", side_effect=RuntimeError("corrupt")):
                result = batch.batch_extract(root, temporary, progress=False)
            self.assertEqual(result.failed, 1)
            self.assertIn("corrupt", result.failures[0].error or "")

    def test_batch_can_stop_on_first_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            root.mkdir()
            (root / "broken.wav").touch()
            with (
                patch.object(batch, "extract", side_effect=RuntimeError("corrupt")),
                self.assertRaisesRegex(BatchProcessingError, "broken.wav"),
            ):
                batch.batch_extract(
                    root,
                    Path(temporary) / "output",
                    progress=False,
                    continue_on_error=False,
                )

    def test_batch_does_not_reprocess_nested_output_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            output = root / "separated"
            output.mkdir(parents=True)
            (root / "song.wav").touch()
            (output / "old-vocals.wav").touch()
            inputs = []

            def fake_extract(input_path: Path, output_dir: Path, **kwargs: Any) -> None:
                del output_dir, kwargs
                inputs.append(input_path)

            with patch.object(batch, "extract", side_effect=fake_extract):
                result = batch.batch_extract(root, output, progress=False)

            self.assertEqual(result.total, 1)
            self.assertEqual([path.name for path in inputs], ["song.wav"])


if __name__ == "__main__":
    unittest.main()
