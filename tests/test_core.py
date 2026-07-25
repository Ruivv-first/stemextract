from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from stemextract import core
from stemextract.audio_io import LoadedAudio
from stemextract.utils import InvalidInputError


class FakeWaveform:
    def clone(self) -> "FakeWaveform":
        return self


class FakeSeparator:
    def update_parameter(self, **kwargs: Any) -> None:
        del kwargs

    def separate_tensor(
        self, waveform: FakeWaveform, *, sr: int
    ) -> tuple[FakeWaveform, Dict[str, str]]:
        del sr
        return waveform, {
            "vocals": "vocals-data",
            "drums": "drums-data",
            "bass": "bass-data",
            "other": "other-data",
        }


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        core.clear_model_cache()

    def test_extract_exports_four_stems_and_returns_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "song.wav"
            source.touch()
            output = Path(temporary) / "output"
            saved: Dict[str, Any] = {}

            def fake_save(
                tracks: Dict[str, Any], output_dir: Path, **kwargs: Any
            ) -> Dict[str, Path]:
                saved.update(tracks)
                return {name: Path(output_dir) / f"{name}.wav" for name in tracks}

            with (
                patch.object(core, "detect_device", return_value="cpu"),
                patch.object(
                    core.audio_io,
                    "load_audio",
                    return_value=LoadedAudio(FakeWaveform(), 44_100),
                ),
                patch.object(
                    core,
                    "_get_separator",
                    return_value=(FakeSeparator(), core.threading.RLock()),
                ),
                patch.object(core.audio_io, "save_tracks", side_effect=fake_save),
            ):
                tracks = core.extract(source, output, progress=False)

            self.assertEqual(tuple(tracks), core.STEMS)
            self.assertEqual(saved, tracks)

    def test_extract_vocals_only_exports_vocals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "song.wav"
            source.touch()
            with (
                patch.object(core, "detect_device", return_value="cpu"),
                patch.object(
                    core.audio_io,
                    "load_audio",
                    return_value=LoadedAudio(FakeWaveform(), 44_100),
                ),
                patch.object(
                    core,
                    "_get_separator",
                    return_value=(FakeSeparator(), core.threading.RLock()),
                ),
                patch.object(core.audio_io, "save_tracks") as save_tracks,
            ):
                tracks = core.extract_vocals(
                    source, Path(temporary) / "output", progress=False
                )

            self.assertEqual(tracks, {"vocals": "vocals-data"})
            self.assertEqual(save_tracks.call_args.args[0], {"vocals": "vocals-data"})

    def test_unknown_stem_is_rejected_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "song.wav"
            source.touch()
            with self.assertRaisesRegex(InvalidInputError, "Unknown stem"):
                core.separate(source, stems=("piano",), progress=False)

    def test_accelerator_oom_retries_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "song.wav"
            source.touch()
            devices = []

            def fake_separate(
                waveform: FakeWaveform,
                sample_rate: int,
                *,
                device: str,
                **kwargs: Any,
            ) -> Dict[str, str]:
                del waveform, sample_rate, kwargs
                devices.append(device)
                if device == "cuda":
                    raise RuntimeError("CUDA out of memory")
                return {stem: f"{stem}-data" for stem in core.STEMS}

            with (
                patch.object(core, "detect_device", return_value="cuda"),
                patch.object(
                    core.audio_io,
                    "load_audio",
                    return_value=LoadedAudio(FakeWaveform(), 44_100),
                ),
                patch.object(core, "_separate_on_device", side_effect=fake_separate),
                patch.object(core, "release_device_memory") as release,
                self.assertWarnsRegex(RuntimeWarning, "retrying"),
            ):
                tracks = core.separate(source, progress=False)

            self.assertEqual(devices, ["cuda", "cpu"])
            release.assert_called_once_with("cuda")
            self.assertEqual(set(tracks), set(core.STEMS))


if __name__ == "__main__":
    unittest.main()
