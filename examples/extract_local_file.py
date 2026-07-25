"""Separate a local WAV file into four stems with stemextract."""

from pathlib import Path

import stemextract

INPUT_FILE = Path("/Users/mac/Desktop/1test.wav")
OUTPUT_DIR = Path("/Users/mac/Desktop/1test_stems")


def main() -> None:
    """Separate the configured audio file and print the results."""
    if not INPUT_FILE.is_file():
        raise SystemExit(f"Input audio file does not exist: {INPUT_FILE}")

    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")

    try:
        tracks = stemextract.extract(
            INPUT_FILE,
            OUTPUT_DIR,
            device="auto",
            progress=True,
        )
    except stemextract.StemExtractError as exc:
        raise SystemExit(f"Stem separation failed: {exc}") from exc

    print("\nSeparation completed successfully:")
    for stem_name in tracks:
        print(f"  {stem_name}: {OUTPUT_DIR / f'{stem_name}.wav'}")


if __name__ == "__main__":
    main()
