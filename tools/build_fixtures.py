#!/usr/bin/env python3
"""Build redistributable deterministic FFmpeg fixtures (no private footage)."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original = args.output / "original.mp4"
    derivative = args.output / "derivative.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=24",
            "-t",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(original),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(original),
            "-vf",
            "scale=480:270,eq=brightness=0.02:saturation=0.9",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-crf",
            "27",
            "-an",
            str(derivative),
        ],
        check=True,
    )
    print(original)
    print(derivative)


if __name__ == "__main__":
    main()
