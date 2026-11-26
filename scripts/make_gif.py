"""Convert the newest Playwright .webm recording into docs/screenshots/demo.gif with ffmpeg.

Two-pass palette method: pass 1 builds an optimal 128-colour palette, pass 2 uses it,
which keeps the GIF sharp and small. Target: <= 20 s and <= 5 MB.
Run:  python scripts/make_gif.py [path/to/ffmpeg]
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "reports" / "videos"
OUT = ROOT / "docs" / "screenshots" / "demo.gif"


def main() -> None:
    ffmpeg = sys.argv[1] if len(sys.argv) > 1 else shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg not found: pass its path as the first argument")
    video = max(VIDEOS.glob("*.webm"), key=lambda p: p.stat().st_mtime)
    filters = "fps=8,scale=960:-1:flags=lanczos"
    palette = VIDEOS / "palette.png"
    subprocess.run(
        [ffmpeg, "-y", "-t", "20", "-i", str(video), "-vf", f"{filters},palettegen=max_colors=128", str(palette)],
        check=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-t",
            "20",
            "-i",
            str(video),
            "-i",
            str(palette),
            "-lavfi",
            f"{filters}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
            str(OUT),
        ],
        check=True,
    )
    print(f"{OUT}  {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
