"""Stitch a sequence of dashboard screenshots into `results/demo.gif`.

    python scripts/stitch_dashboard_gif.py --frames <dir>

This is the second half of producing the dashboard clip. The first half is
capturing the frames, which needs a browser driving the running app:

    1. streamlit run app/streamlit_app.py
    2. point a Chromium-based browser at it with --remote-debugging-port=9222
    3. click Play and screenshot the viewport every second or so

I captured the committed `results/demo.gif` that way. The capture step depends on
a browser and a debugging port, so it is not something the repo can run
unattended — which is exactly why `results/demo_animation.gif` exists alongside
it. That one regenerates from committed code with a single command and no
browser, via `scripts/export_demo_gif.py`.

Both clips show the same real model output. Neither is staged or mocked up.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "results" / "demo.gif"

# The Streamlit sidebar is fixed and identical in every frame, so cropping it
# away keeps the GIF small and puts the chart front and centre in the README.
CROP_LEFT = 300
TARGET_WIDTH = 980
FRAME_MS = 900


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, help="directory of cap_*.png")
    parser.add_argument("--pattern", default="cap_*.png")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    paths = sorted(Path(args.frames).glob(args.pattern))
    if not paths:
        print(f"No frames matching {args.pattern} in {args.frames}", file=sys.stderr)
        return 1

    frames = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image = image.crop((CROP_LEFT, 0, image.width, image.height))
        ratio = TARGET_WIDTH / image.width
        image = image.resize(
            (TARGET_WIDTH, int(image.height * ratio)), Image.LANCZOS
        )
        frames.append(image)

    # A shared palette keeps colours stable between frames — letting each frame
    # pick its own makes the background shimmer. It has to be derived from
    # *every* frame though: built from the first one alone it misses the colours
    # that only appear later (the flagged points, the rebuild markers) and the
    # whole plot area comes out washed grey.
    montage = Image.new("RGB", (frames[0].width, frames[0].height * len(frames)))
    for index, frame in enumerate(frames):
        montage.paste(frame, (0, index * frames[0].height))
    palette_source = montage.quantize(colors=255, method=Image.MEDIANCUT)

    quantised = [f.quantize(palette=palette_source, dither=Image.NONE) for f in frames]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    quantised[0].save(
        out,
        save_all=True,
        append_images=quantised[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"Wrote {out.relative_to(PROJECT_ROOT)} "
          f"({size_mb:.2f} MB, {len(quantised)} frames at {FRAME_MS} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
