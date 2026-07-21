"""Render an animated GIF of the replay, straight from real experiment output.

    python scripts/export_demo_gif.py                  # sudden, all strategies
    python scripts/export_demo_gif.py --stream recurring

Writes `results/demo_animation.gif`. This is the reproducible clip: it reads the
same `data/demo/demo_runs.npz` the dashboard reads and needs no browser, so it
regenerates identically anywhere. The companion `results/demo.gif` is a screen
capture of the live dashboard, produced by `scripts/capture_dashboard_gif.py`.

Every point drawn is real model output. Nothing here is staged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = PROJECT_ROOT / "data" / "demo" / "demo_runs.npz"
OUT_PATH = PROJECT_ROOT / "results" / "demo_animation.gif"

STRATEGIES = ["static", "online-no-reset", "periodic", "drift-triggered"]
COLOURS = {
    "static": "#c1440e",
    "online-no-reset": "#8a8a8a",
    "periodic": "#1f6f8b",
    "drift-triggered": "#2a7f4f",
}
LABELS = {
    "static": "static\n(never updated)",
    "online-no-reset": "online\n(never rebuilt)",
    "periodic": "periodic\n(fixed schedule)",
    "drift-triggered": "drift-triggered",
}

N_FRAMES = 90
FPS = 12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default="sudden")
    parser.add_argument("--sensor", default="Accelerometer1RMS")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    if not DEMO_PATH.exists():
        print(
            f"No demo data at {DEMO_PATH.relative_to(PROJECT_ROOT)}. Run "
            "`python scripts/run_experiment.py` then "
            "`python scripts/build_demo_cache.py`.",
            file=sys.stderr,
        )
        return 1

    with np.load(DEMO_PATH, allow_pickle=False) as data:
        demo = {key: data[key] for key in data.files}

    stream = args.stream
    sensors = list(demo["__sensors__"])
    signal = demo[f"{stream}|signal"][:, sensors.index(args.sensor)]
    change_points = demo[f"{stream}|change_points"]
    note = str(demo[f"{stream}|note"][0])
    n = len(signal)

    runs = {s: {
        "flags": demo[f"{stream}|{s}|flags"],
        "adaptations": demo[f"{stream}|{s}|adaptations"],
        "eval_start": int(demo[f"{stream}|{s}|eval_start"][0]),
    } for s in STRATEGIES}
    eval_start = runs[STRATEGIES[0]]["eval_start"]

    fig, axes = plt.subplots(
        len(STRATEGIES), 1, figsize=(10, 7), sharex=True, dpi=100
    )
    fig.suptitle(
        f"Replaying '{stream}' — {note}\n"
        "dashed red: injected change point   |   vertical bars: model rebuild",
        fontsize=9,
    )

    # Static background, drawn once: the whole trace in pale grey plus the
    # change points. Only the flagged points and the playhead get redrawn.
    stride = max(1, n // 2500)
    x = np.arange(n)
    artists = {}

    for ax, name in zip(axes, STRATEGIES):
        ax.plot(x[::stride], signal[::stride], lw=0.5, color="#d8d8d8", zorder=1)
        ax.axvspan(0, eval_start, color="#f4f4f4", zorder=0)
        for point in change_points:
            ax.axvline(point, color="#d62728", ls="--", lw=1.1, alpha=0.85, zorder=2)
        ax.set_ylabel(LABELS[name], fontsize=8)
        ax.set_xlim(0, n)
        ax.set_ylim(signal.min() - 0.002, signal.max() + 0.002)
        ax.tick_params(labelsize=7)

        artists[name] = {
            "scatter": ax.scatter([], [], s=4, color=COLOURS[name], zorder=4),
            "playhead": ax.axvline(eval_start, color="#111111", lw=1.2, zorder=5),
            "text": ax.text(
                0.995, 0.90, "", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=COLOURS[name], fontweight="bold",
            ),
            "rebuilds": [],
        }

    axes[-1].set_xlabel("stream position", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    positions = np.linspace(eval_start, n - 1, N_FRAMES).astype(int)

    def update(frame_index: int):
        position = positions[frame_index]
        changed = []
        for name in STRATEGIES:
            run = runs[name]
            art = artists[name]

            flagged = np.where(run["flags"][: position + 1] == 1)[0]
            art["scatter"].set_offsets(
                np.column_stack([flagged, signal[flagged]])
                if flagged.size else np.empty((0, 2))
            )
            art["playhead"].set_xdata([position, position])

            seen = run["flags"][run["eval_start"] : position + 1]
            rate = seen.mean() if seen.size else 0.0
            art["text"].set_text(f"{rate:.1%} of points flagged")

            # Rebuild markers appear as the playhead reaches them.
            due = [a for a in run["adaptations"] if a <= position]
            while len(art["rebuilds"]) < len(due):
                marker = due[len(art["rebuilds"])]
                art["rebuilds"].append(
                    axes[STRATEGIES.index(name)].axvline(
                        marker, color=COLOURS[name], lw=0.9, alpha=0.55, zorder=3
                    )
                )
            changed += [art["scatter"], art["playhead"], art["text"]]
        return changed

    print(f"Rendering {N_FRAMES} frames of '{stream}'...")
    animation = FuncAnimation(fig, update, frames=N_FRAMES, blit=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"Wrote {out.relative_to(PROJECT_ROOT)} ({size_mb:.2f} MB, "
          f"{N_FRAMES} frames at {FPS} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
