"""Pack the experiment output into a small file the demo can ship with.

    python scripts/build_demo_cache.py

The full run cache under `data/cache/` is ~9 MB of pickled model output and is
gitignored, so the Streamlit app cannot rely on it — a fresh clone would have
nothing to replay. This writes `data/demo/demo_runs.npz`, which *is* committed:
float32 sensor traces, per-strategy flags as int8, plus change points and
adaptation positions.

Everything in it is real output from `scripts/run_experiment.py`. Nothing here
generates or approximates data; it only re-packs what the models actually
produced, so the demo shows genuine results on a machine that has never run the
experiment.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "runs_full.pkl"
OUT_DIR = PROJECT_ROOT / "data" / "demo"
OUT_PATH = OUT_DIR / "demo_runs.npz"


def main() -> int:
    if not CACHE_PATH.exists():
        print(
            f"No run cache at {CACHE_PATH.relative_to(PROJECT_ROOT)}.\n"
            "Run `python scripts/run_experiment.py` first.",
            file=sys.stderr,
        )
        return 1

    with CACHE_PATH.open("rb") as handle:
        payload = pickle.load(handle)

    table, all_runs = payload["table"], payload["runs"]
    arrays: dict[str, np.ndarray] = {}
    stream_names = []

    for stream_name, (scenario, runs) in all_runs.items():
        stream_names.append(stream_name)
        prefix = f"{stream_name}|"

        arrays[prefix + "signal"] = scenario.frame.to_numpy().astype(np.float32)
        arrays[prefix + "labels"] = scenario.labels.to_numpy().astype(np.int8)
        arrays[prefix + "change_points"] = np.asarray(scenario.change_points, dtype=np.int32)
        arrays[prefix + "has_anomalies"] = np.array([scenario.has_true_anomalies])
        arrays[prefix + "threshold_quantile"] = np.array([scenario.threshold_quantile])
        arrays[prefix + "note"] = np.array([scenario.note])

        for strategy_name, run in runs.items():
            key = f"{prefix}{strategy_name}|"
            arrays[key + "flags"] = run.flags.astype(np.int8)
            arrays[key + "adaptations"] = np.asarray(run.adaptations, dtype=np.int32)
            arrays[key + "detections"] = np.asarray(run.detections, dtype=np.int32)
            arrays[key + "eval_start"] = np.array([run.eval_start], dtype=np.int32)

    arrays["__streams__"] = np.array(stream_names)
    arrays["__sensors__"] = np.array(
        list(all_runs[stream_names[0]][0].frame.columns)
    )
    arrays["__strategies__"] = np.array(
        list(all_runs[stream_names[0]][1].keys())
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, **arrays)

    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"Wrote {OUT_PATH.relative_to(PROJECT_ROOT)} ({size_mb:.2f} MB)")
    print(f"  streams: {', '.join(stream_names)}")
    print(f"  strategies: {', '.join(arrays['__strategies__'])}")
    print(f"  {len(table)} metric rows available in results/metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
