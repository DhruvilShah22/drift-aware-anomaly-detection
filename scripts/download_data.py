"""Download and verify the SKAB dataset, then cache it and write demo samples.

Run this once before anything else:

    python scripts/download_data.py

It prints the shape, columns, and label balance of what it actually read, so a
schema change on the upstream side is visible immediately rather than showing up
later as a strange metric.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (  # noqa: E402
    SENSOR_COLUMNS,
    DatasetError,
    list_skab_files,
    load_anomaly_free,
    load_skab_stream,
    save_sample,
)

# One labelled recording per fault group is enough to build and demo on; the
# rest of SKAB is available through load_skab_stream if a run needs it.
DEMO_STREAMS = [("valve1", "0.csv"), ("valve2", "0.csv"), ("other", "1.csv")]


def main() -> int:
    try:
        print("Checking SKAB availability...")
        for group in ("valve1", "valve2", "other", "anomaly-free"):
            files = list_skab_files(group)
            print(f"  {group}: {len(files)} CSV files, first = {files[0]}")

        print("\nLoading the anomaly-free recording (base signal for injected drift)...")
        clean = load_anomaly_free()
        print(f"  {clean.describe()}")
        print(f"  columns: {list(clean.frame.columns)}")
        print(f"  index: {clean.frame.index[0]} -> {clean.frame.index[-1]}")
        assert list(clean.frame.columns) == SENSOR_COLUMNS
        print(f"  sample written to {save_sample(clean).relative_to(Path.cwd())}")

        print("\nLoading labelled recordings...")
        for group, filename in DEMO_STREAMS:
            stream = load_skab_stream(group, filename)
            print(f"  {stream.describe()}")
            if stream.anomaly_rate == 0:
                raise DatasetError(
                    f"{stream.name} is labelled but contains no anomalies, "
                    "which contradicts the SKAB documentation"
                )
            print(f"  sample written to {save_sample(stream).relative_to(Path.cwd())}")

    except (DatasetError, AssertionError) as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        print(
            "The dataset did not load as expected. Stopping rather than "
            "substituting another source.",
            file=sys.stderr,
        )
        return 1

    print("\nAll datasets verified and cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
