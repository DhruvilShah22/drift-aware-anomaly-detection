"""Loading and verifying the real sensor datasets this project runs on.

The only dataset used here is SKAB (Skoltech Anomaly Benchmark), a set of
labelled multivariate sensor recordings from a water-circulation testbed. Files
are pulled straight from the project's public GitHub repository and cached on
disk, so a download only ever happens once.

Every loader in this module verifies what it read: the expected columns are
present, the labels are binary, and the frame is non-empty. I would rather a run
fail loudly here than have a silently malformed frame propagate into the
experiments.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

# SKAB serves its CSVs raw from GitHub. Pinned to a commit-independent branch
# path; the repository is archived and has not changed in years.
SKAB_RAW_BASE = "https://raw.githubusercontent.com/waico/SKAB/master/data"
SKAB_API_BASE = "https://api.github.com/repos/waico/SKAB/contents/data"

# The groups of recordings SKAB ships. "anomaly-free" is a single clean run,
# useful as a source of undisturbed signal for the injected-drift streams.
SKAB_GROUPS = ("valve1", "valve2", "other", "anomaly-free")

# Sensor channels, in the order SKAB writes them.
SENSOR_COLUMNS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]

LABEL_COLUMN = "anomaly"
CHANGEPOINT_COLUMN = "changepoint"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


class DatasetError(RuntimeError):
    """Raised when a dataset is unreachable or does not match its expected schema."""


@dataclass(frozen=True)
class StreamData:
    """One recording, ready to be replayed as a stream.

    `frame` is time-ordered with a DatetimeIndex. `labels` is 0/1 per row, where
    1 marks a point the SKAB authors annotated as anomalous.
    """

    name: str
    frame: pd.DataFrame
    labels: pd.Series

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def anomaly_rate(self) -> float:
        return float(self.labels.mean())

    def describe(self) -> str:
        return (
            f"{self.name}: {self.n_rows} rows x {self.frame.shape[1]} sensors, "
            f"{int(self.labels.sum())} anomalous points "
            f"({self.anomaly_rate:.1%})"
        )


def list_skab_files(group: str, timeout: int = 30) -> list[str]:
    """Return the CSV filenames SKAB publishes for one group.

    Hits the GitHub contents API rather than assuming filenames, so a change on
    their side surfaces as an error instead of a 404 mid-run.
    """
    if group not in SKAB_GROUPS:
        raise ValueError(f"unknown SKAB group {group!r}; expected one of {SKAB_GROUPS}")

    url = f"{SKAB_API_BASE}/{group}"
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "skab-loader"})
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DatasetError(f"could not list SKAB group {group!r} at {url}: {exc}") from exc

    names = sorted(
        entry["name"]
        for entry in response.json()
        if entry.get("type") == "file" and entry["name"].endswith(".csv")
    )
    if not names:
        raise DatasetError(f"SKAB group {group!r} returned no CSV files")
    return names


def _download_skab_file(group: str, filename: str, timeout: int = 60) -> Path:
    """Fetch one SKAB CSV into `data/raw/`, reusing it if already present."""
    destination = RAW_DIR / group / filename
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    url = f"{SKAB_RAW_BASE}/{group}/{filename}"
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "skab-loader"})
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DatasetError(f"could not download {url}: {exc}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return destination


def _parse_skab_csv(content: bytes | Path, source: str) -> pd.DataFrame:
    """Parse a SKAB CSV and verify it matches the expected schema.

    SKAB uses semicolons as separators and a `datetime` column as the index.
    """
    buffer = io.BytesIO(content) if isinstance(content, bytes) else content
    try:
        frame = pd.read_csv(buffer, sep=";", index_col="datetime", parse_dates=True)
    except Exception as exc:
        raise DatasetError(f"could not parse {source} as a SKAB CSV: {exc}") from exc

    missing = [c for c in SENSOR_COLUMNS if c not in frame.columns]
    if missing:
        raise DatasetError(
            f"{source} is missing expected sensor columns {missing}; "
            f"got {list(frame.columns)}"
        )
    if frame.empty:
        raise DatasetError(f"{source} parsed to an empty frame")

    return frame.sort_index()


def load_skab_stream(group: str, filename: str, use_cache: bool = True) -> StreamData:
    """Load one SKAB recording as a verified `StreamData`.

    Anomaly-free recordings carry no label column; those get an all-zero label
    series so they can be handled uniformly downstream.
    """
    cache_path = CACHE_DIR / f"skab_{group}_{Path(filename).stem}.parquet"
    if use_cache and cache_path.exists():
        frame = pd.read_parquet(cache_path)
    else:
        raw_path = _download_skab_file(group, filename)
        frame = _parse_skab_csv(raw_path, source=f"{group}/{filename}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path)

    if LABEL_COLUMN in frame.columns:
        labels = frame[LABEL_COLUMN].astype(int)
    else:
        # anomaly-free runs ship without labels; by construction nothing is anomalous.
        labels = pd.Series(0, index=frame.index, dtype=int, name=LABEL_COLUMN)

    unexpected = set(labels.unique()) - {0, 1}
    if unexpected:
        raise DatasetError(
            f"{group}/{filename} has non-binary labels: {sorted(unexpected)}"
        )

    sensors = frame[SENSOR_COLUMNS].astype(float)
    if sensors.isna().any().any():
        raise DatasetError(f"{group}/{filename} contains missing sensor values")

    return StreamData(name=f"{group}/{filename}", frame=sensors, labels=labels)


def load_anomaly_free(use_cache: bool = True) -> StreamData:
    """Load the single undisturbed SKAB recording.

    This is the base signal the injected-drift streams are built from: it has no
    annotated faults, so anything the drift injectors add is known exactly.
    """
    return load_skab_stream("anomaly-free", "anomaly-free.csv", use_cache=use_cache)


def save_sample(stream: StreamData, n_rows: int = 2000) -> Path:
    """Write a small slice of a recording to `data/sample/` for the demo.

    The Streamlit app reads from here so it never needs network access or the
    full dataset on a weak machine. Small enough to commit.
    """
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    slug = stream.name.replace("/", "_").replace(".csv", "")
    path = SAMPLE_DIR / f"{slug}.csv"

    sample = stream.frame.head(n_rows).copy()
    sample[LABEL_COLUMN] = stream.labels.head(n_rows)
    sample.to_csv(path)
    return path


def load_sample(slug: str) -> StreamData:
    """Read one of the committed sample slices back."""
    path = SAMPLE_DIR / f"{slug}.csv"
    if not path.exists():
        raise DatasetError(
            f"no committed sample at {path}; run scripts/download_data.py first"
        )
    frame = pd.read_csv(path, index_col="datetime", parse_dates=True)
    labels = frame[LABEL_COLUMN].astype(int)
    return StreamData(name=slug, frame=frame[SENSOR_COLUMNS].astype(float), labels=labels)
