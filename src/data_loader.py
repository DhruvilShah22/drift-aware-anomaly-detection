"""Loading and verifying the real sensor datasets this project runs on.

Three datasets are loaded here. SKAB (Skoltech Anomaly Benchmark) is a set of
labelled multivariate sensor recordings from a water-circulation testbed; NAB is
a univariate collection used to test whether a calibration transfers; and the
UCR Anomaly Archive is 250 univariate series with exactly one anomaly each,
built by Wu and Keogh as a replacement for benchmarks they argued were flawed.
Files are pulled from their public sources and cached on disk, so a download
only ever happens once.

Every loader in this module verifies what it read: the expected columns are
present, the labels are binary, and the frame is non-empty. I would rather a run
fail loudly here than have a silently malformed frame propagate into the
experiments.
"""

from __future__ import annotations

import io
import zipfile
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


# ---------------------------------------------------------------------------
# NAB (Numenta Anomaly Benchmark)
# ---------------------------------------------------------------------------
#
# NAB is the second dataset, used to ask whether the settings chosen on SKAB
# transfer to data they were not tuned on. It differs from SKAB in the two ways
# that matter for that question: it is **univariate** where SKAB has 8 sensors,
# and its anomalies are **rare** (a few percent) where SKAB's valve1 stream is
# about a third anomalous.
#
# Labels come as time *windows* rather than per-row flags, so they are expanded
# against each series' own timestamps.

NAB_RAW_BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
NAB_API_BASE = "https://api.github.com/repos/numenta/NAB/contents/data"
NAB_WINDOWS_PATH = "labels/combined_windows.json"

NAB_VALUE_COLUMN = "value"
NAB_TIME_COLUMN = "timestamp"


def _download_nab_text(relative: str, timeout: int = 60) -> bytes:
    """Fetch a file from the NAB repository, caching it under `data/raw/nab/`."""
    destination = RAW_DIR / "nab" / relative.replace("/", "__")
    if destination.exists() and destination.stat().st_size > 0:
        return destination.read_bytes()

    url = f"{NAB_RAW_BASE}/{relative}"
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "nab-loader"})
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DatasetError(f"could not download {url}: {exc}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return response.content


def load_nab_windows() -> dict[str, list[list[str]]]:
    """The combined anomaly windows NAB publishes, keyed by series path."""
    import json

    raw = _download_nab_text(NAB_WINDOWS_PATH)
    try:
        windows = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"NAB label file is not valid JSON: {exc}") from exc

    if not windows:
        raise DatasetError("NAB label file is empty")
    return windows


def list_nab_series(group: str = "realKnownCause", timeout: int = 30) -> list[str]:
    """List the CSV series NAB publishes in one group."""
    url = f"{NAB_API_BASE}/{group}"
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "nab-loader"})
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DatasetError(f"could not list NAB group {group!r} at {url}: {exc}") from exc

    names = sorted(
        entry["name"]
        for entry in response.json()
        if entry.get("type") == "file" and entry["name"].endswith(".csv")
    )
    if not names:
        raise DatasetError(f"NAB group {group!r} returned no CSV files")
    return names


def load_nab_stream(series_key: str, use_cache: bool = True) -> StreamData:
    """Load one NAB series as a verified `StreamData`.

    `series_key` is the path NAB uses in its label file, for example
    `realKnownCause/machine_temperature_system_failure.csv`.

    NAB labels anomalies as time windows. Every row whose timestamp falls inside
    any window is marked 1. A series whose windows do not overlap its own
    timestamps at all would silently produce an all-zero label vector, so that
    case is treated as an error rather than passed downstream.
    """
    cache_path = CACHE_DIR / f"nab_{series_key.replace('/', '_').replace('.csv', '')}.parquet"

    if use_cache and cache_path.exists():
        frame = pd.read_parquet(cache_path)
    else:
        raw = _download_nab_text(f"data/{series_key}")
        try:
            frame = pd.read_csv(
                io.BytesIO(raw), index_col=NAB_TIME_COLUMN, parse_dates=True
            )
        except Exception as exc:
            raise DatasetError(f"could not parse NAB series {series_key}: {exc}") from exc

        if NAB_VALUE_COLUMN not in frame.columns:
            raise DatasetError(
                f"{series_key} has no '{NAB_VALUE_COLUMN}' column; "
                f"got {list(frame.columns)}"
            )
        frame = frame[[NAB_VALUE_COLUMN]].astype(float).sort_index()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path)

    if frame.empty:
        raise DatasetError(f"{series_key} parsed to an empty frame")
    if frame[NAB_VALUE_COLUMN].isna().any():
        raise DatasetError(f"{series_key} contains missing values")

    windows = load_nab_windows()
    if series_key not in windows:
        raise DatasetError(
            f"{series_key} has no entry in NAB's combined_windows.json"
        )

    labels = expand_windows_to_labels(frame.index, windows[series_key], series_key)
    return StreamData(name=series_key, frame=frame, labels=labels)


def expand_windows_to_labels(
    index: pd.DatetimeIndex,
    windows: list[list[str]],
    source: str = "series",
) -> pd.Series:
    """Turn NAB's [start, end] timestamp windows into a per-row 0/1 label series.

    Both ends of each window are inclusive, matching how NAB defines them.

    A window list that does not overlap the index at all yields an all-zero
    label vector, which downstream would look like a clean series rather than a
    loading bug. That is treated as an error instead.
    """
    labels = pd.Series(0, index=index, dtype=int)
    for window in windows:
        if len(window) != 2:
            raise DatasetError(
                f"{source} has a malformed label window {window!r}; expected [start, end]"
            )
        start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        if start > end:
            raise DatasetError(
                f"{source} has a reversed label window: {start} is after {end}"
            )
        labels.loc[start:end] = 1

    if windows and labels.sum() == 0:
        raise DatasetError(
            f"{source} has {len(windows)} labelled windows but none overlap its "
            "timestamps, which means the labels were not applied correctly"
        )
    return labels


# --------------------------------------------------------------------------
# UCR Anomaly Archive
# --------------------------------------------------------------------------
#
# The third dataset, and the one that exists because of the critique this
# project cites. Wu and Keogh (2021) argued that NAB and benchmarks like it are
# flawed enough to create an illusion of progress, and released this archive as
# a replacement: 250 univariate series, each carrying **exactly one** anomaly,
# with a guaranteed anomaly-free training prefix.
#
# That construction is what makes it worth the download here. On SKAB and NAB
# every strategy catches every labelled event, so event-level recall is 1.0
# across the board and the composite score collapses to a precision ranking.
# With one short anomaly per series, a detector either finds it or it does not,
# and the recall axis finally carries information.
#
# Everything a loader needs is in the file name:
#
#     136_UCR_Anomaly_InternalBleeding16_1200_4187_4197.txt
#     |   |           |                  |    |    |
#     |   |           |                  |    |    anomaly end   (1-based, inclusive)
#     |   |           |                  |    anomaly start (1-based, inclusive)
#     |   |           |                  last index of the clean training prefix
#     |   |           series name
#     |   archive marker
#     index within the archive
#
# The body is one float per line. We parse the name rather than trusting a side
# file, and then check the parse against the data: the anomaly has to lie inside
# the series, after the training prefix, and be a single contiguous run.

UCR_ARCHIVE_URL = (
    "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
    "UCR_TimeSeriesAnomalyDatasets2021.zip"
)
UCR_ZIP = RAW_DIR / "ucr" / "UCR_TimeSeriesAnomalyDatasets2021.zip"
UCR_VALUE_COLUMN = "value"


@dataclass(frozen=True)
class UcrSeries:
    """What the archive's file name says about one series."""

    member: str
    index: int
    name: str
    train_end: int
    anomaly_start: int
    anomaly_end: int

    @property
    def anomaly_length(self) -> int:
        return self.anomaly_end - self.anomaly_start + 1

    @property
    def slug(self) -> str:
        return f"{self.index:03d}_{self.name}"


def _archive_is_complete(path: Path) -> bool:
    """Is this a whole, readable zip rather than a download in progress?

    Size alone does not answer that. A zip stores its central directory at the
    *end* of the file, so a partial download of an 184 MB archive is a perfectly
    plausible 150 MB file that no zip reader can open. We learned that the
    obvious way: a size threshold accepted a file at 84% and the selection step
    died on BadZipFile.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return bool(zf.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


# The archive is served from a university host that drops long connections. A
# single-shot download of 184 MB timed out at 165 MB the first time, so the
# fetch resumes with a Range request instead of starting over.
UCR_DOWNLOAD_ATTEMPTS = 12


def _ensure_ucr_archive(timeout: int = 120, attempts: int = UCR_DOWNLOAD_ATTEMPTS) -> Path:
    """Download the archive once, resuming if the connection drops.

    Bytes accumulate in a `.part` file that is renamed only once it opens as a
    zip, so an interrupted run can never leave a half-file that a later run
    mistakes for a complete one.
    """
    if _archive_is_complete(UCR_ZIP):
        return UCR_ZIP

    UCR_ZIP.parent.mkdir(parents=True, exist_ok=True)
    partial = UCR_ZIP.with_suffix(".zip.part")

    # A previous truncated attempt may have left a full-looking file behind.
    if UCR_ZIP.exists() and not _archive_is_complete(UCR_ZIP):
        UCR_ZIP.unlink()

    last_error: Exception | None = None
    for attempt in range(attempts):
        have = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(
                UCR_ARCHIVE_URL, stream=True, timeout=timeout, headers=headers
            ) as response:
                # 416 means we already asked for everything there is.
                if response.status_code == 416:
                    break
                response.raise_for_status()
                # A server that ignores Range restarts the file, so the local
                # bytes have to be discarded rather than appended to.
                mode = "ab" if (have and response.status_code == 206) else "wb"
                with open(partial, mode) as handle:
                    for chunk in response.iter_content(1 << 20):
                        handle.write(chunk)
        except requests.RequestException as exc:
            last_error = exc
            continue

        if _archive_is_complete(partial):
            break

    if not _archive_is_complete(partial):
        size = partial.stat().st_size if partial.exists() else 0
        raise DatasetError(
            f"the UCR archive stopped at {size} bytes after {attempts} attempts "
            f"and is not a readable zip"
            + (f"; last error: {last_error}" if last_error else "")
        )

    partial.replace(UCR_ZIP)
    return UCR_ZIP


def _parse_ucr_name(member: str) -> UcrSeries | None:
    """Pull the split points out of an archive file name.

    Returns None for anything that does not match the convention, so a stray
    file in the zip is skipped rather than crashing the listing.
    """
    stem = Path(member).stem
    parts = stem.split("_")
    if len(parts) < 6:
        return None
    try:
        index = int(parts[0])
        train_end, start, end = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
    except ValueError:
        return None

    name = "_".join(parts[3:-3]) or stem
    return UcrSeries(
        member=member,
        index=index,
        name=name,
        train_end=train_end,
        anomaly_start=start,
        anomaly_end=end,
    )


def list_ucr_series() -> list[UcrSeries]:
    """Every series in the archive, parsed from its file name."""
    archive = _ensure_ucr_archive()
    try:
        with zipfile.ZipFile(archive) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
    except zipfile.BadZipFile as exc:
        raise DatasetError(f"the UCR archive at {archive} is not a valid zip: {exc}") from exc

    series = [s for s in (_parse_ucr_name(m) for m in members) if s is not None]
    if not series:
        raise DatasetError(
            f"the UCR archive contains {len(members)} .txt files but none matched "
            "the expected <index>_UCR_Anomaly_<name>_<train>_<start>_<end> convention"
        )
    return sorted(series, key=lambda s: s.index)


def load_ucr_stream(series: UcrSeries | str, use_cache: bool = True) -> StreamData:
    """Load one UCR series as a verified `StreamData`.

    Accepts either the parsed record or the archive member name.
    """
    if isinstance(series, str):
        parsed = _parse_ucr_name(series)
        if parsed is None:
            raise DatasetError(f"{series!r} does not match the UCR naming convention")
        series = parsed

    cache_path = CACHE_DIR / f"ucr_{series.slug}.parquet"
    if use_cache and cache_path.exists():
        frame = pd.read_parquet(cache_path)
    else:
        archive = _ensure_ucr_archive()
        with zipfile.ZipFile(archive) as zf:
            raw = zf.read(series.member)
        # One value per line in most files, whitespace-separated in a few.
        try:
            values = [float(tok) for tok in raw.split()]
        except ValueError as exc:
            raise DatasetError(f"{series.member} contains a non-numeric value: {exc}") from exc
        if not values:
            raise DatasetError(f"{series.member} is empty")
        frame = pd.DataFrame({UCR_VALUE_COLUMN: values}, dtype=float)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path)

    n = len(frame)
    if frame[UCR_VALUE_COLUMN].isna().any():
        raise DatasetError(f"{series.member} contains missing values")

    # Check the file name against the file body rather than trusting it. A name
    # that points past the end of its own series would otherwise produce an
    # all-zero label vector and look like a clean stream.
    if not 0 < series.anomaly_start <= series.anomaly_end <= n:
        raise DatasetError(
            f"{series.member} claims an anomaly at [{series.anomaly_start}, "
            f"{series.anomaly_end}] but the series has {n} points"
        )
    if series.train_end >= series.anomaly_start:
        raise DatasetError(
            f"{series.member} places its anomaly at {series.anomaly_start}, inside "
            f"the training prefix that ends at {series.train_end}; the archive "
            "guarantees the prefix is clean"
        )

    labels = pd.Series(0, index=frame.index, dtype=int)
    # The file name uses 1-based inclusive positions.
    labels.iloc[series.anomaly_start - 1 : series.anomaly_end] = 1
    if labels.sum() != series.anomaly_length:
        raise DatasetError(
            f"{series.member} labelled {int(labels.sum())} rows where its name "
            f"implies {series.anomaly_length}"
        )

    return StreamData(name=series.slug, frame=frame, labels=labels)


# Variants of one source signal appear in the archive under names that differ
# only by a corruption prefix or an ordinal. Collapsing them matters: a first
# attempt at the subset came back with five `Tkeep*MARS` series and two `BIDMC`
# series out of twelve, which is a study of two signals wearing a disguise.
UCR_VARIANT_PREFIXES = ("DISTORTED", "NOISE")
UCR_ORDINAL_WORDS = (
    "first", "second", "third", "fourth", "forth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth",
)


def ucr_base_name(name: str) -> str:
    """The source signal a series name refers to, ignoring variant markers.

    `DISTORTEDTkeepFifthMARS`, `TkeepFirstMARS` and `NOISETkeepMARS2` all reduce
    to `tkeepmars`.
    """
    stripped = name
    for prefix in UCR_VARIANT_PREFIXES:
        if stripped.upper().startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    lowered = stripped.lower()
    for word in UCR_ORDINAL_WORDS:
        lowered = lowered.replace(word, "")
    return lowered.rstrip("0123456789_") or stripped.lower()


def select_ucr_series(
    n_series: int = 12,
    max_rows: int = 12_000,
    min_train: int = 1_200,
    max_anomaly_fraction: float = 0.10,
) -> list[UcrSeries]:
    """A deterministic, documented subset of the archive.

    Running the full 250 series through the model-by-policy grid at ten seeds is
    not affordable on one core, so a subset is taken. The rule is fixed and
    stated rather than tuned:

    - the series must fit in `max_rows`, which bounds the runtime;
    - its clean training prefix must be at least `min_train`, so the warm-up
      window never straddles the anomaly;
    - its anomaly must be a small fraction of the series, which is what the
      archive is for and keeps the trivial baseline low;
    - at most one series per base signal, with variant prefixes and ordinals
      normalised away by `ucr_base_name`, so the subset is not five recordings
      of the same instrument;
    - then the lowest archive indices, so the choice does not depend on any
      result.

    A base signal is only consumed once a series actually passes every check.
    An earlier version reserved the name as soon as the cheap filters passed,
    which meant one over-long recording of a signal discarded every shorter
    recording of it too, and the subset came back a third the size it should
    have been.
    """
    chosen: list[UcrSeries] = []
    seen_names: set[str] = set()

    for series in list_ucr_series():
        if len(chosen) >= n_series:
            break
        if series.train_end < min_train:
            continue
        # Cheap bound before paying to read the series: the anomaly cannot end
        # past the last row, so a late anomaly means a long recording.
        if series.anomaly_end > max_rows:
            continue
        base = ucr_base_name(series.name)
        if base in seen_names:
            continue

        stream = load_ucr_stream(series)
        if stream.n_rows > max_rows:
            continue
        if series.anomaly_length / stream.n_rows > max_anomaly_fraction:
            continue

        seen_names.add(base)
        chosen.append(series)

    if len(chosen) < n_series:
        raise DatasetError(
            f"only {len(chosen)} UCR series met the selection rule "
            f"(wanted {n_series}); loosen max_rows or min_train"
        )
    return chosen


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
