"""The UCR Anomaly Archive loader.

The archive is a 184 MB download, so these tests build a small stand-in zip with
the same naming convention and point the loader at it. That keeps the suite
runnable on a fresh clone and still exercises the real parsing, labelling and
verification paths rather than mocking them away.

The naming convention is the whole interface:

    136_UCR_Anomaly_InternalBleeding16_1200_4187_4197.txt

carries the archive index, the series name, the last index of the guaranteed
clean training prefix, and the 1-based inclusive bounds of the single anomaly.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

from src import data_loader
from src.data_loader import (
    DatasetError,
    _archive_is_complete,
    _parse_ucr_name,
    list_ucr_series,
    load_ucr_stream,
    select_ucr_series,
)


def write_series(handle: zipfile.ZipFile, member: str, n_rows: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    handle.writestr(member, "\n".join(f"{v:.6f}" for v in rng.normal(size=n_rows)))


@pytest.fixture
def fake_archive(tmp_path, monkeypatch):
    """A stand-in archive with the same conventions as the real one."""
    path = tmp_path / "ucr.zip"
    with zipfile.ZipFile(path, "w") as zf:
        write_series(zf, "001_UCR_Anomaly_alpha_1500_4000_4100.txt", 6000, seed=1)
        write_series(zf, "002_UCR_Anomaly_DISTORTEDbeta2_1500_3000_3050.txt", 5000, seed=2)
        write_series(zf, "003_UCR_Anomaly_gamma_with_underscores_1500_2000_2010.txt", 4000, seed=3)
        # Too short a training prefix for a 1,000-row warm-up.
        write_series(zf, "004_UCR_Anomaly_delta_300_2000_2010.txt", 4000, seed=4)
        # Anomaly runs past the end of the series.
        write_series(zf, "005_UCR_Anomaly_epsilon_1500_3900_9999.txt", 4000, seed=5)
        # Anomaly inside the training prefix, which the archive forbids.
        write_series(zf, "006_UCR_Anomaly_zeta_3000_2000_2010.txt", 4000, seed=6)
        zf.writestr("README.md", "not a series")

    monkeypatch.setattr(data_loader, "_ensure_ucr_archive", lambda *a, **k: path)
    monkeypatch.setattr(data_loader, "CACHE_DIR", tmp_path / "cache")
    return path


class TestArchiveCompleteness:
    """A zip keeps its central directory at the end of the file, so a partial
    download of a 184 MB archive is a plausible-looking 150 MB file that no zip
    reader can open. A size threshold accepted one at 84% and the run died on
    BadZipFile, so completeness is now decided by trying to read it."""

    def test_a_missing_file_is_not_complete(self, tmp_path):
        assert not _archive_is_complete(tmp_path / "absent.zip")

    def test_an_empty_file_is_not_complete(self, tmp_path):
        path = tmp_path / "empty.zip"
        path.write_bytes(b"")
        assert not _archive_is_complete(path)

    def test_a_truncated_zip_is_not_complete(self, tmp_path):
        whole = tmp_path / "whole.zip"
        with zipfile.ZipFile(whole, "w") as zf:
            write_series(zf, "001_UCR_Anomaly_alpha_1500_4000_4100.txt", 6000)
        truncated = tmp_path / "truncated.zip"
        data = whole.read_bytes()
        truncated.write_bytes(data[: int(len(data) * 0.84)])
        assert truncated.stat().st_size > 0
        assert not _archive_is_complete(truncated)

    def test_a_whole_zip_is_complete(self, tmp_path):
        path = tmp_path / "whole.zip"
        with zipfile.ZipFile(path, "w") as zf:
            write_series(zf, "001_UCR_Anomaly_alpha_1500_4000_4100.txt", 6000)
        assert _archive_is_complete(path)

    def test_a_zip_with_no_members_is_not_complete(self, tmp_path):
        path = tmp_path / "hollow.zip"
        with zipfile.ZipFile(path, "w"):
            pass
        assert not _archive_is_complete(path)


class TestNameParsing:
    @pytest.mark.parametrize(
        "member,name,train,start,end",
        [
            ("136_UCR_Anomaly_InternalBleeding16_1200_4187_4197.txt",
             "InternalBleeding16", 1200, 4187, 4197),
            ("001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt",
             "DISTORTED1sddb40", 35000, 52000, 52620),
            ("250_UCR_Anomaly_weallwalk_2951_5941_5959.txt",
             "weallwalk", 2951, 5941, 5959),
        ],
    )
    def test_reads_the_split_points(self, member, name, train, start, end):
        parsed = _parse_ucr_name(member)
        assert parsed is not None
        assert (parsed.name, parsed.train_end) == (name, train)
        assert (parsed.anomaly_start, parsed.anomaly_end) == (start, end)

    def test_keeps_underscores_inside_a_series_name(self):
        parsed = _parse_ucr_name("007_UCR_Anomaly_a_b_c_1000_2000_2010.txt")
        assert parsed.name == "a_b_c"
        assert parsed.train_end == 1000

    def test_anomaly_length_is_inclusive(self):
        parsed = _parse_ucr_name("008_UCR_Anomaly_x_100_200_209.txt")
        assert parsed.anomaly_length == 10

    @pytest.mark.parametrize(
        "member", ["README.md", "notes.txt", "1_UCR_Anomaly_x.txt", "x_UCR_Anomaly_y_a_b_c.txt"]
    )
    def test_rejects_anything_else(self, member):
        assert _parse_ucr_name(member) is None


class TestListing:
    def test_skips_non_series_files(self, fake_archive):
        names = {s.name for s in list_ucr_series()}
        assert "alpha" in names
        assert len(list_ucr_series()) == 6

    def test_sorted_by_archive_index(self, fake_archive):
        indices = [s.index for s in list_ucr_series()]
        assert indices == sorted(indices)


class TestLoading:
    def test_labels_exactly_the_named_rows(self, fake_archive):
        stream = load_ucr_stream("001_UCR_Anomaly_alpha_1500_4000_4100.txt")
        assert stream.n_rows == 6000
        assert int(stream.labels.sum()) == 101
        # The name is 1-based and inclusive; the frame is 0-based. Rows 4000
        # and 4100 in the name are positions 3999 and 4099 here.
        assert stream.labels.iloc[3999] == 1
        assert stream.labels.iloc[4099] == 1
        assert stream.labels.iloc[3998] == 0
        assert stream.labels.iloc[4100] == 0

    def test_exactly_one_contiguous_event(self, fake_archive):
        from src.evaluate import to_events

        stream = load_ucr_stream("001_UCR_Anomaly_alpha_1500_4000_4100.txt")
        assert len(to_events(stream.labels.to_numpy())) == 1

    def test_the_warm_up_window_is_clean(self, fake_archive):
        """The point of the archive's training prefix. If warm-up saw the
        anomaly the model would learn it as normal."""
        stream = load_ucr_stream("001_UCR_Anomaly_alpha_1500_4000_4100.txt")
        assert stream.labels.iloc[:1000].sum() == 0

    def test_anomaly_rate_is_small(self, fake_archive):
        stream = load_ucr_stream("001_UCR_Anomaly_alpha_1500_4000_4100.txt")
        assert stream.anomaly_rate < 0.05

    def test_an_anomaly_past_the_end_is_rejected(self, fake_archive):
        with pytest.raises(DatasetError, match="but the series has"):
            load_ucr_stream("005_UCR_Anomaly_epsilon_1500_3900_9999.txt")

    def test_an_anomaly_inside_the_training_prefix_is_rejected(self, fake_archive):
        with pytest.raises(DatasetError, match="training prefix"):
            load_ucr_stream("006_UCR_Anomaly_zeta_3000_2000_2010.txt")

    def test_a_malformed_name_is_rejected(self, fake_archive):
        with pytest.raises(DatasetError, match="naming convention"):
            load_ucr_stream("README.md")


@pytest.fixture
def clean_archive(tmp_path, monkeypatch):
    """An archive of well-formed series only.

    `fake_archive` deliberately contains malformed entries to test that loading
    rejects them, and selection loads every candidate, so it would hit one of
    those before it could run out of series. Selection is tested here instead.
    """
    path = tmp_path / "clean.zip"
    with zipfile.ZipFile(path, "w") as zf:
        write_series(zf, "001_UCR_Anomaly_alpha_1500_4000_4100.txt", 6000, seed=1)
        write_series(zf, "002_UCR_Anomaly_DISTORTEDbeta2_1500_3000_3050.txt", 5000, seed=2)
        write_series(zf, "003_UCR_Anomaly_beta7_1500_3000_3050.txt", 5000, seed=3)
        write_series(zf, "004_UCR_Anomaly_delta_300_2000_2010.txt", 4000, seed=4)
        write_series(zf, "005_UCR_Anomaly_omega_1500_2500_2510.txt", 9000, seed=5)
    monkeypatch.setattr(data_loader, "_ensure_ucr_archive", lambda *a, **k: path)
    monkeypatch.setattr(data_loader, "CACHE_DIR", tmp_path / "cache")
    return path


class TestSelection:
    def test_propagates_a_malformed_series(self, fake_archive):
        """A corrupt archive should stop the run, not be quietly skipped."""
        with pytest.raises(DatasetError):
            select_ucr_series(n_series=6, min_train=1_200)

    def test_excludes_a_short_training_prefix(self, clean_archive):
        chosen = select_ucr_series(n_series=3, max_rows=12_000, min_train=1_200)
        assert all(s.train_end >= 1_200 for s in chosen)
        assert "delta" not in {s.name for s in chosen}

    def test_deduplicates_by_base_name(self, clean_archive):
        chosen = select_ucr_series(n_series=3, max_rows=12_000, min_train=1_200)
        bases = [s.name.replace("DISTORTED", "").rstrip("0123456789") for s in chosen]
        assert len(bases) == len(set(bases))

    def test_is_deterministic(self, clean_archive):
        first = [s.member for s in select_ucr_series(n_series=3, min_train=1_200)]
        second = [s.member for s in select_ucr_series(n_series=3, min_train=1_200)]
        assert first == second

    def test_asks_loudly_when_too_few_series_qualify(self, clean_archive):
        with pytest.raises(DatasetError, match="only .* met the selection rule"):
            select_ucr_series(n_series=50, min_train=1_200)
