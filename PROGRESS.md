# Progress log

Running log for this project. A fresh session should be able to resume from this file alone.

## How to resume

```powershell
cd C:\Users\conta\Desktop\Claude\drift-aware-anomaly-detection
.\.venv\Scripts\Activate.ps1        # Python 3.13 venv
python scripts\run_smoke_test.py    # fast end-to-end check (once Phase 5 lands)
```

The venv lives at `.venv` and is built on **Python 3.13** (`C:\Python313`), not the
system default 3.14 — `river` has no cp314 wheels.

## Environment notes

- `river` 0.25.0, scikit-learn 1.9.0, pandas 3.0.3, numpy 2.5.1, streamlit 1.59.2.
- Verified working API signatures (checked directly, not from memory):
  - `anomaly.HalfSpaceTrees(n_trees=10, height=8, window_size=250, limits=None, seed=None)`
  - `drift.ADWIN(delta=0.002, clock=32, max_buckets=5, min_window_length=5, grace_period=10)`
  - `drift.KSWIN(alpha=0.005, window_size=100, stat_size=30, seed=None, window=None)`
  - Detectors are driven with `.update(value)` then read via `.drift_detected`.
- One-off snag: on first import, Windows Application Control blocked
  `river/_river_rust`'s DLL. It cleared on its own after the reputation check
  completed and has been stable since. If it reappears, that is the cause.

## Phase checklist

- [x] **Phase 0 — Setup:** git init, repo skeleton, requirements, .gitignore, PROGRESS.md
- [x] **Phase 1 — Data:** `data_loader.py`, download and verify a real dataset, cache it
- [ ] **Phase 2 — Drift injection:** four drift shapes with known change points + tests
- [ ] **Phase 3 — Model & detectors:** online HST, static IsolationForest, ADWIN/KSWIN wrappers
- [ ] **Phase 4 — Evaluation:** F1, detection timing, false-alarm rate, tested on a toy stream
- [ ] **Phase 5 — Comparison run:** three-strategy experiment, first real metrics + figures
- [ ] **Phase 6 — Streamlit demo:** live dashboard, export `results/demo.gif`
- [ ] **Phase 7 — Light tuning:** small sweep, regenerate table and figures
- [ ] **Phase 8 — Kaggle notebook:** package the heavy run
- [ ] **Phase 9 — Write-up:** finalize README, push

## Log

### 2026-07-21 — Phase 0 complete

Set up the repo skeleton, a Python 3.13 venv, and pinned dependencies. Confirmed
every library in `requirements.txt` installs and imports, and smoke-checked the
three `river` classes the project depends on so later phases build on verified
APIs rather than assumptions.

Decisions made up front:

- Python 3.13 venv (river ships no cp314 wheels).
- Public GitHub repo from the start, pushed after every phase, so an abrupt
  session end never loses work.
- Demo clip produced two ways: a matplotlib animation rendered straight from
  real experiment output (the reliable, reproducible artifact), plus a real
  screen capture of the Streamlit dashboard if that route works cleanly.

### 2026-07-21 — Phase 1 complete

SKAB loads. It is reachable, its schema matches, and the numbers below come from
an actual `python scripts/download_data.py` run:

| group | files | recording used | rows | anomalous |
| --- | --- | --- | --- | --- |
| valve1 | 16 | `valve1/0.csv` | 1147 | 401 (35.0%) |
| valve2 | 4 | `valve2/0.csv` | 1125 | 394 (35.0%) |
| other | 14 | `other/1.csv` | 745 | 188 (25.2%) |
| anomaly-free | 1 | `anomaly-free.csv` | 9405 | 0 (unlabelled by design) |

Schema details worth remembering: SKAB CSVs are **semicolon-separated**, indexed
by a `datetime` column, with 8 sensor channels plus `anomaly` and `changepoint`
label columns. The `anomaly-free` recording ships without label columns, so the
loader synthesises an all-zero label series for it.

The 9405-row anomaly-free run is the base signal for Phase 2's injected-drift
streams — no annotated faults in it, so anything the injectors add is known
exactly. Labelled recordings are much shorter (~1000 rows each).

Caching: raw CSVs land in `data/raw/` and parsed frames in `data/cache/` (both
gitignored); 2000-row slices go to `data/sample/` and **are** committed, so the
Streamlit demo and tests run with no network access.

**Next step:** Phase 2. Write `src/drift_injection.py` with the four drift
shapes — sudden (step), incremental (ramp), gradual (interleaved regimes),
recurring (seasonal) — each returning the modified stream plus the exact indices
where drift was applied, so detection timing can be measured against ground
truth. Build them on top of `load_anomaly_free()`. Cover each shape with pytest
tests in `tests/test_drift_injection.py` asserting the signal actually changes at
the declared points and nowhere else.
