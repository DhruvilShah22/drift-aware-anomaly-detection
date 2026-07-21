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
- [x] **Phase 2 — Drift injection:** four drift shapes with known change points + tests
- [x] **Phase 3 — Model & detectors:** online HST, static IsolationForest, ADWIN/KSWIN wrappers
- [x] **Phase 4 — Evaluation:** F1, detection timing, false-alarm rate, tested on a toy stream
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

### 2026-07-21 — Phase 2 complete

`src/drift_injection.py` implements all four shapes. 16 pytest tests pass.

Design decisions:

- Shifts are sized in **units of each column's own standard deviation**, so one
  `magnitude` setting means the same thing across sensors on very different
  scales (Voltage vs Volume Flow RateRMS). Constant columns get no shift.
- Each injector returns `drift_points` (where a change begins — what a detector
  should flag) *and* a per-row `drift_mask` (which regime each row is in — what
  false-alarm accounting uses). Keeping both separate matters for `recurring`.
- `gradual` shifts whole rows, choosing regimes with rising probability, rather
  than interpolating like `incremental` does. Early in its window the new regime
  looks like sporadic outliers, which is the case that breaks mean-shift logic.
- `recurring` truncates to a whole number of blocks. Without that, a period that
  does not divide the stream length left a boundary a row or two from the end
  that no detector could fairly be scored on.

Verified on the real 9405-row anomaly-free SKAB recording at magnitude 3.0:
sudden → point 4702; incremental → point 3762; gradual → point 3762 with 45.1%
of rows shifted; recurring → points 2351/4702/7053 over 9404 usable rows.

### 2026-07-21 — Phase 3 complete

`src/models.py` and `src/detectors.py` are in. 38 tests pass.

**Scaling decision (resolved by reading river's HST source, not guessing).** The
plan was a `MinMaxScaler` in front of HST. That turns out to be wrong for this
project: river's scalers adapt continuously, so they would absorb the drift
before the model ever saw it — exactly the effect being measured. Instead I take
per-feature min/max from the warm-up window (widened 10%) and pass them as HST's
`limits`, then hold them fixed. Values that drift outside saturate into the
boundary leaves and score as anomalous, which is the wanted behaviour.

**Threshold decision.** HST normalises its raw mass score against a theoretical
maximum that is never approached, so scores sit in a narrow offset band — in a
probe, normal points scored ~0.94 and clearly anomalous ones ~0.995. An absolute
threshold would be meaningless. The threshold is a high quantile (default 0.98)
of warm-up scores. A rolling quantile was rejected for the same reason as the
rolling scaler. Adaptation for every strategy means one thing: re-run `warm_up`
on a recent window, rebuilding model, limits, and threshold together.

**Measured detector behaviour** (5 seeds x 5000 stationary steps, plus a 6-sigma
step at index 600). These are real numbers from a probe run:

| detector | false alarms / 5000 stationary steps | reaction to the step | re-fires? |
| --- | --- | --- | --- |
| ADWIN (delta=0.002) | 0, 0, 0, 0, 0 | index 607, i.e. 7-step delay | no — exactly one alarm |
| KSWIN (alpha=0.005) | 6, 4, 6, 6, 8 | index 628 | yes — also 883 |

So ADWIN is essentially silent when nothing happens and needs no cooldown, while
KSWIN reacts on a shorter window and pays for it with roughly one false alarm per
700 steps plus repeat firing. The cooldown in `DriftMonitor` exists for KSWIN;
ADWIN never engages it. This is a genuine trade-off to report in the write-up,
not a defect — and it is why the evaluation measures false-alarm rate at all.

Detectors monitor the **model's anomaly score**, not a raw sensor channel: a
sensor shift only matters if it degrades the detector, and the score summarises
all eight channels through the model's eyes.

### 2026-07-21 — Phase 4 complete

`src/evaluate.py` is in. 55 tests pass; every metric case is checked against
arithmetic written out in the test, not against the code's own output.

The drift-matching rule, stated once so the reported numbers can be read without
guessing: **each true change point takes the first unused detection at or after
it, within `horizon` (default 750) steps, and before the next change point.
Everything left over is a false alarm.** The "before the next change point"
clause matters on the recurring stream, where boundaries are close together and
a late detection would otherwise be credited to the wrong one.

Degenerate cases are decided deliberately: a strategy that flags nothing scores
precision/recall/F1 of 0 rather than being undefined, so silence loses a
comparison instead of dropping out of it. False alarms are reported per 1000
steps so streams of different lengths compare fairly.

**Next step:** Phase 5. Write `src/experiment.py` running all three strategies
over the same stream — static (warm up once, never adapt), periodic (re-warm
every N steps regardless), drift-triggered (re-warm only when the monitor fires
on the anomaly score). All three share warm-up length and threshold quantile so
the only difference is *when* they adapt. Then `scripts/run_smoke_test.py` for a
fast local run, and first real metrics into `results/metrics.csv` plus figures.

Watch out: two Application Control blocks have now hit on first import of an
unsigned DLL (`river/_river_rust`, then sklearn's `_radius_neighbors`). Both
cleared on a plain retry. If a fresh import fails this way, retry once before
investigating.
