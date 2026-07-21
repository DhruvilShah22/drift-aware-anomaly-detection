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
- [x] **Phase 5 — Comparison run:** three-strategy experiment, first real metrics + figures
- [x] **Phase 6 — Streamlit demo:** live dashboard, export `results/demo.gif`
- [x] **Phase 7 — Light tuning:** small sweep, regenerate table and figures
      *(done early — the first run was visibly miscalibrated and the table would
      have been misleading to leave standing)*
- [x] **Phase 8 — Kaggle notebook:** package the heavy run
- [x] **Phase 9 — Write-up:** finalize README, push

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

### 2026-07-21 — Phases 5 and 7 complete

`src/experiment.py`, `scripts/run_smoke_test.py`, `scripts/run_experiment.py`,
`scripts/run_tuning.py`. 59 tests pass. Smoke test runs in ~1s, the full
experiment in ~100s, the sweep in ~240s. Real results are in
`results/metrics.csv` and `results/tuning.csv`.

Phase 7 got pulled forward because the first run was plainly miscalibrated and
leaving a misleading table committed was worse than doing the sweep early.

#### Three things the code got wrong, all found by running it

1. **The retraining strategies were also learning continuously.** That made
   `periodic` and `drift-triggered` numerically identical to the
   `online-no-reset` control on the labelled stream — continuous learning had
   already absorbed the shift, leaving the rebuild nothing to do. Only the
   control learns between rebuilds now.
2. **The drift monitor watched the model's own anomaly score.** Self-defeating
   for the same reason: a learning model accommodates the shift, its score
   distribution never moves, the detector never fires. Zero detections across
   the entire labelled stream. It now watches mean absolute z-score of the
   inputs against the window the model was last built on.
3. **A feedback loop I was wrong about.** I hypothesised that refitting the
   reference retriggered ADWIN, and added `DriftMonitor.reset()` to stop it.
   Adaptation counts came back *identical*, so the hypothesis was false. The
   real cause is in the next section. The reset is kept as hygiene but is not
   load-bearing for any reported number, and the code says so.

#### The finding that matters most: SKAB's anomaly-free recording is not stationary

Running `drift-triggered` at **magnitude 0.0** — no injected drift at all — still
produces 18 adaptations, at essentially the same positions as with a 3 sd step
injected:

```
magnitude 0.0: detections [1287, 1575, 2023, 2471, 2759, 3239, 3591, 4135, 5095, ...]
magnitude 3.0: detections [1287, 1575, 2023, 2471, 2759, 3239, 3591, 4135, 4711, ...]
```

The injected change at row 4702 adds exactly one detection, at 4711 — a 9-row
delay, which is genuinely fast. Everything else is the detector responding to
real non-stationarity in the recording. "Anomaly-free" means no annotated
faults; it does not mean a fixed distribution.

**Consequence for the metrics:** `false_alarms_per_1k` on the injected streams
counts those as false alarms, which is unfair — the process really was changing,
just not in a way I have ground truth for. That number is an upper bound on the
spurious-detection rate, not a measurement of it, and the write-up must say so.
Detection *delay* on the injected point is unaffected and remains exact.

#### Calibration: the threshold is per-scenario, and deliberately so

The alarm threshold encodes how often anomalies are expected, and the two stream
families differ by more than an order of magnitude — the injected streams have
no true anomalies, the labelled valve1 stream is ~35% anomalous. One number
cannot serve both; forcing it just yields a badly calibrated detector on one.
So: **q=0.98 on the injected streams** (rare-anomaly operating point), **q=0.50
on the labelled stream**. Both are recorded in the results table.

Sweep outcomes: ADWIN `delta=0.002` (the default) detects 6/6 change points at a
mean 192-step delay; `delta=0.2` reacts faster (123) but nearly doubles the
false-alarm rate. HST `n_trees=25` beats 10 marginally and 50 outright, at half
the runtime of 50. Kept both defaults.

#### Full-run results (real, from `results/metrics.csv`)

Injected streams, alarm rate — every flag here is a false alarm by construction:

| stream | static | online-no-reset | periodic | drift-triggered | delay |
| --- | --- | --- | --- | --- | --- |
| sudden | 0.934 | 0.010 | 0.014 | 0.092 | 9 |
| incremental | 0.933 | 0.007 | 0.007 | 0.009 | 181 |
| gradual | 0.934 | 0.013 | 0.043 | 0.035 | 245 |
| recurring | 0.934 | 0.010 | 0.039 | 0.222 | 20 |

Labelled `skab-valve1` (18160 rows, 34.7% anomalous, flag-everything F1 = 0.516):

| strategy | F1 | alarm rate | beats flag-everything |
| --- | --- | --- | --- |
| static | 0.513 | 0.998 | no |
| online-no-reset | 0.357 | 0.293 | no |
| periodic | **0.544** | 0.523 | yes |
| drift-triggered | 0.502 | 0.509 | no |

**Read these honestly.** The two failure modes are demonstrated cleanly: static
flags 93% of points after a shift, the always-learning control goes silent at
1%. But on the labelled stream, point-wise F1 is a weak discriminator — static
"scores" 0.513 purely by flagging 99.8% of points, essentially the
flag-everything baseline. Only `periodic` beats that baseline, and only just.
The defensible claim is about **alarm volume at comparable F1** (periodic and
drift-triggered reach similar F1 while alarming on half as many points), not
about F1 supremacy. Do not overclaim this in the README.

### 2026-07-21 — Phase 6 complete

`app/streamlit_app.py` runs, was driven end to end in a real browser, and both
clips are exported. Launch it with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

**Use `python -m streamlit`, not `streamlit.exe`.** The console script is blocked
by the same Windows Application Control policy that hit the river and sklearn
DLLs; going through the module works.

#### Demo data is committed, not regenerated

The app reads `data/demo/demo_runs.npz` (1.06 MB, committed), packed by
`scripts/build_demo_cache.py` from the full run. `data/cache/runs_full.pkl` is
9.5 MB and gitignored, so a fresh clone would have had nothing to replay. The
npz holds float32 sensor traces, int8 flags, and the change/adaptation
positions — all genuine model output, only re-packed.

#### Two bugs found by actually driving the UI

1. **The position slider froze at 1000 during playback.** A keyed Streamlit
   slider keeps its own state and ignores the `value` argument on rerun. Dropped
   the key; it now follows the playhead and dragging still works when paused.
2. **The whole dashboard rendered permanently dimmed.** Streamlit fades stale
   elements while a rerun is in flight, and playback reruns continuously, so it
   never stopped fading. Fixed with a CSS override on `[data-stale="true"]`.
   This was visible in the first capture attempt as a grey, washed-out image —
   worth knowing it was a real UI defect, not a screenshot artefact.

#### Both clips exported, and they are not the same thing

- `results/demo_animation.gif` (0.96 MB, 90 frames) — **the reproducible one.**
  `python scripts/export_demo_gif.py` regenerates it from committed data with no
  browser. Shows all four strategies stacked, ending at static 93.4% of points
  flagged, online 1.0%, periodic 1.4%, drift-triggered 9.2%.
- `results/demo.gif` (0.15 MB, 11 frames) — **a real screen capture** of the
  live dashboard, driven via CDP on port 9222. `scripts/stitch_dashboard_gif.py`
  does the stitching; the capture half needs a browser and a running app, so it
  is not unattended-reproducible, and the script says so plainly.

A palette detail worth keeping: the shared GIF palette has to be built from a
montage of *all* frames. Built from frame 1 alone it misses the colours that
only appear later and the plot area comes out grey.

### 2026-07-21 — Phase 8 complete

`src/sweep.py`, `scripts/run_sweep.py`, `notebooks/kaggle_experiment.ipynb`.
64 tests pass. The full sweep — 4 shapes x 5 magnitudes x 2 detectors, 40 runs —
took 202s locally; `results/sweep.csv` holds the real output.

The sweep lives in `src/` and the notebook is a thin driver, so Kaggle runs the
same tested code rather than a pasted copy. The notebook takes SKAB from an
attached Kaggle dataset if it finds one under `/kaggle/input`, otherwise
downloads from GitHub, and writes to `/kaggle/working`. Its analysis and
plotting cells were executed locally against the real sweep output before
committing — no untested notebook code.

#### The control changed the conclusion

**`detection_rate` is 1.0 at every magnitude, including 0.0 where nothing was
injected.** The base recording drifts enough on its own that some firing always
lands within the matching horizon of an arbitrary point. Reported alone it would
have looked like a flawless detector. It is worthless on this data and the
notebook says so outright.

What *is* informative is **delay**, and it responds clearly (ADWIN):

| shape | mag 0.0 (control) | 1.0 | 2.0 | 3.0 | 5.0 |
| --- | --- | --- | --- | --- | --- |
| sudden | 393 | 9 | 9 | 9 | 9 |
| incremental | 373 | 277 | 245 | 181 | 117 |
| gradual | 373 | 245 | 245 | 245 | 245 |
| recurring | 201 | 20 | 20 | 20 | 20 |

- **Sudden and recurring saturate instantly** — caught in 9 and 20 rows even at
  1 sd, and a bigger change does not help because it is already immediate.
- **Incremental is the one where magnitude genuinely matters**, falling
  monotonically 277 → 117 as the ramp steepens.
- **Gradual defeats magnitude entirely**, flat at 245 regardless of size —
  early in the transition the new regime looks like sporadic outliers rather
  than a change in level. This is the shape the injector was written to be hard,
  and it is.
- **ADWIN beats KSWIN on both axes**: shorter delays, and ~1.9 vs ~2.8 false
  alarms per 1000 steps.

Honest limitation, recorded in the notebook: `excess_firings` (firings above the
control) is near zero or **negative** in most conditions. Injecting drift changes
*when* the detector fires far more than *how often*. Negative values are left
unclipped on purpose — clipping them to zero would quietly overstate the method.

### 2026-07-21 — Phase 9 complete. All phases done.

README written and checked against `results/metrics.csv` line by line rather
than from memory. That caught two errors worth recording:

1. I had the flag-everything baseline as **0.516**; it is **0.5122**. The base
   rate is 34.4% over the *scored* region (17,160 rows after warm-up), not 34.7%
   over the full 18,160.
2. Consequently I had written that static does not beat the trivial baseline. It
   does — by **0.0009**. The table now gives the signed margin for each policy
   instead of a yes/no, which is both accurate and more informative: static
   +0.001, periodic +0.032, drift-triggered −0.010, online-no-reset −0.156.

The README states plainly that `drift-triggered` does **not** win the headline
comparison — `periodic` edges it on labelled F1. Its defensible advantage is
reacting on evidence rather than schedule, and a 9-row reaction to a sudden
shift. Overclaiming here would have been easy and wrong.

Checked: no forbidden terms, no emoji, no broken relative links, 64 tests pass,
smoke test passes in 1.0s.

## Definition of done — verified

- [x] installs from `requirements.txt` (Python 3.13)
- [x] `python scripts/run_smoke_test.py` runs end to end in ~1s
- [x] `streamlit run app/streamlit_app.py` launches from committed data
- [x] README shows the demo GIF and real results tables
- [x] every number traces to a committed CSV produced by committed code

## If picking this up again

Ideas deliberately left undone rather than half-built:

- A second dataset (NAB) would test whether any of the calibration transfers.
  Right now nothing shows it does.
- Event-level rather than point-wise anomaly scoring would make F1 a real
  discriminator on `valve1`; the 34% base rate and long contiguous fault blocks
  are what make point-wise F1 nearly useless there.
- The gradual-drift delay (flat at 245 rows regardless of magnitude) is the
  clearest open weakness. A detector on a different statistic — variance or a
  two-window KS test on residuals — might do better where ADWIN cannot.

Watch out: two Application Control blocks have now hit on first import of an
unsigned DLL (`river/_river_rust`, then sklearn's `_radius_neighbors`). Both
cleared on a plain retry. If a fresh import fails this way, retry once before
investigating.
