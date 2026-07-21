"""Live demo: replay a sensor stream and watch each strategy react to drift.

    streamlit run app/streamlit_app.py

Everything shown here is real output from `scripts/run_experiment.py`, re-packed
into `data/demo/demo_runs.npz` by `scripts/build_demo_cache.py`. The app replays
recorded per-step model output rather than running the models live: the flags,
detections and rebuild positions are exactly what the models produced. That
keeps the demo instant on a modest laptop, and means what you see on screen is
the same data behind the numbers in `results/metrics.csv`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from src.evaluate import anomaly_metrics  # noqa: E402

DEMO_PATH = PROJECT_ROOT / "data" / "demo" / "demo_runs.npz"

COLOURS = {
    "static": "#c1440e",
    "online-no-reset": "#8a8a8a",
    "periodic": "#1f6f8b",
    "drift-triggered": "#2a7f4f",
}

STRATEGY_BLURB = {
    "static": "Isolation Forest fitted once, never updated. The naive baseline.",
    "online-no-reset": "Keeps learning every step but is never rebuilt. The control.",
    "periodic": "Rebuilt on a fixed schedule, whether or not anything changed.",
    "drift-triggered": "Rebuilt only when the drift detector fires.",
}

# The sudden step is the clearest shape to look at, so it leads.
DEFAULT_STREAM = "sudden"
DEFAULT_STRATEGY = "drift-triggered"

# Background traces are drawn at most this many points wide. Redrawing 18,000
# points every animation frame is what makes a demo feel sluggish on a weak
# machine, and at this width the difference is invisible.
MAX_BACKGROUND_POINTS = 2500


st.set_page_config(page_title="Drift-aware anomaly detection", layout="wide")


@st.cache_data(show_spinner=False)
def load_demo() -> dict:
    if not DEMO_PATH.exists():
        return {}
    with np.load(DEMO_PATH, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def stream_payload(demo: dict, stream: str) -> dict:
    prefix = f"{stream}|"
    return {
        "signal": demo[prefix + "signal"],
        "labels": demo[prefix + "labels"],
        "change_points": demo[prefix + "change_points"],
        "has_anomalies": bool(demo[prefix + "has_anomalies"][0]),
        "threshold_quantile": float(demo[prefix + "threshold_quantile"][0]),
        "note": str(demo[prefix + "note"][0]),
    }


def strategy_payload(demo: dict, stream: str, strategy: str) -> dict:
    prefix = f"{stream}|{strategy}|"
    return {
        "flags": demo[prefix + "flags"],
        "adaptations": demo[prefix + "adaptations"],
        "detections": demo[prefix + "detections"],
        "eval_start": int(demo[prefix + "eval_start"][0]),
    }


def thin(x: np.ndarray) -> slice:
    """Stride that keeps a background trace under MAX_BACKGROUND_POINTS."""
    return slice(None, None, max(1, len(x) // MAX_BACKGROUND_POINTS))


def draw(
    signal: np.ndarray,
    sensor_name: str,
    position: int,
    flags: np.ndarray,
    change_points: np.ndarray,
    adaptations: np.ndarray,
    detections: np.ndarray,
    strategy: str,
    eval_start: int,
):
    fig, (ax_signal, ax_alarm) = plt.subplots(
        2, 1, figsize=(13, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    x = np.arange(len(signal))
    step = thin(x)

    # Whole stream in pale grey so the shape is always visible, with the part
    # already replayed drawn over it.
    ax_signal.plot(x[step], signal[step], lw=0.5, color="#e0e0e0", zorder=1)
    seen = slice(0, position + 1)
    ax_signal.plot(
        x[seen][thin(x[seen])], signal[seen][thin(x[seen])],
        lw=0.6, color="#2b2b2b", zorder=2,
    )

    flagged = np.where(flags[: position + 1] == 1)[0]
    if flagged.size:
        ax_signal.scatter(
            flagged, signal[flagged], s=7, color=COLOURS[strategy],
            zorder=4, label=f"flagged ({flagged.size})",
        )

    for point in change_points:
        ax_signal.axvline(point, color="#d62728", ls="--", lw=1.2, alpha=0.8, zorder=3)
        ax_alarm.axvline(point, color="#d62728", ls="--", lw=1.2, alpha=0.8)

    for adaptation in adaptations[adaptations <= position]:
        ax_signal.axvline(adaptation, color=COLOURS[strategy], lw=1.0, alpha=0.5, zorder=3)

    ax_signal.axvspan(0, eval_start, color="#f2f2f2", zorder=0)
    ax_signal.axvline(position, color="#111111", lw=1.4, zorder=5)
    ax_signal.set_ylabel(sensor_name, fontsize=9)
    ax_signal.set_xlim(0, len(signal))
    if flagged.size:
        ax_signal.legend(loc="upper left", fontsize=8, frameon=False)

    # Alarm density: fraction of points flagged in a trailing window. This is
    # what makes a flood or a silence obvious at a glance.
    window = 200
    density = np.convolve(flags[: position + 1], np.ones(window) / window, mode="same")
    ax_alarm.fill_between(
        np.arange(len(density)), density, color=COLOURS[strategy], alpha=0.65, lw=0
    )
    ax_alarm.set_ylim(0, 1)
    ax_alarm.set_xlim(0, len(signal))
    ax_alarm.set_ylabel("alarm\ndensity", fontsize=8)
    ax_alarm.set_xlabel("stream position")
    ax_alarm.axvline(position, color="#111111", lw=1.4)

    fig.tight_layout()
    return fig


def main() -> None:
    # While a rerun is in flight Streamlit fades the elements it is about to
    # replace. Playback reruns continuously, so without this the whole dashboard
    # sits permanently dimmed and visibly flickers between frames.
    st.markdown(
        """
        <style>
          [data-stale="true"] { opacity: 1 !important; transition: none !important; }
          [data-test-script-state="running"] .stMain { opacity: 1 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    demo = load_demo()
    st.title("Concept-drift-aware streaming anomaly detection")

    if not demo:
        st.error(
            f"No demo data at `{DEMO_PATH.relative_to(PROJECT_ROOT)}`.\n\n"
            "Generate it with:\n\n"
            "```\npython scripts/run_experiment.py\npython scripts/build_demo_cache.py\n```"
        )
        return

    streams = list(demo["__streams__"])
    strategies = list(demo["__strategies__"])
    sensors = list(demo["__sensors__"])

    with st.sidebar:
        st.header("Stream")
        stream = st.selectbox(
            "Drift shape", streams, index=streams.index(DEFAULT_STREAM)
        )
        payload = stream_payload(demo, stream)
        sensor = st.selectbox("Sensor channel", sensors, index=0)

        st.header("Strategy")
        strategy = st.radio(
            "Adaptation policy", strategies,
            index=strategies.index(DEFAULT_STRATEGY),
            captions=[STRATEGY_BLURB[s] for s in strategies],
        )

        st.header("Playback")
        speed = st.slider("Rows per frame", 10, 500, 120, step=10)

    signal = payload["signal"][:, sensors.index(sensor)]
    n = len(signal)
    run = strategy_payload(demo, stream, strategy)

    # Reset the playhead whenever the stream changes, so switching shapes does
    # not leave the position past the end of a shorter stream.
    if st.session_state.get("stream") != stream:
        st.session_state.stream = stream
        st.session_state.position = run["eval_start"]
        st.session_state.playing = False

    st.session_state.setdefault("position", run["eval_start"])
    st.session_state.setdefault("playing", False)

    controls = st.columns([1, 1, 1, 6])
    if controls[0].button("Play", width="stretch"):
        if st.session_state.position >= n - 1:
            st.session_state.position = run["eval_start"]
        st.session_state.playing = True
    if controls[1].button("Pause", width="stretch"):
        st.session_state.playing = False
    if controls[2].button("Reset", width="stretch"):
        st.session_state.position = run["eval_start"]
        st.session_state.playing = False

    # Deliberately no `key` here. A keyed slider keeps its own state and ignores
    # the `value` argument on rerun, which leaves the handle frozen at its
    # starting position while playback advances. Without a key it follows the
    # playhead, and dragging it still works because the returned value is
    # adopted whenever playback is paused.
    position = st.slider(
        "Stream position", 0, n - 1, int(min(st.session_state.position, n - 1))
    )
    if not st.session_state.playing:
        st.session_state.position = position
    position = int(min(st.session_state.position, n - 1))

    st.caption(
        f"**{stream}** — {payload['note']}. "
        f"Alarm threshold at the {payload['threshold_quantile']:.2f} quantile of "
        f"warm-up scores. Shaded region on the left is the warm-up window, before "
        f"the strategy starts predicting."
    )

    st.pyplot(
        draw(
            signal=signal,
            sensor_name=sensor,
            position=position,
            flags=run["flags"],
            change_points=payload["change_points"],
            adaptations=run["adaptations"],
            detections=run["detections"],
            strategy=strategy,
            eval_start=run["eval_start"],
        ),
        clear_figure=True,
    )

    # ---- live readout -----------------------------------------------------
    start = run["eval_start"]
    seen_flags = run["flags"][start : position + 1]
    seen_labels = payload["labels"][start : position + 1]

    cols = st.columns(4)
    alarm_rate = float(seen_flags.mean()) if seen_flags.size else 0.0
    cols[0].metric("Alarm rate so far", f"{alarm_rate:.1%}")

    if payload["has_anomalies"]:
        metrics = anomaly_metrics(seen_labels, seen_flags)
        cols[1].metric("F1", f"{metrics.f1:.3f}")
        cols[2].metric("Precision", f"{metrics.precision:.3f}")
        cols[3].metric("Recall", f"{metrics.recall:.3f}")
    else:
        # No true anomalies on the injected streams, so every flag is a false
        # alarm and F1 would be zero for everyone. Report what is meaningful.
        cols[1].metric("False alarms", f"{int(seen_flags.sum())}")
        passed = payload["change_points"][payload["change_points"] <= position]
        cols[2].metric(
            "Change points passed", f"{len(passed)}/{len(payload['change_points'])}"
        )
        delay = "—"
        if len(passed):
            after = run["detections"][run["detections"] >= passed[-1]]
            if len(after):
                delay = f"{int(after[0] - passed[-1])} rows"
        cols[3].metric("Delay on last change", delay)

    rebuilds = int((run["adaptations"] <= position).sum())
    st.caption(
        f"Model rebuilds so far: **{rebuilds}**. "
        f"Vertical dashed red lines are injected change points; "
        f"solid coloured lines are model rebuilds."
    )

    with st.expander("How every strategy is doing at this position"):
        rows = []
        for name in strategies:
            other = strategy_payload(demo, stream, name)
            flags = other["flags"][other["eval_start"] : position + 1]
            entry = {
                "strategy": name,
                "alarm rate": f"{flags.mean():.1%}" if flags.size else "—",
                "rebuilds": int((other["adaptations"] <= position).sum()),
            }
            if payload["has_anomalies"]:
                labels = payload["labels"][other["eval_start"] : position + 1]
                entry["F1"] = f"{anomaly_metrics(labels, flags).f1:.3f}"
            rows.append(entry)
        st.dataframe(rows, hide_index=True, width="stretch")

        if not payload["has_anomalies"]:
            st.caption(
                "This stream is built on SKAB's anomaly-free recording, so it "
                "contains no true anomalies and every flag is a false alarm. "
                "Note that the recording is anomaly-free but not stationary: "
                "some rebuilds are the detector responding to real changes in "
                "the underlying process, not to the injected step."
            )

    if st.session_state.playing:
        if position >= n - 1:
            st.session_state.playing = False
        else:
            st.session_state.position = min(position + speed, n - 1)
            time.sleep(0.05)
            st.rerun()


if __name__ == "__main__":
    main()
