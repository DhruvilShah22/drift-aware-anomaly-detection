# Concept-Drift-Aware Streaming Anomaly Detector

Anomaly detection on sensor streams whose definition of "normal" keeps moving.

This project is under construction. See [PROGRESS.md](PROGRESS.md) for the current
state and the phase checklist.

## The problem

An anomaly detector trained once on a sensor stream degrades as the process it
watches changes. Left alone, it drifts into one of two failure modes: it floods
the operator with false alarms because normal operation has shifted, or it
quietly adapts to a fault and stops reporting it. The interesting question is
not "can we detect anomalies" but "when should the detector update itself".

## What it does

Replays a sensor stream and compares three adaptation strategies side by side:

1. **Static** — trained once, never updated.
2. **Blind periodic retrain** — updates on a fixed schedule.
3. **Drift-triggered** — a drift detector decides when to adapt.

A Streamlit dashboard plays the stream through and shows the signal, detected
drift points, and flagged anomalies live for each strategy.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Built and tested on Python 3.13.
