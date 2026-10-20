# Sensor Fault vs. Process Anomaly — Synthetic Research Simulations

**Research prototype, version 0.2** | Status: simulated feasibility/robustness study, not a validated fault-diagnosis method.

This reproducible project studies when an apparent anomaly comes from a **measurement sensor** rather than from the monitored **physical/industrial process**. The central Phase-II finding is that *an unmodeled simultaneous process anomaly and sensor failure can induce a confidently wrong forced single-source diagnosis*, and that simple detection of unmodeled events is very sensitive to noise-model mismatch.

**Read [PHASE2_REPORT.md](PHASE2_REPORT.md) first.** The original Phase-I results and caveats are retained in [PHASE1_README.md](PHASE1_README.md).

## Contents

- `src/simulate.py`, `src/robustness.py` — Phase I: static spatial-signature simulations.
- `src/phase2.py` — Phase II: 4-state, 8-sensor dynamic system with time-correlated noise, active verification, unknown faults and compound events.
- `src/theory_check.py` — Validate a standard Gaussian hypothesis-testing error formula by independent Monte Carlo simulations.
- `tests/` — Phase-I and Phase-II sanity/likelihood tests.
- `results/phase2_summary.json` — Complete machine-readable results and parameters.
- `results/phase2_by_seed.csv` — Per-seed accuracy, recall, false-confidence and coverage information.
- `results/gate_by_seed.csv` — Per-seed fit-gate rejection and conditional-accuracy data.
- `results/phase2_*.png` — Publication-preparation figures (NOT a claim of journal-ready evidence).

## Run

Python 3.10+ recommended.

```bash
python -m pip install -r requirements.txt
python src/phase2.py --cases-per-seed 400 --seeds 5
python src/theory_check.py
python -m unittest discover -s tests -v
```

To rerun Phase I:

```bash
python src/simulate.py --seeds 8 --cases-per-seed 800
python src/robustness.py
```

## Limitations

1. All data are mathematically generated; no hardware or real measurements have been included.
2. The inference model assumes four known single-source mechanisms. This is intentionally violated in the stress tests, and the main contribution is to identify the resulting reliability gap.
3. The two-class inference task is only meaningful for single-source cases. Simultaneous process events and sensor faults are recorded as **compound**, not assigned an artificial binary truth label.
4. Policy-specific model-fit thresholds are calibrated on synthetic nominal conditions and are **not distribution-free robustness guarantees**.
5. Auxiliary sensor failures, mismatched process models, different noise levels, and uncertainty over fault onset remain important challenges.
6. Best-fixed is tuned on a *separate* dataset; all active policies have matched request quotas. All-eight is an upper reference with more observations.
7. The results should not be represented as establishing IEEE-level novelty without comparison to modern observer-based fault isolation and active diagnosis methods.

## Maintainer notes

No user credentials, device serial numbers, private monitoring data, or GitHub tokens are included. This project is separate from embedded data-acquisition firmware.
