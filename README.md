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

## Phase III.1 (v0.3.1): explicit compound hypotheses and open-set rejection

**New:** [PHASE3_REPORT.md](PHASE3_REPORT.md) (Chinese study report). This is a fully
synthetic feasibility/stress-test extension; it is **not** a validated new
algorithm and does not establish journal novelty.

Five evaluation truths: normal, process anomaly, sensor fault, simultaneous
process+sensor causes, and unknown (unknown is not trained as a class).

- `src/phase3.py` — four modeled classes, open-set unknown evaluation,
  9-hypothesis marginal likelihood, BIC-penalized fitted-residual baseline,
  separately calibrated confidence- and fit-based rejectors.
- `src/phase3_summary_plots.py` — aggregate available-channel comparisons.
- `tests/test_phase3.py` — posterior normalization, exact Gaussian likelihood
  cross-check, unknown-channel blindness and calibration tests.
- `results/phase3_2channels`, `results/phase3_3channels`, `results/phase3_all8`
  — independent CSV/JSON outputs and plots for 2, 3, 8 observed channels.

Reproduction:

```bash
python -m pip install -r requirements.txt
python src/phase3.py --out results/phase3_2channels --cases-per-class 240 --seeds 4 --channels 0 1
python src/phase3.py --out results/phase3_3channels --cases-per-class 240 --seeds 4 --channels 0 1 4
python src/phase3.py --out results/phase3_all8 --cases-per-class 240 --seeds 4 --channels 0 1 2 3 4 5 6 7
python src/phase3_summary_plots.py
python -m unittest discover -s tests -v
```

Important: explicitly modeling compound events solves most matched, known
compound cases; open-set rejection fails when the fault is outside observed
channels, and a fixed noise covariance produces extreme false rejection under
noise shift. These negative findings motivate Phase III.2.

## Phase III.2 (v0.3.2): identifiability and pre-event noise-scale calibration

Read **[PHASE32_REPORT.md](PHASE32_REPORT.md)** before interpreting results.

- `src/phase32.py`: 24 independent random evaluation dynamical systems, plus 24 disjoint calibration systems; simple Gaussian pairwise distinguishability and Monte Carlo check; normal/known single/compound and two unknown event sources; fixed, pre-event RMS, pre-event robust median, and oracle rejection thresholds.
- `tests/test_phase32.py`: mathematical invariants and Monte Carlo sanity checks.
- `results/phase32/`: machine-readable per-system CSV, aggregate JSON, and three figures.

Reproduce with:

```bash
python src/phase32.py --systems 24 --cases 80
python -m unittest discover -s tests -v
```

**Limitations:** This new Phase III.2 is a deliberately *simpler*, fixed-amplitude known-mean benchmark than Phase III.1. It is not a replacement for the unknown-amplitude/onset-shift analysis. Under scalar stationary noise shift, simple pre-event scale estimation already largely solves the calibration problem; failure under abrupt shifts, untrusted baseline data and unobserved faults is the open issue. No GitHub Actions, field datasets or novel theoretical claims are implied.
