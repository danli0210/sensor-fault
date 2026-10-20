# Sensor Fault vs. Process Anomaly: Distinguishability-Aware Verification

**Status: synthetic proof-of-concept (v0.1); NOT a validated IEEE submission or a real SHM dataset.**

This experiment addresses the following question: **When two different causes produce nearly identical measurements on currently active sensors, does requesting one carefully chosen additional sensor help distinguish a sensor fault from a physical/process event more reliably than random or fixed requests?**

## Mathematical setup

- 4-dimensional latent system state and 8 sensor channels; the sensor observation matrix `H` is specified in `src/simulate.py`.
- Four mutually exclusive event hypotheses: process changes in states 0 or 1 (`P0`, `P1`) and biases on sensors 0 or 1 (`F0`, `F1`). Only one event occurs per trial.
- The latent process events have spatial sensor signatures `H[:, 0]` and `H[:, 1]`; sensor faults have one-hot spatial signatures. Each event is multiplied by the same first-order step response to create an intentionally **difficult fault-versus-process distinction**.
- For a time-window index `t`, observations are

  `y_t = a * phi_t * s_h + epsilon_t`,

  where the unknown amplitude `a` is uniform over `[0.65, 1.35]`, `h` is the hypothesis, `epsilon_t` is temporally independent Gaussian noise with known cross-channel covariance `R`, and `phi_t` is the given time signature.
- Only channels 0 and 1 are initially available. An optional query observes exactly one channel from 2 through 7 for a selected fraction of windows.
- For each hypothesis, the unknown event amplitude is integrated out using 15-point Gauss-Legendre quadrature. The four posterior hypothesis probabilities are pooled into two final decisions: process event vs sensor fault.
- Windows with high hypothesis entropy are allowed to request one additional channel. The proposed **heuristic** selects the sensor with the largest posterior-weighted conditional inter-hypothesis mean dispersion per unit residual-noise variance. **It is not a proven optimal EIG strategy.**

## Files

| Path | Purpose |
|---|---|
| `src/simulate.py` | Main Gaussian model, decision policies, Monte Carlo trials, negative control, and plots |
| `src/robustness.py` | Additional noise and model-mismatch trials |
| `tests/test_simulation.py` | Probability, aliasing, and sensor-choice sanity tests |
| `results/summary.json` | Aggregated accuracy with between-seed 95% normal-approximation intervals |
| `results/metrics_by_seed.csv` | Per-seed and per-observation-budget statistics |
| `results/robustness_summary.json` | Noise/mismatch sensitivity results |
| `results/*.png` and `results/*.svg` | Reproducible figures |
| `RESEARCH_PLAN.md` | Hypotheses, scientific limitations, and recommended next steps |

## Reproduction

```bash
python -m pip install -r requirements.txt
python src/simulate.py --seeds 8 --cases-per-seed 800 --sigma 0.85
python src/robustness.py
python -m unittest discover -s tests -v
```

The main trial runs **6,400 independently generated cases** with 8 seeded replicates, equal prior probability for four hypotheses, and five budgets (0%, 25%, 50%, 75%, 100%). At the **50% extra-sensor budget**, the current generated output is:

| Policy | Mean binary accuracy | Between-seed approximate 95% CI |
|---|---:|---:|
| No extra sensor | 58.13% | 56.66–59.59% |
| Fixed sensor 2 | 67.77% | 66.51–69.03% |
| Random extra sensor | 67.22% | 65.79–68.65% |
| Adaptive extra sensor | **77.36%** | **76.40–78.31%** |

For equal prior mass on exactly aliased P0/F0 and P1/F1 hypotheses, the theoretical best possible accuracy for the *binary source label* is 50% regardless of the number of sensors. The negative-control simulation produces approximately **50.48%**. This validates the *logical limitation*, not the generality of the proposed approach.

## Interpretation and limitations

1. **Matched statistical assumptions**: the main experiment uses the same signature dictionary, amplitude prior, waveform, and Gaussian covariance for both generation and inference. This makes it deliberately favorable to model-based diagnosis.
2. **Process physics is simplified**: there is no full time-varying structural dynamics, temperature effects, unknown loads, sensor calibration error, or coupled failure evolution. "4D state" denotes the observation space design, not a validated dynamic physical system.
3. **Strong priors**: a fixed, known set of event types and only one active hypothesis per trial. Unexpected events, multi-fault cases and unseen disturbances are not modeled.
4. **Auxiliary sensors are assumed healthy**. An auxiliary sensor fault could invalidate a verification result and is an important next experiment.
5. **Risk control has NOT been proven**. Selecting high-entropy windows and selecting a sensor by approximate dispersion are heuristics. Reported accuracy intervals describe repeated simulated samples, not guaranteed worst-case diagnosis risk.
6. **The scenario is intentionally ambiguous** on the first two channels and informative on some unused channels. This is a designed proof-of-concept, not evidence of universal improvements.
7. The mismatch stress test perturbs only true process spatial patterns while keeping the model fixed. It does not establish distribution-free robustness.
8. All source code and figures generated here are original for this study. There are no claims of first-in-literature novelty.

## Candidate technical contribution after deeper work

Jointly characterize **observational distinguishability**, a **statistically calibrated abstention decision**, and **budget-constrained active sensor verification**, comparing against robust residual tests, sparse-fault observers, and established active diagnosis methods. A rigorous theoretical result and public *real* sensor data would be needed to support a serious IEEE TIM / IoT-J submission.
