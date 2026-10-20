"""Reproducible synthetic study of sensor-fault vs process-event discrimination.

Four competing hypotheses: two physical/process events (P0, P1), two sensor
faults (F0, F1). Only sensors 0 and 1 stream by default; querying one of
sensors 2..7 is allowed at a limited observation budget.

The Gaussian observation model and waveform are intentionally transparent.
This is a proof of concept, not a claim of novel theory or hardware validation.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = np.array([
    [1.00, 0.10, 0.00, 0.00],
    [0.12, 1.00, 0.00, 0.00],
    [1.10, 0.10, 0.20, 0.00],
    [0.08, 1.05, 0.20, 0.10],
    [0.40, 0.42, 0.30, -0.05],
    [-0.45, 0.15, 0.00, 0.20],
    [0.15, -0.52, -0.12, 0.00],
    [0.55, 0.25, 0.10, 0.12],
], dtype=float)
BASE = (0, 1)
EXTRA = tuple(range(2, 8))
NAMES = ("process_A", "process_B", "sensor_fault_0", "sensor_fault_1")
AMIN, AMAX = 0.65, 1.35


def templates(alias: bool = False) -> np.ndarray:
    """4 x 8: two state-change signatures H@e_i and two sensor bias signatures."""
    s = np.zeros((4, 8))
    s[0] = H[:, 0]
    s[1] = H[:, 1]
    s[2, 0] = 1.0
    s[3, 1] = 1.0
    if alias:  # Negative control: two process states EXACTLY alias sensor faults.
        s[0] = s[2]
        s[1] = s[3]
    return s


def event_waveform(T: int = 16, onset: int = 4, pole: float = 0.70) -> np.ndarray:
    """First-order state step response, also used for adversarially similar drift faults."""
    t = np.arange(T)
    s = np.where(t >= onset, 1.0 - pole ** np.maximum(t - onset + 1, 0), 0.0)
    return s / s.max()


def covariance(sigma: float = 0.85, corr: float = 0.15) -> np.ndarray:
    """Correlated same-time Gaussian measurement noise covariance."""
    return sigma**2 * ((1.0 - corr) * np.eye(8) + corr * np.ones((8, 8)))


def make_data(n: int, rng: np.random.Generator, sig: np.ndarray,
              R: np.ndarray, phi: np.ndarray):
    labels = rng.integers(0, 4, size=n)
    amplitude = rng.uniform(AMIN, AMAX, size=n)
    # Shared sensor-noise at each time; time samples independent.
    noise = rng.standard_normal((n, len(phi), 8)) @ np.linalg.cholesky(R).T
    waveform = amplitude[:, None, None] * phi[None, :, None] * sig[labels, None, :]
    observed = waveform + noise
    # A sufficient statistic for a known temporal waveform with Gaussian noise.
    z = np.einsum("t,ntj->nj", phi, observed) / np.dot(phi, phi)
    return labels, amplitude, z


def posterior(z: np.ndarray, chosen: tuple[int, ...], sig: np.ndarray,
              R: np.ndarray, phi_energy: float, quad: int = 15):
    """Exact discrete-quadrature approximation to amplitude-marginal likelihood.

    All four hypotheses have identical priors; unknown amplitude is uniformly
    distributed over [AMIN, AMAX]. No ground-truth label is used in inference.
    """
    idx = np.asarray(chosen, dtype=int)
    Z = z[:, idx]
    cov = R[np.ix_(idx, idx)] / phi_energy
    inv = np.linalg.inv(cov)
    nodes, weights = np.polynomial.legendre.leggauss(quad)
    aa = (AMAX + AMIN) / 2 + (AMAX - AMIN) / 2 * nodes
    w = weights / 2  # integrates expectation under uniform amplitude prior
    log_scores = []
    amp_means = []
    for signature in sig[:, idx]:
        d = Z[:, None, :] - aa[None, :, None] * signature[None, None, :]
        quadratic = np.einsum("nqi,ij,nqj->nq", d, inv, d, optimize=True)
        ell = -0.5 * quadratic + np.log(w)[None, :]
        normalizer = logsumexp(ell, axis=1)
        log_scores.append(normalizer)
        amp_means.append(np.sum(np.exp(ell - normalizer[:, None]) * aa[None, :], axis=1))
    scores = np.stack(log_scores, axis=1)
    ph = np.exp(scores - logsumexp(scores, axis=1, keepdims=True))
    ah = np.stack(amp_means, axis=1)
    return ph, ah


def binary_probs(ph: np.ndarray) -> np.ndarray:
    return ph[:, :2].sum(axis=1)  # probability of a process event


def binary_prediction(ph: np.ndarray) -> np.ndarray:
    return (binary_probs(ph) < 0.5).astype(int)  # 0=process, 1=sensor fault


def query_selection(z: np.ndarray, ph: np.ndarray, ah: np.ndarray,
                    sig: np.ndarray, R: np.ndarray, phi_energy: float) -> np.ndarray:
    """Uncertainty-weighted conditional inter-hypothesis separation heuristic.

    This is NOT claimed to be exact expected information gain or risk-optimal.
    Sensor selection depends only on already observed channels and model.
    """
    obs = np.array(BASE)
    Coo_inv = np.linalg.inv(R[np.ix_(obs, obs)])
    weighted_means = np.empty((len(z), len(EXTRA), 4))
    noise_var = np.empty(len(EXTRA))
    for k, sensor in enumerate(EXTRA):
        condition = R[sensor, obs] @ Coo_inv
        # Conditional distribution of z_sensor given observed data and hypothesis.
        # z[:,obs] common across hypotheses; amplitude posterior means are used.
        means = (ah * sig[:, sensor][None, :]
                 + (z[:, obs] @ condition)[:, None]
                 - ah * (sig[:, obs] @ condition)[None, :])
        weighted_means[:, k, :] = means
        noise_var[k] = (R[sensor, sensor] - condition @ R[obs, sensor]) / phi_energy
    mean = np.sum(ph[:, None, :] * weighted_means, axis=2)
    dispersion = np.sum(ph[:, None, :] * (weighted_means - mean[:, :, None]) ** 2, axis=2)
    score = dispersion / noise_var[None, :]
    return np.asarray(EXTRA)[np.argmax(score, axis=1)]


def extra_results(z, selected, requested, p0, sig, R, phi_energy):
    p = p0.copy()
    for sensor in EXTRA:
        mask = requested & (selected == sensor)
        if mask.any():
            p[mask], _ = posterior(z[mask], BASE + (sensor,), sig, R, phi_energy)
    return p


def risk_coverage(ph: np.ndarray, truth_binary: np.ndarray):
    confidence = np.maximum(binary_probs(ph), 1 - binary_probs(ph))
    yhat = binary_prediction(ph)
    order = np.argsort(-confidence)
    n = len(order)
    cs_errors = np.cumsum(yhat[order] != truth_binary[order])
    coverage = np.arange(1, n + 1) / n
    selective_risk = cs_errors / np.arange(1, n + 1)
    return coverage, selective_risk


def single_run(seed: int, n: int, R: np.ndarray, phi: np.ndarray,
               sig: np.ndarray, budgets=(0.0, 0.25, 0.5, 0.75, 1.0)):
    rng = np.random.default_rng(seed)
    label, amp, z = make_data(n, rng, sig, R, phi)
    truth = (label >= 2).astype(int)
    energy = np.dot(phi, phi)
    p0, ah = posterior(z, BASE, sig, R, energy)
    entropy = -np.sum(p0 * np.log(np.maximum(p0, 1e-15)), axis=1)
    ranked = np.argsort(-entropy)
    best = query_selection(z, p0, ah, sig, R, energy)
    random_select = rng.choice(EXTRA, size=n)
    fixed_select = np.full(n, 2)
    rows = []
    for budget in budgets:
        nr = int(round(budget * n))
        request = np.zeros(n, dtype=bool)
        request[ranked[:nr]] = True
        for policy, sel in (("direct", np.full(n, 2)),
                            ("fixed_extra", fixed_select),
                            ("random_extra", random_select),
                            ("active_extra", best)):
            if policy == "direct":
                p = p0
                actual_req = np.zeros(n, dtype=bool)
            else:
                p = extra_results(z, sel, request, p0, sig, R, energy)
                actual_req = request
            yhat = binary_prediction(p)
            rows.append({
                "seed": seed, "noise_sigma": float(np.sqrt(R[0,0])),
                "budget": budget, "policy": policy,
                "accuracy": float(np.mean(yhat == truth)),
                "process_recall": float(np.mean(yhat[truth == 0] == 0)),
                "fault_recall": float(np.mean(yhat[truth == 1] == 1)),
                "requested_fraction": float(actual_req.mean()),
                "queries": int(actual_req.sum()),
            })
    return rows, (truth, p0, label, z, best)


def make_figures(out: Path, rows: list[dict], diag, sig, R, phi):
    out.mkdir(parents=True, exist_ok=True)
    seeds = sorted({r["seed"] for r in rows})
    budgets = sorted({r["budget"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = {"direct": "#777777", "fixed_extra": "#C88B38",
              "random_extra": "#5084B8", "active_extra": "#267B5B"}
    for method in ("direct", "fixed_extra", "random_extra", "active_extra"):
        mean, lower, upper = [], [], []
        for b in budgets:
            samples = [r["accuracy"] for r in rows if r["policy"] == method and r["budget"] == b]
            m = np.mean(samples)
            se = np.std(samples, ddof=1) / np.sqrt(len(seeds))
            mean.append(m); lower.append(m - 1.96 * se); upper.append(m + 1.96 * se)
        label = {"direct":"No additional sensor", "fixed_extra":"Fixed sensor 2",
                 "random_extra":"Random sensor", "active_extra":"Adaptive sensor"}[method]
        ax.plot(budgets, mean, "o-", label=label, color=colors[method])
        if method != "direct":
            ax.fill_between(budgets, lower, upper, color=colors[method], alpha=.13)
    ax.set(xlabel="Fraction of windows allowed one extra sensor", ylabel="Binary diagnostic accuracy",
           title="Identifying sensor faults vs system events")
    ax.set_ylim(.45, 1.01);ax.grid(alpha=.25);ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout();fig.savefig(out/"accuracy_vs_budget.png", dpi=180);fig.savefig(out/"accuracy_vs_budget.svg");plt.close(fig)
    truth, p0, label, z, best = diag
    coverage, risk = risk_coverage(p0, truth)
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.plot(coverage, risk, color="#267B5B", lw=2)
    ax.axhline(np.mean(binary_prediction(p0)!=truth), ls="--", color="gray",label="Always diagnose")
    ax.set(xlabel="Fraction of cases automatically diagnosed",ylabel="Error rate among diagnosed cases",
           title="Selective diagnosis (abstention on ambiguous cases)")
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(out/"risk_coverage.png", dpi=180);fig.savefig(out/"risk_coverage.svg");plt.close(fig)
    fig, ax=plt.subplots(figsize=(7.5,4.3))
    for idx, name in enumerate(NAMES):
        inds=(label==idx)
        counts=np.bincount(best[inds]-2, minlength=6)
        ax.plot(np.array(EXTRA), counts / counts.sum(), marker="o", label=name)
    ax.set(xlabel="Candidate verification sensor",ylabel="Fraction selected",
           title="Adaptive verification sensor choice by underlying scenario",xticks=EXTRA)
    ax.grid(alpha=.25);ax.legend(fontsize=9);fig.tight_layout()
    fig.savefig(out/"sensor_choice.png",dpi=180);fig.savefig(out/"sensor_choice.svg");plt.close(fig)


def alias_control(seed: int, n: int, R, phi):
    # Under process A vs sensor fault 0, full-data distributions are identical.
    rng=np.random.default_rng(seed)
    sig=templates(alias=True)
    label, amp, z=make_data(n, rng, sig, R, phi)
    p,_=posterior(z, tuple(range(8)), sig, R, np.dot(phi,phi))
    truth=(label>=2).astype(int)
    return {"all_sensor_binary_accuracy":float(np.mean(binary_prediction(p)==truth)),
            "theoretical_maximum_under_equal_pair_priors":0.5,
            "note":"The process and fault hypotheses are identical for each pair; classification is unidentifiable."}


def execute(output: Path, seeds=8, n=800, sigma=0.85):
    output.mkdir(parents=True, exist_ok=True)
    sig=templates();R=covariance(sigma=sigma);phi=event_waveform()
    results=[];sample=None
    for i in range(seeds):
        r, diag=single_run(24000+i,n,R,phi,sig)
        results.extend(r)
        if sample is None:sample=diag
    make_figures(output, results, sample, sig, R, phi)
    with (output/"metrics_by_seed.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f, fieldnames=list(results[0]));wr.writeheader();wr.writerows(results)
    summary={
        "design":{"hypotheses":list(NAMES),"seeds":seeds,"cases_per_seed":n,
                  "noise_sigma":sigma,"sensor_count":8,"always_observed":[0,1],
                  "additional_candidates":list(EXTRA),"amplitude_prior":[AMIN,AMAX],
                  "waveform_length":len(phi),"waveform_energy":float(phi@phi),
                  "note":"All observations simulated; labels and Gaussian likelihood use the same specified model."},
        "results":[],"alias_negative_control":alias_control(593,4000,R,phi)
    }
    for budget in sorted({r["budget"] for r in results}):
        for policy in ("direct","fixed_extra","random_extra","active_extra"):
            samples=[r for r in results if r["budget"]==budget and r["policy"]==policy]
            mean=np.mean([r["accuracy"] for r in samples]);std=np.std([r["accuracy"] for r in samples],ddof=1)
            summary["results"].append({"budget":budget,"policy":policy,
                "mean_accuracy":round(float(mean),5),
                "between_seed_95pct_CI":[round(float(mean-1.96*std/np.sqrt(seeds)),5),
                                        round(float(mean+1.96*std/np.sqrt(seeds)),5)],
                "mean_process_recall":round(float(np.mean([r["process_recall"] for r in samples])),5),
                "mean_fault_recall":round(float(np.mean([r["fault_recall"] for r in samples])),5)})
    (output/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print("Experiment complete:",seeds*n,"cases")
    for r in summary["results"]:
        if r["budget"]==0.5:print(r)
    print("Alias negative control:",summary["alias_negative_control"])
    return summary

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]/"results")
    ap.add_argument("--seeds",type=int,default=8)
    ap.add_argument("--cases-per-seed",type=int,default=800)
    ap.add_argument("--sigma",type=float,default=0.85)
    args=ap.parse_args()
    execute(args.out,args.seeds,args.cases_per_seed,args.sigma)
