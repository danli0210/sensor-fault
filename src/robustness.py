"""Stress test sensitivity to observation noise and signature-model mismatch."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from simulate import (templates, covariance, event_waveform, make_data, posterior,
                      query_selection, extra_results, binary_prediction, BASE)


def run(out: Path, n=600, seeds=5):
    out.mkdir(parents=True, exist_ok=True)
    phi=event_waveform(); E=float(phi@phi); nominal=templates()
    records=[]
    for sigma in (0.55,0.85,1.20):
        R=covariance(sigma)
        for mismatch in (0.0,0.20,0.40):
            for run_id in range(seeds):
                seed=35000+1000*round(sigma*100)+100*round(mismatch*10)+run_id
                rng=np.random.default_rng(seed)
                actual=nominal.copy()
                # Perturb process-event patterns only, never use perturbation in inference.
                actual[:2]=nominal[:2]+mismatch*rng.normal(size=(2,8))*np.abs(nominal[:2]).max(axis=1)[:,None]
                label,amp,z=make_data(n,rng,actual,R,phi)
                truth=(label>=2).astype(int)
                p0,ah=posterior(z, BASE, nominal,R,E)
                entropy=-np.sum(p0*np.log(np.clip(p0,1e-15,1)),axis=1)
                request=np.zeros(n,dtype=bool)
                request[np.argsort(-entropy)[:n//2]]=True
                active=query_selection(z,p0,ah,nominal,R,E)
                random_sel=rng.choice(np.arange(2,8),size=n)
                for strategy, sel in [("direct",np.full(n,2)),("random_extra",random_sel),("active_extra",active)]:
                    if strategy=="direct":p=p0
                    else:p=extra_results(z,sel,request,p0,nominal,R,E)
                    records.append({"sigma":sigma,"signature_mismatch":mismatch,"seed":seed,
                                    "policy":strategy,"accuracy":float(np.mean(binary_prediction(p)==truth))})
    with (out/"robustness_by_seed.csv").open("w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    summaries=[]
    for sigma in (0.55,0.85,1.20):
        for mismatch in (0.0,0.20,0.40):
            subset={r["policy"]:[x["accuracy"] for x in records if x["policy"]==r["policy"] and x["sigma"]==sigma and x["signature_mismatch"]==mismatch] for r in records}
            summaries.append({"sigma":sigma,"signature_mismatch":mismatch,
                **{key:round(float(np.mean(value)),4) for key,value in subset.items()}})
    (out/"robustness_summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    for r in summaries:print(r)
    return summaries

if __name__=="__main__":
    run(Path(__file__).resolve().parents[1]/"results")
