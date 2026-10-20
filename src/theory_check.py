"""Check elementary Gaussian simple-hypothesis distinguishability formula.

For fixed KNOWN anomaly amplitudes, equal priors and identical covariance,
minimal binary Bayes error is Phi(-delta/2) with Mahalanobis separation delta.
This is established hypothesis-testing theory, NOT claimed as a new theorem.
"""
import csv
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase2 import make_system,BASE,EXTRAS,idx_for,T


def main(out=None,n=16000):
    out=Path(out) if out is not None else Path(__file__).resolve().parents[1]/'results'
    out.mkdir(parents=True,exist_ok=True)
    sys=make_system()
    rows=[]
    rng=np.random.default_rng(60527)
    for pa,fa in ((0,2),(1,3)):
        for chosen in (BASE,)+tuple(BASE+(j,) for j in EXTRAS):
            ids=idx_for(chosen)
            m0=sys.signature[pa][:,chosen].reshape(-1)
            m1=sys.signature[fa][:,chosen].reshape(-1)
            sig=sys.noise_cov[np.ix_(ids,ids)]
            L=np.linalg.cholesky(sig)
            inv_diff=np.linalg.solve(sig,m0-m1)
            delta=np.sqrt((m0-m1)@inv_diff)
            theory=float(ndtr(-delta/2))
            truth=rng.integers(0,2,n)
            obs=np.where(truth[:,None]==0,m0[None,:],m1[None,:])+rng.normal(size=(n,len(ids)))@L.T
            score=(obs-(m0+m1)/2)@inv_diff
            pred=(score<0).astype(int)
            mc=float(np.mean(pred!=truth))
            rows.append(dict(pair=f'P{pa}_F{fa-2}',sensors=str(chosen),n=n,
                             distance=float(delta),theoretical_bayes_error=theory,
                             simulated_bayes_error=mc))
    with (out/'theory_check.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    fig,ax=plt.subplots(figsize=(7,4.6))
    for pair,color in (('P0_F0','#287E65'),('P1_F1','#507CB3')):
        rr=[r for r in rows if r['pair']==pair]
        xx=[r['theoretical_bayes_error'] for r in rr]
        yy=[r['simulated_bayes_error'] for r in rr]
        ax.scatter(xx,yy,label=pair,color=color,s=42)
    ax.plot([0,.27],[0,.27],ls='--',c='grey',label='Theory = simulation')
    ax.set(xlabel='Theoretical optimal binary error',ylabel='Monte Carlo error',title='Simple Gaussian hypothesis test: analytical check')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'phase2_theory_check.png',dpi=170);plt.close(fig)
    return rows
if __name__=='__main__':main()
