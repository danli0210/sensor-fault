"""Phase III.2: distinguishability + pre-event noise-scale recalibration.

A deliberately restricted, linear-Gaussian research experiment. Not a
field-validated diagnosis method and not a proof of novelty.

Run from project root: python src/phase32.py --systems 24 --cases 80
Outputs results/phase32/. Retains all earlier phase code/results.
"""
from __future__ import annotations
import argparse, csv, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.linalg import solve_discrete_lyapunov, cholesky, solve_triangular
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NSTATE=3; NSENSOR=6; NTIME=9; ONSET=2
NAMES=['normal','process','sensor','both','unknown_process','unknown_sensor']
CHANNEL_CONFIGS={'single':(0,), 'two':(0,1), 'all':tuple(range(NSENSOR))}

@dataclass
class TestSystem:
    index:int
    A:np.ndarray
    C:np.ndarray
    G:np.ndarray
    covariance:np.ndarray
    templates:np.ndarray # (4,NTIME,NSENSOR)
    unknown:np.ndarray # (2,NTIME,NSENSOR)


def make_system(index:int,seed:int=20261020)->TestSystem:
    rng=np.random.default_rng(np.random.SeedSequence([seed,index,71]))
    A=rng.normal(0,0.16,(NSTATE,NSTATE)) + np.diag([.54,.47,.38])
    rad=max(abs(np.linalg.eigvals(A)))
    A=A*(.79/max(.79,rad))
    C=rng.normal(0,.65,(NSENSOR,NSTATE))
    C[0]=np.array([1,.08,0]) + rng.normal(0,.06,NSTATE)
    # Randomize weak or strong cross-channel information, without hand-picking a winner.
    C[1]*=rng.uniform(.15,1.10)
    G=np.array([1,.3,-.15])+rng.normal(0,.2,NSTATE)
    Q=np.diag(rng.uniform(.016,.042,NSTATE))
    z=rng.normal(size=(NSENSOR,NSENSOR)); R=np.diag(rng.uniform(.22,.37,NSENSOR))+.012*z@z.T
    P=solve_discrete_lyapunov(A,Q)
    powers=[np.eye(NSTATE)]
    for i in range(1,NTIME):powers.append(A@powers[-1])
    cov=np.zeros((NTIME*NSENSOR,NTIME*NSENSOR))
    for t in range(NTIME):
        for u in range(NTIME):
            block=C@(powers[t-u]@P if t>=u else P@powers[u-t].T)@C.T
            if t==u:block=block+R
            cov[t*NSENSOR:(t+1)*NSENSOR,u*NSENSOR:(u+1)*NSENSOR]=block
    cov=(cov+cov.T)/2
    process=np.zeros((NTIME,NSENSOR)); x=np.zeros(NSTATE)
    for t in range(NTIME):
        x=A@x+(G if t>=ONSET else 0)
        process[t]=C@x
    sensor=np.zeros_like(process)
    # An intentionally indistinguishable waveform on the first channel alone.
    sensor[:,0]=process[:,0]
    unknown_p=np.zeros_like(process);x=np.zeros(NSTATE)
    g2=rng.normal(size=NSTATE)
    g2-=G*(g2@G)/(G@G)
    g2=g2/max(np.linalg.norm(g2),1e-8)*np.linalg.norm(G)
    for t in range(NTIME):
        x=A@x+(g2 if t>=ONSET else 0)
        unknown_p[t]=C@x
    unknown_f=np.zeros_like(process)
    unknown_f[ONSET:,5]=rng.uniform(1.1,2.0)
    tem=np.stack([np.zeros_like(process),process,sensor,process+sensor])
    return TestSystem(index,A,C,G,cov,tem,np.stack([unknown_p,unknown_f]))


def whitener(sys:TestSystem,channels:tuple[int,...]):
    idx=np.array([t*NSENSOR+s for t in range(NTIME) for s in channels])
    cov=sys.covariance[np.ix_(idx,idx)]
    L=cholesky(cov,lower=True)
    means=sys.templates[:,:,list(channels)].reshape(4,-1)
    mu=solve_triangular(L,means.T,lower=True).T
    unknown=sys.unknown[:,:,list(channels)].reshape(2,-1)
    nu=solve_triangular(L,unknown.T,lower=True).T
    return mu,nu,idx,L


def theory_distance(mu:np.ndarray):
    return float(np.linalg.norm(mu[1]-mu[2]))


def bayes_error_theory(delta:float):
    """Only TWO simple equal-prior Gaussian hypotheses, common covariance."""
    return float(ndtr(-delta/2))


def sample_whitened(mu,nu,n,rng,event_scale=1.,pre_scale=1.,npre=5,pre_corruption=0.,shape_change=0.,return_robust=False):
    """Whitened observation, independent PRE-EVENT baseline windows.

    Event noise has covariance event_scale²*(I + shape_change*uu'), with unit
    vector u supported preferentially on the first observed coordinate.
    For shape_change=0 pre/event covariance differ by scalar only.
    """
    K=len(mu[0]); p=mu.shape[1]
    labels=np.repeat(np.arange(6),n)
    templates=np.vstack([mu,nu])
    means=templates[labels]
    z=rng.standard_normal((len(labels),p))
    if shape_change:
        u=np.zeros(p);u[0]=1
        z+=rng.standard_normal((len(labels),1))*np.sqrt(shape_change)*u
    y=means+event_scale*z
    pre=rng.standard_normal((len(labels),npre,p))*pre_scale
    if pre_corruption:
        flags=rng.random((len(labels),npre))<pre_corruption
        # untrusted pre-event periods with intermittent data corruption
        pre+=flags[:,:,None]*rng.normal(0,3.0,(len(labels),npre,1))
    scalehat=np.sqrt(np.maximum(np.mean(pre**2,axis=(1,2)),1e-9))
    robusthat=np.maximum(np.median(np.abs(pre),axis=(1,2))/.6744897501960817,1e-5)
    perm=rng.permutation(len(labels))
    if return_robust:return y[perm],labels[perm],scalehat[perm],robusthat[perm]
    return y[perm],labels[perm],scalehat[perm]


def predict_and_fit(y,mu):
    # Simple nearest whitened model means. Balanced equal priors.
    scores=np.sum((y[:,None,:]-mu[None,:,:])**2,axis=2)
    return np.argmin(scores,axis=1),np.min(scores,axis=1)


def calibrate_thresholds(systems,seed,n=140,npre=5,coverage=.95):
    """Independent reference known-event calibration. Per-channel pool across systems.
    Excludes unknown events and all evaluation draws.
    """
    rng=np.random.default_rng(np.random.SeedSequence([seed,4444]))
    output={}
    for setting,channels in CHANNEL_CONFIGS.items():
        fixed=[];adaptive=[];robust=[]
        for sys in systems:
            mu,nu,_,_=whitener(sys,channels)
            y,labels,scalehat,robusthat=sample_whitened(mu,nu,n,rng,npre=npre,return_robust=True)
            known=labels<4
            _,fit=predict_and_fit(y[known],mu)
            fixed.extend(fit)
            adaptive.extend(fit/(scalehat[known]**2))
            robust.extend(fit/(robusthat[known]**2))
        output[setting]={'fixed':float(np.quantile(fixed,coverage,method='higher')),
                          'adaptive':float(np.quantile(adaptive,coverage,method='higher')),
                         'robust':float(np.quantile(robust,coverage,method='higher'))}
    return output


def simulate(systems:int=24,cases:int=80,seed:int=20261020,out:Path|None=None):
    if out is None:out=Path(__file__).resolve().parents[1]/'results'/'phase32'
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(np.random.SeedSequence([seed,909]))
    family=[make_system(k,seed) for k in range(systems)]
    # Independent SYSTEM holdout: no calibration system appears in evaluation.
    calibration_family=[make_system(10000+k,seed) for k in range(systems)]
    thresholds=calibrate_thresholds(calibration_family,seed)
    distances=[];rows=[]
    regimes={
      'matched':(1.,1.,0.,0.),
      'stable_noise_1p6':(1.6,1.6,0.,0.),
      'stable_noise_2p0':(2.,2.,0.,0.),
      'abrupt_noise_1p6':(1.6,1.,0.,0.),
      'pre_event_corruption':(1.,1.,.20,0.),
      'covariance_shape_shift':(1.6,1.6,0.,2.),
    }
    for sys in family:
        for setting,channels in CHANNEL_CONFIGS.items():
            mu,nu,_,_=whitener(sys,channels)
            delta=theory_distance(mu)
            theor=bayes_error_theory(delta)
            n_mc=400
            true=rng.integers(0,2,n_mc)
            ypair=mu[true+1]+rng.standard_normal((n_mc,mu.shape[1]))
            v=mu[1]-mu[2]
            pred=np.where((ypair-(mu[1]+mu[2])/2)@v >= 0,1,2)
            empirical=float(np.mean(pred !=true+1))
            distances.append({'system':sys.index,'channels':setting,'delta':delta,
                              'theoretical_pairwise_error':theor,'observed_pairwise_error':empirical})
            for regime,(se,sp,corrupt,shape) in regimes.items():
                y,labels,scalehat,robusthat=sample_whitened(mu,nu,cases,rng,event_scale=se,
                                                    pre_scale=sp,pre_corruption=corrupt,shape_change=shape,return_robust=True)
                pred,fit=predict_and_fit(y,mu)
                known=labels<4;unknown=~known
                for method in ('fixed','adaptive','robust','oracle'):
                    if method=='fixed': stat=fit;thr=thresholds[setting]['fixed']
                    elif method=='adaptive':stat=fit/scalehat**2;thr=thresholds[setting]['adaptive']
                    elif method=='robust':stat=fit/robusthat**2;thr=thresholds[setting]['robust']
                    else:stat=fit/(se**2);thr=thresholds[setting]['fixed']
                    rejected=stat>thr
                    accepted=known&~rejected
                    # Unknown acceptance reported by subtype: unseen channel is
                    # unobservable when channel 5 is not selected.
                    row={'system':sys.index,'channels':setting,'regime':regime,'method':method,
                         'cases':int(len(y)),'scale_hat_mean':float(np.mean(scalehat)), 'robust_scale_hat_mean':float(np.mean(robusthat)),
                         'known_reject':float(np.mean(rejected[known])),
                         'unknown_detect':float(np.mean(rejected[unknown])),
                         'unknown_process_detect':float(np.mean(rejected[labels==4])),
                         'unknown_sensor_detect':float(np.mean(rejected[labels==5])),
                         'known_accepted_error':float(np.mean(pred[accepted]!=labels[accepted])) if np.any(accepted) else np.nan,
                         'known_coverage':float(np.mean(~rejected[known])),
                         'known_class_accuracy':float(np.mean(pred[known]==labels[known]))}
                    rows.append(row)
    def writecsv(path,rows):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    writecsv(out/'per_system.csv',rows)
    writecsv(out/'pairwise_theory.csv',distances)
    agg={}
    for setting in CHANNEL_CONFIGS:
        agg[setting]={}
        for regime in regimes:
            agg[setting][regime]={}
            for method in ('fixed','adaptive','robust','oracle'):
                group=[r for r in rows if (r['channels'],r['regime'],r['method'])==(setting,regime,method)]
                agg[setting][regime][method]={}
                for m in ['known_reject','unknown_detect','unknown_process_detect','unknown_sensor_detect','known_accepted_error','known_coverage','known_class_accuracy','scale_hat_mean','robust_scale_hat_mean']:
                    vals=np.asarray([r[m] for r in group if np.isfinite(r[m])]);
                    if len(vals):
                        agg[setting][regime][method][m]={'mean':float(vals.mean()),'std_across_systems':float(vals.std(ddof=1)) if len(vals)>1 else 0.,
                            'q10':float(np.quantile(vals,.1)),'q90':float(np.quantile(vals,.9))}
    dist_agg={setting:{key:float(np.mean([r[key] for r in distances if r['channels']==setting]))
                       for key in ('delta','theoretical_pairwise_error','observed_pairwise_error')}
              for setting in CHANNEL_CONFIGS}
    result={'metadata':{'systems':systems,'cases_per_class_per_regime_per_system':cases,
       'evaluation_cases':systems*len(CHANNEL_CONFIGS)*len(regimes)*6*cases,
       'seed':seed,'calibration_systems':systems,'evaluation_systems_independent_of_calibration':True,
       'pre_event_windows':5,'target_known_coverage':.95,
       'known_classes':['normal','process','sensor','both'],
       'unknown_subtypes':['unknown_process','unknown_sensor'],
       'limitations':'Fixed known means, stable linear Gaussian dynamics, correctly known baseline covariance shape; oracle uses true event noise scale and is NOT implementable.'},
       'calibration_thresholds':thresholds,'pairwise_summary':dist_agg,'aggregate':agg}
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    make_figures(out,result,distances)
    return result


def make_figures(out,results,distances):
    agg=results['aggregate']; colors={'fixed':'#5a709a','adaptive':'#238066','robust':'#93538d','oracle':'#ae7b39'}
    # fixed / adaptive / oracle across selected operating regimes
    xreg=['matched','stable_noise_1p6','stable_noise_2p0','abrupt_noise_1p6','pre_event_corruption','covariance_shape_shift']
    fig,ax=plt.subplots(figsize=(10,4.8))
    for method in colors:
        vals=[100*agg['all'][r][method]['known_reject']['mean'] for r in xreg]
        ax.plot(range(len(xreg)),vals,'o-',label=method,color=colors[method])
    ax.set_xticks(range(len(xreg)),[r.replace('_','\n') for r in xreg]);ax.set_ylabel('Known-class false rejection (%)')
    ax.set_ylim(bottom=0);ax.grid(alpha=.18);ax.legend()
    fig.tight_layout();fig.savefig(out/'noise_rejection.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for setting,marker in [('single','o'),('two','s'),('all','^')]:
        dat=[r for r in distances if r['channels']==setting]
        ax.scatter([r['delta'] for r in dat],[r['observed_pairwise_error'] for r in dat],alpha=.6,marker=marker,label=setting)
    xx=np.linspace(0,max(8,max(r['delta'] for r in distances)),180)
    ax.plot(xx,ndtr(-xx/2),'k--',label='exact Gaussian Bayes error')
    ax.set(xlabel='Whitened signature distance',ylabel='Pairwise classification error',ylim=(-.03,.56))
    ax.grid(alpha=.17);ax.legend();fig.tight_layout();fig.savefig(out/'distinguishability.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.2,4.5));xs=np.arange(3);names=['single','two','all']; width=.23
    for j,m in enumerate(['fixed','adaptive','oracle']):
        y=[100*agg[s]['stable_noise_1p6'][m]['unknown_process_detect']['mean'] for s in names]
        ax.bar(xs+(j-1)*width,y,width,color=colors[m],label=m)
    ax.set_xticks(xs,names);ax.set_ylim(0,105);ax.set_ylabel('Unknown-process rejection (%)');ax.legend();ax.grid(axis='y',alpha=.16)
    fig.tight_layout();fig.savefig(out/'unknown_process.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--systems',type=int,default=24)
    parser.add_argument('--cases',type=int,default=80)
    parser.add_argument('--seed',type=int,default=20261020)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'results'/'phase32')
    args=parser.parse_args()
    v=simulate(args.systems,args.cases,args.seed,args.out)
    print('Total evaluated windows:',v['metadata']['evaluation_cases'])
    for setting in ('single','two','all'):
        print(setting,'distance',v['pairwise_summary'][setting])
        for regime in ('matched','stable_noise_1p6','abrupt_noise_1p6','pre_event_corruption'):
            print(' ',regime, {m:round(100*v['aggregate'][setting][regime][m]['known_reject']['mean'],2) for m in ('fixed','adaptive','robust','oracle')})
