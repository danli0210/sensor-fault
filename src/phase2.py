"""Phase 2: linear-dynamical generative testbed for sensor-vs-process diagnosis.

The time-correlated normal-process residual covariance is computed from the
stationary state covariance, rather than treating time steps as independent.
All numerical results are synthetic; no empirical measurement claims.

Methods: no query, fixed-best query (trained on separate seeds), random query,
all-hypothesis dispersion query, and source-class-aware query. All querying
policies use the SAME entropy-based case-selection budget.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T=14; D=4; S=8; ONSET=3
BASE=(0,1)
EXTRAS=tuple(range(2,S))
H_NAMES=('process_0','process_1','fault_0','fault_1')

@dataclass
class System:
    A:np.ndarray
    C:np.ndarray
    G:np.ndarray
    Q:np.ndarray
    R:np.ndarray
    signature:np.ndarray # [4,T,8]
    noise_cov:np.ndarray # [T*8,T*8]


def make_system(jitter=0.,rng=None,noise_multiplier=1.):
    if rng is None: rng=np.random.default_rng(0)
    A=np.array([[.71,.09,.02,0],[0,.66,.10,.01],[.01,0,.53,.07],[.02,0,.01,.59]])
    C=np.array([[1,.04,.02,0],[.05,1,.01,0],[.82,.23,.30,-.1],[.16,.88,-.10,.23],
                [.40,-.58,.27,.3],[-.32,.49,.08,.2],[.75,-.31,.18,-.2],[-.17,-.61,.41,-.1]])
    G=np.array([[1.,0.],[0.,1.],[.30,-.15],[-.2,.35]])
    if jitter:
        A=A+rng.normal(0,jitter*.14,A.shape)
        scale=max(1.,np.max(abs(np.linalg.eigvals(A)))/.89)
        A=A/scale
        C=C+rng.normal(0,jitter*.45,C.shape)
        G=G+rng.normal(0,jitter*.12,G.shape)
    Q=np.diag([.012,.014,.009,.01])*noise_multiplier**2
    R=np.diag([.24,.24,.23,.30,.21,.27,.25,.24])*noise_multiplier**2
    R += .017 * noise_multiplier**2 * (np.ones((S,S))-np.eye(S))
    # Deterministic mean process anomaly: an unmeasured forcing step at time ONSET.
    sig=np.zeros((4,T,S))
    for j in range(2):
        x=np.zeros(D)
        for t in range(T):
            x=A@x + (G[:,j] if t >= ONSET else 0)
            sig[j,t]=C@x
    # Sensor faults imitate the local temporal pattern of the corresponding
    # process anomaly. This creates a deliberate but not universal ambiguity.
    sig[2,:,0]=sig[0,:,0]
    sig[3,:,1]=sig[1,:,1]
    P=solve_discrete_lyapunov(A,Q)
    apow=[np.eye(D)]
    for lag in range(1,T):apow.append(A@apow[-1])
    noise=np.zeros((T*S,T*S))
    for t in range(T):
        for u in range(T):
            cross=apow[t-u]@P if t>=u else P@apow[u-t].T
            block=C@cross@C.T
            if t==u:block=block+R
            noise[t*S:(t+1)*S,u*S:(u+1)*S]=block
    noise=(noise+noise.T)*.5
    return System(A,C,G,Q,R,sig,noise)


def idx_for(sensors):
    return np.array([t*S+s for t in range(T) for s in sensors])


class Model:
    def __init__(self, system, amp_std=.22):
        self.sys=system; self.var=amp_std**2
        self.sets={}
        for chosen in (BASE,)+tuple(BASE+(j,) for j in EXTRAS)+(tuple(range(S)),):
            ii=idx_for(chosen)
            cov=system.noise_cov[np.ix_(ii,ii)]
            inv=np.linalg.inv(cov)
            templates=system.signature[:,:,list(chosen)].transpose(1,0,2) if False else \
                system.signature[:,:,list(chosen)].reshape(4,-1)
            # Prior amplitude a~N(1,amp_std^2). Exact Gaussian marginal.
            invs=templates@inv
            s2=np.einsum('hd,hd->h',invs,templates)
            self.sets[chosen]=(ii,inv,templates,invs,s2)
        self.extra_setup={}
        ii,inv,base_s,invs,s2=self.sets[BASE]
        for j in EXTRAS:
            jj=idx_for((j,))
            co=system.noise_cov[np.ix_(jj,ii)]
            proj=co@inv
            rem_cov=system.noise_cov[np.ix_(jj,jj)]-proj@co.T
            rem_cov=(rem_cov+rem_cov.T)*.5
            rem_s=system.signature[:,:,j]-np.einsum('dt,ht->hd',proj,base_s)
            self.extra_setup[j]=(proj,rem_cov,rem_s)

    def posterior(self,ys,chosen=BASE):
        ys=np.atleast_2d(ys)
        ii,inv,st,invs,s2=self.sets[chosen]
        y=ys[:,ii]
        s_y=y@invs.T
        v=1./(1./self.var+s2)
        amean=v[None,:]*(1./self.var+s_y)
        # Log evidence p(y|h) up to common constant using Gaussian integral
        # Exact Gaussian evidence, after marginalizing amplitude a ~ N(1, var).
        # The term -0.5*y^T Sigma^-1*y is common to all hypotheses.
        logit= -.5*(np.log1p(self.var*s2)[None,:] +1./self.var-(s_y+1./self.var)**2*v[None,:])
        p=np.exp(logit-logsumexp(logit,axis=1,keepdims=True))
        return p,amean

    def minimum_whitened_residual(self,ys,chosen=BASE):
        """Minimum generalized least-square reconstruction residual over known hypotheses.

        Diagnostic adequacy statistic (NOT an unknown-fault proof). Under each
        matched hypothesis the unminimized residual obeys chi2(d-1); thresholds
        are empirically calibrated for the whole selection policy.
        """
        yy=np.atleast_2d(ys)
        ii,inv,s,invs,s2=self.sets[chosen]
        y=yy[:,ii]
        norm=np.einsum('nd,df,nf->n',y,inv,y)
        dots=y@invs.T
        amp=np.maximum(dots/np.maximum(s2[None,:],1e-10),0)
        residual=norm[:,None]-2*amp*dots+amp**2*s2[None,:]
        return np.maximum(np.min(residual,axis=1),0.)

    def select(self,ys,prob,amean,method):
        ys=np.atleast_2d(ys)
        n=len(ys)
        if method=='random':
            raise ValueError('Random selection done by seeded generator')
        scores=np.empty((n,len(EXTRAS)))
        yb=ys[:,idx_for(BASE)]
        for pos,j in enumerate(EXTRAS):
            proj,base_cov,rs=self.extra_setup[j]
            # Conditional means for each of four hypotheses, T time samples.
            mean=np.einsum('nh,ht->nht',amean,rs)+(yb@proj.T)[:,None,:]
            var=np.maximum(np.diag(base_cov).mean(),1e-10)
            if method=='dispersion':
                mu=np.einsum('nh,nht->nt',prob,mean)
                scores[:,pos]=np.einsum('nh,nht->n',prob,(mean-mu[:,None,:])**2)/var
            elif method=='class_aware':
                p0=np.clip(prob[:,:2].sum(1),1e-10,1)
                p1=np.clip(prob[:,2:].sum(1),1e-10,1)
                mu0=np.einsum('nh,nht->nt',prob[:,:2],mean[:,:2])/p0[:,None]
                mu1=np.einsum('nh,nht->nt',prob[:,2:],mean[:,2:])/p1[:,None]
                between=(mu0-mu1)**2
                within=(np.einsum('nh,nht->nt',prob[:,:2],(mean[:,:2]-mu0[:,None,:])**2) +
                        np.einsum('nh,nht->nt',prob[:,2:],(mean[:,2:]-mu1[:,None,:])**2))
                scores[:,pos]=np.sum(between/(var+within),axis=1)
            else:raise ValueError(method)
        return np.asarray(EXTRAS)[np.argmax(scores,axis=1)]


def binary_prediction(p):return (p[:,2:].sum(axis=1)>0.5).astype(int)

def entropy(p):return -(p*np.log(np.maximum(p,1e-15))).sum(axis=1)

def draw_dataset(n,system,model,rng,regime):
    # Gaussian baseline noise follows exact dynamic stationary linear-state model.
    sig=system.signature.copy()
    mean_shift=np.zeros((4,T,S))
    if regime=='unknown_fault_shape':
        # New, untrained sensor-fault signatures: piecewise-linear ramp rather
        # than previously modeled transient. Base channels still affected.
        for h,j in ((2,0),(3,1)):
            factor=np.clip((np.arange(T)-ONSET+1)/(T-ONSET),0,1)
            scale=np.max(np.abs(sig[h,:,j]))
            mean_shift[h,:,j]=factor*scale-sig[h,:,j]
    if regime=='auxiliary_fault':
        # Faults shifted to channels 2 and 3, invisible to base sensors.
        for h,old,new in ((2,0,2),(3,1,3)):
            sig[h,:,new]=sig[h,:,old]
            sig[h,:,old]=0
    if regime=='compound':
        # one process event AND one sensor fault; NOT a binary-source label
        sig=np.stack([sig[0]+sig[2],sig[1]+sig[3],sig[0]+sig[3],sig[1]+sig[2]])
    else:
        sig=sig+mean_shift
    labels=rng.integers(0,4,n)
    amplitude=rng.normal(1.,model.var**.5,n)
    L=np.linalg.cholesky(system.noise_cov)
    obs=amplitude[:,None,None]*sig[labels]+(rng.standard_normal((n,T*S))@L.T).reshape(n,T,S)
    return obs.reshape(n,T*S),labels


def query_posteriors(model,ys,old,chosen,requested):
    new=old.copy()
    for j in EXTRAS:
        mask=requested&(chosen==j)
        if np.any(mask):new[mask],_=model.posterior(ys[mask],BASE+(j,))
    return new


def adequacy_statistic(model,ys,selected,requested):
    stat=model.minimum_whitened_residual(ys,BASE)
    for j in EXTRAS:
        mask=requested&(selected==j)
        if np.any(mask):stat[mask]=model.minimum_whitened_residual(ys[mask],BASE+(j,))
    return stat


def policy_masks(model,ys,seed,budget,best_fixed):
    rng=np.random.default_rng(seed)
    p0,amp=model.posterior(ys)
    n=len(ys)
    request=np.zeros(n,bool)
    request[np.argsort(-entropy(p0),kind='stable')[:int(round(budget*n))]]=True
    sel={'direct':np.full(n,best_fixed),'fixed_best':np.full(n,best_fixed),
         'random':rng.choice(EXTRAS,n),
         'dispersion':model.select(ys,p0,amp,'dispersion'),
         'class_aware':model.select(ys,p0,amp,'class_aware')}
    masks={key:(val,np.zeros(n,bool) if key=='direct' else request) for key,val in sel.items()}
    return masks


def calibrate_gate(model,best_fixed,calibration_n=1800,alpha=.01):
    rng=np.random.default_rng(421242)
    yy,_=draw_dataset(calibration_n,model.sys,model,rng,'matched')
    masks=policy_masks(model,yy,62981,.5,best_fixed)
    thresholds={}
    for name,(selected,request) in masks.items():
        ss=adequacy_statistic(model,yy,selected,request)
        # simple held-out empirical upper-tail cutoff, no OOD guarantee
        thresholds[name]=float(np.quantile(ss,1-alpha,method='higher'))
    return thresholds


def gate_dataset(model,ys,labels,seed,best_fixed,thresholds,regime):
    masks=policy_masks(model,ys,seed,.5,best_fixed)
    res={}
    for name,(selected,request) in masks.items():
        statistic=adequacy_statistic(model,ys,selected,request)
        rejected=statistic>thresholds[name]
        res[name]={'rejected_fraction':float(rejected.mean()),
                   'accepted_fraction':float((~rejected).mean()),
                   'threshold':thresholds[name]}
        if regime!='compound':
            p0,_=model.posterior(ys)
            p=p0 if name=='direct' else query_posteriors(model,ys,p0,selected,request)
            truth=(labels>=2).astype(int)
            pred=binary_prediction(p)
            res[name]['accepted_binary_accuracy']=float(np.mean(pred[~rejected]==truth[~rejected])) if np.any(~rejected) else np.nan
            res[name]['confident_wrong_and_accepted']=float(np.mean((pred!=truth)&(~rejected)&(np.max(np.stack([p[:,:2].sum(1),p[:,2:].sum(1)]),axis=0)>.9)))
    return res

def eval_dataset(model,ys,labels,seed,budget=.5,best_fixed=2,regime='matched'):
    rng=np.random.default_rng(seed)
    p0,amp=model.posterior(ys)
    k=int(round(len(ys)*budget))
    requested=np.zeros(len(ys),bool)
    requested[np.argsort(-entropy(p0),kind='stable')[:k]]=True
    sels={'direct':np.full(len(ys),best_fixed),'fixed_best':np.full(len(ys),best_fixed),
          'random':rng.choice(EXTRAS,len(ys)),
          'dispersion':model.select(ys,p0,amp,'dispersion'),
          'class_aware':model.select(ys,p0,amp,'class_aware')}
    results=[]
    for name,ss in sels.items():
        p=p0 if name=='direct' else query_posteriors(model,ys,p0,ss,requested)
        yhat=binary_prediction(p)
        if regime=='compound':
            # No single ground truth class for simultaneous process/fault.
            # Report unjustified confident single-source predictions.
            results.append(dict(policy=name,regime=regime,n=len(ys),
                  confident_single_source_rate=float(np.mean(np.maximum(p[:,:2].sum(1),p[:,2:].sum(1))>=.9)),
                  mean_posterior_confidence=float(np.maximum(p[:,:2].sum(1),p[:,2:].sum(1)).mean()),
                  requested_fraction=float(requested.mean() if name!='direct' else 0)))
        else:
            truth=(labels>=2).astype(int)
            conf=np.maximum(p[:,:2].sum(1),p[:,2:].sum(1))
            results.append(dict(policy=name,regime=regime,n=len(ys),
                  accuracy=float(np.mean(yhat==truth)),
                  macro_recall=float(.5*(np.mean(yhat[truth==0]==0)+np.mean(yhat[truth==1]==1))),
                  high_confidence_wrong=float(np.mean((yhat!=truth)&(conf>=.9))),
                  coverage_90=float(np.mean(conf>=.9)),
                  conditional_risk_90=float(np.mean((yhat!=truth)[conf>=.9]) if np.any(conf>=.9) else np.nan),
                  requested_fraction=float(requested.mean() if name!='direct' else 0)))
    all_p,_=model.posterior(ys,tuple(range(S)))
    if regime!='compound':
        truth=(labels>=2).astype(int)
        results.append(dict(policy='all_eight_oracle',regime=regime,n=len(ys),
               accuracy=float(np.mean(binary_prediction(all_p)==truth)),
               macro_recall=float(.5*(np.mean(binary_prediction(all_p)[truth==0]==0)+np.mean(binary_prediction(all_p)[truth==1]==1))),
               high_confidence_wrong=float(np.mean((binary_prediction(all_p)!=truth)&(np.maximum(all_p[:,:2].sum(1),all_p[:,2:].sum(1))>=.9))),
               coverage_90=float(np.mean(np.maximum(all_p[:,:2].sum(1),all_p[:,2:].sum(1))>=.9)),
               requested_fraction=1.))
    return results


def calibrate_fixed(model,train_n=1000):
    rng=np.random.default_rng(98156)
    yy,labels=draw_dataset(train_n,model.sys,model,rng,'matched')
    truth=(labels>=2).astype(int)
    scores=[]
    for j in EXTRAS:
        p,_=model.posterior(yy,BASE+(j,))
        scores.append(float(np.mean(binary_prediction(p)==truth)))
    return int(EXTRAS[int(np.argmax(scores))]), dict(zip(map(str,EXTRAS),scores))


def run(out:Path,n=400,seeds=5):
    out.mkdir(parents=True,exist_ok=True)
    mod=Model(make_system())
    fixed,fixed_scores=calibrate_fixed(mod)
    regimes=['matched','mismatch_mild','mismatch_strong','higher_noise','unknown_fault_shape','auxiliary_fault','compound']
    gate_thresholds=calibrate_gate(mod,fixed)
    gate_rows=[]
    all_rows=[]
    for regime in regimes:
        for sid in range(seeds):
            rng=np.random.default_rng(302500+sid)
            if regime=='mismatch_mild':sys=make_system(.12,rng)
            elif regime=='mismatch_strong':sys=make_system(.35,rng)
            elif regime=='higher_noise':sys=make_system(0.,rng,1.6)
            else:sys=make_system()
            yy,labels=draw_dataset(n,sys,mod,rng,regime)
            result=eval_dataset(mod,yy,labels,50000+sid,best_fixed=fixed,regime=regime)
            gated=gate_dataset(mod,yy,labels,50000+sid,fixed,gate_thresholds,regime)
            for policy,entry in gated.items():
                gate_rows.append(dict(regime=regime,seed=sid,policy=policy,**entry))
            for row in result:row['seed']=sid;all_rows.append(row)
        print(regime,'done',flush=True)
    with (out/'phase2_by_seed.csv').open('w',newline='') as f:
        fields=sorted({key for row in all_rows for key in row})
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(all_rows)
    with (out/'gate_by_seed.csv').open('w',newline='') as f:
        fields=sorted({key for row in gate_rows for key in row})
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(gate_rows)
    summary={"design":{"T":T,"states":D,"sensors":S,"initial_sensors":list(BASE),"additional_sensor_budget":.5,
              "cases_per_regime":n*seeds,"independent_replicates":seeds,"fixed_sensor_selected_on_training_only":fixed,
              "training_fixed_accuracy_by_sensor":fixed_scores,
              "note":"Synthetic linear dynamical model; Gaussian amplitude prior and exact stationary baseline covariance. All comparison methods use the same active-window selection budget."},
             "results":{},"warnings":["Unknown fault and compound conditions are out of the modeled single-hypothesis dictionary.",
                       "Auxiliary faults are invisible at the initially observed two channels.",
                       "Reported intervals summarize independent seeded synthetic replicates, not distribution-free risk guarantees."]}
    for regime in regimes:
        summary['results'][regime]={}
        for policy in sorted({r['policy'] for r in all_rows if r['regime']==regime}):
            rr=[r for r in all_rows if r['regime']==regime and r['policy']==policy]
            entry={}
            for key in ['accuracy','macro_recall','high_confidence_wrong','coverage_90','conditional_risk_90','confident_single_source_rate','mean_posterior_confidence']:
                vals=np.array([r[key] for r in rr if key in r and np.isfinite(r[key])])
                if len(vals):
                    entry[key]=round(float(vals.mean()),5)
                    if len(vals)>1:
                        se=vals.std(ddof=1)/np.sqrt(len(vals))
                        entry[key+'_CI95']=[round(float(vals.mean()-1.96*se),5),round(float(vals.mean()+1.96*se),5)]
            summary['results'][regime][policy]=entry
    summary['gate_calibration']={'alpha_nominal':.01,'cases':1800,'thresholds':gate_thresholds,
           'note':'Policy-specific 99th percentile thresholds calibrated on separate matched-model synthetic data; no distribution-shift guarantee.'}
    summary['gate_results']={}
    for regime in regimes:
        summary['gate_results'][regime]={}
        for policy in sorted(set(r['policy'] for r in gate_rows)):
            rr=[r for r in gate_rows if r['policy']==policy and r['regime']==regime]
            summary['gate_results'][regime][policy]={}
            for key in ('rejected_fraction','accepted_binary_accuracy','confident_wrong_and_accepted'):
                vals=np.array([r[key] for r in rr if key in r and np.isfinite(r[key])])
                if len(vals): summary['gate_results'][regime][policy][key]=round(float(vals.mean()),5)
    (out/'phase2_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    policies=['direct','fixed_best','random','dispersion','class_aware','all_eight_oracle']
    styles={'direct':'#777777','fixed_best':'#C88B38','random':'#5084B8',
            'dispersion':'#8D67A2','class_aware':'#267B5B','all_eight_oracle':'#333333'}
    regimes_clean=regimes[:-1]
    fig,ax=plt.subplots(figsize=(10,5))
    xx=np.arange(len(regimes_clean)); width=.13
    for j,policy in enumerate(policies):
        data=[summary['results'][reg][policy]['accuracy']*100 for reg in regimes_clean]
        ax.bar(xx+(j-2.5)*width,data,width,label=policy.replace('_',' '),color=styles[policy])
    ax.set_xticks(xx,regimes_clean,rotation=18,ha='right');ax.set_ylabel('Binary accuracy (%)')
    ax.set_ylim(0,101);ax.legend(ncol=3,fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(out/'phase2_robustness.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.7))
    reg=['matched','mismatch_mild','mismatch_strong','higher_noise','unknown_fault_shape']
    for policy in ['direct','fixed_best','random','dispersion','class_aware']:
        vals=[summary['results'][g][policy]['high_confidence_wrong'] for g in reg]
        ax.plot(range(len(reg)),vals,'o-',label=policy.replace('_',' '),color=styles[policy])
    ax.set_xticks(range(len(reg)),reg,rotation=15,ha='right');ax.set_ylabel('Confident incorrect prediction fraction')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'phase2_overconfidence.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.4,4.9))
    reg=['matched','mismatch_mild','mismatch_strong','unknown_fault_shape','auxiliary_fault','compound']
    for pol in ['direct','fixed_best','class_aware']:
        vals=[summary['gate_results'][r][pol]['rejected_fraction'] for r in reg]
        ax.plot(np.arange(len(reg)),vals,'o-',label=pol.replace('_',' '),color=styles[pol])
    ax.set_xticks(np.arange(len(reg)),reg,rotation=15,ha='right')
    ax.set_ylabel('Fraction flagged as not adequately explained')
    ax.set_ylim(0,1.02);ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(out/'phase2_adequacy_gate.png',dpi=170);plt.close(fig)
    print('Fixed sensor:',fixed,flush=True)
    for reg in regimes:
        print(reg, 'accuracy', {p:summary['results'][reg][p].get('accuracy',None) for p in ('direct','fixed_best','class_aware')},
              'gate', {p:summary['gate_results'][reg][p]['rejected_fraction'] for p in ('direct','fixed_best','class_aware')},flush=True)

    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    parser.add_argument('--cases-per-seed',type=int,default=400)
    parser.add_argument('--seeds',type=int,default=5)
    args=parser.parse_args()
    run(args.out,args.cases_per_seed,args.seeds)
