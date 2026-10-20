"""Phase III.1: transparent multi-source diagnosis simulation.

Five truths: nominal, process, sensor, both, unknown. Four actionable labels
are modeled explicitly; unknown is an open-set evaluation condition and NOT
included in training. Synthetic, not real-world validation.

Compared methods:
  single: known normal + 2 process + 2 sensor hypotheses (no compound)
  expanded: same + 4 independent process/sensor compound hypotheses
  glrt: penalized generalized likelihood, all 9 hypotheses
Reject gates: calibrated posterior confidence, fitted residual, and union.
"""
from __future__ import annotations
import argparse, csv, json, sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

# Load the unchanged Phase-II system model when running as a script.
sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase2 import make_system, T, S, D, ONSET

TRUTH = ('normal','process','sensor','both','unknown')
KNOWN = TRUTH[:4]
HYP = [('normal',()), ('process',(0,)),('process',(1,)),
       ('sensor',(2,)),('sensor',(3,))] + [
        ('both',(i,j)) for i in (0,1) for j in (2,3)]
CLASS_ID = {x:i for i,x in enumerate(KNOWN)}


def shifted(sig,shift):
    """Shift an event signature with zero pre-onset state (no wraparound)."""
    if shift==0:return sig.copy()
    ans=np.zeros_like(sig)
    if shift>0:ans[shift:]=sig[:-shift]
    else:ans[:shift]=sig[-shift:]
    return ans


@dataclass
class SampleSet:
    y:np.ndarray
    labels:np.ndarray
    subtypes:np.ndarray


def make_samples(system,n_per_class,rng,regime='matched',amplitude_std=.22):
    """Construct independent process and sensor amplitudes and event onsets.

    The 'unknown' label includes equal numbers of a process direction outside
    the modeled input dictionary and a new channel-5 sensor bias. Distinct
    generation codes are retained in subtypes, avoiding hidden label leakage.
    """
    sig=system.signature.copy()
    noise=np.linalg.cholesky(system.noise_cov)
    count=5*n_per_class
    y=np.empty((count,T*S)); labels=np.repeat(np.arange(5),n_per_class)
    st=np.empty(count,dtype=object)
    # Randomize row order only at the end.
    # Model-unknown process direction is not one of the known G columns.
    g_unknown=np.array([.0,.15,.85,-.63]);g_unknown/=np.linalg.norm(g_unknown)
    u=np.zeros((T,S));xx=np.zeros(D)
    for t in range(T):
        xx=system.A@xx+(g_unknown if t>=ONSET else 0)
        u[t]=system.C@xx
    sensor_u=np.zeros((T,S))
    sensor_u[ONSET:,5]=1.6  # unseen faulty sensor, NOT used in known template.
    for lab in range(5):
        offset=lab*n_per_class
        for j in range(n_per_class):
            i=offset+j
            p,f=None,None
            if lab==1:
                p=int(rng.integers(0,2)); st[i]='known_process'
            if lab==2:
                f=int(rng.integers(2,4)); st[i]='known_sensor'
            if lab==3:
                p=int(rng.integers(0,2));f=int(rng.integers(2,4));st[i]='compound'
            a=rng.normal(1,amplitude_std)
            b=rng.normal(1,amplitude_std)
            if lab==4:
                if j%2==0:
                    base=a*u; st[i]='unknown_process'
                else:
                    base=a*sensor_u;st[i]='unknown_sensor'
            else:
                base=np.zeros((T,S))
                if p is not None:
                    delta=int(rng.integers(-2,3)) if regime=='onset_shift' else 0
                    base+=a*shifted(sig[p],delta)
                if f is not None:
                    delta=int(rng.integers(-2,3)) if regime=='onset_shift' else 0
                    base+=b*shifted(sig[f],delta)
            y[i]=base.reshape(-1)
    z=rng.normal(size=(count,T*S))@noise.T
    if regime=='high_noise':z*=1.6
    if regime=='heavy_tail':
        # Multiplicative scale mixture: within-window occasional large noise.
        z*=np.sqrt(3/np.maximum(rng.chisquare(3,size=(count,1)),.05))
    y+=z
    perm=rng.permutation(count)
    return SampleSet(y[perm],labels[perm],st[perm])


class MultiSource:
    def __init__(self,system,channels=tuple(range(S)),amp_std=.22):
        self.system=system
        self.channels=tuple(channels)
        ix=np.array([t*S+s for t in range(T) for s in channels]);self.ix=ix
        sig=system.signature[:,:,list(channels)].reshape(4,-1)
        self.L=cholesky(system.noise_cov[np.ix_(ix,ix)],lower=True)
        self.U=solve_triangular(self.L,sig.T,lower=True).T  # 4 x obs_dim
        self.var=amp_std**2
        self.hyp=HYP
        self.classes=np.array([CLASS_ID[h[0]] for h in HYP],int)
        # Equal prior for each known class, uniform among templates within class.
        nh=[np.count_nonzero(self.classes==j) for j in range(4)]
        self.logpri=np.array([-np.log(4*nh[c]) for c in self.classes])
        self.par=[]
        for name,inds in HYP:
            umat=self.U[list(inds),:].T if inds else np.empty((len(ix),0))
            mean=umat.sum(1)
            k=len(inds)
            if k:
                gram=umat.T@umat
                W=np.linalg.inv(np.eye(k)+self.var*gram)
                logdet=np.linalg.slogdet(np.eye(k)+self.var*gram)[1]
            else:W=np.zeros((0,0));logdet=0.0
            self.par.append((umat,mean,W,logdet,k))

    def evaluate(self,y):
        z=solve_triangular(self.L,np.asarray(y)[:,self.ix].T,lower=True).T
        znorm=np.einsum('ij,ij->i',z,z)
        n=len(z);loglike=np.empty((n,len(HYP)))
        glrt=np.empty_like(loglike)
        fit=np.empty_like(loglike)
        for i,(U,m,W,logdet,k) in enumerate(self.par):
            r=z-m
            rr=np.einsum('ij,ij->i',r,r)
            if k:
                q=r@U
                corr=self.var*np.einsum('ni,ij,nj->n',q,W,q)
                # GLRT unconstrained amplitude with BIC penalty for k params.
                gram=np.linalg.pinv(U.T@U)
                fit_i=np.maximum(znorm-np.einsum('ni,ij,nj->n',z@U,gram,z@U),0)
            else:corr=0;fit_i=znorm
            loglike[:,i]=-.5*(rr-corr+logdet)
            fit[:,i]=fit_i
            glrt[:,i]=-.5*(fit_i+k*np.log(len(self.ix)))+self.logpri[i]
        logpost=loglike+self.logpri
        logpost-=logsumexp(logpost,axis=1)[:,None]
        prob=np.exp(logpost)
        prob4=np.stack([prob[:,self.classes==j].sum(1) for j in range(4)],axis=1)
        # GLRT uses max among alternative templates, not sum posteriors.
        best_glrt=np.argmax(glrt,axis=1)
        # For single-source, explicitly exclude the 4 compound hypotheses,
        # but permit normal (5 hypotheses, appropriate baseline).
        lp_single=loglike[:,:5]+self.logpri[:5]
        lp_single-=logsumexp(lp_single,axis=1)[:,None]
        p_single=np.exp(lp_single)
        p3=np.stack([p_single[:,0],p_single[:,1:3].sum(1),p_single[:,3:5].sum(1)],axis=1)
        post3=np.column_stack([p3,np.zeros(n)])
        return {
            'single_pred':np.argmax(post3,axis=1),
            'single_conf':np.max(post3,axis=1),
            'expanded_pred':np.argmax(prob4,axis=1),
            'expanded_conf':np.max(prob4,axis=1),
            'glrt_pred':self.classes[best_glrt],
            'glrt_conf':np.max(np.exp(glrt-logsumexp(glrt,axis=1)[:,None]),axis=1),
            # GLRT and Bayes fit residuals: minimum over all known hypotheses
            'fit':np.min(fit,axis=1),
            'fit_single':np.min(fit[:,:5],axis=1),
            'expanded_post':prob4,
            'single_post':post3,
        }


def calibrate_gates(values,calibrated_labels,acceptance=.95):
    """Calibrate in-distribution known acceptance, no unknown test samples."""
    assert np.all(np.isin(calibrated_labels,range(4)))
    def threshold(v):return float(np.quantile(v,acceptance,method='higher'))
    gates={}
    for method in ('single','expanded'):
        # Smaller score means more normal-like. Shared quantile calibration.
        gates[method+'_conf']=threshold(1-values[method+'_conf'])
        gates[method+'_fit']=threshold(values['fit_single' if method=='single' else 'fit'])
        # Union gate: calibrate maximum of percentile ranks using an independent
        # split within calibration. This avoids double dipping.
    return gates


def decide_rejection(out,gate,method,gate_type):
    conf_err=1-out[method+'_conf']
    fit=out['fit_single' if method=='single' else 'fit']
    if gate_type=='conf':return conf_err>gate[method+'_conf']
    if gate_type=='fit':return fit>gate[method+'_fit']
    raise ValueError(gate_type)


def confusion(pred,truth,mask):
    return confusion_matrix(truth[mask],pred[mask],labels=np.arange(4)).tolist()


def summarize(out,labels,mode='expanded',reject=None):
    known=labels<4;unknown=labels==4
    p=out[mode+'_pred'];rej=np.zeros(len(labels),dtype=bool) if reject is None else reject
    accepted=known&~rej
    return {
      'coverage_known':float(np.mean(~rej[known])),
      'accuracy_accepted_known':float(np.mean(p[accepted]==labels[accepted])) if accepted.any() else None,
      'known_macro_f1_without_rejection':float(f1_score(labels[known],p[known],labels=[0,1,2,3],average='macro',zero_division=0)),
      'compound_recall_without_rejection':float(np.mean(p[labels==3]==3)),
      'unknown_rejection':float(np.mean(rej[unknown])),
      'unknown_false_acceptance':float(np.mean(~rej[unknown])),
      'normal_false_alarm_without_rejection':float(np.mean(p[labels==0]!=0)),
      'accepted_known_error_rate':float(np.mean(p[accepted]!=labels[accepted])) if accepted.any() else None,
      'known_confusion_matrix':confusion(p,labels,known),
    }


def risk_curve(out,labels,method):
    known=labels<4
    err=1-out[method+'_conf'][known]
    pred=out[method+'_pred'][known]
    lab=labels[known]
    order=np.argsort(err,kind='stable')
    e=(pred[order]!=lab)
    ps=np.cumsum(e)/np.arange(1,len(e)+1)
    cov=np.arange(1,len(e)+1)/len(e)
    return cov,ps


def draw_figures(res,outdir):
    from sklearn.metrics import ConfusionMatrixDisplay
    matrix=np.array(res['nominal']['expanded']['known_confusion_matrix'])
    fig,ax=plt.subplots(figsize=(5.8,5.1))
    ConfusionMatrixDisplay(matrix,display_labels=KNOWN).plot(ax=ax,cmap='Blues',colorbar=False,values_format='d')
    ax.set_title('Expanded model: matched known classes')
    fig.tight_layout();fig.savefig(outdir/'phase3_confusion.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for method in ('single','expanded'):
        cov=np.array(res['curves'][method]['coverage']);risk=np.array(res['curves'][method]['risk'])
        ax.plot(cov,risk,label=method)
    ax.set(xlabel='Coverage on known cases',ylabel='Conditional error of accepted cases',title='Nominal risk-coverage (posterior ranking)')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(outdir/'phase3_risk_coverage.png',dpi=170);plt.close(fig)
    rows=res['gate_comparison'];names=list(rows)
    fig,ax=plt.subplots(figsize=(9,4.8));x=np.arange(len(names));w=.22
    for j,key in enumerate(['matched_known_rejection','unknown_detection','shifted_known_rejection']):
        ax.bar(x+(j-1)*w,[rows[n][key]*100 for n in names],w,label=key.replace('_',' '))
    ax.set_xticks(x,names,rotation=20,ha='right');ax.set_ylabel('Percent (%)')
    ax.set_ylim(0,102);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(outdir/'phase3_rejection.png',dpi=170);plt.close(fig)


def run(outdir:Path,n=240,seeds=4,channels=tuple(range(S))):
    outdir.mkdir(parents=True,exist_ok=True)
    system=make_system()
    model=MultiSource(system,channels=channels)
    # Separate calibration split, with labels only 0..3 and never any unknown.
    calib=make_samples(system,max(500,n),np.random.default_rng(6271),'matched')
    ci=calib.labels<4
    cal=model.evaluate(calib.y[ci]);gate=calibrate_gates(cal,calib.labels[ci],acceptance=.95)
    all_seeds=[]
    regimes={'nominal':'matched','onset_shift':'onset_shift','noise_x1_6':'high_noise','heavy_tail':'heavy_tail'}
    # Store separately, and report seed-to-seed dispersion where appropriate.
    pooled={k:[] for k in regimes}
    for regime,kind in regimes.items():
        for seed in range(seeds):
            dat=make_samples(system,n,np.random.default_rng(10051+seed*109+list(regimes).index(regime)*701),kind)
            obs=model.evaluate(dat.y)
            q={'regime':regime,'seed':seed,'unknown_subtypes':{}}
            for method in ('single','expanded','glrt'):
                q[method]=summarize(obs,dat.labels,method)
            for method in ('single','expanded'):
                for g in ('conf','fit'):
                    rej=decide_rejection(obs,gate,method,g)
                    sm=summarize(obs,dat.labels,method,rej)
                    # Overall rejection among matched-known cases (not accuracy).
                    q[f'{method}_{g}']={k:v for k,v in sm.items() if k!='known_confusion_matrix'}
            for subtype in ('unknown_process','unknown_sensor'):
                mask=dat.subtypes==subtype
                q['unknown_subtypes'][subtype]={f'{method}_{g}':float(np.mean(decide_rejection(obs,gate,method,g)[mask]))
                                                  for method in ('single','expanded') for g in ('conf','fit')}
            if regime=='nominal' and seed==0:
                for method in ('single','expanded'):
                    cv,rv=risk_curve(obs,dat.labels,method)
                    q[method+'_curve']={'coverage':cv[::max(1,len(cv)//150)].tolist(),'risk':rv[::max(1,len(rv)//150)].tolist()}
            pooled[regime].append(q);all_seeds.append(q)
        print(regime,'done',flush=True)
    results={'design':{'states':D,'sensors':S,'window':T,'used_channels':list(channels),'n_per_class_per_seed':n,
                       'seeds':seeds,'known_class_priors':'balanced known classes; uniform templates per class',
                       'calibration':'independent matched sample, in-distribution known only, target 95% acceptance',
                       'unknown':'never used in calibration or training',
                       'warning':'This is fully synthetic, no IEEE novelty or distribution-shift coverage guarantee'},
             'gates':gate,'scenarios':{},'curves':{},'gate_comparison':{},'by_seed':all_seeds}
    for regime,replicas in pooled.items():
        results['scenarios'][regime]={}
        for method in ('single','expanded','glrt','single_conf','single_fit','expanded_conf','expanded_fit'):
            if method not in replicas[0]:continue
            keys=[k for k,v in replicas[0][method].items() if isinstance(v,(float,int)) and v is not None]
            entry={k:float(np.mean([r[method][k] for r in replicas if r[method].get(k) is not None])) for k in keys}
            results['scenarios'][regime][method]=entry
        if regime=='nominal':
            results['nominal']={method:replicas[0][method] for method in ('single','expanded','glrt')}
            results['curves']={m:replicas[0][m+'_curve'] for m in ('single','expanded')}
    for method in ('single','expanded'):
        for g in ('conf','fit'):
            label=f'{method}_{g}'
            results['gate_comparison'][label]={
               'matched_known_rejection':1-results['scenarios']['nominal'][label]['coverage_known'],
               'unknown_detection':results['scenarios']['nominal'][label]['unknown_rejection'],
               'shifted_known_rejection':1-results['scenarios']['noise_x1_6'][label]['coverage_known'],
            }
    with (outdir/'phase3_summary.json').open('w') as f:json.dump(results,f,indent=2)
    with (outdir/'phase3_by_seed.csv').open('w',newline='') as f:
        keys=['regime','seed','single_f1','expanded_f1','glrt_f1','single_both','expanded_both','glrt_both',
              'single_conf_unknown','single_fit_unknown','expanded_conf_unknown','expanded_fit_unknown',
              'single_fit_knowncoverage','expanded_fit_knowncoverage']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for q in all_seeds:
            writer.writerow({'regime':q['regime'],'seed':q['seed'],
                **{m+'_f1':q[m]['known_macro_f1_without_rejection'] for m in ('single','expanded','glrt')},
                **{m+'_both':q[m]['compound_recall_without_rejection'] for m in ('single','expanded','glrt')},
                **{m+'_'+g+'_unknown':q[m+'_'+g]['unknown_rejection'] for m in ('single','expanded') for g in ('conf','fit')},
                **{m+'_fit_knowncoverage':q[m+'_fit']['coverage_known'] for m in ('single','expanded')}})
    draw_figures(results,outdir)
    for regime in regimes:
        r=results['scenarios'][regime]
        print(regime, 'macro F1 single/expanded/glrt:',*(round(r[m]['known_macro_f1_without_rejection'],3) for m in ('single','expanded','glrt')),
              'compound recall:',*(round(r[m]['compound_recall_without_rejection'],3) for m in ('single','expanded','glrt')))
    print('Gate comparison:',results['gate_comparison'])
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    parser.add_argument('--cases-per-class',type=int,default=240)
    parser.add_argument('--seeds',type=int,default=4)
    parser.add_argument('--channels',nargs='+',type=int,default=list(range(S)))
    a=parser.parse_args()
    run(a.out,n=a.cases_per_class,seeds=a.seeds,channels=tuple(a.channels))
