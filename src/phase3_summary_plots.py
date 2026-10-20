"""Aggregate the three Phase III.1 channel-availability experiments."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(root):
    root=Path(root)
    nums=[2,3,8]
    arr=[json.loads((root/f'phase3_{n}channels'/'phase3_summary.json').read_text()) if n!=8 else
         json.loads((root/'phase3_all8'/'phase3_summary.json').read_text()) for n in nums]
    fig,ax=plt.subplots(figsize=(8,5))
    for method in ('single','expanded','glrt'):
        yy=[a['scenarios']['nominal'][method]['known_macro_f1_without_rejection']*100 for a in arr]
        ax.plot(nums,yy,'o-',label=method)
    ax.set(xlabel='Number of available sensor channels',ylabel='Macro F1 of four known classes (%)',
           title='Known source diagnosis vs. observable channels',xticks=nums,ylim=(45,103))
    ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(root/'phase3_known_f1_by_channels.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for method in ('expanded_conf','expanded_fit'):
        yy=[a['gate_comparison'][method]['unknown_detection']*100 for a in arr]
        ax.plot(nums,yy,'o-',label=method.replace('_',' '))
    ax.set(xlabel='Number of available sensor channels',ylabel='Unknown-event rejection rate (%)',
           title='Open-set detection under calibrated known-data acceptance',xticks=nums,ylim=(-1,102))
    ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(root/'phase3_unknown_by_channels.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for method in ('expanded_conf','expanded_fit'):
        yy=[a['gate_comparison'][method]['shifted_known_rejection']*100 for a in arr]
        ax.plot(nums,yy,'o-',label=method.replace('_',' '))
    ax.set(xlabel='Number of available sensor channels',ylabel='Known-event rejection under noise ×1.6 (%)',
           title='Noise-model mismatch exposes calibration fragility',xticks=nums,ylim=(-1,102))
    ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(root/'phase3_noise_rejection_by_channels.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    run(Path(__file__).resolve().parents[1]/'results')
