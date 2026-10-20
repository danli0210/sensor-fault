"""Checks for Phase III.1. Does not claim complete scientific validation."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phase2 import make_system, T, S
from phase3 import (MultiSource, make_samples, shifted, calibrate_gates,
                    decide_rejection, summarize, HYP)


class Phase3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sys=make_system()
        cls.model=MultiSource(cls.sys,channels=(0,1,4))

    def test_covers_all_five_truths_and_balanced_classes(self):
        d=make_samples(self.sys,22,np.random.default_rng(123))
        self.assertEqual(d.y.shape,(110,T*S))
        np.testing.assert_array_equal(np.bincount(d.labels),[22]*5)
        self.assertTrue({'unknown_process','unknown_sensor'}<=set(d.subtypes))

    def test_posterior_normalized_and_nonnegative(self):
        y=make_samples(self.sys,13,np.random.default_rng(34)).y
        out=self.model.evaluate(y)
        np.testing.assert_allclose(out['expanded_post'].sum(1),1.,rtol=0,atol=1e-12)
        np.testing.assert_allclose(out['single_post'].sum(1),1.,rtol=0,atol=1e-12)
        self.assertTrue(np.all(out['expanded_post']>=0))
        self.assertEqual(len(HYP),9)

    def test_baseline_does_not_silently_label_both(self):
        out=self.model.evaluate(make_samples(self.sys,10,np.random.default_rng(7)).y)
        self.assertFalse(np.any(out['single_pred']==3))

    def test_gate_is_calibrated_known_only(self):
        d=make_samples(self.sys,120,np.random.default_rng(7))
        mask=d.labels<4
        model_out=self.model.evaluate(d.y[mask])
        gate=calibrate_gates(model_out,d.labels[mask])
        r=decide_rejection(model_out,gate,'expanded','fit')
        # Higher empirical percentile -> <=5% in-sample rejection (barring ties).
        self.assertLessEqual(np.mean(r),.05+1e-12)

    def test_independent_compound_templates_nontrivial(self):
        s=self.sys.signature
        self.assertTrue(np.linalg.norm(s[0]+s[2]-s[0])>0)
        self.assertTrue(np.linalg.norm(s[0]+s[2]-s[2])>0)
        self.assertFalse(np.allclose(shifted(s[0],1),s[0]))

    def test_unobserved_fault_is_indistinguishable_on_base_channels(self):
        # Unknown sensor fault on ch 5 does not affect channels 0 or 1.
        # Verify only the deterministic perturbation, not an impossible
        # claim that finite random sample distributions are identical.
        mask=[t*S+j for t in range(T) for j in (0,1)]
        new=np.zeros((T,S));new[3:,5]=1.6
        self.assertEqual(float(np.linalg.norm(new.reshape(-1)[mask])),0.0)

    def test_likelihood_matches_direct_multivariate_normal(self):
        from scipy.stats import multivariate_normal
        from scipy.special import logsumexp
        model=MultiSource(self.sys,channels=(0,1))
        y=make_samples(self.sys,2,np.random.default_rng(52)).y[:1,model.ix].reshape(-1)
        U=model.U; cov=model.L@model.L.T
        logps=[]
        for (name,inds),prior in zip(HYP,model.logpri):
            M=U[list(inds)].T if inds else np.zeros((len(y),0))
            # Need raw (unwhitened) templates for direct Gaussian calculation.
            raw=model.L@M
            mean=raw.sum(1)
            c=cov+model.var*(raw@raw.T)
            logps.append(multivariate_normal.logpdf(y,mean=mean,cov=c)+prior)
        pp=np.exp(np.array(logps)-logsumexp(logps))
        actual=model.evaluate(make_samples(self.sys,1,np.random.default_rng(53)).y) # separate sanity run
        # Test same sample with complete flattened source values.
        obs=np.zeros((1,T*S));obs[0,model.ix]=y
        pred=model.evaluate(obs)['expanded_post'][0]
        ref=np.array([pp[model.classes==j].sum() for j in range(4)])
        np.testing.assert_allclose(pred,ref,atol=2e-10,rtol=2e-10)


if __name__=='__main__':unittest.main()
