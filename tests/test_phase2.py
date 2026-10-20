"""Phase-2 tests. Run: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from phase2 import make_system,Model,BASE,EXTRAS,draw_dataset,policy_masks,calibrate_fixed,adequacy_statistic,binary_prediction

class Phase2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sys=make_system()
        cls.model=Model(cls.sys)

    def test_system_stable_and_covariance_pd(self):
        self.assertLess(np.max(np.abs(np.linalg.eigvals(self.sys.A))),1.)
        np.linalg.cholesky(self.sys.noise_cov)

    def test_probability_normalization(self):
        rng=np.random.default_rng(12)
        yy,_=draw_dataset(50,self.sys,self.model,rng,'matched')
        for sensors in (BASE,BASE+(2,),tuple(range(8))):
            p,a=self.model.posterior(yy,sensors)
            self.assertTrue(np.all(np.isfinite(p)))
            self.assertTrue(np.allclose(p.sum(1),1.,atol=1e-9))
            self.assertTrue(np.all(np.isfinite(a)))

    def test_exact_marginal_likelihood(self):
        from scipy.stats import multivariate_normal
        rng=np.random.default_rng(101)
        yy,_=draw_dataset(2,self.sys,self.model,rng,'matched')
        for sensors in (BASE,BASE+(5,)):
            ids=self.model.sets[sensors][0]
            sigma=self.sys.noise_cov[np.ix_(ids,ids)]
            s=self.model.sets[sensors][2]
            scipy_scores=np.array([multivariate_normal.logpdf(yy[0,ids],mean=m,cov=sigma+self.model.var*np.outer(m,m)) for m in s])
            scipy_p=np.exp(scipy_scores - np.logaddexp.reduce(scipy_scores))
            p,_=self.model.posterior(yy[:1],sensors)
            self.assertTrue(np.allclose(p[0],scipy_p,atol=1e-8))

    def test_observations_and_residual_finite(self):
        rng=np.random.default_rng(11)
        yy,_=draw_dataset(40,self.sys,self.model,rng,'matched')
        stat=self.model.minimum_whitened_residual(yy)
        self.assertTrue(np.all(np.isfinite(stat)))
        self.assertTrue(np.all(stat>=0))

    def test_active_sensor_candidates(self):
        rng=np.random.default_rng(11)
        yy,_=draw_dataset(40,self.sys,self.model,rng,'matched')
        p,a=self.model.posterior(yy)
        for m in ('class_aware','dispersion'):
            selected=self.model.select(yy,p,a,m)
            self.assertTrue(set(selected).issubset(set(EXTRAS)))

    def test_gate_flags_large_unexplained_outlier(self):
        rng=np.random.default_rng(10)
        y,_=draw_dataset(1,self.sys,self.model,rng,'matched')
        bad=y.copy()
        # Unmodeled simultaneous large discontinuities at two base sensors
        for t in range(5,14):
            bad[0,t*8+0]+=10
            bad[0,t*8+1]-=12
        self.assertGreater(self.model.minimum_whitened_residual(bad)[0],self.model.minimum_whitened_residual(y)[0]+100)

if __name__=='__main__':unittest.main()
