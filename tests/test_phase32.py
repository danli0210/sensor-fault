"""Independent sanity tests for Phase III.2 assumptions and geometry."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phase32 import make_system,whitener,theory_distance,bayes_error_theory,sample_whitened,predict_and_fit,calibrate_thresholds

class Phase32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sys=make_system(31)

    def test_stable_state_dynamics_and_spd_covariance(self):
        self.assertLess(max(abs(np.linalg.eigvals(self.sys.A))),1)
        self.assertGreater(np.linalg.eigvalsh(self.sys.covariance)[0],0)

    def test_single_sensor_indistinguishable(self):
        mu,nu,_,_=whitener(self.sys,(0,))
        self.assertEqual(theory_distance(mu),0.)
        self.assertAlmostEqual(bayes_error_theory(0),.5)

    def test_additional_observations_do_not_reduce_simple_gaussian_distance(self):
        mu1,*_=whitener(self.sys,(0,))
        mu2,*_=whitener(self.sys,(0,1))
        mu3,*_=whitener(self.sys,tuple(range(6)))
        self.assertLessEqual(theory_distance(mu1),theory_distance(mu2)+1e-9)
        self.assertLessEqual(theory_distance(mu2),theory_distance(mu3)+1e-9)

    def test_unseen_sensor_fault_invisible_without_its_channel(self):
        for subset in ((0,),(0,1)):
            mu,nu,_,_=whitener(self.sys,subset)
            self.assertTrue(np.allclose(nu[1],mu[0]))
        mu,nu,_,_=whitener(self.sys,tuple(range(6)))
        self.assertGreater(np.linalg.norm(nu[1]),0.)

    def test_pre_event_scale_is_not_computed_from_event(self):
        rng=np.random.default_rng(42)
        mu,nu,*_=whitener(self.sys,(0,1))
        _,labels,hat=sample_whitened(mu,nu,300,rng,event_scale=3.,pre_scale=1.)
        self.assertAlmostEqual(float(np.mean(hat)),1.,delta=.04)
        self.assertEqual(len(labels),6*300)

    def test_distinguishability_prediction_matches_monte_carlo(self):
        rng=np.random.default_rng(483)
        mu,nu,*_=whitener(self.sys,(0,1))
        e=bayes_error_theory(theory_distance(mu))
        n=12000;truth=rng.integers(1,3,n)
        x=mu[truth]+rng.normal(size=(n,mu.shape[1]))
        p=np.where((x-(mu[1]+mu[2])/2)@(mu[1]-mu[2])>=0,1,2)
        self.assertAlmostEqual(float(np.mean(p!=truth)),e,delta=.02)

    def test_model_output_and_calibration(self):
        mu,nu,*_=whitener(self.sys,(0,1))
        y,truth,h=sample_whitened(mu,nu,5,np.random.default_rng(11))
        pred,fit=predict_and_fit(y,mu)
        self.assertTrue(np.all((pred>=0)&(pred<4)))
        self.assertTrue(np.all(fit>=0))
        v=calibrate_thresholds([make_system(101),make_system(102)],2026,n=15)
        self.assertGreater(v['two']['fixed'],0)
        self.assertGreater(v['two']['adaptive'],0)

if __name__=='__main__':unittest.main()
