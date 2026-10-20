"""Fast sanity tests (run with: python -m unittest discover -s tests -v)."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from simulate import (templates, covariance, event_waveform, make_data, posterior,
                      query_selection, BASE, EXTRA)


class SimulationTests(unittest.TestCase):
    def test_valid_probability_and_amplitude(self):
        phi=event_waveform();r=covariance();s=templates();
        _, _,z=make_data(30,np.random.default_rng(42),s,r,phi)
        p,a=posterior(z,BASE,s,r,float(phi@phi))
        np.testing.assert_allclose(p.sum(axis=1),1.0,atol=1e-12)
        self.assertTrue(np.all(p>=0))
        self.assertTrue(np.all((a>=0.65-1e-10)&(a<=1.35+1e-10)))

    def test_active_choice_for_two_different_process_modes(self):
        phi=event_waveform();r=covariance();s=templates()
        z=np.stack([s[0],s[1]])
        p,a=posterior(z,BASE,s,r,float(phi@phi))
        chosen=query_selection(z,p,a,s,r,float(phi@phi))
        self.assertEqual(int(chosen[0]),2)
        self.assertEqual(int(chosen[1]),3)

    def test_exact_alias_has_no_additional_identifiability(self):
        sig=templates(alias=True)
        np.testing.assert_allclose(sig[0],sig[2])
        np.testing.assert_allclose(sig[1],sig[3])
        phi=event_waveform();r=covariance()
        _,_,z=make_data(18,np.random.default_rng(9),sig,r,phi)
        ph,_=posterior(z,tuple(range(8)),sig,r,float(phi@phi))
        np.testing.assert_allclose(ph[:,0],ph[:,2],atol=1e-12)
        np.testing.assert_allclose(ph[:,1],ph[:,3],atol=1e-12)

    def test_event_and_fault_are_nearly_aliased_on_base_channels(self):
        sig=templates()
        self.assertLess(np.linalg.norm(sig[0,list(BASE)]-sig[2,list(BASE)]),0.13)
        self.assertGreater(np.linalg.norm(sig[0]-sig[2]),1.0)
        self.assertEqual(len(EXTRA),6)

if __name__=="__main__":unittest.main()
