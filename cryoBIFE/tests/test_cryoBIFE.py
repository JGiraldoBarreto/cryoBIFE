"""
Unit and regression test for the cryoBIFE package.
"""
# Import package, test suite, and other packages as needed
import cryoBIFE
import numpy as np
import pytest


@pytest.mark.parametrize("kappa", [1.0, 3.0])  # Run for multiple values of prior strength
def test_inv_energy(kappa):
    M = 20    # nodes
    Imgs = 1000  # images
    G = np.arange(M)/M   # dummy free energy profile
    Pmat = np.random.rand(Imgs, M)   # dummy P_{BioEM} matrix (probabilities in [0,1])
    Neg_Post = cryoBIFE.neglogpost_cryobife(G, kappa, Pmat)
    G = G+1.7     # overall energy shift, note answer invariant
    Neg_Post_shifted = cryoBIFE.neglogpost_cryobife(G, kappa, Pmat)

    assert(np.isclose(Neg_Post, Neg_Post_shifted))
