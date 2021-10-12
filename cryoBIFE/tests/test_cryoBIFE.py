"""
Unit and regression test for the cryoBIFE package.
"""
# Import package, test suite, and other packages as needed
import sys

import pytest

import cryoBIFE

import numpy as np

def test_cryoBIFE_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "cryoBIFE" in sys.modules

# ======== main: crucial to provide a self-test/demo with every function.

def test_inv_energy():

    kappa = 2.0   # prior strength
    M = 20    # nodes
    I = 1000  # images
    G = np.arange(M)/M   # dummy free energy profile
    Pmat = np.random.rand(I,M)   # dummy P_{BioEM} matrix (probabilities in [0,1])
    Neg_Post = cryoBIFE.neglogpost_cryobife(G,kappa,Pmat)
    G = G+1.7     # overall energy shift, note answer invariant
    Neg_Post_shifted = cryoBIFE.neglogpost_cryobife(G,kappa,Pmat)
   
    assert(np.isclose(Neg_Post,Neg_Post_shifted))


# note, to run main:
# import neglogpost_cryobife
# after changes, use:
# import importlib
# importlib.reload(neglogpost_cryobife)

