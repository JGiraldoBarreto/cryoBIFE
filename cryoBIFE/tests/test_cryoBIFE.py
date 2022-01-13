"""
Unit and regression test for the cryoBIFE package.
"""
# Import package, test suite, and other packages as needed
import cryoBIFE
import numpy as np
import pytest
import torch


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


def test_torch_numpy_agreement():
    # M = 20    # nodes
    M = 3    # nodes
    Imgs = 10  # images
    G = np.arange(M)/M   # dummy free energy profile
    Pmat = np.random.rand(Imgs, M)   # dummy P_{BioEM} matrix (probabilities in [0,1])
    kappa = 1.0

    numpy_negpost = cryoBIFE.neglogpost_cryobife(G, kappa, Pmat)
    # torch_negpost = cryoBIFE.neglogpost_cryobife_pytorch(torch.from_numpy(G).double(),
    #                                                      kappa,
    #                                                      torch.from_numpy(Pmat).double())
    torch_negpost = cryoBIFE_pytorch.neglogpost_cryobife_pytorch(torch.from_numpy(G),
                                                         kappa,
                                                         torch.from_numpy(Pmat))
    torch_negpost = torch_negpost.numpy()
    assert(np.isclose(numpy_negpost, torch_negpost))
