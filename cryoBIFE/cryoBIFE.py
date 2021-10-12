"""Provide the primary functions."""
import numpy as np

def neglogpost_cryobife(G,kappa,Pmat):
    """
    Negative log posterior of 1D discretized free-energy profile given a set
    of image probabilities, from cryo-BIFE paper.

    Parameters
    ----------
    G : numpy array
        1d numpy array (size M) of free energy values at the M nodes
    kappa : float
        Strength of log prior (scalar >0)
    Pmat : numpy array
        size I*M numpy matrix of marginalized likelihood of an image
        having derived from each node 1...M of the 1D path on which
        G is discretized. Eg from BioEM or toy Gaussian model. p(w_i|x_m)
        in Eq.(9) of paper.

    Returns
    -------
    neg_log_posterior: float
        negative log posterior of free energy profile given prior and image
        likelihoods (scalar)

    Notes
    -----
    Implements minus the log of Eq.(9) in the paper
    http://dx.doi.org/10.1038/s41598-021-92621-1
    with Z_1 the discrete normalization of rho over the nodes,
    and prior from the Methods (MCMC) section, p.10, except with an extra
    factor of kappa not in that paper.

    Authors: Julian Giraldo-Barreto, Pilar Cossio; docs and rewrite
    by Alex Barnett. 10/8/21
    """
    M = np.size(G)
    mathcalG = sum(np.diff(G)**2)  # note matches paper notation, not confusing
    logprior = kappa * np.log(1/mathcalG**2)    # note kappa scales *log* prior
    rho = np.exp(-G)           # density vec
    rho = rho/np.sum(rho)      # normalize, Eq.(8)
    loglik = np.sum(np.log(np.dot(Pmat,rho)))    # sum here since iid images
    neg_log_posterior = -(loglik + logprior)
    return(neg_log_posterior)             # check logprior sign error?


def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
