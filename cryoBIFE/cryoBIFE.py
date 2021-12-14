import numpy as np


def neglogpost_cryobife(G, kappa, Pmat, log_prior_fxn=None):
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
    """
    if log_prior_fxn is None:  # Default is the prior from the paper
        log_prior_fxn = integrated_prior
    log_prior = kappa * log_prior_fxn(G)
    rho = np.exp(-G)           # density vec
    rho = rho/np.sum(rho)      # normalize, Eq.(8)
    log_likelihood = np.sum(np.log(np.dot(Pmat, rho)))    # sum here since iid images
    neg_log_posterior = -(log_likelihood + log_prior)
    return(neg_log_posterior)             # check log_prior sign error?


def integrated_prior(G):
    mathcal_G = sum(np.diff(G)**2)
    log_prior = np.log(1/mathcal_G**2)    # note kappa scales *log* prior
    return log_prior


def normal_prior(G):
    return - sum(np.diff(G)**2)


if __name__ == "__main__":
    kappa = 2.0   # prior strength
    M = 20    # nodes
    num_images = 1000  # images
    G = np.arange(M)/M   # dummy free energy profile
    Pmat = np.random.rand(num_images, M)   # dummy P_{BioEM} matrix (probabilities in [0,1])
    Neg_Post = neglogpost_cryobife(G, kappa, Pmat)
    print("Negative Posterior:", Neg_Post)
