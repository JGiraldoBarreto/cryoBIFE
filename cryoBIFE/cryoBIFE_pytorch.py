import torch


def neglogpost_cryobife_pytorch(G, kappa, Pmat, log_prior_fxn=None):
    """
    Calculates the negative log posterior using pytorch.  
    API Matches the  numpy version
    """
    if log_prior_fxn is None:  # Default is the prior from the paper
        log_prior_fxn = normal_prior_pytorch
    log_prior = kappa * log_prior_fxn(G)
    rho = torch.exp(-G)           # density vec
    rho = rho/torch.sum(rho)      # normalize, Eq.(8)
    log_likelihood = torch.sum(torch.log(Pmat @ rho))    # sum here since iid images
    neg_log_posterior = -(log_likelihood + log_prior)
    return(neg_log_posterior)             # check log_prior sign error?


def normal_prior_pytorch(G):
    return - torch.sum(torch.diff(G)**2)


def integrated_prior(G):
    mathcal_G = sum(torch.diff(G)**2)
    log_prior = torch.log(1/mathcal_G**2)    # note kappa scales *log* prior
    return log_prior
