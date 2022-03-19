import numpy as np
import random as r
from numba import jit
from arviz import rhat

@jit
def Gibss(trials, nmodels, nframes, mcsteps, Prob_matrix):

    """ Metropolis/MCMC-based cryo-BIFE algorithm. Provides cryo-bife's Log-posterior, and the infered free energy values, related to the BioEM's
    probability matrix, as described in 10.1038/s41598-021-92621-1.

    :param trials: Number of independent replicas of the Metropolis/MCMC process.
    :param nmodels: Number of models of the structural path.
    :param nframes: Total number of (cryoEM) images.
    :param mcsteps: Number of Montecarlo steps.
    :param Prob_matrix: Array with the probability values of generate an image w from a model m.
                 Shape must be (nframes, nmodels).

    :returns: Array with the infered free energy values and cryo-BIFE logposterior per Montecarlo step.
    """    

    Acc_FE = np.zeros((trials,mcsteps,nmodels))
    Log_Posterior = np.zeros((trials,mcsteps))
    Gstart = np.zeros((trials,nmodels))
    G = np.zeros((mcsteps,nmodels))

    for i in range(trials):

        pold = -9999999999

        Gold = 4.*np.random.random(nmodels)-2.0
        Gnew = np.zeros(nmodels)         
        Gstart[i] = Gold

        for mc in range(mcsteps): 

            prior = 0
            log_posterior = 0
            likelihood = np.zeros(nframes)

            rnum2 = r.randint(0,nmodels)
            dis = -0.5 + r.random()
            Gnew = np.copy(Gold)
            Gnew[rnum2] = Gold[rnum2] + dis

            Gmean = np.sum(Gnew)/nmodels
            Gnew = Gnew - Gmean
            norm = np.sum(np.exp(-Gnew))

            prior = np.sum(np.diff(Gnew)**2)
            log_prior = np.log(1 / prior**2)

            likelihood = np.dot(Prob_matrix, np.exp(-Gnew)/norm)
            log_posterior = np.sum(np.log(likelihood))+ log_prior
           
            rr=np.log(r.random())

            if (log_posterior > pold):

                Log_Posterior[i,mc] = log_posterior

                pold = log_posterior

                Gold = np.copy(Gnew)

            elif (rr < -(pold-log_posterior)):

                Log_Posterior[i,mc] = log_posterior

                pold = log_posterior

                Gold = np.copy(Gnew)

            else:

                Log_Posterior[i,mc] = pold

            G[mc] = Gold 
            #print('Steps',i,mc,Log_Posterior[i,mc],log_posterior,np.log(1/prior**2))

        Acc_FE[i] = G
        
    return(Acc_FE, Log_Posterior)

BioEM = np.loadtxt('Post_Matrix_orange_T_1_S_1')
#BioEM = np.loadtxt('Post_Matrix_black_T_1_S_1')

#Number of independent replicas
trials = 2
#number of nodes along the pathls
nmodels = BioEM.shape[1]
#number of cryo-EM particles
nframes = BioEM.shape[0]
#MC steps
mcsteps = 200000
#Taking samples ever mccor to avoid correlation
mccor= 1
#Where we start averaging (convergence time)
mccut= 200000

Gb, LogProbs = Gibss(trials, nmodels, nframes, mcsteps, BioEM)

Log_hat = rhat(LogProbs)
print('rhat test =', Log_hat)

f = open('Gs','w')

for i in range(trials):
    for k in range(mcsteps):
     
        if k%mccor == 0:
            f.write(' ' + str(i) + ' ' + str(k))

            for j in range(nmodels):
                f.write(' ' + str(Gb[i,k,j]))

            f.write('\n')

f.close()