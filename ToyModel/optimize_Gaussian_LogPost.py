# Run as python3 optimize_Gaussian_LogPost.py sigma_value Matrix_with_number_of_images_per_model (e.g python3 optimize_Gaussian_LogPost.py 0.4 Num_Images_grid-3well)
# The necessary files must be in the same folder

import sys
import numpy as np
import numpy.random as r
import scipy.optimize as so
import matplotlib.pyplot as plt

#### The following 3 libraries are included in the same folder this file is. Docs within each file. ####

from cryoBIFE.Calculate_PostProb_Toymodel import Post_prob
from cryoBIFE import neglogpost_cryobife, normal_prior, integrated_prior
from cryoBIFE.Generate_Gaussian_Images_Toymodel import Gaussian_Images, sample_grid_data

##### MAIN #####

factor = 1  # Scaling factor of the nuber of images (See Docs in "Generate_Gaussian_Images_Toymodel.py").
kappa = 1.0  # Prior scaling factor (See Docs in "neglogpost_cryobife.py").
sigma = np.float(sys.argv[1])  # Standard deviation of the gaussian model (See Docs in "Generate_Gaussian_Images_Toymodel.py").
Num_images = np.loadtxt(sys.argv[2])  # Matrix with number of images per model (See Docs in "Generate_Gaussian_Images_Toymodel.py").

# ERIK: TODO: replace these relative imports with something more principled...
Black = np.loadtxt('../cryoBIFE/data/Black')  # Black path in coordinate space.
Black = Black - np.ones((len(Black), 2))  # correction to match python notation.
number_of_nodes = Black.shape[0]  # Number of nodes of Black path.

Orange = np.loadtxt('../cryoBIFE/data/Orange')  # Orange path in coordinate space.
Orange = Orange - np.ones((len(Orange), 2))  # correction to match python notation.

Grid = np.loadtxt('../cryoBIFE/data/grid')  # Coordinate points representing all the models of the system. 400 models going from the point (1,1) to (20,20).
Grid = Grid - np.ones((len(Grid), 2))  # correction to match python notation.

G_Vec = Gaussian_Images(Grid, Num_images, factor, sigma)  # Function to generate the gaussian images from the toy system model. (See Docs in "Generate_Gaussian_Images_Toymodel.py").
Post_Matrix_Black = Post_prob(Black, G_Vec, sigma)  # Generate the probability matrix comparing each image in G_Vec, whit each node of the Black Path. (See Docs in "Calculate_PostProb_Toymodel.py").
Post_Matrix_Orange = Post_prob(Orange, G_Vec, sigma)  # Same as above, but using Orange path instead Black path.

# np.savetxt('Prob_Matrix_Black',Post_Matrix_Black)
# np.savetxt('Prob_Matrix_Orange',Post_Matrix_Orange)

G_old = np.zeros(number_of_nodes)  # Array of initial random Free energy values.
for x in range(0, number_of_nodes):  # Initial G values to start the minimize methods.

    G_old[x] = 4.*r.random()-2.0

##### LogPost Black Path #####

G_op1 = []  # Saves the G optimum values.
max_Like_val = 0  # Saves the LogPosterior values for optim. method

G_op = so.minimize(neglogpost_cryobife, G_old, method='L-BFGS-B', args=(kappa, Post_Matrix_Black,))  # Scipy minimization function, using the L-BFGS-B method and an initial vector with random values of G.

G_op1.append(G_op.x)
G_op1 = np.array(G_op1)
G_op1 = G_op1.reshape(14)
max_Like_val = -1*G_op.fun  # Here, we return the Likelihood to the correct sign after minimization, and save it.

mathcal_G_op1 = sum(np.diff(G_op1)**2)  # note matches paper notation, not confusing
log_prior_G_op1 = kappa * np.log(1/mathcal_G_op1**2)    # note kappa scales *log* prior
rho_G_op1 = np.exp(-G_op1)           # density vec
rho_G_op1 = rho_G_op1/np.sum(rho_G_op1)      # normalize, Eq.(8)
log_likelihood_G_op1 = np.sum(np.log(np.dot(Post_Matrix_Black, rho_G_op1)))    # sum here since iid images

print('Path: Black', '|', 'Sigma: ', sigma, '|', 'LogPosterior:', max_Like_val, '|', 'Av. Probability per model:', np.sum(Post_Matrix_Black)/Post_Matrix_Black.shape[0], '|', 'Sum of Prob. Values:', np.sum(Post_Matrix_Black), '|', 'Calculated LogPost:', -1*neglogpost_cryobife(G_op1, kappa, Post_Matrix_Black), '|', 'Calculated Prior:', log_prior_G_op1, '|', 'Calculated Likelihood:', log_likelihood_G_op1, '|', 'Max Probability value:', np.max(Post_Matrix_Black), '|', 'Min. Probability value:', np.min(Post_Matrix_Black), '\n')  # ,'\n Free Energy:',G_op1,'\n') # Visualizing the outputs.

###### FE plot from Black path results ######

plt.figure(figsize=(6.5, 6))
#plt.plot([i for i in range(len(G_op1))],G_op1-np.min(G_op1), 'o-')
plt.plot([i for i in range(len(G_op1))], G_op1, 'o-')
plt.title('Free enegy Projection - Toy Model $\sigma =$%s | Black Path' % sigma)
plt.xlabel('Path CV', fontsize=14)
plt.ylabel('Free energy', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid()
plt.savefig('Black_FE_sigma_%s.png' % sigma, bbox_inches='tight')
plt.close()

##### LogPost Orange Path #####

G_op2 = []  # Saves the G optimum values.
max_Like_val2 = 0  # Saves the LogPosterior values for optim. method

G_op = so.minimize(neglogpost_cryobife, G_old, method='L-BFGS-B', args=(kappa, Post_Matrix_Orange,))  # Scipy minimization function, using the L-BFGS-B method and an initial vector with random values of G.

G_op2.append(G_op.x)
G_op2 = np.array(G_op2)
G_op2 = G_op2.reshape(14)
max_Like_val2 = -1*G_op.fun  # Here, we return the Likelihood to the correct sign after minimization, and save it.

mathcal_G_op2 = sum(np.diff(G_op2)**2)  # note matches paper notation, not confusing
log_prior_G_op2 = kappa * np.log(1/mathcal_G_op2**2)    # note kappa scales *log* prior
rho_G_op2 = np.exp(-G_op2)           # density vec
rho_G_op2 = rho_G_op2/np.sum(rho_G_op2)      # normalize, Eq.(8)
log_likelihood_G_op2 = np.sum(np.log(np.dot(Post_Matrix_Orange, rho_G_op2)))    # sum here since iid images


print('Path: Orange', '|', 'Sigma: ', sigma, '|', 'LogPosterior:', max_Like_val2, '|', 'Av. Probability per model:', np.sum(Post_Matrix_Orange)/Post_Matrix_Orange.shape[0], '|', 'Sum of Prob. Values:', np.sum(Post_Matrix_Orange), '|', 'Calculated LogPost:', -1*neglogpost_cryobife(G_op2, kappa, Post_Matrix_Orange), 'Calculated Prior:', log_prior_G_op2, '|', 'Calculated Likelihood:', log_likelihood_G_op2, '|', 'Max Probability value:', np.max(Post_Matrix_Orange), '|', 'Min. Probability value:', np.min(Post_Matrix_Orange), '\n')  # ,'Free Energy:',G_op2,'\n') # Visualizing the outputs.

###### FE plot from Orange path results ######

plt.figure(figsize=(6.5, 6))
#plt.plot([i for i in range(len(G_op2))],G_op2-np.min(G_op2), 'o-')
plt.plot([i for i in range(len(G_op2))], G_op2, 'o-')
plt.title('Free enegy Projection - Toy Model $\sigma =$%s | Orange Path' % sigma)
plt.xlabel('Path CV', fontsize=14)
plt.ylabel('Free energy', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid()
plt.savefig('Orange_FE_sigma_%s.png' % sigma, bbox_inches='tight')
plt.close()

###### 2D histogram of the Gaussian Images ######

plt.figure(figsize=(8.9, 7.0))

plt.gca().set_facecolor('navy')
plt.title('2D FE Landscape $\sigma =$%s' % sigma, fontsize=30)
plt.xlabel('CMA', fontsize=25)
plt.ylabel('CMB', fontsize=25)

plt.hist2d(G_Vec[:, 1], G_Vec[:, 0], bins=(40, 40), cmap=plt.cm.jet)
plt.plot(Black[:, 1], Black[:, 0], 'o-', color='black', linewidth=4)
plt.plot(Orange[:, 1], Orange[:, 0], 'o-', color='orange', linewidth=4)
plt.colorbar()
plt.xlim([1, 18.5])
plt.savefig('2d_histo_gaussian_images_sigma_%s.png' % sigma)
plt.close()

###### 2D histogram of the Gaussian Images ######

z1 = Post_Matrix_Black
Y = [i for i in range(Post_Matrix_Black.shape[0])]
X = [i for i in range(Post_Matrix_Black.shape[1])]
fig, ax1 = plt.subplots(figsize=(15, 13))
ax1.contour(X, Y, z1)  # z1-np.min(z1), levels=14, linewidths=0.5, colors='k')
cntr1 = ax1.contourf(X, Y, z1)  # -np.min(z1), levels=14, cmap="RdBu_r")
plt.xlabel('CMA', fontsize=15)
plt.ylabel('CMB', fontsize=15)
plt.grid()
fig.colorbar(cntr1, ax=ax1)
plt.savefig('ColorMap_Prob_Black_sigma_%s.png' % sigma)
plt.close()


###### 2D histogram of the Gaussian Images ######

z1 = Post_Matrix_Orange
Y = [i for i in range(Post_Matrix_Orange.shape[0])]
X = [i for i in range(Post_Matrix_Orange.shape[1])]
fig, ax1 = plt.subplots(figsize=(15, 13))
ax1.contour(X, Y, z1)  # z1-np.min(z1), levels=14, linewidths=0.5, colors='k')
cntr1 = ax1.contourf(X, Y, z1)  # -np.min(z1), levels=14, cmap="RdBu_r")
plt.xlabel('CMA', fontsize=15)
plt.ylabel('CMB', fontsize=15)
plt.grid()
fig.colorbar(cntr1, ax=ax1)
plt.savefig('ColorMap_Prob_Orange_sigma_%s.png' % sigma)
plt.close()
