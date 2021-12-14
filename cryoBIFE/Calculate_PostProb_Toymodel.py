
# Run as python3 Calculate_PostProb_Toymodel.py

import numpy as np
import numpy.random as r


def Post_prob(Path, Gauss_Image_vector, sigma):
    """
    Posterior probability of the gaussian images respect the nodes of a path.
    Parameters
    ----------
    Path : numpy array
           size L*2 numpy matrix of the nodes of a path with length L, in the coordinate space.
    Gauss_Image_vector : numpy array
           size I*2 numpy matrix of the gaussian images I, in the coordinate space.
    sigma : float
           standar deviation of the gaussian model.
    Returns:
    --------
    PostProb_Matrix : numpy array
           size I*L numpy matrix of the probability of each image being generated from each node of the path
    """
    number_of_nodes = Path.shape[0]
    number_of_images = Gauss_Image_vector.shape[0]
    PostProb_Matrix = np.zeros((number_of_images, number_of_nodes))
    for i in range(number_of_nodes):
        Index0 = np.int(Path[i, 0])
        Index1 = np.int(Path[i, 1])
        for j in range(number_of_images):
            PostProb_Matrix[j, i] = (1/(2*np.pi*sigma**2))*np.exp(-((Index0 - Gauss_Image_vector[j, 0])**2 + (Index1 - Gauss_Image_vector[j, 1])**2)/(2*sigma**2))
    return(PostProb_Matrix)


if __name__ == "__main__":

    Total_Images = np.random.randint(50, size=(20, 20))
    nm_x = Total_Images.shape[0]
    nm_y = Total_Images.shape[1]
    Gaussian_Images = []
    for x in range(0, nm_x):
        for y in range(0, nm_y):
            number_of_images = int(Total_Images[x, y])
            for k in range(0, number_of_images):
                dx = np.random.normal(0, 2)
                dy = np.random.normal(0, 2)
                Gaussian_Images.append([x+dx, y+dy])  # Calcuates the Gaussian-Images vector
    Gaussian_Images = np.array(Gaussian_Images)
    sigma = 1.0
    Path = np.random.randint(20, size=(14, 2))
    Post_Matrix_Path = Post_prob(Path, Gaussian_Images, sigma)
    print('Probability Matrix: \n', Post_Matrix_Path)
