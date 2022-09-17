import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

class compress_image():

    def __init__(self,img,method) -> None:
        self.img = img
        self.method = method


    def compress(self):
        if self.method == 'PCA':
            return self.pca()
        elif self.method == 'KNN':
            return self.knn()
        else:
            print('Invalid method')

    def pca(self):
        
        # center the image
        centered_img = self.img - np.mean(self.img, axis = 0)

        # compute the covariance matrix
        cov = np.cov(centered_img , rowvar = False)

        eigen_values , eigen_vectors = np.linalg.eigh(cov)

        rescon_loss = []

        #sort the eigenvalues in descending order
        sorted_index = np.argsort(eigen_values)[::-1]
        
        sorted_eigenvalue = eigen_values[sorted_index]
        #similarly sort the eigenvectors 
        sorted_eigenvectors = eigen_vectors[:,sorted_index]

        n = self.img.size

        for k in range(1,500):   # Ideally k search space is from 1 to n
            # select the top k eigenvectors
            Vk = sorted_eigenvectors[:,0:k]

            # project the data onto the k eigenvectors
            img_proj = np.dot(self.img,Vk)

            # reconstruct the image
            img_recon = np.dot(img_proj,Vk.T)

            # find relative error
            rescon_loss.append(np.linalg.norm(self.img - img_recon)/np.linalg.norm(self.img))
    
        best_k = np.argmin(rescon_loss) + 1
        print('Best dimensions = ',best_k)

        fig = px.line(x = range(1,500), y = rescon_loss)
        fig.update_layout(
            title="Relative Reconstruction loss vs number of components",
            xaxis_title="Number of components",
            yaxis_title="Relative Reconstruction loss",
        )
        fig.show()

        # compress the image
        Vk = sorted_eigenvectors[:,0:best_k]
        compressed_image = np.dot(self.img,Vk)
        reconstructed_image = np.dot(compressed_image,Vk.T)


        return compressed_image,reconstructed_image
