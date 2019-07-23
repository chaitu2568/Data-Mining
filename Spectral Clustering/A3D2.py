import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from A1D1 import eucledian
from scipy.linalg import eig

plotTrue = True
def get_data(file):

    X = file[:, :-1]
    Y = file[:, 2]
    return X, Y


class K_Means:
    def __init__(self, clu=2, con=1e-20, cost = eucledian, no_iter=1000):
        self.clu=clu
        self.no_iter=no_iter
        self.con=con
        self.sse = 0
        self.cost = cost

    def fit(self, d):
        self.centers={}
        # for i in range(self.clu):
        #     self.centers[i]=d[np.random.choice(len(d), replace=False)] #ntializing clusters centres randomly
            # self.centers[i]=d[i] # intializing clusters centres sequentially
        self.centers[0]=d[np.random.choice(len(d), replace=False)]
        for i in range(1,self.clu):
            mai=[]
            for row in d:
                dist=[]
                for center in self.centers:
                    dist.append(self.cost(self.centers[center],row))
                maxi=max(dist)
                mai.append(maxi)
            max_index=mai.index(max(mai))
            self.centers[i]=d[max_index]
        for i in range(1,self.clu):
            self.centers[i]=d[np.random.choice(len(d), replace=False)] #ntializing clusters centres randomly
            # self.centers[i]=d[i] # intializing clusters centres sequentially
        for i in range(self.no_iter):
            self.classes={}
            for i in range(self.clu):
                self.classes[i]=[]
            for row in d:
                space=[]
                for center in self.centers:
                    #cost is a paramter which is given to find Cosine or Eucledian
                    space.append(self.cost(self.centers[center], row))
                center_index=space.index(min(space))
                self.classes[center_index].append(row)
            old_centers=dict(self.centers)
            for i in self.classes:
                if len(self.classes[i])!=0:
                    self.centers[i]=np.mean(self.classes[i], axis=0)#caluculating mean of all the points to form the new centres
            convergence = True
            for i in self.centers:
                actu_center=old_centers[i]
                pre_center=self.centers[i]
                if(np.abs(np.sum(((actu_center - pre_center)/pre_center * 100))) > self.con):
                    convergence=False
            if convergence:
                break
        self.K_index=self.classes
        self.clu_cen=self.centers


    def spectralCall(self, X):
        self.fit(X)
        Y = np.zeros(X.shape[0])
        self.classesTuple = dict()
        for i in range(len(X)):
            j = 0
            for key, values in self.classes.items():
                for val in values:
                    if all(X[i] == val):
                        Y[i] = j
                j+=1
        return Y

class SpectralClustering:
    def __init__(self, file, K=5):
        self.X, self.Y = get_data(file)
        self.K = K
        self.s = 1

    def clustering(self, k=3):
        len = self.X.shape[0]
        pairwise_dists = squareform(pdist(self.X, 'euclidean'))
        pairwise_dists = np.exp(-(pairwise_dists ** 2) / (2 * self.s ** 2))
        self.simMat = pairwise_dists
        for i in range(len):
            dist_with_index = zip(pairwise_dists[i], range(len))
            dist_with_index = sorted(dist_with_index, key=lambda x: x[0])
            k_close = [dist_with_index[m][0] for m in range(self.K)][-1]
            for j in range(len):
                if pairwise_dists[i, j] < k_close:
                    self.simMat[i, j] = -1 * pairwise_dists[i, j]
        for i in range(len):
            for j in range(i, len):
                if self.simMat[i, j] + self.simMat[j, i] < 0:
                    self.simMat[i, j] = self.simMat[j, i] = 0
                elif self.simMat[i, j] + self.simMat[j, i] == 0:
                    self.simMat[i, j] = self.simMat[j, i] = abs(self.simMat[i, j])
        self.degreeMatrix = np.diag(np.array(self.simMat.sum(axis=1)))
        self.laplacianMat = self.degreeMatrix - self.simMat
        eigen_values, eigen_vectors = eig(self.laplacianMat, self.degreeMatrix)
        valuesSort = np.argsort(eigen_values)
        self.transformedData = np.zeros((len, k))
        for i in range(k):
            self.transformedData[:, i] = eigen_vectors[:,valuesSort[i]]
        model = K_Means(clu=k)
        pred = model.spectralCall(self.transformedData)
        print(pred)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=pred)
        plt.title("prediction using spectral clustering")
        plt.show()



colors = 10*["r","b","c"]
coln=['f1','f2','Y']

x=pd.read_csv('d2.csv',names=coln)
cl0=x.loc[x['Y']==0.0]
cl1=x.loc[x['Y']==1.0]
cl2=x.loc[x['Y']==2.0]
cla0=cl0.values[:,:-1]
cla1=cl1.values[:,:-1]
cla2=cl2.values[:,:-1]
plt.scatter(cla0[:,0],cla0[:,1], color='red')
plt.scatter(cla1[:,0],cla1[:,1], color='blue')
plt.scatter(cla2[:,0],cla2[:,1], color='black')
plt.show()
xy=x.values
clutab=xy[:,:-1]
model = SpectralClustering(xy)
model.clustering(k=2)


