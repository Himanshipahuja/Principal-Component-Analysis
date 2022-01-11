import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import seaborn as sn
from sklearn.preprocessing import StandardScaler

# loading dataset into Pandas DataFrame
df = pd.read_csv(r'D:\Himanshi work\BE CSBS-Sem 2\UCT201\Pizza.csv')

# split data table into data X and brand of pizza
X  = df.iloc[:,2:9]
Brand = df.iloc[:,0]

#pizza brands
pizza_dict = {1: 'A',2: 'B',3: 'C', 4:'D',5:'E',6:'F',7:'G',8:'H',9:'I',10:'J'}
#columns in dataset
columns_dict = {0: 'mois',1: 'prot',2: 'fat',3: 'ash',4:'sodium',5:'carb',6:'cal'} 

#standardization is done to reduce the differences between the ranges
X_std = StandardScaler().fit_transform(X)

def PCA(X):
    samp_data=X_std
    cov_matrix=np.matmul(samp_data.T,samp_data)
    e_values,e_vectors=eigh(cov_matrix,eigvals=(5,6))
    e_vectors=e_vectors.T
    new_mat=np.matmul(e_vectors,samp_data.T)
    new_mat=np.vstack((new_mat,Brand)).T
    finaldataframe=pd.DataFrame(data=new_mat,columns=["principal_component_1",'principal_component_2','Brand'])
    return finaldataframe

#final dataframe
dfinal=PCA(X_std)
print(dfinal)

#plotting the graph
sn.FacetGrid(dfinal,hue="Brand",height=4).map(plt.scatter,'principal_component_1','principal_component_2').add_legend()
plt.show()
