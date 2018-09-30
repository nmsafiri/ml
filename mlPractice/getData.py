import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

df = pd.read_csv('/home/nmbwamb/Documents/ml/mlPractice/iris.data', header = None)#load data to dataframe  
y = df.iloc[0:100, 4].values #get the values on column 4 row 0 to 100 of the data frame 
y = np.where(y == 'Iris-setosa', -1,1) #rename the classes to use 1 and -1 
X = df.iloc[0:100, [0, 2]].values # extract features out of the main df to contain only two 0 and 2 for samples 0 to 100 
plt.scatter (X[:50,0], X[:50,1], color = 'red', marker = 'o', label = 'sentosa')#visualize all labeled sentosa 
plt.scatter (X[50:,0], X[50:,1], color= 'blue', marker = 'x', label = 'versicolor')#visualize all labeled versicolor 

plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc = 'upper left')
plt.show () 


ppn = Perceptron(eta = 0.1, n_iter = 10) `
