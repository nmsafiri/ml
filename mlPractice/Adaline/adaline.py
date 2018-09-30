import numpy as np 
import pandas as pd 


class AdalineGD (object): 
  #Adaptive Linear Neuron  Classifier 
  #constructor 
  def __init__(self, eta = 0.01, n_iter = 50):
     self.eta = eta #learning rate 
     self.n_iter = n_iter #number of iterations through the data set 

  def fit(self, X, y):
     self.w_ = np.zeros(1+X.shape[1])
     self.cost_ = []
     for i in range (self.n_iter):
        output = self.net_input(X)
        errors = (y-output)
        print ("this is ",output)
        self.w_[1:] +=self.eta*X.T.dot(errors)
        self.w_[0] += self.eta*errors.sum()
        cost = (errors**2).sum() /2.0
        self.cost_.append(cost)
     return self

  def net_input(self,X):
     return np.dot(X, self.w_[1:]) + self.w_[0]

  def activation(self, X):
     return self.net_input (X)

  def predict(self, X):
     return np.where(self.activation(X) >= 0.0, 1, -1)

df = pd.read_csv('/home/nmbwamb/Documents/ml/mlPractice/iris.data', header = None)
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa', 1, -1) # create the target values 
X = df.iloc[0:100, [0,2]].values # create the samples with only two features 

ada1 = AdalineGD(eta = 0.01, n_iter = 10).fit(X,y)

