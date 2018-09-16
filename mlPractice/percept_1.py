import numpy as np

class Perceptron(object):

  def __init__(self,eta = 0.01, n_iter = 10): #eta: is the learning rate
    self.eta = eta
    self.n_iter = n_iter # n_iter: defines the number times the lerning will happen over the data set 

  def fit(self, X, y): #X is an array defined by number of samples and features 
    self.w_ = np.zeros(1 + X.shape[1]) # creating an array with a size of number of weights + wo
    self.errors_ = []# a list where we will append all occurance of missclassification 



    
