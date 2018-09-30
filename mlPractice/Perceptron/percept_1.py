import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from  matplotlib.colors import ListedColormap

class Perceptron(object):

  def __init__(self,eta = 0.01, n_iter = 10): #eta: is the learning rate
    self.eta = eta
    self.n_iter = n_iter # n_iter: defines the number times the lerning will happen over the data set 

  def fit(self, X, y): #X is an array defined by number of samples and features 
    self.w_ = np.zeros(1 + X.shape[1]) # creating an array with a size of number of weights + wo
    self.errors_ = []# a list where we will append all occurance of missclassification 
    for _ in range(self.n_iter):
        errors = 0
        for xi, target in zip (X,y):
            update = self.eta * (target - self.predict(xi))# eta * (y-y') on each iteration 
            self.w_[1:] += update * xi # all weights of the given sample gets updated automaticall on @ iter
            self.w_[0] += update #update wo
            errors += int(update != 0.0)
        self.errors_.append(errors)# create a list of errors in  missclassifictions 
    return self
  
  def net_input(self, X):
    return np.dot(X, self.w_[1:]) + self.w_[0]

  def predict(self,X):
    return np.where(self.net_input(X) >= 0.0, 1, -1)


df = pd.read_csv('/home/nmbwamb/Documents/ml/mlPractice/iris.data', header = None)
y = df.iloc[0:100,4].values #create a list of classes 
y = np.where(y == 'Iris-setosa', 1, -1) # change class names to either 1 or -1 
X = df.iloc[0:100, [0,2]].values # create the samples with only two features 


ppn = Perceptron (eta = 0.1, n_iter = 10) # create a perceptron object 
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

def plot_decision_regions(X,y,classifier, resolution = 0.02):
  #setup marker generator and color map 
  markers = ('s','x', 'o', '^','v') # a tuple of variaties of markers 
  colors = ('red','blue','lightgreen', 'grey', 'cyan')#a tuple of variaties of colors 
  cmap = ListedColormap (colors[:len(np.unique(y))])#create color map with colors depinding on the unique classes 
  
   


 
