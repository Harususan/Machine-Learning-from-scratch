# Polynomial regression 
# Features --> polynomial transformation --> Linear Regression --> Label

"""Library Imports"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import random
from sklearn.model_selection import train_test_split

"""Importing class LinReg from linear regression module."""
class LinReg(object):
	"""
	Linear Regression Model
	=======================
	y = X@w
	X : A feature matrix
	w : weight vector
	y: label vector
	"""
	def __init__(self):
		self.t0 = 200
		self.t1 = 1000000

	def predict(self,X:np.ndarray) -> np.ndarray:
		"""
		Prediction of output label for the given values.

		Args:
		X: Feature matrix of the given input.

		Returns:
		y : Predicted label vector predicted by the model.
		"""
		y = X @ self.w
		return y 

	def loss(self, X:np.ndarray, y:np.ndarray) -> float:
		"""
		Calculates the loss for a model on known labels.

		Args:
		X: Feature matrix of inputs
		y: Output label vector predicted by model

		Returns:
		Loss
		"""
		e = y - self.predict(X)
		return (1/2)*(np.transpose(e) @ e)

	def rmse(self, X:np.ndarray, y:np.ndarray) -> float:
		"""
		Calculates the loss for a model on known labels.

		Args:
		X: Feature matrix of inputs
		y: Output label vector predicted by model

		Returns:
		Loss
		"""

		return np.sqrt((2/X.shape[0])* self.loss(X,y))

	def fit(self,X:np.ndarray,y:np.ndarray) -> np.ndarray:
		"""
		Estimates parameters of linear regression model with normal equation.

		Args:
		X: Feature matrix for given vector
		y: Output label vector

		Returns:
		Weight vector
		"""
		self.w = np.linalg.pinv(X) @ y
		return self.w 

	def calculate_gradient(self,X:np.ndarray,y:np.ndarray) -> np.ndarray:
		"""
		Calculates gradients of loss function w.r.t weight vector on training set.

		Args:
		X: Feature matrix for training data
		y: Label vector for training data

		Returns:
		A vector of gradients.
		"""

		return np.transpose(X) @ (self.predict(X) - y)

	def update_weights(self, grad:np.ndarray, lr:float) -> np.ndarray:
		"""
		Updates the weights based on the gradient of loss function.

		Weight updates are carried out with the following formula:
		w_new = w_old - lr * grad

		Args:
		grad: gradient pf loss w.r.t w
		lr: learning rate

		Returns:
		Updated weight vector
		"""
		return (self.w - lr*grad)

	def learning_schedule(self,t):
		return self.t0/(t + self.t1)


	def gd(self, X:np.ndarray,y:np.ndarray,num_epochs:int,lr:float) -> np.ndarray:
		"""
		Estimates parameters of linear regression through gradient descent algorithm.

		Args:
		X: Feature matrix for training data
		y: label vector for trainig data
		num_epochs: Number of training steps
		lr: Learing rate

		Returns:
		Weight vector: Final weight vector.
		"""
		self.w = np.zeros((X.shape[1]))
		self.w_all = []
		self.arr_all = []
		for i in np.arange(0,num_epochs):
			dJdW = self.calculate_gradient(X,y)
			self.w_all.append(self.w)
			self.arr_all.append(self.loss(X,y))
			self.w = self.update_weights(dJdW, lr)

		return self.w

	def mbgd(self, X:np.ndarray,y:np.ndarray,num_epochs:int,batch_size:int):
		"""
		Estimates parameters of linear regression through gradient descent algorithm.

		Args:
		X: Feature matrix for training data
		y: label vector for trainig data
		num_epochs: Number of training steps
		batch_size: Number of samples in a batch.

		Returns:
		Weight vector: Final weight vector.
		"""
		self.w = np.zeros((X.shape[1]))
		self.w_all = []
		self.err_all = []
		mini_batch_id = 0

		for epoch in range(num_epochs):
			shuffled_indices = np.random.permutation(X.shape[0])
			X_shuffled = X[shuffled_indices]
			y_shuffled = y[shuffled_indices]
			for i in range(0, X.shape[0]):
				mini_batch_id+=1
				xi = X_shuffled[i:i+batch_size]
				yi = y_shuffled[i:i+batch_size]

				self.w_all.append(self.w)
				self.err_all.append(self.loss(xi,yi))

				dJdW = 2/batch_size * self.calculate_gradient(xi,yi)
				self.w = self.update_weights(dJdW,self.learning_schedule(mini_batch_id))

		return self.w 	

	def sgd(self, X:np.ndarray,y:np.ndarray,num_epochs:int):
		"""
		Estimates parameters of linear regression through gradient descent algorithm.

		Args:
		X: Feature matrix for training data
		y: label vector for trainig data
		num_epochs: Number of training steps
		batch_size: Number of samples in a batch.

		Returns:
		Weight vector: Final weight vector.
		"""
		self.w = np.zeros((X.shape[1]))
		self.w_all = []
		self.err_all = []

		for epoch in range(num_epochs):
			for i in range(0, X.shape[0]):
				random_index = np.random.randint(X.shape[0])
				xi = X[random_index:1+random_index]
				yi = y[random_index:1+random_index]

				self.w_all.append(self.w)
				self.err_all.append(self.loss(xi,yi))

				gradients = 2 * self.calculate_gradient(xi,yi)
				lr = self.learning_schedule(epoch*X.shape[0]+i)
				self.w = self.update_weights(gradients,lr)

		return self.w 

# Polynomial transformation.
# Steps -- 1) Generating combination of input features of length = 0,1,...M
#          2) Perform multiplication operation between features to obtaion new features.

import itertools
import functools

def get_combination(x,degree):
    return itertools.combinations_with_replacement(x,degree)

def compute_new_feature(items):
    # reduce (lambda x,y:x*y,[1,2,3,4,5]) calculates ((((1*2)*3)*4)*5)
    return functools.reduce(lambda x,y:x*y,items)

def polynomial_transform(x,degree,logging=False):
    if x.ndim == 1:
        x = x[:,None]

    x_t = x.transpose()
    features = [np.ones(len(x))]

    if logging:
        print("Input:",x)
    for degree in range(1,degree+1):
        for items in get_combination(x_t,degree):
            features.append(compute_new_feature(items))
            if logging:
                print(items,":", compute_new_feature(items))
    if logging:
        print(np.asarray(features).transpose())

    return np.asarray(features).transpose()

# print(polynomial_transform(np.array([2,3]),3,logging=True))

# Non-linear data generation
def create_non_lin_training_set(func,sample_size,std):
    x = np.linspace(0,1,sample_size)
    y = func(x) + np.random.normal(scale=std,size=x.shape)
    return x,y 
def nonlin(x):
    return np.sin(2*np.pi*x)

# Creating non-linear data
num_samples = 100
X,y = create_non_lin_training_set(nonlin,num_samples,0.2)

# Model fitment

# degree1 = 2
# print("degree: ",degree1)
# X_transform = polynomial_transform(X,degree1)
# lin_reg = LinReg()
# lin_reg.fit(X_transform,y)
# print("Weight: ",lin_reg.w)
# print("Loss: ",lin_reg.rmse(X_transform,y))
# print()

# degree2 = 3
# print("degree: ",degree2)
# X_transform = polynomial_transform(X,degree2)
# lin_reg = LinReg()
# lin_reg.fit(X_transform,y)
# print("Weight: ",lin_reg.w)
# print("Loss: ",lin_reg.rmse(X_transform,y))
# print()

# degree3 = 4
# print("degree: ",degree3)
# X_transform = polynomial_transform(X,degree3)
# lin_reg = LinReg()
# lin_reg.fit(X_transform,y)
# print("Weight: ",lin_reg.w)
# print("Loss: ",lin_reg.rmse(X_transform,y))
# print()

# degree4 = 5
# print("degree: ",degree4)
# X_transform = polynomial_transform(X,degree4)
# lin_reg = LinReg()
# lin_reg.fit(X_transform,y)
# print("Weight: ",lin_reg.w)
# print("Loss: ",lin_reg.rmse(X_transform,y))
# print()

# Model Selection
# Process -- 
# 1) Fix the list of degrees you want to experiment with.
# 2) Divide the data into training, validation and test.
# 3) For each degree m -> a) train poynomial regression ,b) calculate traing and validation accuracy.
# Select the model with lowest traing and validation accuracy.







