import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import random
from sklearn.model_selection import train_test_split

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

def generate_data(n,w0,w1):
	"""
	Generates random data with n points.

	Args:
	n: Number of data points
	w1: Value of slope
	w0 : slope of intercept

	Returns:
	(X,y): pair of feature matrix and label vector.
	"""

	X = 10* np.random.randn(n,)
	y = w0 + w1 * X + np.random.randn(n,)
	
	return X,y

def split_data(X,y):
	"""
	Args:
	(X,y): Feature matrix
	Returns:
	(X_train,y_train,X_test,y_test): splitted vectors
	"""
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

	return (X_train,y_train,X_test,y_test)


def add_dummy_feature(x):
	"""
	Adds dummy feature to the dataset.
	Args:
	x: Training dataset
	Returns:
	Training dataset with addition with dummyset.
	"""
	return np.column_stack((np.ones(x.shape[0]),x))

(X,y) = generate_data(100,6,3)
X = add_dummy_feature(X)
(X_train,y_train,X_test,y_test) = split_data(X,y)


# Normal equation
lin_reg = LinReg()
lin_reg.fit(X_train,y_train)
print("Weight by normal method: ",lin_reg.w)
print("Loss by normal method: ",lin_reg.rmse(X,y))
print()

# Using gd
lin_reg.gd(X_train,y_train,1000,lr=1e-4)
print("Weight by gd method: ",lin_reg.w)
print("Loss by gd method: ",lin_reg.rmse(X,y))
print()

# Using mbgd
lin_reg.mbgd(X_train,y_train,1000,16)
print("Weight by mbgd method: ",lin_reg.w)
print("Loss by mbgd method: ",lin_reg.rmse(X,y))
print()

# Using sgd
lin_reg.sgd(X_train,y_train,1000)
print("Weight by sgd method: ",lin_reg.w)
print("Loss by sgd method: ",lin_reg.rmse(X,y))
print()	



