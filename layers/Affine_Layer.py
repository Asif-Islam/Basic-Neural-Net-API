import numpy as np
from layers import Layer

class Affine_Layer(object):

	def __init__(self, neuron_dim, prev_layer_neuron_dim, reg_strength, learning_rate, optim):
		"""
		Inputs:
			neuron_dim: (Integer) Number of neurons in the layer
			prev_layer_neuron_dim: (Integer) Number of neurons in the previous layer
			reg_strength: (Float) Regularization strength for L2 regularization
			learning_rate: (Float) Learning_rate for optimization
			optimizer: (String)  Describes the optimizer for the neural network;

		"""

		self.cache = {}
		self.weight = np.random.randn(prev_layer_neuron_dim, neuron_dim) / np.sqrt(prev_layer_neuron_dim);
		self.bias = np.zeros(neuron_dim)
		self.reg_strength = reg_strength
		
		#Instantiate an optimizer class based what optimizer is
		if optim == 'sgd':
			optimizer = SGD_updater()
		elif optim == 'adam':
			optimizer = ADAM_updater()
		elif optimizer = 'rmsprop':
			optimizer = RMSPROP_updater()
		else:
			raise ValueError('Requested optimizer was not found')


	def forward_pass(self, X):
		"""
		Inputs:
			X: (N, D) Numpy array of the input into the layer through forward propagation

		"""

		output = X.dot(self.weight) + self.bias
		self.cache['X'] = X

		return output


	def backward_pass(self, dout):
		"""
		Inputs:
			dout: (N, M) Numpy array of the gradient of the loss wrt the forward output 

		"""

		X = cache['X'];
		dX = dout.dot(self.weight.T)
		dW = X.T.dot(dout)
		db = np.sum(dout, axis=0)

		#Call optimizer on dW and db


		return dX
