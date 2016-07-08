import numpy as np 
from layers import Layer

class ReLU_Activation(object):

	def __init__(self, neuron_dim):
		self.cache = {}
		self.neuron_dim = neuron_dim

	def forward_pass(self, X, mode):
		out = np.maximum(0, X)
		self.cache['X'] = X

		return out

	def backward_pass(self, dout, mode):
		X = self.cache['X']
		dX = dout
		dX[X <= 0] = 0

		return dX