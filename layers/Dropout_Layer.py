import numpy as np

class Dropout_Layer(object):

	def __init__(self, dropout_prob, neuron_dim):
		self.cache = {}
		self.dropout_prob = dropout_prob
		self.neuron_dim = neuron_dim

	def forward_pass(self, X, mode):
		if (mode == 'train'):
			mask = np.random.randn((*X.shape) < self.dropout_prob) / self.dropout_prob
			out = X * mask
			self.cache['mask'] = mask
			return out
		else:
			return X


	def backward_pass(self, dout, mode):
		if (mode == 'train'):
			mask = self.cache['mask']
			dX = dout
			dX[mask == 0] = 0
			dX /= self.dropout_prob
			return dX
		else:
			return dout

