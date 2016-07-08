import numpy as np

class SGD_updater(object):

	def __init__(self, learning_rate,):
		self.learning_rate = learning_rate
	

	def optim_step(self, W, dW):
		W -= self.learning_rate * dW
		return W

