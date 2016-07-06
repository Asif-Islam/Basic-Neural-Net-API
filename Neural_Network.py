import numpy as np 

class Neural_Network(object):

	def __init__(self):
		self.layers = []
		self.num_layers = 0
		self.batch_size = 0


	def AddLayer(self, layer, **kwargs):
		#layer is a string = the type of layer that can be placed
		#kwargs is any number of arguments needed for that given layer; I.e. dropout probability

		#Do a switch statement over the different types of layers

		pass

	def forward_propagate(self):
		#Loop over self.layers and call that layer's forward_pass function
		pass

	def backward_propagate(self):
		#Loop over self.layers in reverse and call that layer's backward_pass function
		pass

	def train(self, X, y):
		pass

	def predict(self, X):
		pass
