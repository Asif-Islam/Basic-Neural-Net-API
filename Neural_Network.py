import numpy as np 

class Neural_Network(object):

	def __init__(self, batch_size, feature_dim, reg_strength, learning_rate, optimizer):

		'''
		Constructor for the Neural Network

		Inputs:
			batch_size: (Integer) Standard batch_size used for training the neural network; 
			reg_strength: (Float) Regularization strength for L2 Regularization
			learning_rate: (Float) Learning_rate for optimization
			optimizer: (String) Describes the optimizer for the neural network; Choose from
						'sgd', 'rmsprop' or 'adam'

		#################################  EXAMPLE  ######################################

		NN = Neural_Network(BATCH_SIZE, FEATURE_DIM, REG_STRENGTH, LEARNING_RATE 'sgd')

		##################################################################################

		'''
		if (batch_size <= 0) or (feature_dim <= 0):
			raise ValueError('No inputs can be less than or equal to 0');

		self.layers = []
		self.num_layers = 0
		self.batch_size = batch_size
		self.feature_dim = feature_dim
		self.reg_strength = reg_strength
		self.learning_rate = learning_rate
		self.optimizer = optimizer
		self.loss_history = []

	def AddLayer(self, layer, **kwargs):

		'''
		Appends a layer object into the neural network

		Inputs:
			layer: (String) Type of layer to be added;
			**kwargs: Additional arguments to instantiate the layer, dependent on the type chosen

		#################################  EXAMPLE  ######################################

		NN.AddLayer('Affine', {neuron_dim: NEURON_DIM}) ; => NEURON_DIM (Integer)
		NN.AddLayer('ReLU');
		NN.AddLayer('Sigmoid');
		NN.AddLayer('Dropout', {dropout_prob: DROPOUT_PROB}); => DROPOUT_PROB (Float)
		NN.AddLayer('Softmax');

		##################################################################################

		'''

		if (layer == 'Affine'):			#Affine Layer

			neuron_dim = kwargs.pop('neuron_dim', 10);
			verifyValidAddLayer(kwargs)

			if (len(self.layers) == 0):
				prev_layer_neuron_dim = self.feature_dim			
			else:
				prev_layer_neuron_dim = self.layers[-1].neuron_dim
				layers.append(Affine_Layer(neuron_dim, prev_layer_neuron_dim, self.reg_strength, self.learning_rate, self.optimizer))

		elif (layer == 'ReLU'):			#ReLU Activation Layer

			verifyValidAddLayer(kwargs)

			if len(self.layers) == 0:
				raise StandardError('Cannot be placed as the first layer')
			else:
				layers.append(ReLU_Activation(self.layers[-1].neuron_dim))

		elif (layer == 'Sigmoid'):		#Sigmoid Activation Layer

			verifyValidAddLayer(kwargs)

			if len(self.layers) == 0:
				raise StandardError('Cannot be placed as the first layer')
			else:
				layers.append(Sigmoid_Activation(self.layers[-1].neuron_dim))

		elif (layer == 'Dropout'):		#Dropout Layer

			dropout_prob = kwargs.pop('dropout_prob', 0.0)
			verifyValidAddLayer(kwargs)

			if len(self.layers == 0):
				raise StandardError('Cannot be placed as the first layer')
			else:
				layers.append(Dropout_Layer(dropout_prob, self.layers[-1].neuron_dim))

		elif (layer == 'Softmax'):		#Softmax Layer

			verifyValidAddLayer(kwargs)

			if len(self.layers == 0):
				raise StandardError('Cannot be placed as the first layer')
			else:
				layers.append(Softmax_Layer(self.layers[-1].neuron_dim))



	def verifyValidAddLayer(**kwargs):
		"""
		Verify that the call to AddLayer was done so appopriately

		Inputs:
			**kwargs - Dictionary of optional inputs after popping all relevant values

		"""

		if len(kwargs) > 0:
      		extra = ', '.join(''%s'' % k for k in kwargs.keys())
      		raise ValueError('Unrecognized arguments %s' % extra) 

	def forward_propagate(self, mode, X):
		"""
		Calls a forward_propagation through all layers of the neural network

		Inputs:
			mode: 'train' or 'test'
			X: (N, D) Numpy array of the input features into the neural net

		"""

		if (mode != 'train' or mode != 'test'):
			raise ValueError('Unrecognized mode %s', % mode)

		if (mode == 'test' and y is not None):
			raise ValueError('predicted outputs should not be included in test time')

		output = X;
		for layer in self.layers:
			output = layer.forward_pass(output, mode)

		return output 		#PREDICTED SCORES


	def backward_propagate(self, predicted_scores, y):
		"""
		Calls a backward propagation through all layers of the neural for weight tuning

		Inputs:
			predicted_Scores: (N, T) Collection of predicted scores for the training examples,
							  where N is the # of training examples and T is the # of labels

			y: (N,) Expected label output for the N training examples; 
				Only applicable during train time to compute gradients

		"""
		self.layers[-1].backward_pass(predicted_labels, y)
		for layer in reversed(self.layers[:-1]):
			dout = layer.backward_pass(dout)


	def train(self, X, y):
		"""
		Calls forward and back propagation to tune the network's weights

		Inputs:
			X: (N, D) Input feature matrix
			y: (N,) True labels
		"""

		predicted_scores = forward_propagate('train', X)
		backward_propagate(predicted_scores, y)

		#Convert predicted_scores into predicted_labels, i.e. using np.argmax
		accuracy = np.mean(predicted_labels == y)
		return 'running accuracy through mini-batch is %f' % accuracy


	def predict(self, X):
		predicted_scores = forward_propagate('test', X)
		#Convert predicted_scores into predicted_labels, i.e. using np.argmax
		return predicted_labels
