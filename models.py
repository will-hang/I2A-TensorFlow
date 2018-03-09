import tensorflow.contrib.layers as layers
from tensorflow import nn
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected
import numpy as np

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def natureCNN(images):
	# we apply the basic Nature feature extractor
	conv1_1 = conv2d(images, 32, 8, stride=4)
	conv2_1 = conv2d(conv1_1, 64, 4, stride=2)
	conv3_1 = conv2d(conv2_1, 64, 3, stride=1)
	conv3_1 = layers.flatten(conv3_1)
	hidden_1 = fully_connected(conv3_1, 512)
	# return a 512 dimensional embedding
	return hidden_1

def linear(inputs):
	hidden1 = layers.fully_connected(inputs, 256)
	hidden2 = layers.fully_connected(hidden1, 256)
	hidden3 = layers.fully_connected(hidden2, 128)
	return hidden3

def lstm_cell(hidden_dim):
	return tf.contrib.rnn.BasicLSTMCell(hidden_dim)

def dynamic_rnn(layers, data, hidden_dim):
	stacked = tf.contrib.rnn.MultiRNNCell(
		[lstm_cell(hidden_dim) for _ in range(layers)])
	outputs, state = tf.nn.dynamic_rnn(cell=stacked,
									inputs=data,
									dtype=tf.float32)
	return outputs, state

class Policy():
	def __init__(self, stem, config, scope):
		self.stem = stem
		self.config = config
		self.scope = scope

	def forward(self, x):
			features = self.stem(x)
			features = layers.flatten(features)
			logits = fully_connected(features, self.config.n_actions,
				activation_fn=None,
				weights_initializer=normalized_columns_initializer(0.01),
				biases_initializer=None)
			pi = nn.softmax(logits)
			actions = tf.squeeze(tf.multinomial(logits, 1))
			vf = tf.squeeze(fully_connected(features, 1, 
				activation_fn=None,
				weights_initializer=normalized_columns_initializer(1.0),
				biases_initializer=None))
			return logits, pi, actions, vf