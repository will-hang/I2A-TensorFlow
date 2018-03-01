# pi.x you need to figure this one out
# DELETE policy.get_initial_features from universe-starter-agent

import tensorflow.contrib.layers as layers
from tensorflow.contrib.layers import conv2d, fully_connected

class I2A(object):
	def __init__(self, config, scope):
		self.config = config
		self.scope = scope
		self.imagination_core = ImaginationCore(config)
		self.encoder = Encoder(config)
		self.mf_policy = Policy(natureCNN, config)
		# TODO: write the linear policy function class
		self.i2a_policy = Policy(linear, config)

		# for universe-starter-agent
		self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

	def act(self, state):
		rollouts_s, rollouts_r = self.rollout(state)
		sess.run(
			[self.logits, self.value],
			feed_dict={
				self.inputs_s: rollouts_s,
				self.inputs_r: rollouts_r
			})

	def rollout(self, state):
		# initialize state for batch rollouts
		states = np.stack([state] * self.config.n_actions, axis=0)
		# roll everything out and put it in a placeholder
		rollouts_s = [states]
		rollouts_r = []
		for i in range(self.config.rollout_length):
			# on the first timestep, each rollout will take its own action
			# afterwards, each rollout will take randomly sampled actions
			next_states, rewards = self.imagination_core.predict(states, init=i == 0)
			# add to our rollout
			rollouts_s.append(next_states)
			rollouts_r.append(rewards)
			# advance
			states = next_states
		# TODO: we need to fully process these rollouts before we can pass them into the encoder
		#		process the rollouts so that they can be accepted by an LSTM
		#		the current shape is [n_actions, rollout_length, *observation_dim]
		return rollouts_s, rollouts_r

	def build_graph(self):
		config = self.config
		self.inputs_s = tf.placeholder(tf.float32, [config.n_actions] + [config.rollout_length] + config.input_dims)
		self.inputs_r = tf.placeholder(tf.float32, [config.n_actions] + [config.rollout_length])
		self.state = tf.placeholder(tf.float32, [1] + config.input_dims)
		rollout_encodings = sess.run(
			[self.encoder.encodings],
			feed_dict={
				self.encoder.inputs_s: self.inputs_s,
				self.encoder.inputs_r: self.inputs_r
			})
		# output here should be [n_actions, hidden_dim], so we should aggregate (flatten completely)
		aggregate = tf.reshape(rollout_encodings, shape=[-1])
		# aggregate should be shape [-1]
		mf_logits = sess.run(
			[self.mf_policy.logits],
			feed_dict={
				self.mf_policy.inputs: self.state
			})[0]
		# output here should be [-1]
		inputs = tf.concat([aggregate, mf_logits], axis=[-1])
		self.logits, self.value = sess.run(
			[self.i2a_policy.logits, self.i2a_policy.vf],
			feed_dict={
				self.i2a_policy.inputs: inputs
			})[0]

########## BRANDON'S SECTION ##########
class ImaginationCore(object):
	def __init__(self, config):
		self.policy = Policy(natureCNN, config)
		# you need to make the EnvironmentModel class
		self.env_model = EnvironmentModel(config)

	def predict(self, states, init=False):
		# TODO: i give you a batch of states, and you give me the predictions for the next state
		#		if init == True, then each action for each rollout should be fixed, one distinct action per rollout
		#		if init == False, then sample random actions for each rollout
		#		use sess.run to retrieve actions from self.policy
		#		make your own environment model and use it however you want!

		# states dims: [n_actions, *state_dims]
		# expected output dims: 
		# 	next_state: [n_actions, *state_dims]
		#	reward: [n_actions]

class EnvironmentModel(object):
	def __init__(self, config):
		pass

########## END BRANDON'S SECTION ##########

########## KEVIN'S SECTION ##########

class Encoder(object):
	def __init__(self, config):
		pass

	# TODO: i give you a list of states and rewards, and you do an encoding in the reverse order
	#		since this is part of the computation graph, you need to define the inputs to this class as tf.placeholder instead of
	#			function parameters. you also can't return anything, you need to store the result in the object

	# states dims: [n_actions, rollout_length, *state_dims]
	# reward dims: [n_actions, rollout_length]
	# expected output dims:
	# 	encodings: [n_actions, hidden_dim]

########## END KEVIN'S SECTION ##########

def natureCNN(images):
	# we apply the basic Nature feature extractor
	conv1_1 = conv2d(images, 32, 8, stride=4)
	conv2_1 = conv2d(conv1_1, 64, 4, stride=2)
	conv3_1 = conv2d(conv2_1, 64, 3, stride=1)
	conv3_1 = layers.flatten(conv3_1)
	hidden_1 = fully_connected(conv3_1, 512)
	# return a 512 dimensional embedding
	return hidden_1

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class Policy():
	def __init__(self, stem, config):
		self.inputs = tf.placeholder(tf.float32, [None] + config.input_dims)
		with tf.variable_scope(config.scope):
			features = stem(self.inputs)
			features = layers.flatten(features)
			self.logits = fully_connected(features, config.output_dims,
				activation_fn=None,
				weights_initializer=normalizedColumnsInitializer(0.01),
				biases_initializer=None)
			self.pi = nn.softmax(self.logits)
			self.actions = tf.squeeze(tf.multinomial(self.logits, 1))
			self.vf = tf.squeeze(fully_connected(outputs, 1, 
				activation_fn=None,
				weights_initializer=normalizedColumnsInitializer(1.0),
				biases_initializer=None))