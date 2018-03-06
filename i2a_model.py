# pi.x you need to figure this one out
# DELETE policy.get_initial_features from universe-starter-agent

import tensorflow.contrib.layers as layers
from tensorflow import nn
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected

TEST = True

sess = tf.Session()
		
########## BRANDON'S SECTION ##########
class ImaginationCore(object):
	def __init__(self, config, policy):
		self.policy = policy
		# you need to make the EnvironmentModel class
		self.env_model = EnvironmentModel(config)

		self.states = tf.placeholder(tf.float32, [config.n_actions, *config.state_dims])
		_, _, self.actions, _ = policy.forward(self.states)

	def predict(self, states, init=False):
		# TODO: i give you a batch of states, and you give me the predictions for the next state
		#		if init == True, then each action for each rollout should be fixed, one distinct action per rollout
		#		if init == False, then sample random actions for each rollout
		#		use sess.run to retrieve actions from self.policy
		#		make your own environment model and use it however you want!
		if TEST:
			return np.zeros(states.shape), np.zeros(states.shape[0])

		# imagined_actions = None
		# if init:
		# 	imagined_actions = np.array(range(config.n_actions))
		# else:
		# 	imagined_actions = sess.run(
		# 		[self.actions]
		# 		feed_dict={
		# 			self.states: states
		# 		})

		# action_tile = np.zeros((config.n_actions, *config.state_dims, config.n_actions))
		# for idx, act in enumerate(imagined_actions.tolist()):
		# 	action_tile[idx, :, :, act] = 1
		# stacked = np.concatenate([states, action_tile], axis=-1)

		# imagined_next_states, imagined_rewards = sess.run(
		# 	[self.env_model.pred_state, self.env_model.pred_reward],
		# 	feed_dict={
		# 		self.env_model.x: stacked
		# 	})

		# return imagined_next_states, imagined_rewards

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
	def __init__(self, stem, config, scope):
		self.stem = stem
		self.config = config
		self.scope = scope

	# TODO: i give you a list of states and rewards, and you do an encoding in the reverse order
	#		since this is part of the computation graph, you need to define the inputs to this class as tf.placeholder instead of
	#			function parameters. you also can't return anything, you need to store the result in the object

	# states dims: [-1, n_actions, rollout_length, *state_dims]
	# reward dims: [-1, n_actions, rollout_length]
	# expected output dims:
	# 	encodings: [n_actions, hidden_dim]
	def forward(self, x):
		return tf.get_variable('shit', [1, config.n_actions, config.hidden_dim])
		# with tf.variable_scope(self.scope):
			# x = tf.reshape(x, [-1] + config.state_dims)
			# feats = self.stem(x)
			# feats = tf.reshape(x, [-1, config.n_actions, config.rollout_length])






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

def linear(inputs):
	hidden1 = layers.fully_connected(inputs, 256)
	hidden2 = layers.fully_connected(hidden1, 256)
	hidden3 = layers.fully_connected(hidden2, 128)
	return hidden3

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class Policy():
	def __init__(self, stem, config, scope):
		self.stem = stem
		self.config = config
		self.scope = scope

	def forward(self, x):
		with tf.variable_scope(self.scope):
			features = self.stem(x)
			features = layers.flatten(features)
			logits = fully_connected(features, config.n_actions,
				activation_fn=None,
				weights_initializer=normalizedColumnsInitializer(0.01),
				biases_initializer=None)
			pi = nn.softmax(logits)
			actions = tf.squeeze(tf.multinomial(logits, 1))
			vf = tf.squeeze(fully_connected(features, 1, 
				activation_fn=None,
				weights_initializer=normalizedColumnsInitializer(1.0),
				biases_initializer=None))
			return logits, pi, actions, vf

class I2A(object):
	def __init__(self, config, scope):
		self.config = config
		self.scope = scope
		self.encoder = Encoder(natureCNN, config, 'worker')
		self.mf_policy = Policy(natureCNN, config, 'worker')
		# TODO: write the linear policy function class
		self.i2a_policy = Policy(linear, config, 'worker')
		self.rollout_policy = Policy(natureCNN, config, 'worker')
		self.imagination_core = ImaginationCore(config, self.rollout_policy)
		# for universe-starter-agent
		self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
		# build some graph nodes
		self.inputs_s = tf.placeholder(tf.float32, [-1] + [config.n_actions] + [config.rollout_length] + config.state_dims)
		self.inputs_r = tf.placeholder(tf.float32, [-1] + [config.n_actions] + [config.rollout_length])
		self.states = tf.placeholder(tf.float32, [-1] + config.state_dims)

		encodings = self.encoder.forward(self.inputs_s)
		mf_logits = self.mf_policy.forward(self.states)

		aggregate = tf.reshape(encodings, shape=[-1] + [config.n_actions, config.hidden_dim])
		# we can experiment with this next line of code
		# you can either concat, add, or multiply
		i2a_inputs = tf.concat([aggregate, self.mf_policy.logits], axis=[-1])

		self.logits, i2a_pi, i2a_actions, self.vf = self.i2a_policy.forward(i2a_inputs)
		rp_logits, rp_pi, rp_actions, rp_vf = self.rollout_policy.forward(self.states)

		# self.aux_policy_loss = tf.nn.sparse_softmax_with_logits(labels=i2a_actions, logits=rp_logits)
		# self.loss = None

		# you need to do these things:
		# 	make sure that the rollout policy is defined in i2a and passed into imagination core
		#		during training, run actions through the rollout policy and set the loss to negative log likelihood with
		#		the i2a policy

	def act(self, state):
		# i2a gets a 1-batch of states
		# [84, 84, 4]
		rollouts_s, rollouts_r = self.rollout(state)
		sess.run(
			[self.logits, self.value],
			feed_dict={
				self.inputs_s: rollouts_s,
				self.inputs_r: rollouts_r
			})

	def rollout(self, state):
		# initialize state for batch rollouts
		# this is of shape [n_actions, 84, 84, 4]
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
		# you now have something of [rollout_length, n_actions, 84, 84, 4]
		# TODO: we need to fully process these rollouts before we can pass them into the encoder
		#		process the rollouts so that they can be accepted by an LSTM
		#		the current shape is [n_actions, rollout_length, *observation_dim]
		return rollouts_s, rollouts_r

	# def train(self):
		# we train on monster batches of [-1, rollout_length, n_actions, 84, 84, 4]


class config():
	n_actions = 6
	state_dims = [84, 84, 4]
	rollout_length = 15
	hidden_dim = 64

config = config()

i2a = I2A(config, 'worker')

state = np.ones((1, 84, 84, 4))
i2a.act(state)
